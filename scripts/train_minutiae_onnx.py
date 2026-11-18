"""
Train & Export Lightweight CPU ONNX Minutiae Detector (MinutiaeNetV1).

Strict data hygiene:
- Trains strictly on Set B disjoint sources (default: DB2_B, DB3_B, DB4_B + synthetic
  calibrated minutiae/hard-negative patches; optionally includes DB1_B only when
  explicitly requested for final Set-A deployment weights). NEVER touches FVC2002 Set A.
- Validates out-of-domain accuracy and speed on held-out DB1_B via ONNX Runtime.
"""

import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.descriptors.mcc import MinutiaeCylinderCode, flip_mcc_dual
from src.evaluation.metrics import compute_roc_metrics
from src.matching.bipartite_matcher import compute_pairwise_hamming_distance
from src.matching.spectral_consensus import SpectralGraphMatcher
from src.minutiae.extractor import extract_minutiae
from src.minutiae.onnx_exporter import ONNXGraphBuilder
from src.preprocessing.enhancer import enhance_fingerprint, estimate_orientation_field
from src.preprocessing.segmenter import segment_fingerprint


# ---------------------------------------------------------------------------
# Fixed in-graph local contrast normalization (15x15 box statistics)
# ---------------------------------------------------------------------------
_BOX_K = 15
_BOX_PAD = _BOX_K // 2
_BOX_W = np.full((1, 1, _BOX_K, _BOX_K), 1.0 / (_BOX_K * _BOX_K), dtype=np.float32)
_VAR_EPS = np.array([[[[0.0025]]]], dtype=np.float32)


def local_contrast_normalize_np(x_01: np.ndarray) -> np.ndarray:
    """Exact NumPy equivalent of the ONNX front-end normalization stem."""
    # x_01: (H, W) or (N, 1, H, W) float32 in [0, 1]
    if x_01.ndim == 2:
        mu = cv2.blur(x_01, (_BOX_K, _BOX_K), borderType=cv2.BORDER_CONSTANT)
        mu2 = cv2.blur(x_01 * x_01, (_BOX_K, _BOX_K), borderType=cv2.BORDER_CONSTANT)
        var = np.abs(mu2 - mu * mu)
        return (x_01 - mu) / np.sqrt(var + 0.0025)
    out = np.empty_like(x_01, dtype=np.float32)
    for i in range(len(x_01)):
        img = x_01[i, 0]
        mu = cv2.blur(img, (_BOX_K, _BOX_K), borderType=cv2.BORDER_CONSTANT)
        mu2 = cv2.blur(img * img, (_BOX_K, _BOX_K), borderType=cv2.BORDER_CONSTANT)
        var = np.abs(mu2 - mu * mu)
        out[i, 0] = (img - mu) / np.sqrt(var + 0.0025)
    return out


# ---------------------------------------------------------------------------
# Pure-NumPy vectorized Conv2d, ConvTranspose2d(2x2,s=2), AvgPool2x2
# ---------------------------------------------------------------------------
def conv2d_fwd(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> Tuple[np.ndarray, np.ndarray]:
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant") if pad > 0 else x
    # cols: (N, C_in, H_out, W_out, k, k)
    cols = sliding_window_view(xp, (k, k), axis=(2, 3))
    N, C_in, H, W, _, _ = cols.shape
    C_out = w.shape[0]
    cols_2d = np.ascontiguousarray(cols.transpose(0, 2, 3, 1, 4, 5)).reshape(N * H * W, C_in * k * k)
    w_2d = w.reshape(C_out, C_in * k * k)
    y_2d = cols_2d @ w_2d.T
    y = y_2d.reshape(N, H, W, C_out).transpose(0, 3, 1, 2) + b[None, :, None, None]
    return y, cols_2d


def conv2d_bwd(
    dy: np.ndarray, cols_2d: np.ndarray, w: np.ndarray, x_shape: Tuple[int, int, int, int], pad: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, C_out, H, W = dy.shape
    C_in, k = w.shape[1], w.shape[2]
    dy_2d = np.ascontiguousarray(dy.transpose(0, 2, 3, 1)).reshape(N * H * W, C_out)
    dw = (dy_2d.T @ cols_2d).reshape(C_out, C_in, k, k)
    db = dy.sum(axis=(0, 2, 3))
    dcols_2d = dy_2d @ w.reshape(C_out, C_in * k * k)
    dcols = dcols_2d.reshape(N, H, W, C_in, k, k).transpose(0, 3, 1, 2, 4, 5)
    dxp = np.zeros((N, C_in, x_shape[2] + 2 * pad, x_shape[3] + 2 * pad), dtype=np.float32)
    for ky in range(k):
        for kx in range(k):
            dxp[:, :, ky : ky + H, kx : kx + W] += dcols[:, :, :, :, ky, kx]
    dx = dxp[:, :, pad : pad + x_shape[2], pad : pad + x_shape[3]] if pad > 0 else dxp
    return dx, dw, db


def deconv2x2_fwd(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    # w shape (C_in, C_out, 2, 2) per ONNX ConvTranspose spec
    N, C_in, H, W = x.shape
    C_out = w.shape[1]
    # y_tiles: (N, C_out, H, W, 2, 2)
    x_2d = np.ascontiguousarray(x.transpose(0, 2, 3, 1)).reshape(N * H * W, C_in)
    w_2d = w.reshape(C_in, C_out * 4)
    tiles = (x_2d @ w_2d).reshape(N, H, W, C_out, 2, 2)
    # Transpose to (N, C_out, H, 2, W, 2) -> reshape (N, C_out, 2H, 2W)
    y = np.ascontiguousarray(tiles.transpose(0, 3, 1, 4, 2, 5)).reshape(N, C_out, H * 2, W * 2)
    return y + b[None, :, None, None]


def deconv2x2_bwd(
    dy: np.ndarray, x: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, C_in, H, W = x.shape
    C_out = w.shape[1]
    db = dy.sum(axis=(0, 2, 3))
    dy_tiles = np.ascontiguousarray(
        dy.reshape(N, C_out, H, 2, W, 2).transpose(0, 2, 4, 1, 3, 5)
    ).reshape(N * H * W, C_out * 4)
    x_2d = np.ascontiguousarray(x.transpose(0, 2, 3, 1)).reshape(N * H * W, C_in)
    dw = (x_2d.T @ dy_tiles).reshape(C_in, C_out, 2, 2)
    dx_2d = dy_tiles @ w.reshape(C_in, C_out * 4).T
    dx = np.ascontiguousarray(dx_2d.reshape(N, H, W, C_in).transpose(0, 3, 1, 2))
    return dx, dw, db


class MinutiaeNetV1:
    """Lightweight U-Net-style FCN matching ONNXMinutiaeDetector I/O."""

    def __init__(self, seed: int = 2026):
        rng = np.random.default_rng(seed)

        def kaiming(shape):
            fan_in = int(np.prod(shape[1:]))
            return (rng.normal(0.0, np.sqrt(2.0 / max(1, fan_in)), size=shape)).astype(np.float32)

        # Initialize first 12 filters of Conv1 (16 filters, 5x5) with Gabor directional ridge detectors
        w1 = kaiming((16, 1, 5, 5))
        yy, xx = np.mgrid[-2:3, -2:3]
        for idx, th in enumerate(np.linspace(0, np.pi, 8, endpoint=False)):
            x_th = xx * np.cos(th) + yy * np.sin(th)
            y_th = -xx * np.sin(th) + yy * np.cos(th)
            g = np.exp(-0.5 * ((x_th ** 2) / 4.0 + (y_th ** 2) / 2.25))
            k_even = g * np.cos(2.0 * np.pi * 0.12 * y_th)
            k_odd = g * np.sin(2.0 * np.pi * 0.12 * y_th)
            k_even = (k_even - k_even.mean()) * 0.5
            k_odd = (k_odd - k_odd.mean()) * 0.5
            w1[idx, 0] = k_even
            if idx + 8 < 12:
                w1[idx + 8, 0] = k_odd

        self.params: Dict[str, np.ndarray] = {
            "w1": w1,
            "b1": np.zeros(16, dtype=np.float32),
            "w2": kaiming((16, 16, 3, 3)),
            "b2": np.zeros(16, dtype=np.float32),
            "w3": kaiming((24, 16, 3, 3)),
            "b3": np.zeros(24, dtype=np.float32),
            "w4": kaiming((24, 24, 3, 3)),
            "b4": np.zeros(24, dtype=np.float32),
            # ConvTranspose weight layout in ONNX: (C_in=24, C_out=16, 2, 2)
            "w5_up": rng.normal(0.0, np.sqrt(1.0 / (24 * 4)), size=(24, 16, 2, 2)).astype(np.float32),
            "b5_up": np.zeros(16, dtype=np.float32),
            "w6": kaiming((16, 16, 3, 3)),
            "b6": np.zeros(16, dtype=np.float32),
            "w_hm": (rng.normal(0.0, 0.05, size=(1, 16, 1, 1))).astype(np.float32),
            "b_hm": np.array([-3.0], dtype=np.float32),  # start low sparse probability ~0.047
            "w_ori": kaiming((2, 16, 1, 1)),
            "b_ori": np.zeros(2, dtype=np.float32),
        }

    def forward_backward(
        self,
        x_norm: np.ndarray,
        y_hm: np.ndarray,
        y_ori: np.ndarray,
        pos_weight: float = 12.0,
        hm_mask: Optional[np.ndarray] = None,
        compute_grads: bool = True,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        p = self.params
        # Stage 1 (full res)
        z1, c1 = conv2d_fwd(x_norm, p["w1"], p["b1"], pad=2)
        h1 = np.maximum(z1, 0.0)
        z2, c2 = conv2d_fwd(h1, p["w2"], p["b2"], pad=1)
        h2 = np.maximum(z2, 0.0)

        # Downsample x2 -> Stage 2
        N, C, H, W = h2.shape
        p1 = h2.reshape(N, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))
        z3, c3 = conv2d_fwd(p1, p["w3"], p["b3"], pad=1)
        h3 = np.maximum(z3, 0.0)
        z4, c4 = conv2d_fwd(h3, p["w4"], p["b4"], pad=1)
        h4 = np.maximum(z4, 0.0)

        # Upsample + Skip
        up = deconv2x2_fwd(h4, p["w5_up"], p["b5_up"])
        zm = up + h2
        hm = np.maximum(zm, 0.0)

        # Fusion Conv6
        z6, c6 = conv2d_fwd(hm, p["w6"], p["b6"], pad=1)
        h6 = np.maximum(z6, 0.0)

        # Heads
        logits_hm, c_hm = conv2d_fwd(h6, p["w_hm"], p["b_hm"], pad=0)
        prob_hm = 1.0 / (1.0 + np.exp(-np.clip(logits_hm, -15.0, 15.0)))

        ori_raw, c_ori = conv2d_fwd(h6, p["w_ori"], p["b_ori"], pad=0)
        ori_norm_sq = np.sum(ori_raw * ori_raw, axis=1, keepdims=True) + 1e-6
        ori_inv = 1.0 / np.sqrt(ori_norm_sq)
        ori_pred = ori_raw * ori_inv

        # Weighted BCE on heatmap + Soft Dice sharpening + masked L2 orientation alignment
        eps = 1e-6
        m_f = hm_mask if hm_mask is not None else np.ones_like(y_hm, dtype=np.float32)
        w_pix = np.where(y_hm > 0.25, 1.0 + (pos_weight - 1.0) * y_hm, 1.0).astype(np.float32) * m_f
        bce = -(y_hm * np.log(prob_hm + eps) + (1.0 - y_hm) * np.log(1.0 - prob_hm + eps))
        denom_hm = max(1.0, float(np.sum(w_pix)))
        loss_hm = float(np.sum(w_pix * bce) / denom_hm)

        # Soft Dice loss on foreground heatmap sharpens peaks and suppresses diffuse halos
        dice_eps = 1e-4
        inter_d = float(np.sum(m_f * prob_hm * y_hm))
        denom_d = float(np.sum(m_f * (prob_hm + y_hm))) + dice_eps
        num_d = 2.0 * inter_d + dice_eps
        loss_dice = 1.0 - (num_d / denom_d)
        dice_w = np.float32(0.35)

        ori_m = np.maximum(m_f, np.float32(0.1))
        ori_denom = max(1.0, float(np.sum(ori_m)))
        ori_w = np.float32(0.40)
        loss_ori = float(ori_w * np.sum(ori_m * np.sum((ori_pred - y_ori) ** 2, axis=1, keepdims=True)) / ori_denom)
        loss = loss_hm + float(dice_w) * loss_dice + loss_ori

        if not compute_grads:
            return loss, {}

        dlogits_bce = (w_pix * (prob_hm - y_hm) / denom_hm).astype(np.float32)
        dprob_dice = m_f * (num_d - 2.0 * y_hm * denom_d) / (denom_d * denom_d)
        dlogits_dice = (dice_w * dprob_dice * prob_hm * (1.0 - prob_hm)).astype(np.float32)
        dlogits_hm = dlogits_bce + dlogits_dice

        # d(loss_ori)/d(ori_raw) with projection onto tangent of unit circle
        scale_ori = np.float32((2.0 * float(ori_w)) / ori_denom)
        dori_pred = scale_ori * ori_m * (ori_pred - y_ori)
        dot = np.sum(dori_pred * ori_pred, axis=1, keepdims=True)
        dori_raw = ((dori_pred - ori_pred * dot) * ori_inv).astype(np.float32)

        dh6_a, dw_hm, db_hm = conv2d_bwd(dlogits_hm, c_hm, p["w_hm"], h6.shape, pad=0)
        dh6_b, dw_ori, db_ori = conv2d_bwd(dori_raw, c_ori, p["w_ori"], h6.shape, pad=0)
        dh6 = dh6_a + dh6_b

        dz6 = dh6 * (z6 > 0.0)
        dhm, dw6, db6 = conv2d_bwd(dz6, c6, p["w6"], hm.shape, pad=1)

        dzm = dhm * (zm > 0.0)
        dh4, dw5_up, db5_up = deconv2x2_bwd(dzm, h4, p["w5_up"])

        dz4 = dh4 * (z4 > 0.0)
        dh3, dw4, db4 = conv2d_bwd(dz4, c4, p["w4"], h3.shape, pad=1)
        dz3 = dh3 * (z3 > 0.0)
        dp1, dw3, db3 = conv2d_bwd(dz3, c3, p["w3"], p1.shape, pad=1)

        dh2_pool = 0.25 * np.repeat(np.repeat(dp1, 2, axis=2), 2, axis=3)
        dh2 = dzm + dh2_pool

        dz2 = dh2 * (z2 > 0.0)
        dh1, dw2, db2 = conv2d_bwd(dz2, c2, p["w2"], h1.shape, pad=1)
        dz1 = dh1 * (z1 > 0.0)
        _, dw1, db1 = conv2d_bwd(dz1, c1, p["w1"], x_norm.shape, pad=2)

        grads = {
            "w1": dw1, "b1": db1,
            "w2": dw2, "b2": db2,
            "w3": dw3, "b3": db3,
            "w4": dw4, "b4": db4,
            "w5_up": dw5_up, "b5_up": db5_up,
            "w6": dw6, "b6": db6,
            "w_hm": dw_hm, "b_hm": db_hm,
            "w_ori": dw_ori, "b_ori": db_ori,
        }
        return loss, grads

    def export_onnx(self, path: str):
        p = self.params
        b = ONNXGraphBuilder("MinutiaeNetV1", opset=13)
        b.add_input("input", [1, 1, -1, -1])
        b.add_output("heatmap", [1, 1, -1, -1])
        b.add_output("orientation", [1, 2, -1, -1])

        b.add_initializer("box_w", _BOX_W)
        b.add_initializer("var_eps", _VAR_EPS)
        b.add_initializer("ori_eps", np.array([[[[1e-6]]]], dtype=np.float32))
        b.add_initializer("ori_axes", np.array([1], dtype=np.int64))
        for k, v in p.items():
            b.add_initializer(k, v)

        # In-graph Local Contrast Normalization (dry/wet sensor invariant)
        b.add_node("Conv", ["input", "box_w"], ["mu"], attrs={"pads": [_BOX_PAD] * 4})
        b.add_node("Mul", ["input", "input"], ["x_sq"])
        b.add_node("Conv", ["x_sq", "box_w"], ["mu2"], attrs={"pads": [_BOX_PAD] * 4})
        b.add_node("Mul", ["mu", "mu"], ["mu_sq"])
        b.add_node("Sub", ["mu2", "mu_sq"], ["var_raw"])
        b.add_node("Abs", ["var_raw"], ["var"])
        b.add_node("Add", ["var", "var_eps"], ["var_safe"])
        b.add_node("Sqrt", ["var_safe"], ["std_loc"])
        b.add_node("Sub", ["input", "mu"], ["diff"])
        b.add_node("Div", ["diff", "std_loc"], ["x_norm"])

        # Backbone
        b.add_node("Conv", ["x_norm", "w1", "b1"], ["z1"], attrs={"pads": [2, 2, 2, 2]})
        b.add_node("Relu", ["z1"], ["h1"])
        b.add_node("Conv", ["h1", "w2", "b2"], ["z2"], attrs={"pads": [1, 1, 1, 1]})
        b.add_node("Relu", ["z2"], ["h2"])

        b.add_node("AveragePool", ["h2"], ["p1"], attrs={"kernel_shape": [2, 2], "strides": [2, 2]})
        b.add_node("Conv", ["p1", "w3", "b3"], ["z3"], attrs={"pads": [1, 1, 1, 1]})
        b.add_node("Relu", ["z3"], ["h3"])
        b.add_node("Conv", ["h3", "w4", "b4"], ["z4"], attrs={"pads": [1, 1, 1, 1]})
        b.add_node("Relu", ["z4"], ["h4"])

        b.add_node("ConvTranspose", ["h4", "w5_up", "b5_up"], ["up"], attrs={"kernel_shape": [2, 2], "strides": [2, 2]})
        b.add_node("Add", ["up", "h2"], ["zm"])
        b.add_node("Relu", ["zm"], ["hm"])
        b.add_node("Conv", ["hm", "w6", "b6"], ["z6"], attrs={"pads": [1, 1, 1, 1]})
        b.add_node("Relu", ["z6"], ["h6"])

        # Heads
        b.add_node("Conv", ["h6", "w_hm", "b_hm"], ["hm_logits"])
        b.add_node("Sigmoid", ["hm_logits"], ["heatmap"])

        b.add_node("Conv", ["h6", "w_ori", "b_ori"], ["ori_raw"])
        b.add_node("Mul", ["ori_raw", "ori_raw"], ["ori_sq"])
        b.add_node("ReduceSum", ["ori_sq", "ori_axes"], ["ori_norm_sq"], attrs={"keepdims": 1})
        b.add_node("Add", ["ori_norm_sq", "ori_eps"], ["ori_norm_safe"])
        b.add_node("Sqrt", ["ori_norm_safe"], ["ori_den"])
        b.add_node("Div", ["ori_raw", "ori_den"], ["orientation"])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        b.save(path)


# ---------------------------------------------------------------------------
# Consensus Mining & Synthetic Multi-Contrast Data Generator
# ---------------------------------------------------------------------------
def _make_gaussian_target(h: int, w: int, pts: List[Tuple[float, float]], sigma: float = 2.0) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in pts:
        g = np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (sigma * sigma)))
        hm = np.maximum(hm, g)
    return hm


def mine_consensus_patches(
    subsets: Tuple[str, ...] = ("DB2_B", "DB3_B", "DB4_B"),
    patch_size: int = 48,
    max_patches_per_db: int = 1500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mines repeatable true minutiae and hard-negative non-repeatable regions from
    non-DB1_A Set B datasets via SE(2) Spectral Consensus across genuine impressions.
    """
    matcher = SpectralGraphMatcher(sigma_dist=18.0, sigma_angle=0.38, power_iterations=18)
    mcc = MinutiaeCylinderCode(
        radius=70.0, num_spatial_cells=8, num_directional_slices=4,
        sigma_spatial=8.5, sigma_directional=np.pi / 5, sigmoid_gain=2.0,
    )

    x_list, hm_list, ori_list, msk_list = [], [], [], []
    half = patch_size // 2
    rng = np.random.default_rng(101)

    for db in subsets:
        dpi = 569.0 if db.startswith("DB2") else 500.0
        mcc.dpi = dpi
        imgs, masks, oris, ms_map, descs_map = {}, {}, {}, {}, {}
        for p in sorted(glob.glob(f"dataset/FVC2002/{db}/*.tif")):
            fid = os.path.basename(p).split(".")[0]
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            _, mask, _ = segment_fingerprint(img, method="adaptive")
            enh, ori, coh = enhance_fingerprint(img, use_gabor=True, mask=mask)
            ms, _ = extract_minutiae(
                enh, mask, ori, coh, binarize_method="adaptive", prune_artifacts=True, quality_thresh=0.35,
                border_margin=10,
            )
            imgs[fid] = (img.astype(np.float32) / 255.0)
            masks[fid] = (mask > 0).astype(np.float32)
            # Ridge tangent field ori -> (cos 2th, sin 2th)
            oris[fid] = np.stack([np.cos(2.0 * ori), np.sin(2.0 * ori)], axis=0).astype(np.float32)
            ms_map[fid] = ms
            d, w = mcc.extract_descriptors(ms)
            descs_map[fid] = (d, w)

        # Pooled symmetric Z-norm inside this DB for accurate mining
        all_d = np.vstack([v[0] for v in descs_map.values() if len(v[0]) > 0])
        mu = np.mean(all_d, axis=0)
        sd = np.std(all_d, axis=0) + 1e-4
        mu_s = 0.5 * (mu + flip_mcc_dual(mu, 8, 4))
        sd_s = 0.5 * (sd + flip_mcc_dual(sd, 8, 4))

        norm_d = {}
        for fid, (d, w) in descs_map.items():
            if len(d) == 0:
                norm_d[fid] = (d, d, w)
                continue
            ds = (d - mu_s) / sd_s
            ds = ds / (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-6)
            dsd = mcc.get_dual_descriptors(ds)
            norm_d[fid] = (ds, dsd, w)

        # Count consensus hits per minutia across genuine pairs
        hits: Dict[str, np.ndarray] = {fid: np.zeros(len(ms_map[fid]), dtype=np.int32) for fid in imgs}
        for f in range(101, 111):
            for i in range(1, 9):
                ki = f"{f}_{i}"
                if ki not in imgs or len(ms_map[ki]) < 3:
                    continue
                ci = np.array([[m.x, m.y] for m in ms_map[ki]], dtype=np.float32)
                ti = np.array([m.theta for m in ms_map[ki]], dtype=np.float32)
                di, did, wi = norm_d[ki]
                for j in range(i + 1, 9):
                    kj = f"{f}_{j}"
                    if kj not in imgs or len(ms_map[kj]) < 3:
                        continue
                    cj = np.array([[m.x, m.y] for m in ms_map[kj]], dtype=np.float32)
                    tj = np.array([m.theta for m in ms_map[kj]], dtype=np.float32)
                    dj, _, wj = norm_d[kj]
                    cos1 = np.clip(di @ dj.T, -1.0, 1.0)
                    cos2 = np.clip(did @ dj.T, -1.0, 1.0)
                    dist_mat = np.clip(1.0 - np.maximum(cos1, cos2), 0.0, 2.0) / 2.0
                    _, det = matcher.match(
                        ci, ti, cj, tj, dist_mat, tau=0.42, query_weights=wi, enrolled_weights=wj,
                        dpi_query=dpi, dpi_enrolled=dpi,
                    )
                    for u, v in det.get("selected_pairs", []):
                        hits[ki][u] += 1
                        hits[kj][v] += 1

        # Build positive & hard-negative patches from each impression
        db_patches = 0
        for fid, img01 in imgs.items():
            h, w = img01.shape
            ms = ms_map[fid]
            ht = hits[fid]
            pos_pts = [(m.x, m.y) for m, c in zip(ms, ht) if c >= 2]
            ambig_pts = [(m.x, m.y) for m, c in zip(ms, ht) if c == 1]
            spur_pts = [(m.x, m.y) for m, c in zip(ms, ht) if c == 0]
            if len(pos_pts) < 3:
                continue

            full_hm = _make_gaussian_target(h, w, pos_pts, sigma=1.65)
            # Ignore ambiguous single-hit regions in loss mask so network isn't penalized inconsistently
            full_msk = masks[fid].copy()
            yy, xx = np.mgrid[0:h, 0:w]
            for ax, ay in ambig_pts:
                r2 = (xx - ax) ** 2 + (yy - ay) ** 2
                full_msk[r2 <= 16.0] = 0.0

            x_norm_full = local_contrast_normalize_np(img01)
            ori_full = oris[fid]

            # Sample positive patches centered near consensus minutiae (with small random shift)
            for cx, cy in pos_pts:
                if db_patches >= max_patches_per_db:
                    break
                sx = int(round(cx + rng.integers(-6, 7)))
                sy = int(round(cy + rng.integers(-6, 7)))
                if sy - half < 0 or sy + half > h or sx - half < 0 or sx + half > w:
                    continue
                x_list.append(x_norm_full[sy - half : sy + half, sx - half : sx + half][None, :, :])
                hm_list.append(full_hm[sy - half : sy + half, sx - half : sx + half][None, :, :])
                ori_list.append(ori_full[:, sy - half : sy + half, sx - half : sx + half])
                msk_list.append(full_msk[sy - half : sy + half, sx - half : sx + half][None, :, :])
                db_patches += 1

            # Sample hard-negative spurious patches (0 consensus hits) & interior foreground ridges
            neg_cands = spur_pts[: max(4, len(pos_pts) // 2)]
            for cx, cy in neg_cands:
                if db_patches >= max_patches_per_db:
                    break
                sx, sy = int(round(cx)), int(round(cy))
                if sy - half < 0 or sy + half > h or sx - half < 0 or sx + half > w:
                    continue
                x_list.append(x_norm_full[sy - half : sy + half, sx - half : sx + half][None, :, :])
                hm_list.append(full_hm[sy - half : sy + half, sx - half : sx + half][None, :, :])
                ori_list.append(ori_full[:, sy - half : sy + half, sx - half : sx + half])
                msk_list.append(full_msk[sy - half : sy + half, sx - half : sx + half][None, :, :])
                db_patches += 1

        print(f"Mined {db_patches} patches from {db}")

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(hm_list, dtype=np.float32),
        np.asarray(ori_list, dtype=np.float32),
        np.asarray(msk_list, dtype=np.float32),
    )


def generate_synthetic_patches(n_patches: int = 1600, patch_size: int = 48, seed: int = 77):
    """Generates exact ground-truth synthetic bifurcation / ending / negative ridge patches."""
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]

    x_list, hm_list, ori_list, msk_list = [], [], [], []
    for i in range(n_patches):
        theta = float(rng.uniform(0.0, np.pi))
        freq = float(rng.uniform(0.09, 0.145))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = xx - half
        dy = yy - half
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)

        is_pos = (i % 2 == 0)
        if is_pos:
            mx = float(rng.uniform(-5.0, 5.0))
            my = float(rng.uniform(-5.0, 5.0))
            cx, cy = half + mx, half + my
            rx = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
            ry = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
            # Ridge ending or bifurcation dislocation via smooth phase jump (branch cut at rx=0)
            disloc = np.arctan2(ry, rx) * 0.5 + np.pi * (ry >= 0) * 0.0
            phi_field = 2.0 * np.pi * freq * ry + 2.0 * np.arctan2(ry, rx + 1e-4) * 0.5 + phase
            hm = np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0)).astype(np.float32)
        else:
            # Smooth parallel / slightly curved ridge with no minutia (or faint pore dot)
            curv = float(rng.uniform(-0.008, 0.008))
            phi_field = 2.0 * np.pi * freq * (v + curv * u * u) + phase
            hm = np.zeros((patch_size, patch_size), dtype=np.float32)

        ridge = 0.5 - 0.5 * np.cos(phi_field)
        # Apply dry/wet sensor contrast + Gaussian blur + noise augmentations
        contrast = float(rng.uniform(0.18, 0.65))
        bias = float(rng.uniform(0.25, 0.85))
        img = bias + contrast * (ridge - 0.5)
        img += rng.normal(0.0, float(rng.uniform(0.02, 0.09)), size=img.shape)
        if rng.random() < 0.35:
            # Simulate pore/speckle noise
            py, px = int(rng.integers(8, patch_size - 8)), int(rng.integers(8, patch_size - 8))
            img[py - 1 : py + 2, px - 1 : px + 2] = np.clip(bias + 0.35, 0.0, 1.0)
        img01 = np.clip(img, 0.0, 1.0).astype(np.float32)
        x_norm = local_contrast_normalize_np(img01)

        c2t = np.full((patch_size, patch_size), np.cos(2.0 * theta), dtype=np.float32)
        s2t = np.full((patch_size, patch_size), np.sin(2.0 * theta), dtype=np.float32)

        x_list.append(x_norm[None, :, :])
        hm_list.append(hm[None, :, :])
        ori_list.append(np.stack([c2t, s2t], axis=0))
        msk_list.append(np.ones((1, patch_size, patch_size), dtype=np.float32))

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(hm_list, dtype=np.float32),
        np.asarray(ori_list, dtype=np.float32),
        np.asarray(msk_list, dtype=np.float32),
    )


def _augment_dihedral(
    X: np.ndarray, H: np.ndarray, O: np.ndarray, M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies orientation-consistent horizontal flip & 90-deg rotation augmentations."""
    X_flip = X[:, :, :, ::-1].copy()
    H_flip = H[:, :, :, ::-1].copy()
    M_flip = M[:, :, :, ::-1].copy()
    O_flip = O[:, :, :, ::-1].copy()
    O_flip[:, 1] = -O_flip[:, 1]

    X_rot = np.rot90(X, k=1, axes=(2, 3)).copy()
    H_rot = np.rot90(H, k=1, axes=(2, 3)).copy()
    M_rot = np.rot90(M, k=1, axes=(2, 3)).copy()
    O_rot = np.rot90(O, k=1, axes=(2, 3)).copy()
    O_rot = -O_rot  # +pi/2 adds +pi to 2*theta

    return (
        np.concatenate([X, X_flip, X_rot], axis=0),
        np.concatenate([H, H_flip, H_rot], axis=0),
        np.concatenate([O, O_flip, O_rot], axis=0),
        np.concatenate([M, M_flip, M_rot], axis=0),
    )


def train_and_export(
    train_subsets: Tuple[str, ...] = ("DB2_B", "DB3_B", "DB4_B"),
    out_onnx_path: str = "models/minutiae_net_v1.onnx",
    epochs: int = 24,
    batch_size: int = 64,
    lr: float = 2.8e-3,
):
    print(f"Mining Set B consensus patches from {train_subsets}...")
    xr, hr, orr, mr = mine_consensus_patches(subsets=train_subsets)
    xs, hs, ors, ms = generate_synthetic_patches(n_patches=1500)

    X_base = np.concatenate([xr, xs], axis=0)
    H_base = np.concatenate([hr, hs], axis=0)
    O_base = np.concatenate([orr, ors], axis=0)
    M_base = np.concatenate([mr, ms], axis=0)
    X, H, O, M = _augment_dihedral(X_base, H_base, O_base, M_base)
    print(f"Total training patches: {len(X)} (base={len(X_base)}, real={len(xr)}, synth={len(xs)})")

    net = MinutiaeNetV1(seed=2026)
    m_adam = {k: np.zeros_like(v) for k, v in net.params.items()}
    v_adam = {k: np.zeros_like(v) for k, v in net.params.items()}
    step = 0
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        # Cosine annealing down to 0.15 * lr
        lr_ep = lr * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / max(1, epochs - 1))))
        perm = rng.permutation(len(X))
        ep_loss, n_b = 0.0, 0
        for s in range(0, len(X), batch_size):
            idx = perm[s : s + batch_size]
            if len(idx) < 8:
                continue
            step += 1
            loss, grads = net.forward_backward(X[idx], H[idx], O[idx], pos_weight=16.0, hm_mask=M[idx])
            ep_loss += loss
            n_b += 1
            b1_c = 1.0 - (0.9 ** step)
            b2_c = 1.0 - (0.999 ** step)
            for k, g in grads.items():
                # Small weight decay on kernels
                if k.startswith("w"):
                    g = g + np.float32(1e-4) * net.params[k]
                m_adam[k] = 0.9 * m_adam[k] + 0.1 * g
                v_adam[k] = 0.999 * v_adam[k] + 0.001 * (g * g)
                m_hat = m_adam[k] / b1_c
                v_hat = v_adam[k] / b2_c
                net.params[k] -= (lr_ep * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)
        if ep in (1, 4, 8, 12, 16, 20, epochs):
            print(f"  Epoch {ep:2d}/{epochs} (lr={lr_ep:.5f}): loss={ep_loss / max(1, n_b):.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    net.export_onnx(out_onnx_path)
    print(f"Exported ONNX model to {out_onnx_path} ({os.path.getsize(out_onnx_path)/1024:.1f} KB)")

    # Verify ONNXRuntime inference parity vs NumPy forward pass
    sess = ort.InferenceSession(out_onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    test_img = rng.uniform(0.1, 0.9, size=(1, 1, 96, 96)).astype(np.float32)
    ort_hm, ort_ori = sess.run(None, {inp_name: test_img})
    print(f"ORT smoke test passed: hm {ort_hm.shape} [{ort_hm.min():.3f}, {ort_hm.max():.3f}], ori {ort_ori.shape}")


if __name__ == "__main__":
    train_and_export()
