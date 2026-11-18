import numpy as np
import scipy.stats as stats
import cv2
import time
import os
import sys

from src.pipeline import CancelableFingerprintPipeline
from src.graph.delaunay import build_delaunay_triangles
from src.graph.descriptors import extract_all_descriptors
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.evaluation.metrics import compute_roc_metrics
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs

print("Loading DB1_A dataset...")
ds = load_fvc_dataset("DB1_A", auto_download=False)
gen_pairs, imp_pairs = generate_fvc_protocol_pairs(ds)
print(f"Loaded {len(gen_pairs)} genuine, {len(imp_pairs)} impostor pairs.")

pipe = CancelableFingerprintPipeline(in_dim=9, dpi=ds["dpi"], use_gpu=True)

# Extract minutiae
print("Extracting minutiae for 800 images...")
min_cache = {}
for item in ds["all_images"]:
    p = item["path"]
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    feat = pipe.extract_features(img)
    min_cache[p] = feat["minutiae"]

# Extract descriptors
templates = {}
all_d = []
for p, mins in min_cache.items():
    tris = build_delaunay_triangles(mins, max_edge_length=120.0)
    d, w = extract_all_descriptors(tris, dpi=ds["dpi"], include_types=False)
    templates[p] = {"descs": d, "weights": w}
    if len(d) > 0:
        all_d.append(d)

stacked = np.vstack(all_d)
mean = np.mean(stacked, axis=0)
std = np.std(stacked, axis=0)
W = derive_orthonormal_matrix(pipe.master_secret, salt="salt_DB1_A", app_id="fvc2002_DB1_A", in_dim=9, out_dim=32)

for p in templates:
    d = templates[p]["descs"]
    if len(d) > 0:
        proj = project_descriptors(d, W, mean=mean, std=std)
        bits = quantize_descriptors(proj)
    else:
        bits = np.empty((0, 32), dtype=np.uint8)
    templates[p]["bits"] = bits

print("Running cfg1 (baseline) vs cfg3 (coverage normalized)...")
# Run on genuine pairs
g_scores_cfg1 = []
g_scores_cfg3 = []
for p in gen_pairs:
    t_q = templates[p["probe_path"]]
    t_e = templates[p["gallery_path"]]
    s1 = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22, coverage_penalty_exponent=1.0, min_matches=0)
    s3 = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22, coverage_penalty_exponent=1.2, min_matches=3)
    g_scores_cfg1.append(s1)
    g_scores_cfg3.append(s3)

# Run on impostor pairs
i_scores_cfg1 = []
i_scores_cfg3 = []
for p in imp_pairs:
    t_q = templates[p["probe_path"]]
    t_e = templates[p["gallery_path"]]
    s1 = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22, coverage_penalty_exponent=1.0, min_matches=0)
    s3 = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22, coverage_penalty_exponent=1.2, min_matches=3)
    i_scores_cfg1.append(s1)
    i_scores_cfg3.append(s3)

g1 = np.array(g_scores_cfg1)
g3 = np.array(g_scores_cfg3)
i1 = np.array(i_scores_cfg1)
i3 = np.array(i_scores_cfg3)

roc1 = compute_roc_metrics(g1, i1, warn_on_inversion=False)
roc3 = compute_roc_metrics(g3, i3, warn_on_inversion=False)

print(f"CFG1 Baseline:   EER={roc1['eer']*100:.2f}%, AUC={roc1['auc']:.4f}")
print(f"CFG3 CovNorm:    EER={roc3['eer']*100:.2f}%, AUC={roc3['auc']:.4f}")

# Paired Wilcoxon Signed-Rank Test on Impostor score suppression
w_imp = stats.wilcoxon(i1, i3, alternative='greater')
print(f"Wilcoxon Impostor Suppression (i1 > i3): stat={w_imp.statistic}, p-value={w_imp.pvalue:.4e}")

# Paired Wilcoxon on Genuine scores
w_gen = stats.wilcoxon(g1, g3)
print(f"Wilcoxon Genuine difference: stat={w_gen.statistic}, p-value={w_gen.pvalue:.4e}")

# Paired Bootstrap for delta AUC and delta EER
np.random.seed(42)
B = 1000
delta_aucs = []
delta_eers = []
n_g = len(g1)
n_i = len(i1)
for _ in range(B):
    idx_g = np.random.randint(0, n_g, size=n_g)
    idx_i = np.random.randint(0, n_i, size=n_i)
    r1 = compute_roc_metrics(g1[idx_g], i1[idx_i], warn_on_inversion=False)
    r3 = compute_roc_metrics(g3[idx_g], i3[idx_i], warn_on_inversion=False)
    delta_aucs.append(r3['auc'] - r1['auc'])
    delta_eers.append((r3['eer'] - r1['eer']) * 100)

ci_d_auc = np.percentile(delta_aucs, [2.5, 97.5])
ci_d_eer = np.percentile(delta_eers, [2.5, 97.5])
print(f"Delta AUC (CovNorm - Baseline): {roc3['auc'] - roc1['auc']:+.4f} (95% CI: [{ci_d_auc[0]:+.4f}, {ci_d_auc[1]:+.4f}])")
print(f"Delta EER (CovNorm - Baseline): {(roc3['eer'] - roc1['eer'])*100:+.2f}% (95% CI: [{ci_d_eer[0]:+.2f}%, {ci_d_eer[1]:+.2f}%])")

# Save arrays for reproducibility
np.save("experiments/results/wp2/db1_a_g_scores_cfg1.npy", g1)
np.save("experiments/results/wp2/db1_a_g_scores_cfg3.npy", g3)
np.save("experiments/results/wp2/db1_a_i_scores_cfg1.npy", i1)
np.save("experiments/results/wp2/db1_a_i_scores_cfg3.npy", i3)
print("Saved DB1_A paired score arrays.")
