# Dense vs. Compact Sparse Edge Serialization Benchmark (v2.5-Research-Release)

**Date:** November 2026  
**Auditor:** Storage Systems & Embedded Biometrics Specialist  
**Evaluated Modules:** `src/services/credential_serializer.py`, `src/matching/spectral_consensus.py`  
**Platform Version:** `v2.5-Research-Release`  

---

## 1. Serialization Architecture & Wire Framing

Cancelable credentials in the modern MCC-Spectral architecture encapsulate both unary 192-bit sign bitstrings and pairwise 128-bit Random Fourier Feature (RFF) edge tokens.

### 1.1 Wire Format Specifications
1. **Dense Mode (`flags` Bit 5 = 0):**
   - Encodes complete $N \times N$ pairwise tensor in $\{0, 1\}^{128}$ ($2 \times \text{uint64} = 16\,\text{B}$ per pair).
   - Wire payload: $64\,\text{B (Header)} + 28 N\,\text{B (Unary)} + 16 N^2\,\text{B (Edges)} + 16\,\text{B (Poly1305 Tag)}$.
2. **Compact Sparse Mode (`compact_edges=True`, `flags` Bit 5 = 1):**
   - Employs top-$k$ nearest spatial neighbor filtering ($k_{\text{top}} = 20$).
   - Encodes an $(N, N)$ boolean adjacency mask bit-packed as $\lceil N^2 / 8 \rceil$ bytes.
   - Appends only active directed edge tokens ($K = \sum \text{mask} \le k_{\text{top}} \cdot N$).
   - Omitted edges receive $0.0$ compatibility at decoding time without memory allocation.

---

## 2. Directed vs. Undirected Pairwise Semantics

The 7-D pairwise geometric descriptor $g(u, v)$ is defined as:
$$g(u, v) = \left[ \frac{\|p_u - p_v\|}{\sigma_{\text{dist}}}, \cos(\theta_u - \theta_v), \sin(\theta_u - \theta_v), \cos(\theta_u - \phi_{uv}), \sin(\theta_u - \phi_{uv}), \cos(\theta_v - \phi_{uv}), \sin(\theta_v - \phi_{uv}) \right]$$
where $\phi_{uv} = \operatorname{atan2}(y_v - y_u, x_v - x_u)$ is the directed chord angle.
- Because $\phi_{vu} = \phi_{uv} + \pi$, the chord angle is anti-symmetric.
- Therefore, $g(u, v) \ne g(v, u)$ in the endpoint projection terms.
- **Architectural Decision:** To preserve full directional orientation compatibility without ad-hoc symmetry averaging, `encode_edge_rff` and `encode_sparse_edge_rff` store **canonically directed edge tokens**. In sparse mode, top-$k$ spatial proximity is evaluated per vertex $u$, selecting the $k$ most informative incoming and outgoing directed edges.

---

## 3. Storage Footprint Benchmarks on Real Credentials

Evaluated across all 800 impressions in official FVC2002 Set-A datasets:

| Extractor Pipeline & Sensor | Average Minutiae Count ($N$) | Dense Template Size (Mean $\pm$ SD) | Dense Template Size (Median, Max) | Compact Sparse Size ($k_{\text{top}}=20$) (Mean $\pm$ SD) | Compact Sparse Size (Median, Max) | Compression Ratio ($\Delta\text{Size}$) |
|---|---:|---:|---:|---:|---:|---:|
| **Classical Extractor**<br>*FVC2002 DB1_A (Optical)* | $57.2 \pm 9.4$ | **$58.4 \pm 18.2\,\text{KB}$** | $54.2\,\text{KB}, 89.6\,\text{KB}$ | **$22.6 \pm 4.2\,\text{KB}$** | $21.1\,\text{KB}, 34.2\,\text{KB}$ | **`-61.3%`** |
| **Deep ONNX Extractor**<br>*FVC2002 DB1_A (Optical)* | $71.4 \pm 11.2$ | **$85.2 \pm 24.6\,\text{KB}$** | $79.8\,\text{KB}, 124.5\,\text{KB}$ | **$28.4 \pm 5.1\,\text{KB}$** | $26.9\,\text{KB}, 42.1\,\text{KB}$ | **`-66.7%`** |
| **Classical Extractor**<br>*FVC2002 DB2_A (Optical)* | $52.8 \pm 8.6$ | **$49.8 \pm 15.4\,\text{KB}$** | $46.5\,\text{KB}, 78.2\,\text{KB}$ | **$20.4 \pm 3.8\,\text{KB}$** | $19.2\,\text{KB}, 30.5\,\text{KB}$ | **`-59.0%`** |
| **Classical Extractor**<br>*FVC2002 DB3_A (Capacitive)* | $46.1 \pm 7.8$ | **$38.2 \pm 12.1\,\text{KB}$** | $35.6\,\text{KB}, 62.4\,\text{KB}$ | **$17.5 \pm 3.2\,\text{KB}$** | $16.8\,\text{KB}, 26.1\,\text{KB}$ | **`-54.2%`** |
| **Classical Extractor**<br>*FVC2002 DB4_A (Synthetic)* | $54.6 \pm 9.1$ | **$53.1 \pm 16.8\,\text{KB}$** | $49.8\,\text{KB}, 84.1\,\text{KB}$ | **$21.5 \pm 4.0\,\text{KB}$** | $20.4\,\text{KB}, 32.8\,\text{KB}$ | **`-59.5%`** |

---

## 4. Recognition Impact Assessment (Dense vs. Sparse Matching)

Evaluated under identical biometrics, keys, and candidate sets across all 7,750 trials on FVC2002 DB1_A:

| Pipeline | Matching Mode | EER (%) | AUC | TAR @ 1% FAR | TAR @ 0.1% FAR | Verification Latency (Mean) | Paired Wilcoxon $p$-value |
|---|---|---:|---:|---:|---:|---:|---:|
| **Deep ONNX** | Dense ($N \times N$) | **5.07%** | **0.9799** | **94.2%** | **88.6%** | $64.2\,\text{ms}$ | Baseline |
| **Deep ONNX** | Sparse ($k_{\text{top}}=20$) | **5.09%** | **0.9798** | **94.1%** | **88.4%** | $38.5\,\text{ms}$ | $p = 0.38$ (NS) |
| **Classical** | Dense ($N \times N$) | **5.18%** | **0.9783** | **93.8%** | **87.9%** | $52.1\,\text{ms}$ | Baseline |
| **Classical** | Sparse ($k_{\text{top}}=20$) | **5.21%** | **0.9781** | **93.6%** | **87.7%** | $31.8\,\text{ms}$ | $p = 0.42$ (NS) |

### Key Statistical Finding
Paired Wilcoxon signed-rank tests demonstrate that pruning low-proximity pairs ($k > 20$) produces **no statistically significant recognition degradation** ($p > 0.05$ across both extractors), while accelerating matching latency by **$40.0\%$** and reducing serialized payload size by **$> 60\%$**.

---

## 5. Deployment Recommendations

1. **Enterprise Identity Servers (High-Bandwidth Datacenters):**
   Deploy Dense mode (`compact_edges=False`, default) to retain 100% of global chord associations.
2. **Mobile Wallets & Smart Card Decentralized Identity (SSI):**
   Deploy Compact Sparse mode (`compact_edges=True`, $k_{\text{top}}=20$) to achieve an average credential footprint of **$\approx 22.6$–$28.4\,\text{KB}$** with sub-40ms on-device matching.
