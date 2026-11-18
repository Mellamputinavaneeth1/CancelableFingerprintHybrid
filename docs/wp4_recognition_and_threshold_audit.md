# Work Package 4 (WP4) — Recognition and Threshold-Calibration Validity Audit

**Document Status:** Complete & Verified  
**Revision:** 2.2-Hardened  
**Date:** November 2026  
**Applicability:** FVC2002 DB1_A, DB1_B, DB2_B, DB3_B, DB4_B, Multi-Sensor Benchmark Suite  

---

## 1. Executive Summary

Work Package 4 (WP4) provides an exhaustive, mathematically rigorous audit of the biometric recognition pipeline, matcher equivalence, score normalization, and threshold calibration across the `CancelableFingerprintHybrid` platform.

The canonical baseline on the primary optical benchmark (**FVC2002 DB1_A**) achieves:
$$\text{EER} = 41.20\% \quad [95\%\text{ CI: } 38.32\%, 44.42\%], \quad \text{AUC} = 0.6249 \quad [95\%\text{ CI: } 0.5840, 0.6698], \quad \text{TAR}_{@1\%\text{ FAR}} = 5.54\%$$

This performance limitation is structural rather than an implementation bug. This audit establishes:
1. **Pipeline Bottlenecks:** Minutiae extraction achieves $0.0\%$ FTE across all 800 DB1_A impressions. The primary failure mode is **elastic skin displacement and Delaunay edge-flipping**, which destroys $69.12\%$ of genuine Delaunay simplices, leaving only $30.88\%$ surviving triangles. Quantization to 1-bit hypercube representations does *not* degrade performance; instead, it regularizes feature jitter, boosting AUC by $+0.064$ over unquantized continuous Euclidean matching.
2. **Matcher Non-Equivalence & Suboptimality:** The Hungarian assignment on the augmented slack cost matrix is the unique exact reference for global bipartite matching. The alternative matcher is mathematically designated the **"Greedy one-to-one partial-matching heuristic"**. It is *not* generally globally optimal and can be trapped in local minima (worst-case approximation factor $\le 1/2$). We construct a minimal counterexample where Greedy and Hungarian select completely disjoint edges, resulting in a substantial cost suboptimality gap ($0.65$ vs $0.43$) and score gap ($0.35$ vs $0.57$).
3. **Coverage-Aware Normalization:** Score damping with $\gamma = 1.2$ and $M_{\min} = 3$ penalizes low-overlap impostors, yielding modest gains on DB1_A (EER $41.20\% \to 40.62\%$, AUC $0.6249 \to 0.6301$). Paired Wilcoxon signed-rank tests confirm significant impostor score suppression ($p < 10^{-10}$), but paired bootstrap analysis demonstrates that the delta ($\Delta\text{AUC} = +0.0052$) is modest and does not resolve the underlying recognition bottleneck.
4. **Underpowered Low-FAR Threshold Calibration:** Development Set B partitions contain only $N_{\text{imp}} = 45$ impostor comparisons, where each trial accounts for $1/45 \approx 2.22\%$ empirical FAR. Calibrating $1\%$ or $0.1\%$ FAR thresholds on 45 trials is statistically invalid. When zero false accepts are observed, the Clopper-Pearson 95% upper bound is $6.46\%$ and the Wilson score bound is $7.86\%$. All development-calibrated low-FAR operational thresholds must be reported as **exploratory and statistically underpowered**.
5. **Sensor Provenance & Benchmark Separation:** DB4 (synthetic SFinGe generative model) must be strictly separated from physical sensors (DB1 optical Identix, DB2 optical Biometrika, DB3 capacitive Precise). The historical pooled benchmark that combined held-out Set A test trials with Set B development trials ($3,920$ gen / $5,130$ imp) is invalid; Set A and Set B partitions must never be pooled.

---

## 2. Comprehensive Pipeline Bottleneck Diagnostics (Section 5.1)

To isolate the failure mechanisms driving the DB1_A baseline ($41.20\%$ EER), each pipeline stage was measured under deterministic settings across FVC2002 partitions:

### 2.1 Minutiae Extraction and Failure to Enroll (FTE)
Minutiae extraction yield and FTE rates were evaluated across all 5 benchmark partitions:
- **DB1_A (Optical Identix TouchView II):** 800 images, $\text{FTE} = 0.0\%$ (0/800). Mean minutiae count per print: $41.2 \pm 9.4$ (range: 14 to 76).
- **DB1_B (Optical Identix TouchView II):** 80 images, $\text{FTE} = 0.0\%$ (0/80). Mean minutiae: $42.5 \pm 10.1$.
- **DB2_B (Optical Biometrika FX2000):** 80 images, $\text{FTE} = 0.0\%$ (0/80). Mean minutiae: $38.7 \pm 8.6$.
- **DB3_B (Capacitive Precise Biometrics 100 SC):** 80 images, $\text{FTE} = 0.0\%$ (0/80). Mean minutiae: $34.1 \pm 7.9$.
- **DB4_B (Synthetic SFinGe v2.51):** 80 images, $\text{FTE} = 0.0\%$ (0/80). Mean minutiae: $45.8 \pm 11.2$.

*Diagnostic Conclusion:* Minutiae extraction and FTE are **not** the bottleneck. Robust Gabor ridge filtering, local orientation estimation, and neighborhood non-maximum suppression reliably produce $\ge 14$ minutiae per impression, well above the $N \ge 3$ simplex formation threshold.

### 2.2 Topological Simplex Stability (Delaunay vs. $k$-NN)
The structural stability of triangle graphs under intra-class skin elasticity was evaluated on 280 DB1_B genuine impression pairs:
- **Standard Delaunay Triangulation:**
  - Total triangles formed: 39,979
  - Recoverable triangles (all 3 vertices detected in both impressions): 2,367
  - Surviving simplices (identical vertex triplet forms a Delaunay face in both impressions): 731
  - **Simplex Survival Rate:** **$30.88\%$**
  - **Edge-Flipping Rate:** **$69.12\%$**
  - **Gross Triangle Survival Rate:** **$1.83\%$** (731 / 39,979)
- **Degeneracy-Filtered Delaunay (min edge 10 px, min area 25 px², max aspect 4.0):**
  - Total triangles formed: 37,104
  - Recoverable triangles: 2,277
  - Surviving simplices: 721
  - Simplex Survival Rate: **$31.66\%$** (Edge-Flipping Rate: $68.34\%$)
  - Gross Triangle Survival Rate: $1.94\%$
- **$k$-NN Redundant Triplets ($k=5$):**
  - Total triangles formed: 140,967 ($3.53\times$ expansion)
  - Recoverable triangles: 8,477
  - Surviving simplices: 4,312
  - Simplex Survival Rate: **$50.87\%$** (Edge-Flipping Rate: $49.13\%$)
  - Gross Triangle Survival Rate: $3.06\%$

*Diagnostic Conclusion:* **Delaunay edge-flipping is the primary recognition bottleneck.** Non-linear skin stretching displaces minutiae by $5\text{--}15$ pixels, altering empty circumcircle tests and causing $69.12\%$ of valid triangles to flip their connectivity. While $k$-NN ($k=5$) increases surviving triangles to $50.87\%$, the $3.5\times$ template bloat creates severe combinatorial false matches and increases matching latency by $26\times$ without improving overall ROC AUC.

### 2.3 Feature Drift and Signal-to-Noise Ratio (SNR)
Measuring feature variance across genuine vs. impostor pairs reveals significant disparities in discriminative power:
$$\text{SNR}_k = \frac{|\mu_{\text{imp}, k} - \mu_{\text{gen}, k}|}{\sqrt{\frac{1}{2}(\sigma_{\text{imp}, k}^2 + \sigma_{\text{gen}, k}^2)}}$$

- **Perimeter ($P_{\text{mm}}$):** $\text{SNR} = \mathbf{0.648}$ (Highest discriminative power). Genuine error: $0.548 \pm 0.492$ mm vs Impostor diff: $0.867 \pm 0.835$ mm.
- **Side Ratios ($r_1, r_2$):** $\text{SNR} = 0.118$ and $0.241$.
- **Internal Angles ($\phi_0, \phi_1, \phi_2$):** $\text{SNR} = 0.078\text{--}0.205$ (Substantially degraded by non-affine shear).
- **Mutual Minutiae Orientations ($d\theta_{01}, d\theta_{12}, d\theta_{20}$):** $\text{SNR} = 0.067\text{--}0.225$.

### 2.4 Quantization Noise Filtering
Comparing continuous normalized Euclidean distance against 1-bit hypercube sign quantization $\mathbf{b} = \operatorname{sgn}(\mathbf{W}\mathbf{z})$:
- Unprotected continuous Euclidean matching on DB1_A: $\text{EER} = 45.53\%, \text{AUC} = 0.5605$.
- Protected 1-bit 32-bit quantized matching on DB1_A: $\text{EER} = 41.20\%, \text{AUC} = 0.6249$.
- AUC improvement: **$+0.0644$** ($+11.5\%$ relative gain).

*Diagnostic Conclusion:* 1-bit random projection acts as a **regularizer**. Continuous Euclidean distances are corrupted by high-dimensional jitter; mapping to discrete halfspaces dampens intra-class noise while preserving broad geometric cluster separations.

---

## 3. Matcher Equivalence Audit: Hungarian Reference vs. Greedy Heuristic (Section 5.2)

### 3.1 Terminology and Mathematical Definitions
- **Exact Reference:** Hungarian assignment via the augmented cost matrix $\mathbf{C}_{\text{aug}} \in \mathbb{R}^{(N_q + N_e) \times (N_q + N_e)}$.
- **Approximate Baseline:** **Greedy one-to-one partial-matching heuristic**. Under no circumstances should this heuristic be referred to as "optimal Greedy" or asserted to be identical to Hungarian.

#### Augmented Cost Matrix Formulation
For query descriptors $\mathbf{q}_i$ ($i \in [N_q]$) and enrolled descriptors $\mathbf{e}_j$ ($j \in [N_e]$) with distance matrix $D_{ij} \in [0, 1]$ and distance threshold $\tau$:
$$\mathbf{C}_{\text{aug}} = \begin{pmatrix} \mathbf{D}_{\text{valid}} & \mathbf{U}_q \\ \mathbf{U}_e & \mathbf{0}_{N_e \times N_q} \end{pmatrix}$$
where:
$$\mathbf{D}_{\text{valid}}[i, j] = \begin{cases} D_{ij} & \text{if } D_{ij} < \tau \\ \infty & \text{if } D_{ij} \ge \tau \end{cases}, \quad \mathbf{U}_q[i, k] = \begin{cases} \tau / 2 & \text{if } i = k \\ \infty & \text{if } i \ne k \end{cases}, \quad \mathbf{U}_e[l, j] = \begin{cases} \tau / 2 & \text{if } l = j \\ \infty & \text{if } l \ne j \end{cases}$$

For any 1-to-1 matching $\pi$ selecting $k$ valid real pairs $\mathcal{M} = \{(i_r, j_r)\}_{r=1}^k$ with $D_{i_r, j_r} < \tau$, the total augmented cost is:
$$J(\pi) = \sum_{r=1}^k D_{i_r, j_r} + (N_q - k) \frac{\tau}{2} + (N_e - k) \frac{\tau}{2} = \frac{N_q + N_e}{2} \tau - \sum_{r=1}^k (\tau - D_{i_r, j_r})$$
Minimizing $J(\pi)$ is mathematically equivalent to maximizing the total savings $\sum_{r=1}^k (\tau - D_{i_r, j_r})$ over all possible 1-to-1 partial matchings.

#### Greedy One-to-One Partial-Matching Heuristic
1. Enumerate all pairs $(i, j)$ where $D_{ij} < \tau$.
2. Sort pairs by distance $D_{ij}$ in ascending order (stable sort).
3. Maintain sets of matched query indices $\mathcal{S}_q = \emptyset$ and enrolled indices $\mathcal{S}_e = \emptyset$.
4. Iteratively select candidate pair $(i, j)$: if $i \notin \mathcal{S}_q$ and $j \notin \mathcal{S}_e$, accept match, add $i \to \mathcal{S}_q, j \to \mathcal{S}_e$; otherwise discard.

### 3.2 Proof of Suboptimality & Concrete Counterexample
In maximum weight bipartite matching, greedy orderings achieve a worst-case approximation guarantee of $1/2$. A greedy choice made early can preclude two or more subsequent matches whose aggregate utility exceeds the greedy selection.

#### Counterexample Specification:
Let $N_q = 2, N_e = 2$, $\tau = 0.50$, and distance matrix:
$$\mathbf{D} = \begin{pmatrix} 0.20 & 0.22 \\ 0.21 & 0.45 \end{pmatrix}$$
- **Greedy Selection:**
  - Candidate pairs sorted by distance: $(0, 0): 0.20$, $(1, 0): 0.21$, $(0, 1): 0.22$, $(1, 1): 0.45$.
  - Step 1: Greedy picks $(0, 0)$ with $D_{0,0} = 0.20$. Sets $\mathcal{S}_q = \{0\}, \mathcal{S}_e = \{0\}$.
  - Step 2: Candidates $(1, 0)$ and $(0, 1)$ conflict with consumed indices $\{0\}$ and are discarded.
  - Step 3: Candidate $(1, 1)$ is available with $D_{1,1} = 0.45 < 0.50$. Accepted.
  - **Greedy Matched Pairs:** $\{(0, 0), (1, 1)\}$.
  - Real distance sum: $0.20 + 0.45 = \mathbf{0.65}$.
  - Total augmented cost: $0.65 + 0 = \mathbf{0.65}$.
  - Distance savings: $(0.50 - 0.20) + (0.50 - 0.45) = \mathbf{0.35}$.
  - Resulting Biometric Similarity Score ($w=1$): $\frac{2 \times [ (1 - 0.20/0.50) + (1 - 0.45/0.50) ]}{4} = \frac{2 \times [0.60 + 0.10]}{4} = \mathbf{0.3500}$.
- **Hungarian Global Optimal Assignment:**
  - Hungarian evaluates all permutations and selects: $\{(0, 1), (1, 0)\}$.
  - Real distance sum: $0.22 + 0.21 = \mathbf{0.43}$.
  - Total augmented cost: $0.43 + 0 = \mathbf{0.43}$.
  - Distance savings: $(0.50 - 0.22) + (0.50 - 0.21) = \mathbf{0.57}$.
  - Resulting Biometric Similarity Score ($w=1$): $\frac{2 \times [ (1 - 0.22/0.50) + (1 - 0.21/0.50) ]}{4} = \frac{2 \times [0.56 + 0.58]}{4} = \mathbf{0.5700}$.

**Suboptimality Proof Summary:**
$$\text{Cost}_{\text{Hungarian}} = 0.43 < 0.65 = \text{Cost}_{\text{Greedy}}, \quad \text{Score}_{\text{Hungarian}} = 0.5700 > 0.3500 = \text{Score}_{\text{Greedy}}$$
Greedy chooses completely disjoint edges and incurs a $+51.2\%$ cost error and a $-38.6\%$ score depression. This definitively refutes mathematical equivalence.

### 3.3 Empirical Benchmark Across Template Sizes ($N \in \{10, \dots, 200\}$)
While theoretically sub-optimal, Greedy 1-to-1 exhibits high empirical correlation on fingerprint descriptor matrices due to sparse compatibility. Benchmarking across template sizes on real and synthetic templates ($200$ trials per size):

| Template Size $N$ | Hungarian Latency (ms) | Greedy 1-to-1 Latency (ms) | Speedup Factor | Score MAE | Max Score Diff | Pearson $r$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 0.052 ms | 0.032 ms | **$1.47\times$** | 0.0000 | 0.0000 | 1.0000 |
| **25** | 0.105 ms | 0.062 ms | **$1.60\times$** | 0.0000 | 0.000025 | 1.0000 |
| **50** | 0.323 ms | 0.157 ms | **$1.94\times$** | 0.000007 | 0.001238 | 0.9999 |
| **100** | 2.281 ms | 0.770 ms | **$2.95\times$** | 0.000040 | 0.003448 | 0.9999 |
| **144** (Standard) | 6.425 ms | 1.581 ms | **$4.22\times$** | 0.000020 | 0.000908 | 0.9999 |
| **200** | 13.465 ms | 2.518 ms | **$5.85\times$** | 0.000074 | 0.002693 | 0.9999 |

On real sensor trials (**DB1_B**, 325 trials):
- Score Pearson $r = 0.9903$, MAE $= 0.0109$, Max difference $= 0.0225$.
- **Decision Agreement:** At $\tau = 0.3751$, Hungarian and Greedy agree on **$82.77\%$** of decisions, while **$17.23\%$** of decisions differ.
- Full DB1_A ROC evaluation: Hungarian EER $= 41.20\%$, AUC $= 0.6249$; Greedy EER $= 41.25\%$, AUC $= 0.6258$.

*Architectural Recommendation:* Hungarian assignment must remain the gold-standard reference in all formal reporting and security proofs. Greedy 1-to-1 partial matching is approved as a fast edge-deployment heuristic yielding a $4.2\times$ speedup at $N=144$ with negligible population ROC degradation ($\Delta\text{EER} = +0.05\%$).

---

## 4. Coverage-Aware Score Normalization Audit & Paired Difference Tests (Section 5.3)

### 4.1 Formulation and Mathematical Verification
To penalize spurious high-similarity matches between fingerprint impressions with minimal physical overlap, the score normalization function is defined as:
$$S_{\text{norm}} = S_{\text{raw}} \times \left( \frac{2 M_{\text{matched}}}{N_q + N_e} \right)^\gamma \times \psi(M_{\text{matched}}, M_{\min})$$
where:
$$\psi(M, M_{\min}) = \begin{cases} 1.0 & \text{if } M \ge M_{\min} \\ \left( \frac{M}{M_{\min}} \right)^2 & \text{if } M < M_{\min} \end{cases}$$
and $S_{\text{raw}} = \frac{2 \sum_{(u, v) \in \mathcal{M}} s_{uv}}{\sum w_q + \sum w_e}$ represents the weighted base matching similarity.

#### Mathematical Invariant Verification:
1. **Score Direction:** Preserved ($S_{\text{norm}} \ge 0$, higher indicates greater biometric confidence).
2. **Range Bounds:** Since $S_{\text{raw}} \in [0, 1]$, $\frac{2 M}{N_q + N_e} \in [0, 1]$, and $\psi \in [0, 1]$, $S_{\text{norm}} \in [0, 1]$.
3. **Query/Enrolled Symmetry:** Invariant under $(q, e) \leftrightarrow (e, q)$ index transposition since $N_q + N_e$ and $\sum w_q + \sum w_e$ are symmetric sums.
4. **Degenerate Behavior:** When $M_{\text{matched}} = 0$, $S_{\text{norm}} \equiv 0.0$.
5. **Development-Only Parameter Calibration:** Parameters $\gamma = 1.2$ and $M_{\min} = 3$ were calibrated exclusively on development partition DB1_B to maximize the separation ratio $( \mu_{\text{gen}} - \mu_{\text{imp}} ) / \sqrt{\sigma_{\text{gen}}^2 + \sigma_{\text{imp}}^2}$. They were frozen prior to running held-out DB1_A.

### 4.2 Paired Performance Difference Analysis on Held-Out DB1_A
Evaluating DB1_A (2,800 genuine, 4,950 impostor comparisons) under Baseline ($\gamma = 1.0, M_{\min} = 0$) vs. Coverage-Normalized ($\gamma = 1.2, M_{\min} = 3$):

| Configuration | EER (%) [Clustered 95% CI] | AUC [Clustered 95% CI] | TAR @ 1% FAR (%) | Impostor Mean | Impostor Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (cfg1)** | 41.20% [38.32, 44.42] | 0.6249 [0.5840, 0.6698] | 5.54% | 0.3681 | 0.0314 |
| **Coverage Normalized (cfg3)** | 40.62% [37.64, 44.10] | 0.6301 [0.5890, 0.6738] | 5.79% | 0.3542 | 0.0329 |
| **Paired Delta ($\Delta$)** | **$-0.58\%$** | **$+0.0052$** | **$+0.25\%$** | **$-0.0139$** | — |

#### Statistical Significance Testing:
- **Wilcoxon Signed-Rank Test on Impostor Suppression:** $W = 1.22 \times 10^7, p < 10^{-15}$. Impostor scores are systematically depressed by coverage damping when overlap is sparse.
- **Paired Bootstrap for $\Delta\text{AUC}$ ($B=1,000$ resamples):** $\Delta\text{AUC} = +0.0052$ ($95\%\text{ CI: } [+0.0018, +0.0089]$).
- **Paired Bootstrap for $\Delta\text{EER}$ ($B=1,000$ resamples):** $\Delta\text{EER} = -0.58\%$ ($95\%\text{ CI: } [-1.15\%, -0.05\%]$).

*Audit Finding:* Coverage-aware normalization produces a statistically significant, reproducible reduction in impostor false matches ($p < 10^{-15}$), improving DB1_A EER from $41.20\%$ to $40.62\%$ and AUC from $0.6249$ to $0.6301$. However, this $+0.0052$ AUC improvement is **modest** and must not be characterized as solving the fundamental recognition bottleneck.

---

## 5. Statistical Power Deficit of Development Threshold Calibration (Section 5.4)

### 5.1 Sample Size Granularity in FVC2002 Set B
Under the standardized FVC2002 evaluation protocol, Set B partitions contain exactly 10 fingers with 8 impressions each (80 total images):
- **Genuine Trials:** $\binom{8}{2} = 28$ pairs per finger $\times 10$ fingers = **280 genuine comparisons**.
- **Impostor Trials:** Comparing the first impression of each finger against the first impression of every other finger: $\binom{10}{2} = \mathbf{45}$ **impostor comparisons**.

#### Mathematical Granularity Limit:
Because $N_{\text{imp}} = 45$, empirical False Acceptance Rate on Set B can only take discrete values in steps of:
$$\Delta\text{FAR} = \frac{1}{45} \approx \mathbf{2.222\%}$$

| Observed False Accepts ($k$) | Empirical FAR (%) |
| :---: | :---: |
| 0 | 0.00% |
| 1 | 2.22% |
| 2 | 4.44% |
| 3 | 6.67% |

It is **mathematically impossible** to observe an empirical FAR of $1.00\%$ or $0.10\%$ on 45 trials. Any threshold selected on Set B to target "1% FAR" or "0.1% FAR" must necessarily observe exactly $k=0$ false accepts (or interpolate between 0 and 1).

### 5.2 Uncertainty Bounds for Zero False Accepts ($k=0, n=45$)
When zero false accepts are observed on 45 impostor trials, the statistical confidence interval for the true population FAR is wide:
1. **Rule of Three (approximate upper 95% bound):**
   $$\text{FAR}_{95\%} \approx \frac{3}{n} = \frac{3}{45} = \mathbf{6.67\%}$$
2. **Exact Clopper-Pearson Binomial 95% Upper Bound:**
   $$\text{FAR}_{95\%} = 1 - (0.05)^{1/45} = 1 - 0.9354 = \mathbf{6.46\%}$$
3. **Wilson Score 95% Upper Bound:**
   $$\text{FAR}_{95\%} = \frac{z^2}{n + z^2} = \frac{1.96^2}{45 + 1.96^2} = \frac{3.8416}{48.8416} = \mathbf{7.86\%}$$

*Audit Conclusion:* Observing 0 false accepts on 45 trials cannot establish that the true false acceptance rate is below $1\%$, let alone $0.1\%$; it only bounds the error rate below $\approx 6.5\text{--}7.9\%$. Claiming "calibrated operational thresholds at 1% and 0.1% FAR" from DB1_B is **statistically underpowered and invalid**. All such thresholds must be explicitly documented as **exploratory operational estimates**.

### 5.3 Transfer Audit: DB1_B Calibration Transferred to Held-Out DB1_A
To audit how the underpowered threshold behaves under adequate statistical sample sizes, the DB1_B threshold calibrated at $k=0$ false accepts ($\tau = 0.4545$) was transferred unconditionally to held-out DB1_A ($N_{\text{imp}} = 4,950$ impostor comparisons):
- On DB1_A, threshold $\tau = 0.4545$ produces exactly **$FP = 20$** false accepts out of 4,950 trials.
- Realized Test FAR on DB1_A:
  $$\text{FAR}_{\text{realized}} = \frac{20}{4950} = \mathbf{0.404\%} \quad [95\%\text{ CI: } 0.25\%, 0.62\%]$$
- Realized Test TAR on DB1_A:
  $$\text{TAR}_{\text{realized}} = \frac{72}{2800} = \mathbf{2.57\%} \quad [95\%\text{ CI: } 2.01\%, 3.23\%]$$
- Operating Confusion Accounting: $TP = 72, FN = 2,728$ ($TP + FN = 2,800$); $FP = 20, TN = 4,930$ ($FP + TN = 4,950$).

The realized operating point on DB1_A is approximately $0.40\%$ FAR (not $1.00\%$), yielding an extremely low verification rate ($\text{TAR} = 2.57\%$). True $1.00\%$ FAR operation on DB1_A occurs at threshold $\tau = 0.4385$ (producing $\text{TAR} = 5.54\%$, $FP = 50$).

---

## 6. Sensor Provenance Separation & Pooled Benchmark Correction (Section 5.5)

### 6.1 Physical vs. Synthetic Sensor Categorization
The benchmark incorporates four distinct image generation mechanisms that must be clearly differentiated:
1. **FVC2002 DB1:** **Optical capture** using Identix TouchView II (Total Internal Reflection prism, 500 DPI, $388 \times 374$ px). Real human finger impressions.
2. **FVC2002 DB2:** **Optical capture** using Biometrika FX2000 (CMOS optical sensor, 569 DPI, $296 \times 560$ px). Real human finger impressions.
3. **FVC2002 DB3:** **Capacitive capture** using Precise Biometrics 100 SC (solid-state capacitive silicon, 500 DPI, $300 \times 300$ px). Real human finger impressions.
4. **FVC2002 DB4:** **Synthetic generation model** using SFinGe v2.51 (Synthetic Fingerprint Generator, Univ. of Bologna, 500 DPI, $288 \times 384$ px). **Synthetic mathematically generated ridges; not a physical sensor.**

Under no circumstances should DB4 be reported as a physical capture device or conflated with hardware sensor evaluations.

### 6.2 Elimination of Flawed Historical Pooled Benchmark
The historical report contained a "Multi-Sensor Pooled Benchmark" claiming:
$$N_{\text{gen}} = 3,920, \quad N_{\text{imp}} = 5,130, \quad \text{EER} = 41.90\%$$
Auditing the trial indices revealed that this pool was constructed by adding:
- Held-out test set **DB1_A:** 2,800 genuine, 4,950 impostor
- Development set **DB1_B:** 280 genuine, 45 impostor
- Development set **DB2_B:** 280 genuine, 45 impostor
- Development set **DB3_B:** 280 genuine, 45 impostor
- Development set **DB4_B:** 280 genuine, 45 impostor
$$\text{Total Genuine} = 2800 + 4 \times 280 = 3,920; \quad \text{Total Impostor} = 4950 + 4 \times 45 = 5,130$$

**Audit Ruling:** Pooling held-out test data (Set A) with development tuning data (Set B) violates fundamental machine learning hygiene and biometric testing standards (ISO/IEC 19795-1). It is strictly prohibited.

The corrected reporting protocol requires:
1. **Development Set B:** Reported independently for DB1_B, DB2_B, DB3_B, and DB4_B.
2. **Held-Out Set A:** Reported independently for DB1_A, DB2_A, DB3_A, and DB4_A.
3. **Canonical Pooled Set A:** If a pooled multi-sensor benchmark is reported, it must pool all four Set A partitions exclusively ($4 \times 2,800 = 11,200$ genuine, $4 \times 4,950 = 19,800$ impostor; total 31,000 comparisons), with clear disclosure of physical vs. synthetic partitions.

---

## 7. WP4 Deliverables & Verification Status

| Deliverable | Location | Status |
| :--- | :--- | :---: |
| Recognition Bottleneck Report | `experiments/results/phase1/bottleneck/` & `docs/wp4_recognition_and_threshold_audit.md` | **VERIFIED** |
| Hungarian vs. Greedy Difference Audit | `experiments/results/wp2/hungarian_vs_greedy_benchmark.json` | **VERIFIED** |
| Recalibrated Threshold Registry | `experiments/results/wp2/wp2_recognition_report.json` | **VERIFIED** |
| Paired Performance Difference Analysis | `scratch_paired_diff.py` & Section 4 of this report | **VERIFIED** |
| Sensor Provenance & Benchmark Separation | `multisensor_set_a_report.json` & Section 6 of this report | **VERIFIED** |
| Unit and Property Test Suite | `tests/test_wp4_recognition_audit.py` | **VERIFIED** |
