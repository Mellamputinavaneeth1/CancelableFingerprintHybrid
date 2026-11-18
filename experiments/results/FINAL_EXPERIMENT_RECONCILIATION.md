# Master Experiment & Protocol Reconciliation Report

**Release Identity:** `v2.5-Research-Release`  
**Date:** November 2026  
**Auditor:** Statistical Evaluator & Biometric Protocol Auditor  

---

## 1. SOCOFing Protocol Reconciliation & Trial Accounting

### 1.1 Resolution of the 600 vs. 590 Trial Discrepancy
Historical project reports cited both a 600-trial evaluation ($EER = 24.55\%$, $AUC = 0.8234$) and a clean 590-trial protocol ($EER = 26.14\%$, $AUC = 0.8038$). 

Exhaustive verification of the raw trial ledger [experiments/results/socofing_full_canonical/results/pooled_oof_scores.csv](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/experiments/results/socofing_full_canonical/results/pooled_oof_scores.csv) confirms the exact relationship:
- Total rows in raw score ledger: **`600`**
- Number of same-image self-matches (`query_path == gallery_path`): **`10`** (all producing score $1.0000$)
- Distinct-impression genuine pairs: **`90`**
- Distinct-subject impostor pairs: **`500`**

| Protocol Variant | Genuine Trials | Impostor Trials | Total Trials | EER | AUC | Operational Purpose |
|---|---:|---:|---:|---:|---:|---|
| **Canonical Verification Protocol (Clean)** | **`90`** | **`500`** | **`590`** | **`26.14%`** | **`0.8038`** | **Authoritative Verification Benchmark** (Excludes trivial same-image self-matches) |
| Historical Protocol (with Self-Matches) | `100` | `500` | `600` | `24.55%` | `0.8234` | Legacy Sanity Protocol (Demonstrates $1.0$ score on identical image files) |

**Conclusion:** Including 10 identical-image comparisons artificially depressed EER by $1.59$ percentage points and inflated AUC by $0.0196$. The canonical benchmark is strictly established as the **`590-trial protocol`** with $N_{gen} = 90$ and $N_{imp} = 500$.

### 1.2 Five-Fold Subject-Disjoint Cross-Validation
Across the 10 evaluated subjects (100 fingerprint images across Real and Altered variants):
- Folds $1$–$5$ enforce strict subject-level isolation:
  - Fold 1: Test Subjects `[1, 2]`, Dev-Fit `[3, 4, 5, 6, 7]`, Dev-Cal `[8, 9, 10]`
  - Fold 2: Test Subjects `[3, 4]`, Dev-Fit `[5, 6, 7, 8, 9]`, Dev-Cal `[1, 2, 10]`
  - Fold 3: Test Subjects `[5, 6]`, Dev-Fit `[1, 2, 7, 8, 9]`, Dev-Cal `[3, 4, 10]`
  - Fold 4: Test Subjects `[7, 8]`, Dev-Fit `[1, 2, 3, 4, 9]`, Dev-Cal `[5, 6, 10]`
  - Fold 5: Test Subjects `[9, 10]`, Dev-Fit `[1, 2, 3, 4, 5]`, Dev-Cal `[6, 7, 8]`
- **Subject Leakage:** Exactly $0.0\%$ (all intersection sets $\emptyset$).
- **Alteration Stratified EER on Clean 590 Trials:**
  - Easy Alterations: **`14.29%`**
  - Medium Alterations: **`26.67%`**
  - Hard Alterations: **`36.67%`**
  Confirming expected monotonic degradation as physical distortion severity increases.

---

## 2. FVC2002 Benchmark Accounting & Hygiene

### 2.1 Set-A Official Protocol (Zero Set-A Leakage)
For each of the four benchmark databases (`DB1_A`–`DB4_A`):
- Total images: $800$ ($100$ fingers $\times$ $8$ impressions).
- Genuine comparisons: Every pair of impressions $(i, j)$ with $j > i$ for each finger:
  $$N_{\text{gen}} = 100 \times \frac{8 \times 7}{2} = 2,800\,\text{trials}$$
- Impostor comparisons: First impression ($i=1$) compared across all distinct finger pairs:
  $$N_{\text{imp}} = \frac{100 \times 99}{2} = 4,950\,\text{trials}$$
- Total trials per database: **`7,750` trials** (conserved exactly: $TP + FN = 2800$, $FP + TN = 4950$).

### 2.2 Set-B Calibration Protocol
- Database size: $80$ images ($10$ fingers $\times$ $8$ impressions).
- Genuine comparisons: $10 \times \binom{8}{2} = 280$ trials.
- Standard Impostor comparisons: $\binom{10}{2} = 45$ trials.
- All-Impostor protocol: $10 \times 9 \times 32 = 2,880$ trials.
- **Hygiene Enforcement:** All feature scalers, Set-B symmetric ZCA whitening ($\lambda = 1.0$), and operating thresholds $T_{\text{EER}}$ are fitted strictly on Set B before evaluating Set A.

---

## 3. Deep Minutiae Model Provenance (`MinutiaeNetV1`)

- **Architecture:** Local Contrast Normalization ($15 \times 15$, $\epsilon=0.0025$) $\to$ 16-channel Gabor-initialized directional Conv1 $\to$ 24-channel bottleneck U-Net $\to$ Dual heads (heatmap logits + double-angle orientation).
- **Weight Checkpoint:** `models/minutiae_net_v1.onnx` ($64,234\,\text{bytes}$, SHA-256: `18fbe902bca487b3...`).
- **Training Set:** Strictly trained on `DB2_B + DB3_B + DB4_B` + synthetic ridge patterns for 24 epochs using cosine LR annealing.
- **Contamination Check:** **Zero exposure** to any Set-A images (`DB1_A`–`DB4_A`) or held-out validation `DB1_B`.

---

## 4. Comprehensive FVC2002 Set-A Experimental Results

All experiments evaluated on official 7,750-trial Set-A partitions with zero hyperparameter tuning on test sets:

| Dataset | Modality / Sensor / Resolution | Pipeline & Feature Mode | Prot. EER | 95% Clustered EER CI | Prot. AUC | Unprot. Baseline EER / AUC | Paired $\Delta\text{EER}$ | TAR @ 1% FAR | TAR @ 0.1% FAR | FTE Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **`DB1_A`** | Optical Prism (TouchView II) / 500 DPI | **MCC-Spectral Deep ONNX** | **`5.07%`** | `[4.64%, 11.23%]` | **`0.9799`** | `4.74%` / `0.9831` | **`+0.34%`** | **`92.50%`** | **`87.96%`** | **`0.0%` (`0`)** |
| **`DB1_A`** | Optical Prism (TouchView II) / 500 DPI | **MCC-Spectral Classical** | **`5.18%`** | `[4.53%, 9.35%]` | **`0.9783`** | `4.35%` / `0.9832` | **`+0.83%`** | **`93.25%`** | **`89.04%`** | **`0.0%` (`0`)** |
| `DB1_A` | Optical Prism (TouchView II) / 500 DPI | Legacy 9-D Delaunay Baseline | `41.20%` | — | `0.6249` | `45.53%` / `0.5841` | `-4.33%` | — | — | `0.0%` (`0`) |
| **`DB2_A`** | Optical CMOS (FX2000) / 569 DPI | MCC-Spectral Classical | `14.17%` | `[11.34%, 17.37%]` | `0.8975` | `12.29%` / `0.9053` | `+1.88%` | `52.79%` | `28.29%` | `1.4%` (`11`) |
| **`DB3_A`** | Capacitive (100 SC) / 500 DPI | MCC-Spectral Classical | `18.29%` | `[13.01%, 21.78%]` | `0.8867` | `13.90%` / `0.8869` | `+4.38%` | `65.14%` | `44.96%` | `4.4%` (`35`) |
| **`DB4_A`** | Synthetic (SFinGe v2.51) / 500 DPI | MCC-Spectral Classical | `19.75%` | `[13.44%, 23.98%]` | `0.8550` | `16.36%` / `0.8518` | `+3.39%` | `46.21%` | `26.93%` | `6.5%` (`52`) |

---

## 5. Cross-Sensor Operating Threshold Transport Analysis

When evaluating credentials under frozen Set-B operating thresholds:
- `DB1_A` Deep ONNX using optical `DB1_B` threshold ($T = 0.020485$): **`TAR = 93.96%` @ `FAR = 2.40%`**
- `DB1_A` Classical using optical `DB1_B` threshold ($T = 0.020817$): **`TAR = 94.46%` @ `FAR = 3.07%`**
- When the optical threshold is transferred directly to heterogeneous sensors:
  - `DB2_A` CMOS Optical: `TAR = 88.29%` @ **`FAR = 33.56%`**
  - `DB3_A` Capacitive: `TAR = 81.71%` @ **`FAR = 12.69%`**
  - `DB4_A` Synthetic: `TAR = 80.25%` @ **`FAR = 17.47%`**

**Scientific Finding & Recommendation:**
Single-threshold transport across different sensor modalities fails in heterogeneous environments due to sensor-specific contrast, noise floor, and minutiae count variations. Production multi-sensor systems must employ sensor-specific threshold calibration or quality-aware score normalization fitted on development partitions.
