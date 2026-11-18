# Implementation Plan: Updated Next-Stage Research and Engineering Plan
## 9-D Pure Geometric Cancelable Fingerprint Protection System

This implementation plan formalizes the execution strategy for the **Updated Next-Stage Research and Engineering Plan**, responding directly to the comprehensive audit findings, experimental corrections, and work packages (WP0 through WP9) specified in the work order.

---

## 1. Goal Description

The primary objective is to execute an **evidence-first validation, recognition-improvement, and security-hardening program**, addressing the core technical findings identified in the audit:
1. **Recognition Improvement as Primary Technical Priority:** The reported FVC2002 DB1_A EER of 41.20% and AUC of 0.6249 represent a prototype baseline; the system requires measurable genuine/impostor separation improvements (especially at low FAR) through Delaunay stability, triangle filtering, and coverage scoring before claiming operational verification utility.
2. **Uncertainty-Gated Security & Privacy Reporting:** Qualify all linkability and privacy claims by their confidence intervals ($D_{\text{sys}}^{\text{max}} = 0.0375$ [0.0296, 0.1333]; cross-epoch $D_{\text{sys}} = 0.0689$ [0.0046, 0.2014]). Replace all unqualified "fully unlinkable" language with ISO/IEC 30136:2018 **Low Residual Linkability** and pre-registered upper-bound acceptance gates.
3. **Reproducibility Lock & Arithmetic Reconciliation (WP0):** Reconcile all internal reporting discrepancies (sensor count vs partition count, AEAD storage arithmetic 64 + 16 + 576 + 288 = 944 B, DB1_A confidence interval definitions, operating points vs ROC curves, SOCOFing fold-1 vs pooled cell counts, matcher throughput workloads, and manifest counts).
4. **Mathematical & Descriptor Correctness (WP1):** Establish orientation conventions ($[0, 2\pi)$ directed vs $[0, \pi)$ axial), deterministic canonical vertex ordering with tie-breaking, cyclic permutation equivariance, reflection chirality, and execute a controlled ablation of a reduced 5-D shape parameterization vs the 9-D candidate.

---

## 2. User Review Required

> [!IMPORTANT]
> **Sequential Gate Enforcement (WP0 & WP1 as Blocking Prerequisite):**
> Following Section 9 of the work order, work will proceed in strict order:
> 1. **Step 1 (WP0):** Reproducibility lock, trial-level table generation, and resolution of all 11 immediate audit discrepancies.
> 2. **Step 2 (WP1):** Descriptor property tests (rotation, translation, permutation, reflection) and 5-D reduced geometric ablation.
> 3. **Step 3 (WP2–WP9):** Recognition matching optimization, multi-sensor Set A evaluation, robustness separation, security inversion, and standards mapping.

> [!WARNING]
> **Acceptance Gate Policy for Linkability ($D_{\text{sys}}$):**
> A point estimate of $D_{\text{sys}} \le 0.05$ does **not** satisfy a conservative unlinkability gate when the upper 95% bootstrap confidence interval reaches $0.1333$. In compliance with ISO/IEC 30136:2018, the system is strictly reported as exhibiting **Low Residual Linkability** across evaluated threat models, and automated test assertions will gate on pre-registered upper confidence bounds.

> [!NOTE]
> **Data & Sensor Provenance Naming:**
> - FVC2002 DB1, DB2, DB3, DB4 will be reported as **four evaluation database families** (DB1/DB2: optical; DB3: capacitive; DB4: synthetic SFinGe generator), not "five sensor technologies".
> - SOCOFing will be strictly designated as **synthetic digital alterations**, not "physical alterations".

---

## 3. Open Questions for the User

1. **Reduced Dimensionality Candidate (WP1):** For the 5-D reduced geometric descriptor ablation, do you prefer parameterization via:
   - **Option A (Recommended):** Two independent normalized side-length ratios ($L_1, L_2$ with $L_3 = 1 - L_1 - L_2$) + 3 cyclic relative minutiae orientation differences ($\Delta \theta_1, \Delta \theta_2, \Delta \theta_3$).
   - **Option B:** Two interior angles ($\alpha_1, \alpha_2$ with $\alpha_3 = \pi - \alpha_1 - \alpha_2$) + 3 cyclic relative orientations.
2. **Set A Availability for DB2, DB3, DB4 (WP3):** DB1_A (800 images) is present in the workspace. Are external Set A image files for DB2_A, DB3_A, and DB4_A to be downloaded/deposited, or should WP3 generate synthetic/mock evaluation manifests until those licensed archives are deposited?
3. **1:N Identification vs 1:1 Verification (Section 4.3):** The current primary focus is 1:1 verification under application-scoped keys. Do you want to include the per-subject key derivation ablation for 1:N screening in WP7, or retain application-scoped keys as the sole operational design?

---

## 4. Execution Work Packages (WP0 through WP9)

### WP0 — Reproducibility Lock & Report Reconciliation (Priority: P0 — Blocking)
- **Trial-Level Canonical Table:** Build unified row-level manifest (`trial_scores.parquet` / CSV) containing every trial across all 23 dataset manifests (14,755 trials) with trial ID, subject ID, probe/gallery IDs, sensor, genuine/impostor label, score, decision, and threshold version.
- **Audit Findings Resolution (11 Items):**
  1. *Sensor-count wording:* Update text to "four database families / five evaluation partitions" (DB4 clearly labeled synthetic SFinGe).
  2. *Cross-sensor evidence:* Clarify DB2_B, DB3_B, DB4_B as 10-finger development sets.
  3. *EER confidence intervals:* Standardize on single finger-clustered bootstrap with documented seeds.
  4. *Operating points:* Strictly bifurcate (a) frozen Set B developmental thresholds ($T_{\text{EER}}=0.3751, T_{1\%}=0.4497, T_{0.1\%}=0.4705$) from (b) descriptive ROC points.
  5. *SOCOFing trial accounting:* Disaggregate Fold 1 ($N=180$, cell totals TP=30, FP=15) from Fold 2 ($N=180$) and Pooled 10-subject ($N=360$).
  6. *Linkability gate:* Qualify all $D_{\text{sys}}$ statements with upper confidence interval bounds ($D_{\text{sys}}^{\text{max}} = 0.0375$ [0.0296, 0.1333]); label as Low Residual Linkability.
  7. *AEAD storage arithmetic:* Reconcile formula: 64 B header (containing 12 B nonce) + 16 B Poly1305 tag + 576 B descriptors + 288 B weights = **944 Bytes** (for $N=144$). Generate storage tables directly from parser byte lengths.
  8. *Matcher throughput reconciliation:* Disclose exact workload conditions for match-only (Hungarian 369.6 vs Greedy 306.4 comp/s) vs 1:N impostor screening (Greedy 1,242.4 comp/s) vs end-to-end interactive latency (86.04 ms).
  9. *Stage latency sum:* Reconcile individual stage times with end-to-end monotonic timers.
  10. *Manifest count index:* Publish explicit manifest index confirming all 23 manifests and 14,755 accounted comparisons.
  11. *Standards language:* Use "evaluated with reference to ISO/IEC 24745:2022 and ISO/IEC 30136:2018"; remove any implication of formal certification.

### WP1 — Mathematical and Descriptor Correctness Audit (Priority: P1 — Critical)
- **Orientation Wrap Convention:** Validate whether minutiae extractor outputs directed angles in $[0, 2\pi)$ or axial ridge orientations in $[0, \pi)$; ensure $\Delta \theta_i$ formula matches extractor range.
- **Canonical Vertex Ordering:** Formulate deterministic vertex ordering: sort sides $l_1 \le l_2 \le l_3$; define explicit lexicographic tie-breaking for isosceles/equilateral triangles ($l_i = l_j$).
- **Cyclic Equivariance & Chirality:** Prove planar cyclic equivariance under $C_3$; evaluate reflection chirality ($O(2) \setminus SO(2)$) to document whether preserving mirror asymmetry is intended.
- **Dimensionality Ablation:** Implement and evaluate 5-D reduced geometric descriptor ($L_1, L_2, \Delta\theta_1, \Delta\theta_2, \Delta\theta_3$) against 9-D candidate on DB1_B to measure redundancy, collision rates, and discrimination.

### WP2 — Recognition and Matching Improvement (Priority: P1 — Critical)
- **Delaunay Simplex Stability:** Investigate edge flipping under elastic skin distortion (18.1% triangle loss); test local triangle selection and area/side degeneracy filters.
- **Coverage-Aware Scoring:** Prevent small sets of high-scoring triangles from dominating score aggregation; benchmark coverage ratio penalties.
- **Matcher Evaluation:** Benchmark Hungarian vs Greedy 1-to-1 on identical cost matrices, reporting pair-level agreement and exact speedup boundaries.
- **Error Breakdown:** Analyze worst genuine rejections and highest impostor false matches to isolate failure causes.

### WP3 — Multi-Sensor & External-Dataset Validation (Priority: P1 — Critical)
- **Set A Benchmarking:** Execute held-out Set A evaluations on DB1_A, DB2_A, DB3_A, and DB4_A (synthetic SFinGe).
- **Frozen Transfer:** Evaluate DB1_B-calibrated operating thresholds transferred out-of-domain to all Set A datasets without retuning.
- **Clustered Bootstrap:** Compute paired finger-clustered bootstrap intervals for all dataset comparisons.

### WP4 — Realistic Robustness Evaluation (Priority: P2 — High)
- **Evidence Stratification:** Clearly separate (1) algebraic property tests, (2) in-silico image nuisances, (3) SOCOFing digital alterations, and (4) real repeated-session captures.
- **SOCOFing 2-Fold CV:** Maintain subject-clustered uncertainty and report degradation across Easy $\to$ Medium $\to$ Hard $\to$ Obliteration.

### WP5 — Security, Privacy, and Revocation Evaluation (Priority: P2 — High)
- **HiGHS MILP / QP Pre-image Inversion:** Re-run 3,000 trials on 9-D vs 12-D candidates; verify continuous Chebyshev center radius ($r^* = 0.0772$) and cross-key re-issuance ASR (0.0%).
- **Subject-Disjoint Attribute Inference:** Evaluate 11,450 real triangles under 5-fold CV with Benjamini-Hochberg FDR control; confirm zero explicit type bits stored.
- **Unlinkability Uncertainty Gating:** Recompute Gómez-Barrero $D_{\text{sys}}$ per bootstrap replicate; enforce pre-registered upper-bound gates.
- **Revocation Lifecycle:** Benchmark registry blocking (100%), cryptographic bypass ASR (0.0%), and cross-epoch linkability ($D_{\text{sys}}^{\text{epoch}} = 0.0689$).

### WP6 — Cryptographic and Storage Hardening (Priority: P2 — High)
- **KMS / HSM Isolation:** Ensure KMS boundaries are enforced with clear separation between root secrets and derived keys.
- **AEAD Nonce Discipline:** Enforce 96-bit unique nonces per encryption; test 1-bit tampering detection and replay/rollback rejection.
- **Storage Parser Audit:** Compute credential byte sizes directly from serialized envelopes across all enrolled templates.

### WP7 — Performance and Deployment Profiling (Priority: P3 — High)
- **Monotonic Profiling:** Profile feature pipeline, cryptographic derivation, pairwise matching, and 1:N screening throughput under identical workloads and hardware.
- **Concurrency Stress:** Validate multi-threaded execution (8 workers, 100 queries) with 0.0% error rate.

### WP8 — Test Architecture & Independent Reproduction (Priority: P3 — High)
- **Testing Pyramid (102 Tests across 5 Layers):** Maintain Layer 1 (Unit), Layer 2 (Integration), Layer 3 (Security Regressions), Layer 4 (Concurrency), Layer 5 (Reproducibility).
- **Automated Assertions:** Re-verify $TP + FN \equiv N_{\text{gen}}$ and $FP + TN \equiv N_{\text{imp}}$ across all 23 manifests.

### WP9 — Final Academic Manuscript & Standards Mapping (Priority: P4 — Final)
- Author formal 10-chapter academic manuscript (`final_research_report.md` / `report.md`) with clause-by-clause mapping to ISO/IEC 24745:2022, ISO/IEC 30136:2018, and ISO/IEC 19795-1:2021.
- Package clean standalone archive `CancelableFingerprintHybrid.zip`.

---

## 5. Verification Plan

### Automated Test Suite
```powershell
cd C:\Users\mella\.gemini\antigravity\scratch\CancelableFingerprintHybrid
python -m pytest tests/ -q
```
*Pass Criterion:* All 102 tests pass cleanly with 0 failures and 0 warnings.

### Baseline Reproduction Harness (WP0)
```powershell
python experiments/reproduce_baseline.py
```
*Pass Criterion:* All 8 verification checks pass, validating 76 SHA-256 digests, trial accounting identities, SOCOFing harmonization, and KDE boundary assertions.

### Production Service Demonstrator
```powershell
python demo/production_demo.py
```
*Pass Criterion:* All 7 production lifecycle steps pass on real sensor images from FVC2002 DB1_B.

### Concurrency Stress & Trial Accounting Suite (WP8)
```powershell
python experiments/run_phase5.py
```
*Pass Criterion:* 100 queries across 8 threads (0.0% error rate), 14,755 trials accounted across 23 manifests.
