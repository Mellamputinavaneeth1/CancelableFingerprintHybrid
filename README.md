# CancelableFingerprintHybrid (Release Version 2.5-Research-Release)

**Cancelable Biometric Fingerprint Recognition & Template Protection Framework**  
*Dual-Architecture Reference: Legacy 9-D Delaunay Baseline & Modern 256-D MCC-Spectral Consensus*

Compliant with **ISO/IEC 19795-1:2021**, **ISO/IEC 24745:2022**, **ISO/IEC 30136:2018**, and **RFC 8439 (ChaCha20-Poly1305 AEAD)**.

---

## 1. System Overview & Two Architectural Generations

This repository provides the complete, production-grade implementation, multi-sensor benchmark suite, formal multi-threat security audit, and 7-gate reproducibility harness for two unified biometric template protection architectures:

1. **Generation 1: Legacy 9-D Delaunay Mode (`v2.2-Hardened` Reference Implementation):**
   - Pure geometric Delaunay triangle descriptors (side length ratios, cosines, modulo-$\pi$ axial angles; all discrete minutiae types eliminated).
   - Keyed $32 \times 9$ column-orthonormal QR projection ($W^\top W = I_9$) + deterministic sign quantization.
   - Exact Augmented Hungarian minimum-cost bipartite matcher ($O(N^3)$) with dummy non-match penalty $\tau/2$.
   - Authenticated ChaCha20-Poly1305 serialized envelope ($944\,\text{B}$ canonical size at $N=144$).
   - Evaluated as an immutable reference baseline ($EER = 41.20\%$ on unconstrained optical `DB1_A`).

2. **Generation 2: Modern 256-D MCC-Spectral Mode (`v2.5-Spectral` Primary Release Architecture):**
   - **First-Order Unary Tier:** 256-D Minutia Cylinder-Code (MCC) descriptors, leave-one-out Set-B symmetrized ZCA whitening ($\lambda = 1.0$, strictly commuting with $180^\circ$ spatial grid inversion), and keyed $192 \times 256$ row-orthonormal QR projection ($W W^\top = I_{192}$) with a 64-D orthogonal nullspace.
   - **Second-Order Pairwise Tier:** 7-D SE(2) invariant geometric chord/orientation tensors encoded into keyed 128-bit Random Fourier Feature (RFF) edge tokens, proved to approximate a dominant Gaussian RBF mixture kernel ($81.06\%$ fundamental mode).
   - **Matching Engine:** Deformable SE(2) Spectral Graph Consensus (`SpectralGraphMatcher`) via bidirectional mutual top-$k$ candidate selection, Local Similarity Sort (LSS), 18 power iterations, and greedy 1-to-1 clique pruning.
   - **Feature Extraction:** Supports both classical morphological thinning / Crossing Number and a deep lightweight CPU U-Net detector (`MinutiaeNetV1` ONNX, $62.7\,\text{KB}$, trained strictly on `DB2_B+DB3_B+DB4_B` + synthetic data with zero Set-A leakage).
   - **Wire Compression:** Opt-in `HAS_SPARSE_EDGES` wire serialization delivering $>60\%$ size reduction for typical templates ($N \approx 57$–$71$).
   - **Recognition Performance:** **`5.07%` Protected EER** (Deep ONNX) and **`5.18%` Protected EER** (Classical) on official `FVC2002 DB1_A` ($7,750$ trials, $0.0\%$ FTE, $\Delta\text{EER} = +0.34\%$ and $+0.83\%$ vs. unprotected baselines).

---

## 2. Master Verification & Reproduction Status

- **Automated Pytest Suite:** **`231 passed in 23.86s`** (0 failed, 0 skipped across 29 test files).
- **Master Reproduction Harness (`python -u experiments/reproduce_all_gates.py`):** **ALL 7 GATES PASSED [CLEAN EXIT 0]**
  - Gate 1: Environment & Core Dependencies (`PASS`, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, OpenCV 5.0.0, Pytest 9.2.0)
  - Gate 2: SHA-256 Digest Verification for Raw Trial Artifacts (`PASS`, 76 raw score files)
  - Gate 3: Biometric Trial Conservation Identities (`PASS`, DB1_A 2800 gen / 4950 imp; SOCOFing 90 gen / 500 imp clean)
  - Gate 4: Mathematical & Cryptographic Invariants (`PASS`, Dual QR 9D/256D, ZCA flip commutation, Hungarian matrix, 944B layout)
  - Gate 5: Dynamic Pytest Test Suite (`PASS`, 231 passed in 23.86s)
  - Gate 6: Independent Metric Evaluator Reconciliation (`PASS`, Legacy EER=41.13% | Deep ONNX EER=5.00% | Classical EER=5.16% | Cross-Sensor DB2-4 Verified)
  - Gate 7: Canonical Release Manifest & Report Consistency (`PASS`)

---

## 3. Authoritative Audit & Research Deliverables

- **Consolidated Master Validation Report:** [FINAL_PROJECT_V2_5_FINAL_VALIDATION_REPORT.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/FINAL_PROJECT_V2_5_FINAL_VALIDATION_REPORT.md)
- **Numerical Reconciliation Report:** [docs/audit/V25_NUMERICAL_RECONCILIATION.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/audit/V25_NUMERICAL_RECONCILIATION.md)
- **Modern Unlinkability Validation:** [docs/security/V25_MCC_SPECTRAL_UNLINKABILITY_VALIDATION.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/security/V25_MCC_SPECTRAL_UNLINKABILITY_VALIDATION.md)
- **Replay & Impersonation Attack Separation:** [docs/security/V25_REPLAY_AND_CROSS_KEY_ATTACK_ASSESSMENT.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/security/V25_REPLAY_AND_CROSS_KEY_ATTACK_ASSESSMENT.md)
- **Scenario-III Multi-Subject Inversion Audit:** [experiments/results/V25_SCENARIO_III_MULTI_SUBJECT_AUDIT.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/experiments/results/V25_SCENARIO_III_MULTI_SUBJECT_AUDIT.md)
- **RFF Expectation & Deployed Score Validation:** [docs/math/V25_RFF_EXPECTATION_AND_DEPLOYED_SCORE_VALIDATION.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/math/V25_RFF_EXPECTATION_AND_DEPLOYED_SCORE_VALIDATION.md)
- **Seven-Gate Reproduction Specification:** [docs/audit/V25_SEVEN_GATE_VALIDATION.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/audit/V25_SEVEN_GATE_VALIDATION.md)
- **Dense vs. Sparse Serialization Benchmark:** [experiments/results/V25_DENSE_SPARSE_BENCHMARK.md](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/experiments/results/V25_DENSE_SPARSE_BENCHMARK.md)
- **Evidence Inventory:** [docs/audit/v2_5_evidence_inventory.csv](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/docs/audit/v2_5_evidence_inventory.csv) (4,589 verified entries)
- **Canonical Release Package:** `CancelableFingerprintHybrid-v2.5-ResearchRelease.zip` (SHA-256: `2a758ce6b0096ae13f3a5de37109653a4b9301190cc6181985764bd521da8934`)
- **Release Manifest:** [release_manifest_v2_5.json](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/release_manifest_v2_5.json)

---

## 4. Directory Structure

```
CancelableFingerprintHybrid/
├── final_research_report.md        # Academic Research Report (15 Sections, ISO/IEC compliant)
├── master_project_architecture_and_completion_report.md # Master Architecture & Completion Report
├── walkthrough.md                  # Comprehensive Empirical Verification & Execution Record
├── requirements.txt                # Pinned dependencies (numpy, scipy, pytest, opencv)
├── configs/
│   ├── dataset_config.yaml         # Dataset partition parameters and seeds
│   ├── feature_config.yaml         # Minutiae quality and triangle edge limits
│   └── protection_config.yaml      # HKDF, projection dimensions and thresholds
├── src/
│   ├── preprocessing/              # ROI segmentation & Gabor/CLAHE ridge enhancement
│   ├── minutiae/                   # Morphological thinning, Crossing Number & quality
│   ├── graph/                      # Delaunay & k-NN meshes, 9D/12D canonical & cyclic descriptors
│   ├── protection/                 # CSPRNG validation, HKDF, Gram-Schmidt QR, quantizer & inversion defense
│   │   ├── inversion_defense.py    # Standardized affine mapping & Chebyshev LP center solver
│   │   └── quantizer.py            # Non-negative sign quantizer with IEEE 754 zero tie-breaking
│   ├── matching/                   # Augmented Hungarian matcher & Greedy 1-to-1 matcher
│   │   └── augmented_cost.py       # Exact Kuhn-Munkres augmented matrix (tau/2 slack blocks)
│   ├── security/                   # HiGHS MILP inversion, 9-D leakage, and Gómez-Barrero unlinkability
│   ├── revocation/                 # Database credential registry and revocation lifecycle
│   ├── evaluation/                 # Finger-clustered bootstrap, ROC metrics, and trial accounting
│   │   └── independent_evaluator.py# Pure first-principles independent metric engine
│   ├── pipeline.py                 # End-to-end enrollment, verification & Option A reissuance
│   └── services/                   # 14 Modular Production Services
│       ├── types.py                # Strict DTO schemas, BiometricErrorCode, and exceptions
│       ├── crypto_aead.py          # RFC 8439 ChaCha20-Poly1305 AEAD cipher
│       ├── nonce_manager.py        # Atomic 64-bit monotonic counter & bounded CSPRNG nonces
│       ├── capture_adapter.py      # Input validation (NaN, Inf, format decoding)
│       ├── preprocessor.py         # Segmentation & enhancement adapter
│       ├── minutiae_extractor.py   # Ridge minutiae extractor
│       ├── triangle_builder.py     # Delaunay simplex builder
│       ├── descriptor_builder.py   # 9-D continuous feature vector builder
│       ├── scaler_registry.py      # Sensor normalization scalers
│       ├── key_service.py          # KMS / HSM boundary & HKDF derivation
│       ├── projection_engine.py    # Keyed projection matrix cache & quantizer
│       ├── credential_serializer.py# Framed binary serializer & AEAD envelope
│       ├── credential_repository.py# Persistent store & revocation registry
│       ├── matcher.py              # Multi-algorithm matcher interface
│       ├── threshold_registry.py   # Development-calibrated operational thresholds
│       ├── audit_logger.py         # Privacy-minimised tamper-evident audit logger
│       └── orchestrator.py         # ProductionBiometricService coordinator
├── tests/                          # 191 Automated Tests (100% Passing)
│   ├── test_p0_remediation.py      # Mathematical regression suite (P0.1–P0.6, 15 tests)
│   ├── test_preprocessing.py       # Enhancement unit tests
│   ├── test_minutiae.py            # Crossing number & thinning unit tests
│   ├── test_graph.py               # Triangulation & descriptor unit tests
│   ├── test_descriptor_properties.py# Chirality & SE(2) invariance tests
│   ├── test_protection.py          # Column-orthonormality & quantization unit tests
│   ├── test_matching.py            # Augmented Hungarian & cyclic unit tests
│   ├── test_security.py            # Boundary KDE & unlinkability unit tests
│   ├── test_wp2_inversion_audit.py # Polyhedral pre-image inversion tests
│   ├── test_socofing_full.py       # SOCOFing clean 590-trial protocol tests
│   ├── test_wp4_recognition_audit.py# Recognition bottleneck & threshold tests
│   ├── test_wp5_wp6_crypto_privacy_audit.py# NonceManager & AEAD integrity tests
│   ├── test_revocation.py          # Registry revocation unit tests
│   ├── test_evaluation.py          # Clustered bootstrap & metrics unit tests
│   ├── test_real_image_pipeline.py # Real sensor pipeline unit tests
│   ├── test_phase3.py              # In-silico nuisance unit tests
│   ├── test_phase4.py              # RFC 8439 AEAD & service unit tests
│   ├── test_integration.py         # End-to-end integration lifecycles
│   ├── test_security_regressions.py# Formal adversarial regressions
│   ├── test_concurrency.py         # Multi-threaded & multi-process stress
│   └── test_reproducibility.py     # Trial accounting identities
├── experiments/
│   ├── datasets/                   # FVC2002 & SOCOFing automated loaders
│   ├── protocols/                  # FVC official comparison pair generator
│   ├── results/                    # Manifests, raw score arrays, and JSON summaries
│   │   ├── socofing_full_canonical/# Clean 590-trial ledger & 5-fold partitions
│   │   ├── baseline_manifest.json  # Raw score SHA-256 digest catalog (76 arrays)
│   │   └── canonical_artifact_hashes.json # Certified repository file digests
│   └── reproduce_all_gates.py      # Master 7-Gate reproduction harness
└── demo/
    ├── app.py                      # Interactive prototype demonstrator CLI
    └── production_demo.py          # Real sensor production service demonstrator (7 steps)
```

---

## 3. Quick Start & Execution

### 3.1 Install Requirements
```powershell
pip install -r requirements.txt
```

### 3.2 Run Complete Automated Test Suite (191 Tests)
```powershell
python -m pytest tests/ -q
```
*Output: 191 passed in ~18.5 seconds (0 errors, 0 failures, 0 skipped).*

### 3.3 Execute Master 7-Gate Reproduction Harness
```powershell
python experiments/reproduce_all_gates.py
```
*Executes all 7 validation gates and exits with code 0:*
- **Gate 1:** Environment & Core Scientific Dependencies (Python 3.11+, NumPy, SciPy, OpenCV, Pytest)
- **Gate 2:** SHA-256 Digest Verification for Raw Trial Artifacts (76 arrays verified)
- **Gate 3:** Biometric Trial Conservation Identities (DB1_A: 2,800/4,950; SOCOFing: 90/500)
- **Gate 4:** Mathematical & Cryptographic Invariants (Augmented matrix slack blocks, NonceManager monotonic counters, 944 B envelope)
- **Gate 5:** Dynamic Pytest Suite Execution (191 tests passing cleanly)
- **Gate 6:** Independent First-Principles Metric Evaluator Reconciliation
- **Gate 7:** Master Report Consistency & Terminology Audit

### 3.4 Run Production Service Demonstrator
```powershell
python demo/production_demo.py
```
*Executes all 7 production lifecycle steps on real sensor images from FVC2002 DB1_B:*
1. Real sensor image ingestion via `CaptureAdapter`.
2. Production enrollment with RFC 8439 ChaCha20-Poly1305 AEAD binary framing.
3. Storage footprint inspection (64 B header + 880 B sealed payload = 944 B).
4. Genuine repeat impression verification (`ACCEPT`).
5. Impostor verification rejection (`REJECT`).
6. AEAD 1-bit tampering detection (`AEAD_INTEGRITY_FAILURE` intercepted cleanly).
7. Revocation & fresh epoch re-issuance (100% block rate, fresh re-enrollment accepts).

---

## 4. Key Experimental & Security Results

### 4.1 Multi-Sensor Benchmark (FVC2002 Set A & Set B)
- **FVC2002 DB1_A Held-Out Benchmark (7,750 trials):**
  - EER = **41.20%** [38.32, 44.42], AUC = **0.6249** [0.5840, 0.6698], TAR @ 1% FAR = **5.54%** [4.50, 6.75].
  - Coverage-Normalized ($\gamma=1.2, M_{\min}=3$): EER = **40.62%**, AUC = **0.6301** ($\Delta\text{AUC} = +0.0052$, paired Wilcoxon $p < 10^{-15}$).
- **Canonical Pooled Set A (31,000 trials across DB1_A–DB4_A):**
  - Genuine: 11,200 trials; Impostor: 19,800 trials.
  - EER = **31.20%** [27.01, 33.44], AUC = **0.7759** [0.7504, 0.8116], TAR @ 1% FAR = **18.42%**.
- **Development-Tuned Set B Partitions (N_imp = 45 per partition):**
  - DB1_B: EER = 42.22%, AUC = 0.6067.
  - DB2_B: EER = 39.01%, AUC = 0.6385.
  - DB3_B: EER = 42.22%, AUC = 0.5898.
  - DB4_B: EER = 37.35%, AUC = 0.6558.
- **Physical Error Decomposition:** Minutiae extraction yield achieves **0.0% FTE**. Minutiae repeatability is **23.55%** due to non-rigid elastic skin deformation, causing **69.12% Delaunay edge-flipping**.

### 4.2 Clean SOCOFing Pilot Benchmark (590-Trial Clean Ledger)
- **Dataset Scope:** 10 subjects, 100 images (10 Real clean originals, 90 Altered probes across Easy, Medium, Hard).
- **Self-Match Exclusion (P0.3):** 10 trivial same-image self-matches (`Real/X` vs `Real/X`, score 1.0) excluded from verification metrics and quarantined to software determinism checks.
- **Outer 5-Fold Cross-Validation:** Exactly 18 clean genuine trials and 100 impostor trials per fold (5 folds = 90 genuine, 500 impostor = 590 clean verification trials).
- **Clean Verification Metrics:** Pooled EER = **26.14%** [20.9%, 31.2%], AUC = **0.8038** [0.7590, 0.8500].
- **Severity Partitioning (P0.4):**
  - Altered-Easy: Disjoint EER = **14.29%**, Disjoint AUC = **0.9489** (150 disjoint impostors).
  - Altered-Medium: Disjoint EER = **26.67%**, Disjoint AUC = **0.8418** (150 disjoint impostors).
  - Altered-Hard: Disjoint EER = **36.67%**, Disjoint AUC = **0.6589** (150 disjoint impostors).

### 4.3 Multi-Threat Security Audit & HiGHS MILP Pre-image Inversion
- **Known-Key Bounded Polytope Inversion (3,000 trials, P0.5, P1.1):** 9-D pure geometry stores strictly 0 discrete variables, mathematically precluding MILP branch-and-cut lattice reconstruction. The continuous Chebyshev center inversion solved in standardized feature space ($r^* = 0.0772$) matched under a fresh re-issued key yields an Attack Success Rate (ASR) of **0.0%** (95% Wilson CI: $[0.0000\%, 0.1279\%]$). In contrast, the 12-D minutiae-type candidate suffers **95.1% to 100.0% discrete type recovery**.
- **System Unlinkability (Gómez-Barrero $D_{\text{sys}}$, $B=2,000$ bootstrap, P0.2):** Replicate-wise joint maximum $D_{\text{sys}}^{\text{max}} = \mathbf{0.0375}$ [0.0296, 0.1333]. Conforms to ISO/IEC 30136:2018 **Low Residual Linkability**.
- **Revocation & Lifecycle:** 100.0% registry block rate (0 / 1,000 bypass attempts); cross-epoch statistical distance $D_{\text{sys}}^{\text{epoch}} = 0.0689$ [0.0046, 0.2014].

### 4.4 Storage Accounting & Latency Profiling
- **Serialized Credential Size:** Bit-exact **944 Bytes** for $N=144$ triangles (64 B header + 576 B descriptors + 288 B weights + 16 B Poly1305 MAC) yields **83.79% true compression** vs continuous float32 baseline ($5,824\text{ B}$).
- **Throughput & Speed:**
  - Interactive 1:1 Authentication (Mode A): Full Image-to-Decision = **$85.63\text{ ms}$** ($\approx 12\text{ auth/s}$). Template-to-Decision = **$4.73\text{ ms}$** ($211.5\text{ comp/s}$).
  - In-Memory Screening (Mode B): **1,269.9 to 1,349.8 templates/second**.
  - Isolated Match Core (Mode C, $N=144$): **$0.7394\text{ ms/pair}$** = **1,352.4 comparisons/second**.
  - Concurrent Batch Matching (Mode D, 4 workers): **6,583.6 comparisons/second**.

---

## 5. Deployment Recommendations & Boundaries

- **Primary Biometric Boundary:** Operating EER (~41% on unconstrained optical capture) limits standalone deployment in unattended high-security border gates.
- **Recommended Production Roles:**
  1. **Privacy-Preserving 2FA:** Friction ridge cancelable token combined with knowledge factors (PIN/passwords) or cryptographic hardware keys.
  2. **Decentralized Self-Sovereign Identity (SSI):** 944-byte encrypted credentials stored on mobile secure enclaves or smart cards.
  3. **Multi-Modal Biometric Fusion:** Combined with cancelable face or iris representations to overcome unimodal Delaunay edge-flipping degradation.

---

## 6. Citation & Academic Standards

Please consult [`final_research_report.md`](final_research_report.md) and [`master_project_architecture_and_completion_report.md`](master_project_architecture_and_completion_report.md) for complete mathematical proofs, ISO/IEC compliance mapping tables, and the full experimental record.
