# Mathematical Verification: Subspace Projections (QR) & Symmetrized ZCA Equivariance

**Module References:** [src/protection/projection.py](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/src/protection/projection.py), [src/services/mcc_builder.py](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/src/services/mcc_builder.py), [src/descriptors/mcc.py](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/src/descriptors/mcc.py)  
**Release Version:** `v2.5-Research-Release`  
**Author:** Mathematical Optimization & Biometric Security Auditor  

---

## 1. Dual-Architecture QR Subspace Projections

The codebase uses two distinct projection dimensionalities for its two architectural generations. They must not be conflated into a single ambiguous formula.

### 1.1 Legacy Delaunay Mode: Column-Orthonormal Embedding ($32 \times 9$)
In the legacy 9-D Delaunay architecture:
- Input feature dimension: $d_{\text{in}} = 9$.
- Output projection dimension: $d_{\text{out}} = 32$.
- Here $d_{\text{out}} \ge d_{\text{in}}$.

**Construction:**
1. A pseudo-random Gaussian matrix $A \in \mathbb{R}^{32 \times 9}$ is generated deterministically from HKDF key bytes:
   $$A_{ij} \sim \mathcal{N}(0, 1), \quad 1 \le i \le 32, \; 1 \le j \le 9$$
2. Thin QR decomposition yields $A = Q R$ where $Q \in \mathbb{R}^{32 \times 9}$ and $R \in \mathbb{R}^{9 \times 9}$ upper triangular.
3. Canonical diagonal sign correction enforces deterministic positive diagonal elements:
   $$s_j = \operatorname{sgn}(R_{jj}), \quad Q_{\text{canon}} = Q \cdot \operatorname{diag}(s_1, \dots, s_9)$$
4. The projection matrix is:
   $$W_{\text{legacy}} = Q_{\text{canon}} \in \mathbb{R}^{32 \times 9}$$

**Orthogonality Invariant:**
Because $Q$ has orthonormal columns in $\mathbb{R}^{32}$:
$$W_{\text{legacy}}^\top W_{\text{legacy}} = I_9 \in \mathbb{R}^{9 \times 9}$$
*(Note: $W_{\text{legacy}} W_{\text{legacy}}^\top \ne I_{32}$ because $\operatorname{rank}(W) \le 9 < 32$.)*

---

### 1.2 Modern MCC-Spectral Mode: Row-Orthonormal Subspace Projection ($192 \times 256$)
In the primary 256-D MCC-Spectral architecture:
- Input cylinder feature dimension: $d_{\text{in}} = 256$.
- Output projection dimension: $d_{\text{out}} = 192$.
- Here $d_{\text{out}} < d_{\text{in}}$.

**Construction:**
1. A pseudo-random Gaussian matrix $A \in \mathbb{R}^{256 \times 192}$ is generated deterministically from HKDF key bytes:
   $$A_{ij} \sim \mathcal{N}(0, 1), \quad 1 \le i \le 256, \; 1 \le j \le 192$$
2. Thin QR decomposition yields $A = Q R$ where $Q \in \mathbb{R}^{256 \times 192}$ and $R \in \mathbb{R}^{192 \times 192}$.
3. Canonical diagonal sign correction enforces positive diagonal signs:
   $$s_j = \operatorname{sgn}(R_{jj}), \quad Q_{\text{canon}} = Q \cdot \operatorname{diag}(s_1, \dots, s_{192})$$
4. The projection matrix is defined as the transpose:
   $$W_{\text{MCC}} = Q_{\text{canon}}^\top \in \mathbb{R}^{192 \times 256}$$

**Orthogonality Invariant:**
Because $Q_{\text{canon}}^\top Q_{\text{canon}} = I_{192}$:
$$W_{\text{MCC}} W_{\text{MCC}}^\top = (Q^\top) (Q^\top)^\top = Q^\top Q = I_{192} \in \mathbb{R}^{192 \times 192}$$
This guarantees that $W_{\text{MCC}}$ is an exact orthogonal projection operator from $\mathbb{R}^{256}$ onto a 192-dimensional subspace.

**Nullspace Ambiguity $\ker(W_{\text{MCC}})$:**
The nullspace dimension is:
$$\dim(\ker(W_{\text{MCC}})) = 256 - 192 = 64$$
For any vector $v \in \ker(W_{\text{MCC}})$ and any standardized descriptor $z$, $W(z + v) = Wz$, providing a continuous 64-dimensional information-theoretic pre-image ambiguity under zero-threshold sign quantization.

---

## 2. MCC Symmetrized ZCA Whitening & Dual-Flip Equivariance

### 2.1 The Spatial Grid Inversion Permutation $P$
In Minutia Cylinder-Code (MCC), each minutia centers an $8 \times 8 \times 4 = 256$ cell cylinder. Under an axial ridge orientation flip $\theta_i \mapsto \theta_i + \pi$, the local coordinate axes invert:
$$(x' - x_i, y' - y_i) \mapsto -(x' - x_i, y' - y_i)$$
This spatial inversion corresponds to reversing both the horizontal and vertical grid axes in each directional slice:
$$P: \mathbb{R}^{256} \to \mathbb{R}^{256}, \quad P^2 = I_{256}$$
Because $P$ is a pure permutation of coordinates, it is an orthogonal matrix:
$$P^\top = P^{-1} = P, \quad \|P v\|_2 = \|v\|_2 \quad \forall v \in \mathbb{R}^{256}$$

---

### 2.2 Standardization Equivariance
Let $D \in \mathbb{R}^{N \times 256}$ be raw MCC descriptors. In [src/services/mcc_builder.py](file:///C:/Users/mella/.gemini/antigravity/scratch/CancelableFingerprintHybrid/src/services/mcc_builder.py), Set-B channel statistics $(\mu_B, \sigma_B)$ are symmetrized via:
$$\mu_{\text{sym}} = \frac{1}{2}\big(\mu_B + P \mu_B\big), \qquad \sigma_{\text{sym}} = \frac{1}{2}\big(\sigma_B + P \sigma_B\big)$$
Applying $P$ to $\mu_{\text{sym}}$:
$$P \mu_{\text{sym}} = \frac{1}{2}\big(P \mu_B + P^2 \mu_B\big) = \frac{1}{2}\big(P \mu_B + \mu_B\big) = \mu_{\text{sym}}$$
and similarly $P \sigma_{\text{sym}} = \sigma_{\text{sym}}$.
Consequently, coordinatewise z-score standardization strictly commutes with the flip operator:
$$\operatorname{Standardize}(P D) = \frac{P D - \mu_{\text{sym}}}{\sigma_{\text{sym}}} = \frac{P D - P \mu_{\text{sym}}}{P \sigma_{\text{sym}}} = P\left(\frac{D - \mu_{\text{sym}}}{\sigma_{\text{sym}}}\right) = P \operatorname{Standardize}(D)$$

---

### 2.3 Symmetrized ZCA Covariance Commutation
Let $Z = \operatorname{Standardize}(D)$ and $Z_f = P Z$ be standardized Set-B descriptors and their duals. The empirical symmetrized covariance matrix is:
$$C_{\text{sym}} = \frac{1}{2N}\big(Z^\top Z + Z_f^\top Z_f\big) = \frac{1}{2N}\big(Z^\top Z + P Z^\top Z P\big)$$
Conjugating $C_{\text{sym}}$ by $P$:
$$P C_{\text{sym}} P = \frac{1}{2N}\big(P Z^\top Z P + P^2 Z^\top Z P^2\big) = \frac{1}{2N}\big(P Z^\top Z P + Z^\top Z\big) = C_{\text{sym}}$$
Because $P C_{\text{sym}} = C_{\text{sym}} P$, $C_{\text{sym}}$ and $P$ commute and share a common eigenbasis $V$:
$$C_{\text{sym}} = V \Lambda V^\top, \qquad W_{\text{ZCA}} = V (\Lambda + \lambda I)^{-1/2} V^\top \quad (\lambda = 1.0)$$
Therefore, the regularized ZCA whitening matrix $W_{\text{ZCA}}$ commutes with $P$:
$$P W_{\text{ZCA}} = W_{\text{ZCA}} P \iff (P z) W_{\text{ZCA}} = P (z W_{\text{ZCA}})$$

---

### 2.4 L2 Normalization & Dual-Path Verification
For whitened vector $\tilde{z} = z W_{\text{ZCA}}$:
$$\|\tilde{z} P\|_2 = \|\tilde{z}\|_2$$
so L2 normalization preserves exact bit equivariance:
$$\operatorname{Normalize}(P \tilde{z}) = P \operatorname{Normalize}(\tilde{z})$$

At verification time, the matcher checks both the direct query bitstrings $b_q$ and the dual query bitstrings $b_q^{\text{dual}} = \operatorname{Quantize}(W \cdot P \tilde{z})$ against enrolled bitstrings:
$$D_{ij} = \min\big(\operatorname{Ham}(b_q^{(i)},\, b_e^{(j)}),\;\; \operatorname{Ham}(b_q^{\text{dual}(i)},\, b_e^{(j)})\big)$$
This guarantees $100\%$ detection invariance against $180^\circ$ orientation flips while preventing cross-finger spurious matches.
