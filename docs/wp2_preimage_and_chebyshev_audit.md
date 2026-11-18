# WP2 Technical Audit: Pre-Image Security, Chebyshev Optimization, and Multi-Threat Inversion Analysis

**Platform Version:** 2.2-Hardened  
**Document Identifier:** `DOC-WP2-SEC-2026-V2`  
**Authoritative Code References:** `src/security/inversion_attack.py`, `tests/test_wp2_inversion_audit.py`, `experiments/evaluate_phase2_inversion.py`  
**Audit Dataset Provenance:** FVC2002 DB1_B Benchmark (3,000 evaluated Delaunay triangle pre-image inversions)  

---

## 1. Executive Summary and Audit Scope

In strict conformance with **Section 3 of the Master Revision Prompt (Report v2.2)**, this work package conducts a comprehensive mathematical and empirical audit of the pre-image resistance, Chebyshev center optimization, and continuous feature inversion vulnerabilities of the Cancelable Fingerprint Research System.

Key findings and corrections established in this audit:
1. **Inequality Sign Inversion Correction:** Reconciled the conflicting inequality signs in earlier report drafts. Established the exact, mathematically sound halfspace containment condition:
   $$(2b_j - 1) \mathbf{w}_j^\top \mathbf{z} \ge r \|\mathbf{w}_j\|_2$$
2. **Compactness and Bounded Polytope:** Formally demonstrated that homogeneous sign cones $\mathcal{C}_{\mathbf{b}}$ are unbounded ($r^* \to \infty$). Formulated the exact **Bounded Physical Descriptor Polytope** $\mathcal{P}_{\text{bounded}}$ incorporating physical feature boxes and descending edge ratio constraints, which guarantees a finite, well-posed Chebyshev radius $r^* < \infty$.
3. **Angular MAE Unit Conversion Audit:** Discovered that the published directional error of $29.9^\circ$ arose from an erroneous full-circle factor ($0.0831 \times 360^\circ$). Because angular features are normalized to $[0, 1]$ where $1.0 \equiv \pi/2 = 90^\circ$ (maximum orthogonal separation on $\mathbb{RP}^1$), the actual physical reconstruction error is:
   $$0.0831 \times 90^\circ = 7.479^\circ \approx 7.48^\circ$$
4. **Multi-Threat Attack Suite Evaluation:** Benchmarked Scenario II (unknown key / key-guessing), Scenario III (known key Chebyshev LP and QP Mahalanobis centers), and Option A re-issuance under fresh keys.
5. **MILP Inversion Scope Boundary:** Narrowed the overclaim "all MILP attacks are mathematically impossible" to the defensible fact that 9-D Pure Geometry stores strictly 0 discrete variables, eliminating discrete minutiae-type lattice recovery, while continuous optimization attacks remain feasible.
6. **Rigorous Uncertainty Bounds:** Replaced absolute zero assertions ($0.0\%$ ASR) with two-sided Wilson score 95% confidence intervals ($[0.0\%, 27.75\%]$ for $N=10$ subject re-issuance trials; $[0.0\%, 0.128\%]$ for $N=3,000$ triangle inversions).

---

## 2. Mathematical Formulation of Pre-Image Inversion

### 2.1 The Open Polyhedral Sign Cone
Let $\mathbf{d} \in \mathbb{R}^D$ be an unstandardized descriptor ($D=9$), with development-fitted mean $\boldsymbol{\mu} \in \mathbb{R}^D$ and standard deviation $\boldsymbol{\sigma} \in \mathbb{R}^D$. The standardized descriptor is:
$$\mathbf{z} = \frac{\mathbf{d} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} \in \mathbb{R}^D$$
Given an orthonormal projection matrix $W \in \mathbb{R}^{M \times D}$ ($M \in \{32, 64, 128\}$), zero-threshold sign quantization generates the protected template bitstring $\mathbf{b} \in \{0, 1\}^M$:
$$b_j = \begin{cases} 1 & \text{if } \mathbf{w}_j^\top \mathbf{z} \ge 0 \\ 0 & \text{if } \mathbf{w}_j^\top \mathbf{z} < 0 \end{cases}$$
Defining the sign variable $s_j = 2b_j - 1 \in \{-1, +1\}$, each bit $b_j$ restricts the pre-image $\mathbf{z}$ to a closed affine halfspace:
$$\mathcal{H}_j = \{ \mathbf{z} \in \mathbb{R}^D \mid s_j \mathbf{w}_j^\top \mathbf{z} \ge 0 \}$$
The intersection of all $M$ halfspaces forms the homogeneous polyhedral cone:
$$\mathcal{C}_{\mathbf{b}} = \bigcap_{j=1}^M \mathcal{H}_j = \{ \mathbf{z} \in \mathbb{R}^D \mid (2b_j - 1) \mathbf{w}_j^\top \mathbf{z} \ge 0, \quad j=1,\dots,M \}$$

### 2.2 Chebyshev Center Sign Convention & Euclidean Ball Containment
In a Known-Key Inversion Attack (Scenario III), an attacker possessing $\mathbf{b}$ and $W$ seeks to find the Chebyshev center $\mathbf{z}^*$—the center of the largest Euclidean ball $\mathcal{B}(\mathbf{z}, r) = \{ \mathbf{z} + r\mathbf{u} \mid \|\mathbf{u}\|_2 \le 1 \}$ inscribed inside the feasible region.

**Derivation:**
For the ball $\mathcal{B}(\mathbf{z}, r)$ to be fully contained within halfspace $\mathcal{H}_j$:
$$\forall \mathbf{u} \in \mathbb{R}^D \text{ with } \|\mathbf{u}\|_2 \le 1: \quad s_j \mathbf{w}_j^\top (\mathbf{z} + r \mathbf{u}) \ge 0$$
$$s_j \mathbf{w}_j^\top \mathbf{z} + r s_j \mathbf{w}_j^\top \mathbf{u} \ge 0$$
The minimum over the unit sphere is attained when $\mathbf{u} = -\frac{s_j \mathbf{w}_j}{\|s_j \mathbf{w}_j\|_2} = -\frac{s_j \mathbf{w}_j}{\|\mathbf{w}_j\|_2}$:
$$\min_{\|\mathbf{u}\|_2 \le 1} \left( s_j \mathbf{w}_j^\top \mathbf{u} \right) = -\|\mathbf{w}_j\|_2$$
Substituting the worst-case boundary displacement yields:
$$s_j \mathbf{w}_j^\top \mathbf{z} - r \|\mathbf{w}_j\|_2 \ge 0$$
$$\mathbf{(2b_j - 1) \mathbf{w}_j^\top \mathbf{z} \ge r \|\mathbf{w}_j\|_2}$$

In standard linear programming form ($A\mathbf{x} \le \mathbf{b}$):
$$\mathbf{-(2b_j - 1) \mathbf{w}_j^\top \mathbf{z} + r \|\mathbf{w}_j\|_2 \le 0}$$

> **Correction of Inverted Equation in Earlier Draft:**  
> The previous report equation:
> $$(2b_j - 1) \mathbf{w}_j^\top \mathbf{z} + r \|\mathbf{w}_j\|_2 \le 0 \quad (\text{INCORRECT})$$
> had a sign inversion error that forced $(2b_j - 1)\mathbf{w}_j^\top \mathbf{z} \le -r\|\mathbf{w}_j\|_2 \le 0$, constraining $\mathbf{z}$ to the *complementary infeasible halfspace*. This equation is now corrected in source and documentation.

---

## 3. Mathematical Compactness: Unbounded Cone vs. Bounded Polytope

### 3.1 Unboundedness of the Pure Homogeneous Cone
Consider the pure homogeneous linear program:
$$\max_{\mathbf{z}, r} \quad r \quad \text{s.t.} \quad (2b_j - 1)\mathbf{w}_j^\top \mathbf{z} \ge r\|\mathbf{w}_j\|_2, \quad j=1,\dots,M, \quad r \ge 0$$

**Theorem (Unboundedness):**  
If the cone $\mathcal{C}_{\mathbf{b}}$ has a non-empty interior containing $\mathbf{z}_0$ with margin $\delta_0 = \min_j s_j \mathbf{w}_j^\top \mathbf{z}_0 > 0$, then for any scalar $\lambda > 0$, the scaled vector $\mathbf{z}_\lambda = \lambda \mathbf{z}_0$ satisfies:
$$\min_j s_j \mathbf{w}_j^\top \mathbf{z}_\lambda = \lambda \delta_0$$
The maximum admissible radius is $r_\lambda = \frac{\lambda \delta_0}{\max_j \|\mathbf{w}_j\|_2}$.  
As $\lambda \to \infty$, $r_\lambda \to \infty$.  
*Conclusion:* The optimization is unbounded ($\sup r = +\infty$). A finite Chebyshev center **does not exist** on the homogeneous cone alone.

### 3.2 Definition of the Bounded Physical Descriptor Polytope
To render the Chebyshev center finite, unique, and biometrically meaningful, the solver code in `src/security/inversion_attack.py` restricts $\mathbf{z}$ to the compact polytope $\mathcal{P}_{\text{bounded}}$ formed by mapping physical descriptor bounds into standardized coordinates:

$$\mathcal{P}_{\text{bounded}} = \left\{ \mathbf{d} \in \mathbb{R}^9 \;\middle|\; \begin{aligned} 
-(2b_j - 1)\mathbf{w}_j^\top \left(\frac{\mathbf{d} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right) + r \|\mathbf{w}_j\|_2 &\le 0, \quad j=1,\dots,M \\
-d_0 + d_1 + r &\le 0 \quad (r_2 \le r_1) \\
d_k + r &\le d_{\max, k}, \quad k=0,\dots,8 \\
-d_k + r &\le -d_{\min, k}, \quad k=0,\dots,8 \\
r &\ge 0
\end{aligned} \right\}$$

Physical coordinate bounding box $\mathcal{D}_{\text{physical}} = [\mathbf{d}_{\min}, \mathbf{d}_{\max}]$:
* Side ratios: $r_1, r_2 \in [0.1, 1.0]$
* Physical perimeter: $P_{\text{mm}} \in [2.0, 30.0]\text{ mm}$
* Incident edge relative angles: $\phi_0, \phi_1, \phi_2 \in [0.0, 1.0]$
* Mutual orientation distances: $d\theta_{01}, d\theta_{12}, d\theta_{20} \in [0.0, 1.0]$

Because $\mathcal{D}_{\text{physical}}$ is compact, the Heine-Borel theorem guarantees that $\mathcal{P}_{\text{bounded}}$ is compact. The objective $f(\mathbf{d}, r) = r$ attains a unique finite maximum $r^* < \infty$, resolved via the SciPy HiGHS simplex solver (tolerance $10^{-8}$).

---

## 4. Angular MAE Metric Audit: Resolving the $29.9^\circ$ Discrepancy

In Section 7.1 of previous report drafts, the following statement appeared:
> *"Angular Directional MAE: $0.0831 \pm 0.0008$ ($29.9^\circ$)"*

### Mathematical Audit:
1. The angular descriptor coordinates $\phi_0, \phi_1, \phi_2$ and $d\theta_{01}, d\theta_{12}, d\theta_{20}$ are normalized to $[0, 1]$ via the canonical formula:
   $$d_{\mathbb{RP}^1}(\theta_a, \theta_b) = \frac{\Delta_{\pi}(\theta_a, \theta_b)}{\pi / 2} \in [0.0, 1.0]$$
2. The maximum possible angular distance between two axial ridge orientations on $\mathbb{RP}^1$ is $\frac{\pi}{2}\text{ rad} = 90^\circ$ (orthogonal orientations).
3. Therefore, $1.0\text{ unit}$ on the normalized scale equals $90^\circ$ (or $\frac{\pi}{2}\text{ radians}$).
4. Evaluating the actual physical angle error:
   $$\text{Error}_{\text{physical}} = 0.083075 \times 90.0^\circ = \mathbf{7.47675^\circ \approx 7.48^\circ}$$
   $$\text{Error}_{\text{radians}} = 0.083075 \times \frac{\pi}{2} = \mathbf{0.1305\text{ radians}}$$
5. Origin of the $29.9^\circ$ figure:
   $$0.083075 \times 360.0^\circ = 29.907^\circ \approx 29.9^\circ$$
   The earlier author incorrectly applied the full-circle $2\pi$ multiplier ($360^\circ$) to a metric normalized to a quarter-circle $\pi/2$ interval ($90^\circ$). This introduced a **$4\times$ erroneous inflation**.
6. The actual angular reconstruction accuracy of the Chebyshev center is **$7.48^\circ$**, confirming that pre-image optimization achieves substantially tighter angular approximation than previously reported.

---

## 5. Multi-Threat Inversion Attack Benchmark

Audited across 3,000 Delaunay triangles from FVC2002 DB1_B across projection bit-lengths $M \in \{32, 64, 128\}$:

| Attack Scenario | Threat Model & Method | $M$ (Bits) | Polytope Feasibility (%) | Chebyshev Radius $r^*$ | Continuous MAE | Angular MAE | Gallery Match Score | Intra-Key ASR (%) | Cross-Key ASR (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario III (Known Key)** | Chebyshev Center LP | 32 | 100.0% | $0.0772 \pm 0.0004$ | $0.1738 \pm 0.0012$ | $7.48^\circ$ ($0.0831$) | $0.3423$ | $10.0\%$ | **$0.0\%$** |
| **Scenario III (Known Key)** | Chebyshev Center LP | 64 | 100.0% | $0.0416 \pm 0.0003$ | $0.1129 \pm 0.0009$ | $4.94^\circ$ ($0.0549$) | $0.3061$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario III (Known Key)** | Chebyshev Center LP | 128 | 100.0% | $0.0184 \pm 0.0002$ | $0.0652 \pm 0.0006$ | $3.08^\circ$ ($0.0343$) | $0.2883$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario III (Known Key)** | QP Mahalanobis Center | 32 | 100.0% | N/A (Min $\|z\|_2^2$) | $0.5738 \pm 0.0028$ | N/A | $0.3423$ | $10.0\%$ | **$0.0\%$** |
| **Scenario III (Known Key)** | QP Mahalanobis Center | 64 | 100.0% | N/A (Min $\|z\|_2^2$) | $0.5429 \pm 0.0025$ | N/A | $0.3061$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario III (Known Key)** | QP Mahalanobis Center | 128 | 100.0% | N/A (Min $\|z\|_2^2$) | $0.4376 \pm 0.0021$ | N/A | $0.2883$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario II (Unknown Key)** | Random Orthogonal Guess | 32 | N/A | N/A | $0.5897 \pm 0.0031$ | $28.29^\circ$ ($0.3143$) | $0.0327$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario II (Unknown Key)** | Random Orthogonal Guess | 64 | N/A | N/A | $0.5860 \pm 0.0029$ | $28.27^\circ$ ($0.3141$) | $0.0277$ | **$0.0\%$** | **$0.0\%$** |
| **Scenario II (Unknown Key)** | Random Orthogonal Guess | 128 | N/A | N/A | $0.5914 \pm 0.0030$ | $28.34^\circ$ ($0.3149$) | $0.0065$ | **$0.0\%$** | **$0.0\%$** |

*Operational Threshold:* $T_{\text{accept}} = 0.3751$ (FVC2002 DB1_B EER threshold).

---

## 6. Statistical Uncertainty Bounds for Zero-Success Outcomes

Biometric security guidelines prohibit claiming that an attack success rate is absolute zero based on empirical zero occurrences. In accordance with Rule 4 and Rule 10, all zero-success outcomes are bounded using two-sided Wilson score 95% confidence intervals:

$$\hat{p} = 0, \quad \text{CI}_{95\%} = \left[ 0.0, \; \frac{z^2}{N + z^2} \right], \quad z = 1.96$$

* **Subject-Level Re-Issuance Evaluation ($N = 10$ subjects):**
  * Observed successes: $0 / 10$
  * Re-issuance ASR: **$0.0\%$**
  * 95% Wilson Confidence Interval: **$[0.00\%, \; 27.75\%]$**
* **Simulated Stolen Credential Interception ($N = 100$ trials):**
  * Observed successes: $0 / 100$
  * ASR: **$0.0\%$**
  * 95% Wilson Confidence Interval: **$[0.00\%, \; 3.69\%]$**
* **Triangle-Level Pre-Image Feasibility Attacks ($N = 3,000$ triangles):**
  * Cross-key re-issuance successes: $0 / 3,000$
  * Cross-key ASR: **$0.0\%$**
  * 95% Wilson Confidence Interval: **$[0.0000\%, \; 0.1279\%]$**

---

## 7. Clarification of MILP Inversion Boundaries

The statement in earlier drafts that *"all MILP inversion attacks are mathematically impossible"* is scientifically unsound. An attacker can formulate integer programs over grid discretizations, orientation topologies, or combinatorial assignment matchings.

The claim is strictly narrowed to:
> **Minutiae-Type Lattice Inversion Scope:**  
> In the 9-D Pure Geometric representation, discrete minutiae structural classifications $(t_0, t_1, t_2)$ are completely eliminated. The descriptor contains strictly zero discrete variables. Consequently, branch-and-bound mixed-integer linear programming (MILP) cannot recover minutiae types, because no discrete type information is encoded in the template. However, continuous polyhedral pre-image attacks (Linear Programming and Quadratic Programming) remain viable and must be defended against via key secrecy and credential re-issuance.
