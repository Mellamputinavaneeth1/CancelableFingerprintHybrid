# WP1 Technical Specification: Canonical Descriptor Mathematics, Invariance Guarantees, and Manifold Dimensionality Analysis

**Platform Version:** 2.2-Hardened  
**Document Identifier:** `DOC-WP1-MATH-2026-V2`  
**Authoritative Code References:** `src/graph/descriptors.py`, `tests/test_descriptor_properties.py`, `experiments/evaluate_wp1_descriptors.py`  
**Audit Dataset Provenance:** FVC2002 DB1_B Development Split (11,450 evaluated Delaunay simplices)  

---

## 1. Executive Summary and Scope

This technical document provides the complete, mathematically audited, and reproducible specification of the geometric minutiae triangle descriptors utilized throughout the Cancelable Fingerprint Research System.

In strict compliance with the **Master Revision Prompt (v2.2-Hardened)**, this document:
1. Formalizes the coordinate frames, conventions, and continuous-to-discrete mappings.
2. Provides a comprehensive feature specification table linking mathematical symbols to source functions and automated verification test IDs.
3. Formulates the exact canonical vertex ordering and tie-breaking hierarchy, reconciling historical documentation errors regarding Cartesian sort keys.
4. Mathematically refutes the alleged algebraic redundancy of the 9th dimension ($d\theta_{20}$) via metric counterexamples on the project space $\mathbb{RP}^1$.
5. Details the constructive 8-fold sign ambiguity collision inherent in the 6-D parameterization, explaining the observed recognition collapse.
6. Documents sensor DPI handling and resolution fallback policies.
7. Evaluates the full 5-candidate dimensionality spectrum (5-D, 6-D, 8-D, 9-D, 12-D).

---

## 2. Coordinate Systems, Conventions, and Mathematical Foundations

### 2.1 Coordinate Frame and Units
* **Image Space:** $\mathbb{R}^2$ with continuous Euclidean coordinates $(x, y)$, where $x$ increases from left to right along the horizontal axis, and $y$ increases from top to bottom along the vertical axis.
* **Spatial Distances:** Raw pixel distances $l = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \in [0, \infty)$ pixels.
* **Physical Metric Conversion:** Fingerprint ridge period and spatial features are physically normalized using the sensor optical resolution $\text{DPI} \in \mathbb{R}^+$ (dots per inch):
  $$l_{\text{mm}} = l_{\text{pixels}} \times \frac{25.4}{\text{DPI}} \quad [\text{mm}]$$
  When sensor metadata is absent, unparseable, or invalid ($\le 0$), the pipeline enforces a frozen fallback resolution of $\text{DPI} = 500.0$ ($0.0508\text{ mm/pixel}$), conforming to the FBI CJIS-RS-0010 biometric standard.

### 2.2 Minutiae Representation and Axial Ridge Flow
A minutia $m_i$ is defined as the 5-tuple:
$$m_i = (x_i, y_i, \theta_i, t_i, q_i)$$
where:
* $(x_i, y_i) \in [0, W) \times [0, H)$ are sub-pixel spatial coordinates.
* $\theta_i \in [0, \pi)$ is the local ridge orientation on the real projective line $\mathbb{RP}^1 \cong \mathbb{R} / \pi\mathbb{Z}$. Fingerprint ridge orientation exhibits axial symmetry: ridges are undirected flow lines where $\theta \equiv \theta + k\pi, \forall k \in \mathbb{Z}$.
* $t_i \in \{0, 1\}$ represents minutia structural type ($0 = \text{ridge termination}, 1 = \text{ridge bifurcation}$).
* $q_i \in (0, 1.0]$ is the localized quality score derived from directional clarity and ridge contrast.

### 2.3 Distance Metric on the Axial Projective Line $\mathbb{RP}^1$
The shortest angular distance between two axial orientations $\theta_a, \theta_b \in [0, \pi)$ is defined by the function `_axial_diff`:
$$\Delta_{\pi}(\theta_a, \theta_b) = \min\left(|\theta_a - \theta_b| \pmod \pi, \; \pi - (|\theta_a - \theta_b| \pmod \pi)\right) \in \left[0, \frac{\pi}{2}\right]$$
Normalized to the unit interval $[0, 1]$ by dividing by the maximum orthogonal separation $\frac{\pi}{2}$:
$$d_{\mathbb{RP}^1}(\theta_a, \theta_b) = \frac{\Delta_{\pi}(\theta_a, \theta_b)}{\pi / 2} = \frac{2}{\pi} \Delta_{\pi}(\theta_a, \theta_b) \in [0.0, 1.0]$$
* Identical orientations ($\theta_a = \theta_b$ or $\theta_a = \theta_b \pm \pi$) yield $d_{\mathbb{RP}^1} = 0.0$.
* Orthogonal orientations ($|\theta_a - \theta_b| = \pi / 2$) yield $d_{\mathbb{RP}^1} = 1.0$.

---

## 3. Comprehensive Feature Specification Table

The authoritative 9-D descriptor vector $\mathbf{d} = (d_0, d_1, \dots, d_8)^\top \in \mathbb{R}^9$ characterizes an ordered triangle simplex $T = (v_0, v_1, v_2)$.

| Index | Symbol | Mathematical Formulation | Range | Units | Invariance Properties | Intrinsic Redundancy / Dependency | Source Function | Verification Test ID |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0$ | $r_1$ | $\frac{e_{\text{mid}}}{e_{\max}}$ | $(0.0, 1.0]$ | Dimensionless | $SE(2)$, Scale, $S_3$ Permutation | Independent (determines triangle shape ratio) | `extract_triangle_descriptor` | `test_scalene_all_six_permutations` |
| $1$ | $r_2$ | $\frac{e_{\min}}{e_{\max}}$ | $(0.0, 1.0]$ | Dimensionless | $SE(2)$, Scale, $S_3$ Permutation | Independent (triangle inequality: $r_1 + r_2 > 1$) | `extract_triangle_descriptor` | `test_law_of_cosines_interior_angle_determination` |
| $2$ | $P_{\text{mm}}$ | $(e_{\max} + e_{\text{mid}} + e_{\min}) \cdot \frac{25.4}{\text{dpi}}$ | $(0.0, \infty)$ | Millimeters (mm) | $SE(2)$, $S_3$ Permutation; **Scales linearly** ($s \cdot P$) | Independent (sets physical Euclidean scale) | `extract_triangle_descriptor` | `test_9d_uniform_scaling_behavior` |
| $3$ | $\phi_0$ | $\frac{2}{\pi} \min(|\theta_0 - \alpha_{0 \to 1}|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | Discards orientation sign ($\pm$ relative to directed edge) | `extract_triangle_descriptor` | `test_9d_rigid_rotation_and_translation_invariance` |
| $4$ | $\phi_1$ | $\frac{2}{\pi} \min(|\theta_1 - \alpha_{1 \to 2}|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | Discards orientation sign ($\pm$ relative to directed edge) | `extract_triangle_descriptor` | `test_9d_rigid_rotation_and_translation_invariance` |
| $5$ | $\phi_2$ | $\frac{2}{\pi} \min(|\theta_2 - \alpha_{2 \to 0}|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | Discards orientation sign ($\pm$ relative to directed edge) | `extract_triangle_descriptor` | `test_9d_rigid_rotation_and_translation_invariance` |
| $6$ | $d\theta_{01}$ | $\frac{2}{\pi} \min(|\theta_0 - \theta_1|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | Mutual distance on $\mathbb{RP}^1$ | `extract_triangle_descriptor` | `test_axial_diff_maximum_orthogonal` |
| $7$ | $d\theta_{12}$ | $\frac{2}{\pi} \min(|\theta_1 - \theta_2|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | Mutual distance on $\mathbb{RP}^1$ | `test_axial_diff_symmetry_and_bounds` |
| $8$ | $d\theta_{20}$ | $\frac{2}{\pi} \min(|\theta_2 - \theta_0|_{\pi}, \pi - |\dots|)$ | $[0.0, 1.0]$ | Normalized rad / $(\pi/2)$ | $SE(2)$, Scale, $S_3$ Permutation | **Metric-constrained, non-redundant** (Refutes 8-D claim) | `extract_triangle_descriptor` | `test_8d_absolute_distance_constraint_counterexample` |

*Note on Directed Angles:* For edge $(v_i, v_j)$, the incident edge orientation is $\alpha_{i \to j} = \operatorname{atan2}(y_j - y_i, x_j - x_i) \pmod \pi \in [0, \pi)$.

---

## 4. Canonical Vertex Ordering and Deterministic Tie-Breaking Hierarchy

### 4.1 Permutation Group $S_3$ Symmetries
An unordered triplet of Delaunay minutiae $\{m_A, m_B, m_C\}$ has $|S_3| = 3! = 6$ possible permutations. To achieve genuine biometric rotation and registration invariance, the feature extractor must canonically order the vertices such that:
$$\forall \sigma \in S_3, \quad \operatorname{ExtractDescriptor}(\sigma(m_A, m_B, m_C)) = \operatorname{ExtractDescriptor}(m_A, m_B, m_C)$$

### 4.2 Authoritative Sorting Algorithm and Implementation Truth
In historical drafts of the research report, an erroneous statement claimed that Cartesian coordinates $(x_i, y_i)$ were used as a tertiary tie-breaker. Such a key would destroy rigid rotation invariance ($SO(2)$) because Cartesian coordinates change under rotation.

The audited source code in `src/graph/descriptors.py` (`extract_triangle_descriptor_with_ordered_minutiae`) implements a strictly invariant lexicographical key:

```python
# Canonical Sorting Key Hierarchy in src/graph/descriptors.py
perm_key = (
    tuple(rounded_geom_key),      # 1. Primary: Descending side lengths (e_max, e_mid, e_min)
    chirality_key,                 # 2. Secondary: Planar cross-product chirality (+1, -1, 0)
    tuple(unrounded_geom_key),    # 3. Tertiary: Exact unrounded floating-point edge lengths
    quality_key,                   # 4. Quaternary: Descending minutiae extraction quality (q0, q1, q2)
    type_key                       # 5. Quinary: Discrete minutiae structural type (t0, t1, t2)
)
```

1. **Primary Key (`rounded_geom_key`):**
   Edge lengths are rounded to tolerance $\tau = \epsilon_{\text{rel}} \cdot \max(l_0, l_1, l_2)$ (default $\epsilon_{\text{rel}} = 0.03$). For scalene triangles, this completely and uniquely orders the vertices:
   $$v_0 = \text{opposite to } e_{\max}, \quad v_1 = \text{opposite to } e_{\text{mid}}, \quad v_2 = \text{opposite to } e_{\min}$$
2. **Secondary Key (`chirality_key`):**
   When two or three side lengths are equal within tolerance $\tau$ (isosceles or equilateral triangles), planar signed area (chirality) resolves the order:
   $$\chi = \operatorname{sgn}\left((x_1 - x_0)(y_2 - y_0) - (y_1 - y_0)(x_2 - x_0)\right) \in \{+1, -1, 0\}$$
   *Crucial Mathematical Invariance:* Because the 2D cross product of vector differences is invariant under translations and rigid rotations in $SO(2)$, $\chi$ preserves strict $SE(2)$ invariance!
3. **Tertiary Key (`unrounded_geom_key`):**
   Full IEEE 754 64-bit float precision side lengths breaking numerical near-ties.
4. **Quaternary Key (`quality_key`):**
   Descending minutia sensor quality scores $(q_0, q_1, q_2)$.
5. **Quinary Key (`type_key`):**
   Minutia ridge structural classification $(t_0, t_1, t_2) \in \{0, 1\}^3$.

*Zero Cartesian Dependence:* Cartesian coordinates $(x_i, y_i)$ are **never** evaluated in the sort tuple. SE(2) invariance is mathematically preserved across all triangles.

---

## 5. Mathematical Analysis of Candidate Dimensionalities

### 5.1 The 8-D Candidate and Refutation of the Algebraic Redundancy Claim
In early theoretical treatments, it was hypothesized that the 9th dimension ($d\theta_{20}$) could be dropped because the sum of directed angular differences around a closed loop is zero modulo $\pi$:
$$(\theta_0 - \theta_1) + (\theta_1 - \theta_2) + (\theta_2 - \theta_0) = 0 \pmod \pi$$

**Mathematical Refutation:**
The descriptor components $d\theta_{01}, d\theta_{12}, d\theta_{20}$ are **not** directed signed angles; they are **undirected geodesic distances** on the metric space $\mathbb{RP}^1 = \mathbb{R} / \pi\mathbb{Z}$.
On metric spaces, geodesic distances satisfy triangle inequalities rather than deterministic scalar additions:
$$|d\theta_{01} - d\theta_{12}| \le d\theta_{20} \le \min\left(d\theta_{01} + d\theta_{12}, \; 2.0 - (d\theta_{01} + d\theta_{12})\right)$$

**Constructive Counterexample (verified in `test_8d_absolute_distance_constraint_counterexample`):**
* **Case A:** Let $\theta = \left(0, \; \frac{\pi}{4}, \; \frac{\pi}{2}\right)$.
  * $d\theta_{01} = \frac{2}{\pi} \Delta_{\pi}(0, \pi/4) = 0.5$
  * $d\theta_{12} = \frac{2}{\pi} \Delta_{\pi}(\pi/4, \pi/2) = 0.5$
  * $d\theta_{20} = \frac{2}{\pi} \Delta_{\pi}(\pi/2, 0) = 1.0$
* **Case B:** Let $\theta = \left(\frac{\pi}{4}, \; 0, \; \frac{\pi}{4}\right)$.
  * $d\theta_{01} = \frac{2}{\pi} \Delta_{\pi}(\pi/4, 0) = 0.5$
  * $d\theta_{12} = \frac{2}{\pi} \Delta_{\pi}(0, \pi/4) = 0.5$
  * $d\theta_{20} = \frac{2}{\pi} \Delta_{\pi}(\pi/4, \pi/4) = 0.0$

In both cases, $(d\theta_{01}, d\theta_{12}) = (0.5, 0.5)$, yet $d\theta_{20}$ takes diametrically opposed extreme values ($1.0$ vs $0.0$).
*Conclusion:* $d\theta_{20}$ is **not** an algebraic function of $(d\theta_{01}, d\theta_{12})$. Dropping $d\theta_{20}$ discards genuine topological ordering information on $\mathbb{RP}^1$, causing empirical EER degradation from $39.73\%$ to $41.07\%$.

### 5.2 The 6-D Candidate and the 8-Fold Sign Ambiguity Collision
The 6-D candidate descriptor is defined as:
$$\mathbf{d}_{\text{6D}} = \left(r_1, \; r_2, \; P_{\text{mm}}, \; \phi_0, \; \phi_1, \; \phi_2\right)$$
where each $\phi_i = d_{\mathbb{RP}^1}(\theta_i, \alpha_{i \to i+1})$ measures the unsigned distance between minutia $i$'s ridge orientation and the directed incident edge $v_i \to v_{(i+1)\bmod 3}$.

**Sign Ambiguity Analysis:**
Because $d_{\mathbb{RP}^1}$ discards the sign of the angular displacement:
$$\phi_i(\alpha + \delta) = \phi_i(\alpha - \delta) = \frac{2}{\pi}\delta \quad \text{for } \delta \in \left(0, \frac{\pi}{2}\right)$$
For any fixed triangle geometry, each vertex admits two distinct physical ridge orientations:
$$\theta_i^{(s_i)} = (\alpha_{i \to i+1} + s_i \delta_i) \pmod \pi, \quad s_i \in \{+1, -1\}$$
Selecting sign choices $(s_0, s_1, s_2) \in \{+1, -1\}^3$ yields $2^3 = 8$ distinct orientation configurations $(\theta_0, \theta_1, \theta_2) \in [0, \pi)^3$.

**Empirical Verification:**
As verified in `test_6d_sign_ambiguity_eightfold_collision` and `audit_wp1_mathematics.py`:
* All 8 configurations produce identical 6-D descriptors:
  $$\mathbf{d}_{\text{6D}} = (0.9610, \; 0.9307, \; 9.2033\text{ mm}, \; 0.5000, \; 0.7000, \; 0.3000)$$
* Yet in 9-D space, all 8 configurations produce mutually distinct mutual orientation triplets $(d\theta_{01}, d\theta_{12}, d\theta_{20})$.
* Consequence: Dropping mutual orientations forces distinct biometric features to collide into identical representations. This causes catastrophic recognition failure:
  $$\text{DB1\_B EER}_{\text{6D}} = 55.56\%, \quad \text{AUC}_{\text{6D}} = 0.4441 \quad (\text{genuine scores become worse than impostor scores!})$$

---

## 6. Empirical Dimensionality Evaluation Summary

Evaluated on the canonical FVC2002 DB1_B development benchmark under identical preprocessing, Delaunay triangulation, and 1:1 matching protocols:

| Descriptor Candidate | Dimension ($D$) | Formulation / Features Included | DB1_B EER (%) | DB1_B AUC | Genuine-Impostor Separation ($\Delta\mu$) | Condition Number $\kappa$ | Unique Bitstrings (%) | Failure / Diagnostic Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **5-D** | 5 | $(r_1, r_2, d\theta_{01}, d\theta_{12}, d\theta_{20})$ | **35.56%** | **0.7019** | **+0.0378** | $142.3$ | 28.31% | Fully scale-invariant; best unweighted baseline |
| **6-D** | 6 | $(r_1, r_2, P_{\text{mm}}, \phi_0, \phi_1, \phi_2)$ | 55.56% | 0.4441 | -0.0056 | $892.1$ | 39.82% | **Score Inversion** due to 8-fold sign collision |
| **8-D** | 8 | $(r_1, r_2, P_{\text{mm}}, \phi_0, \phi_1, \phi_2, d\theta_{01}, d\theta_{12})$ | 41.07% | 0.6140 | +0.0089 | $1,210.5$ | 32.97% | Loss of $d\theta_{20}$ reduces mutual constraint |
| **9-D Baseline** | 9 | $(r_1, r_2, P_{\text{mm}}, \phi_0..2, d\theta_{01}..20)$ | 39.73% | 0.6404 | +0.0137 | $1,799.9$ | 28.39% | Production baseline; full rank covariance ($9/9$) |
| **12-D** | 12 | 9-D + discrete minutiae types $(t_0, t_1, t_2)$ | 42.22% | 0.6251 | +0.0134 | $2,488.2$ | 28.03% | Type noise across impressions degrades matching |

---

## 7. Mathematical Audit Sign-off

1. **Dimensionality Determination:** The 9-D descriptor covariance matrix has full rank ($9/9$), confirming that no single feature is an exact linear combination of the others.
2. **Deterministic Tie-Breaking:** Verified $SE(2)$ invariance across all permutations $S_3$ using rotation-invariant signed area chirality.
3. **8-D Non-Redundancy:** Rigorously proven via metric counterexample on $\mathbb{RP}^1$.
4. **6-D Collapse:** Verified via constructive 8-to-1 sign ambiguity proof.
5. **Regression Protection:** All properties protected by 26 automated property tests in `tests/test_descriptor_properties.py`.
