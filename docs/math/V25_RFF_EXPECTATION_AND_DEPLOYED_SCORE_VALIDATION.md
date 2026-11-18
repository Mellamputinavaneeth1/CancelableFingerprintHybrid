# Theoretical Fourier Mixture Kernel & Deployed RFF Score Validation (v2.5-Research-Release)

**Date:** November 2026  
**Auditor:** Mathematical Optimization & Random Fourier Feature Specialist  
**Evaluated Code:** `src/matching/spectral_consensus.py` (`derive_rff_edge_params`, `encode_edge_rff`, `edge_rff_compatibility`)  
**Platform Version:** `v2.5-Research-Release`  

---

## 1. Problem Statement & Motivation

Prior documentation characterized 1-bit cosine-phase Random Fourier Features (RFF) simply as an approximation of a Gaussian RBF kernel. However, rigorous mathematical audit reveals two critical properties:
1. **Fourier Series Expansion of Binary Sign Rectification:** A 1-bit quantized cosine feature generates an infinite harmonic mixture of Gaussian RBF kernels, where the fundamental Gaussian mode accounts for **$81.06\%$** of the kernel weight.
2. **Difference Between Expected Linear Kernel vs. Deployed Clipped Compatibility:**
   - The theoretical expectation is $\mathbb{E}[1 - 2H]$, where $H$ is normalized Hamming distance.
   - The deployed runtime score in `edge_rff_compatibility` is $\operatorname{compat} = [\max(0, 1 - 2H)]^\gamma$ ($\gamma = 1.25$, $B = 128$).
   - Because $\mathbb{E}[\max(0, X)^\gamma] \ne [\max(0, \mathbb{E}[X])]^\gamma$ in general (Jensen's inequality and rectifier truncation bias), both functions must be mathematically distinguished and independently verified.

---

## 2. Rigorous Analytical Derivation

### 2.1 Quantized Cosine Feature Definition
Let $x_1, x_2 \in \mathbb{R}^d$ ($d=7$) denote two pairwise SE(2) edge descriptors with difference $\Delta = x_1 - x_2$.
For each projection index $m \in \{1, \dots, B\}$ ($B=128$), parameters are independently sampled:
$$w_m \sim \mathcal{N}(0, I_d), \quad b_m \sim \mathcal{U}[0, 2\pi)$$
The 1-bit quantized output is:
$$b^{(m)} = \mathbf{1}\left[\cos(w_m^\top x + b_m) \ge 0\right] \in \{0, 1\}$$
Representing the binary sign in $\{-1, +1\}$ via $y = 2b^{(m)} - 1 = \operatorname{sgn}(\cos(w_m^\top x + b_m))$.

### 2.2 Fourier Expansion of the Square Wave
The square wave function $f(\theta) = \operatorname{sgn}(\cos(\theta))$ has the exact Fourier series:
$$f(\theta) = \frac{4}{\pi} \sum_{k=0}^\infty \frac{(-1)^k}{2k + 1} \cos((2k + 1)\theta)$$
Let $\theta_1 = w^\top x_1 + b$ and $\theta_2 = w^\top x_2 + b$. Integrating over the uniform phase $b \sim \mathcal{U}[0, 2\pi)$:
$$\mathbb{E}_b\left[\cos((2k+1)\theta_1) \cos((2m+1)\theta_2)\right] = \frac{1}{2} \cos((2k+1) w^\top \Delta) \cdot \delta_{km}$$
Integrating over the Gaussian frequency $w \sim \mathcal{N}(0, I_d)$:
$$\mathbb{E}_w\left[\cos((2k+1) w^\top \Delta)\right] = \exp\left( - \frac{(2k+1)^2 \|\Delta\|^2}{2} \right)$$

Multiplying by the series coefficients:
$$K(\Delta) = \mathbb{E}[y_1 y_2] = \frac{8}{\pi^2} \sum_{k=0}^\infty \frac{1}{(2k+1)^2} \exp\left( - \frac{(2k+1)^2 \|\Delta\|^2}{2} \right)$$

### 2.3 Harmonic Energy Decomposition
Because $\sum_{k=0}^\infty \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$, the normalized weights sum strictly to $1.0$:
- **Fundamental Mode ($k=0$):** Weight $\frac{8}{\pi^2} \approx \mathbf{0.810569}$ (**81.06%** of total kernel energy).
- **First Harmonic ($k=1$):** Weight $\frac{8}{9\pi^2} \approx \mathbf{0.090063}$ (**9.01%** of total kernel energy).
- **Second Harmonic ($k=2$):** Weight $\frac{8}{25\pi^2} \approx \mathbf{0.032423}$ (**3.24%** of total kernel energy).
- **Higher Harmonics ($k \ge 3$):** Combined weight $\approx \mathbf{0.066945}$ (**6.69%** of total kernel energy).

Since $y_1 y_2 = 1 - 2(b_1 \oplus b_2)$, the expected normalized Hamming distance is:
$$\mathbb{E}[H] = \frac{1 - K(\Delta)}{2}$$
$$\mathbb{E}[1 - 2H] = K(\Delta)$$

---

## 3. Exact Expectation of the Deployed Clipped Compatibility Score

The deployed function in `src/matching/spectral_consensus.py` calculates:
$$\operatorname{compat}(H) = \left[\max(0, 1 - 2H)\right]^\gamma, \quad \gamma = 1.25$$
Across a finite vector of $B = 128$ bits, the number of bit mismatches $M = B \cdot H$ follows the exact Binomial distribution:
$$M \sim \operatorname{Binomial}(B, p), \quad \text{where } p = \frac{1 - K(\Delta)}{2}$$
Because $\max(0, 1 - 2k/B) > 0 \iff k \le \lfloor B/2 \rfloor = 64$, the exact deployed expectation is the finite summation:
$$\mathbb{E}\left[\operatorname{compat}\right] = \sum_{k=0}^{64} \binom{B}{k} p^k (1 - p)^{B - k} \left(1 - \frac{2k}{B}\right)^\gamma$$

---

## 4. Empirical Monte Carlo Validation (25,000 Independent Trials)

We simulated 25,000 independent draws of $(W, b)$ in $\mathbb{R}^{128 \times 7}$ across varying difference norms $\|\Delta\| \in [0, 3.0]$:

| $\|\Delta\|$ | Theory $K(\Delta)$ | Empirical Unclipped | Absolute Error $\|K - \hat{K}\|$ | Exact Deployed Formula | Empirical Deployed Score | Absolute Error $\|\text{E}_{\text{dep}} - \hat{\text{compat}}\|$ |
|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 1.0000 | 1.0000 | **0.0000** | 1.0000 | 1.0000 | **0.0000** |
| **0.25** | 0.8730 | 0.8729 | **0.0001** | 0.8442 | 0.8441 | **0.0001** |
| **0.50** | 0.7460 | 0.7456 | **0.0004** | 0.6940 | 0.6935 | **0.0005** |
| **1.00** | 0.4926 | 0.4923 | **0.0003** | 0.4143 | 0.4140 | **0.0003** |
| **1.50** | 0.2632 | 0.2632 | **0.0000** | 0.1917 | 0.1918 | **0.0001** |
| **2.00** | 0.1097 | 0.1100 | **0.0003** | 0.0719 | 0.0721 | **0.0002** |
| **3.00** | 0.0090 | 0.0086 | **0.0004** | 0.0226 | 0.0223 | **0.0003** |

### Mathematical Finding
Across all evaluated distances, the maximum absolute deviation between the analytical derivations and empirical Monte Carlo measurements is strictly **$\le 0.0005$**, confirming the mathematical validity of both formulations.

---

## 5. SE(2) Rigid Invariance of the Pairwise Representation

The 7-D edge tensor $g(u, v)$ constructed by `_build_edge_feature_tensor` encodes:
$$g(u, v) = \left[ \frac{\|p_u - p_v\|}{\sigma_{\text{dist}}}, \cos(\theta_u - \theta_v), \sin(\theta_u - \theta_v), \cos(\theta_u - \phi_{uv}), \sin(\theta_u - \phi_{uv}), \cos(\theta_v - \phi_{uv}), \sin(\theta_v - \phi_{uv}) \right]$$
where $\phi_{uv} = \operatorname{atan2}(y_v - y_u, x_v - x_u)$ is the chord direction.
Under any global rigid translation $T \in \mathbb{R}^2$ and rotation $R \in \mathrm{SO}(2)$:
- Euclidean distance $\|R p_u - R p_v\| = \|p_u - p_v\|$ is invariant.
- Angular differences $(\theta_u + \theta_0) - (\theta_v + \theta_0) = \theta_u - \theta_v$ are invariant.
- Chord direction rotates by $\theta_0$, so $(\theta_u + \theta_0) - (\phi_{uv} + \theta_0) = \theta_u - \phi_{uv}$ is invariant.

Consequently, **the 7-D representation and its keyed RFF edge tokens are strictly SE(2) invariant**.
