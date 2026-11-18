# Work Packages 5 & 6 (WP5/WP6) — Cryptography, AEAD Hardening, and Privacy Audit

**Document Status:** Complete & Verified  
**Revision:** 2.2-Hardened  
**Date:** November 2026  
**Applicability:** Core Cryptographic Services, AEAD Framing, Key Hierarchy, Linkability, and Lifecycle  

---

## 1. Executive Summary

Work Packages 5 and 6 provide an exhaustive cryptographic and privacy audit of the `CancelableFingerprintHybrid` platform, addressing key management scopes, AEAD authenticated framing, nonce allocation guarantees, terminology corrections regarding forward secrecy, and empirical linkability evaluation under ISO/IEC 30136:2018 and Gómez-Barrero et al. (IEEE TIFS 2018).

Key findings and architectural hardening verified in this audit:
1. **Key Scoping & 1:N Screening Architecture:**
   - **Application-Scoped Keying:** Used in all benchmark evaluations (FVC2002 DB1_A, DB1_B, DB2_B, DB3_B, DB4_B). A single projection matrix $\mathbf{W}_{\text{app}}$ is derived per application. Queries are projected once ($O(1)$) and screened against all $N$ gallery templates in $O(N)$ binary Hamming operations ($\approx 1,000\text{--}1,350\text{ matches/s}$).
   - **Credential-Scoped Keying:** Derives a unique matrix $\mathbf{W}_{\text{cred}}$ per credential salt. While providing maximal cryptographic isolation, in an unindexed 1:N identification workflow, each gallery comparison requires reprojecting the raw query descriptors under that credential's specific matrix ($N$ projections). Benchmarks demonstrate this introduces a $1.00\times\text{--}1.06\times$ screening time overhead for small templates, but scales linearly with gallery size unless candidate indexing is applied.
2. **Forward Secrecy Terminology Correction:**
   - Derivation of epoch-specific keys from a retained static master secret $K_{\text{master}}$ via HKDF provides domain-separated key diversification. It does **not** provide forward secrecy.
   - If $K_{\text{master}}$ is compromised, all past and future epoch keys can be derived. The mechanism is officially designated **"epoch-specific key separation"**. True forward secrecy requires key ratcheting with irreversible erasure of earlier states.
3. **ChaCha20-Poly1305 Nonce Allocation Architecture:**
   - A simulation of 100,000 nonces without collisions does *not* prove uniqueness.
   - A dedicated `NonceManager` is implemented supporting two production strategies:
     - **Monotonic Counter Allocation:** 4-byte worker namespace + 8-byte atomic monotonic counter ($96$ bits total) with persistent state synchronization, surviving restarts and multi-threaded concurrency.
     - **Random CSPRNG Bounded Invocations:** Generates 96-bit random nonces with per-key invocation tracking, enforcing a hard cap $q_{\max} \le 2^{32}$ (collision probability $P \le 2^{-33} \approx 1.16 \times 10^{-10}$) and mandatory key rotation.
4. **AEAD Header AAD Integrity Binding:**
   - The entire 64-byte binary header is passed as Authenticated Associated Data (AAD) to ChaCha20-Poly1305.
   - Every security-sensitive field (`MAGIC`, `version`, `flags`, `in_dim`, `credential_id`, `salt`, `num_triangles`, `nonce`) is authenticated by Poly1305.
   - Header CRC32 is strictly an unkeyed transport corruption check; cryptographic integrity is exclusively enforced by Poly1305.
5. **Mock HSM Boundary Disclosure:**
   - The Python stack-inspection guard (`inspect.currentframe()`) in `KeyService` is an in-process demonstrator guard, not a hardware security module. Production deployments require physical HSM interfaces (PKCS#11 / FIPS 140-3).
6. **System Linkability & Privacy Interpretation:**
   - The $0.05$ threshold is a **project-defined engineering acceptance criterion**, not a universal ISO/IEC 30136 pass/fail standard.
   - Joint maximum linkability across threat models is evaluated using replicate-wise maxima: $D_{\text{sys}}^{\text{max}} = \mathbf{0.0375}$ with clustered 95% CI $[\mathbf{0.0296}, \mathbf{0.1333}]$.
   - Because the upper confidence limit ($0.1333$) exceeds $0.05$, the system status is formally designated **LOW RESIDUAL LINKABILITY** (not an unconditional pass).
   - Cross-epoch linkability ($D_{\text{sys}}^{\text{epoch}} = \mathbf{0.0689}$ [95% CI: $0.0046, 0.2014$]) exhibits measurable residual linkability and must not be described as "confirming robust isolation".
7. **Decoupling Lifecycle Properties:**
   - Registry revocation (100% access rejection) is strictly decoupled from cryptographic unlinkability ($D_{\text{sys}}^{\text{epoch}} > 0$).
8. **Credential Storage Layout:**
   - Serialized credential size for $N=144$ triangles is bit-exact: $64\text{ B header} + 576\text{ B descriptors} + 288\text{ B weights} + 16\text{ B tag} = \mathbf{944\text{ Bytes}}$.

---

## 2. Key Scoping and Derivation Hierarchy (Section 6.1)

### 2.1 Cryptographic Entropy and Derivation Chain
The platform enforces a strict cryptographic derivation chain grounded in a 256-bit CSPRNG master secret $K_{\text{master}}$:
$$K_{\text{master}} \xrightarrow[\text{RFC 5869 HKDF-Extract}]{\text{salt}} \text{PRK} \xrightarrow[\text{RFC 5869 HKDF-Expand}]{\text{info}} K_{\text{app}} \xrightarrow[\text{HMAC-SHA256 Counter PRF}]{\text{Box-Muller}} \mathbf{A} \xrightarrow{\text{QR}} \mathbf{W}$$

- **Entropy Assumptions:** $K_{\text{master}}$ is generated via OS CSPRNG (`secrets.token_bytes(32)`) with 256 bits of min-entropy.
- **Domain Separation Strings:** Info strings follow structured namespacing:
  $$\text{info} = \text{"CancelableBio|app="} \parallel \text{app\_id} \parallel \text{"|purpose="} \parallel \text{purpose} \parallel \text{"|ver=1.0"}$$
  For credential-specific derivation:
  $$\text{info} = \text{"CancelableBio|subject="} \parallel \text{cred\_id} \parallel \text{"|app="} \parallel \text{app\_id} \parallel \text{"|purpose="} \parallel \text{purpose} \parallel \text{"|ver=1.0"}$$
  No plaintext personal identifiable information (PII) is included; pseudonymous UUIDs are used.
- **Deterministic PRF to Gaussian Mapping:** A stateless HMAC-SHA256 counter-mode PRF generates 32-bit integers $v \in [0, 2^{32}-1]$, mapped to the open interval $u = (v + 0.5) / 2^{32} \in (0, 1)$, preventing Box-Muller singularities ($\ln(0)$ or $\sqrt{\text{negative}}$).
- **Orthonormalization:** Matrix $\mathbf{A} \in \mathbb{R}^{32 \times 9}$ is factored via thin QR decomposition: $\mathbf{A} = \mathbf{W}\mathbf{R}$, yielding column-orthonormal $\mathbf{W} \in \mathbb{R}^{32 \times 9}$ satisfying $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_9$.

### 2.2 Comparison of Key Scoping Models and 1:N Architectural Impact

| Scoping Model | Derivation Context ($\text{info}$ & $\text{salt}$) | Mathematical Matrix | 1:1 Verification Cost | 1:N Screening Overhead | Security Isolation Scope | Usage in Platform |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Application-Scoped** | $\text{salt} = \text{AppSalt}, \text{info} = \text{AppID}$ | Single $\mathbf{W}_{\text{app}}$ | $O(1)$ projection + $O(1)$ match | **Optimal ($1.00\times$):** Query projected once; $N$ binary Hamming matches | Application-wide; key compromise affects all users in app | Canonical Benchmark (FVC2002 DB1_A, DB1_B, DB2_B, DB3_B, DB4_B) |
| **Credential-Scoped** | $\text{salt} = \text{CredSalt}_i, \text{info} = \text{CredID}_i$ | Unique $\mathbf{W}_i$ per user | $O(1)$ projection + $O(1)$ match | **Overhead ($1.00\text{--}1.06\times$ small; scales linearly with $N$):** Query must be projected $N$ times | Per-credential; compromise of $\mathbf{W}_i$ isolates other users | High-Security Zero-Trust Enclaves |
| **Epoch-Specific** | $\text{salt} = \text{CredSalt} \parallel \text{EpochID}$ | Rotated $\mathbf{W}_{\text{epoch}}$ | $O(1)$ projection + $O(1)$ match | Same as scoped mode | Epoch-isolated; enables cryptographic revocation | Template Re-issuance & Revocation |

#### Empirical Screening Throughput Benchmark Across Gallery Sizes ($N$):
Benchmarked on synthetic continuous templates ($N \in \{10, 50, 100, 500, 1000\}$ gallery items):
- $N=10$: App-Scoped = $0.0098$ s (1,018.1 m/s) vs Cred-Scoped = $0.0101$ s (989.8 m/s); Overhead = $1.03\times$.
- $N=100$: App-Scoped = $0.0988$ s (1,012.4 m/s) vs Cred-Scoped = $0.0993$ s (1,007.6 m/s); Overhead = $1.00\times$.
- $N=1000$: App-Scoped = $0.9731$ s (1,027.7 m/s) vs Cred-Scoped = $1.0110$ s (989.2 m/s); Overhead = $1.04\times$.

*Architectural Resolution:* The benchmark evaluations in Sections 4, 5, and 6 strictly utilize **Application-Scoped Keying**. When describing credential-scoped keying, the report explicitly notes that an unindexed 1:N screening workflow requires $N$ separate projection operations.

---

## 3. Forward-Secrecy Terminology Correction (Section 6.2)

### 3.1 Mathematical Fact: Retained Master Secret Prevents True Forward Secrecy
In symmetric key derivation:
$$\mathbf{W}_{\text{epoch}} = \operatorname{KDF}(K_{\text{master}}, \text{salt} \parallel \text{epoch})$$
If an adversary compromises $K_{\text{master}}$ at Epoch $t$, the adversary can retroactively compute all prior epoch keys $\{\mathbf{W}_1, \dots, \mathbf{W}_{t-1}\}$ because the derivation function is deterministic and the master secret was never erased.

### 3.2 Formal Threat Model
1. **Master Secret Compromise ($K_{\text{master}}$ exposed):**
   - Result: All past, present, and future projection matrices and AEAD storage keys are exposed.
   - Status: **NO FORWARD SECRECY**.
2. **Child Key Compromise ($\mathbf{W}_{\text{epoch}}$ exposed, $K_{\text{master}}$ secure):**
   - Result: Pre-image attack surface on current epoch templates; earlier and subsequent epoch templates remain protected by the pre-image resistance of HMAC-SHA256.
   - Status: **EPOCH-SPECIFIC KEY SEPARATION**.

*Terminology Hardening:* All instances of "cryptographic forward secrecy" in reference to HKDF derivation from a retained master secret have been replaced with **"epoch-specific key separation"**.

---

## 4. ChaCha20-Poly1305 Nonce Management Architecture (Section 6.3)

### 4.1 Nonce Uniqueness Guarantees Under RFC 8439
RFC 8439 ChaCha20-Poly1305 requires that a 96-bit nonce **never repeat** under the same key. A repeat allows an attacker to XOR ciphertexts, compute the Poly1305 one-time key $\mathbf{r}$, and forge messages.

### 4.2 NonceManager Production Allocation Strategies
To guarantee uniqueness, `NonceManager` (`src/services/nonce_manager.py`) implements:

#### Strategy A: Persistent Monotonic Counter Mode (Default for Stateful Enclaves)
- **Nonce Structure (12 Bytes / 96 Bits):**
  $$\text{Nonce} = \underbrace{\text{NamespaceID}}_{4\text{ Bytes (uint32)}} \parallel \underbrace{\text{MonotonicCounter}}_{8\text{ Bytes (uint64, big-endian)}}$$
- **Thread Safety:** Atomic increments protected by a re-entrant mutex lock.
- **Persistence & Crash Recovery:** The current counter state is atomically written to an fsync'd state file (`os.replace`). Upon service restart or crash recovery, the counter resumes from the last saved state, strictly preventing counter rollback.
- **Capacity:** $2^{64}$ nonces per worker namespace before key rotation is required.

#### Strategy B: Random CSPRNG Bounded Invocations Mode (Default for Stateless Services)
- **Nonce Generation:** 96 bits of cryptographic randomness via `os.urandom(12)`.
- **Collision Probability (Birthday Bound):**
  $$P(\text{collision}) \approx \frac{q^2}{2 \times 2^{96}} = \frac{q^2}{2^{97}}$$
- **Enforced Usage Cap:** The manager tracks invocations under each key fingerprint and enforces a hard limit $q_{\max} = 2^{32}$ ($4.29 \times 10^9$ encryptions). At $q = 2^{32}$:
  $$P(\text{collision}) \le \frac{2^{64}}{2^{97}} = 2^{-33} \approx 1.16 \times 10^{-10}$$
- **Key Rotation Enforcement:** Attempting to allocate beyond $2^{32}$ nonces raises `KeyUsageLimitExceeded`, forcing key rotation.

---

## 5. AEAD Header Authentication and AAD Integrity Binding (Section 6.4)

### 5.1 Serialized Credential Binary Framing
The serialized credential envelope is framed as follows:

```
+-----------------------------------------------------------------------------------+
| 64-BYTE AUTHENTICATED HEADER (Passed as AAD to ChaCha20-Poly1305)                 |
| Offset  0: Magic Bytes (4B, "CFPB")                                              |
| Offset  4: Schema Version (2B, uint16 = 1)                                       |
| Offset  6: Flags (2B, uint16: bit0=topo, bit1=cipher, bit2=enc, bit8-15=out_dim) |
| Offset  8: Input Dimension (1B, uint8 = 9)                                       |
| Offset  9: Reserved0 (1B, uint8 = 0)                                             |
| Offset 10: Reserved1 (2B, bytes = 0x0000)                                        |
| Offset 12: Credential ID UUID (16B, binary UUID)                                 |
| Offset 28: Presentation Salt (16B, ASCII canonical salt)                         |
| Offset 44: Triangle Count N (4B, uint32)                                         |
| Offset 48: Nonce (12B, binary 96-bit nonce)                                      |
| Offset 60: Header Framing CRC32 (4B, uint32)                                     |
+-----------------------------------------------------------------------------------+
| ENCRYPTED PAYLOAD (CIPHERTEXT, Length = 576 + 288 = 864 Bytes for N=144)          |
| Descriptors: N x 4 Bytes bit-packed uint32 (576 Bytes)                            |
| Weights:     N x 2 Bytes float16 (288 Bytes)                                      |
+-----------------------------------------------------------------------------------+
| POLY1305 AUTHENTICATION TAG (16 Bytes)                                            |
+-----------------------------------------------------------------------------------+
Total Serialized Size (N=144) = 64 + 576 + 288 + 16 = 944 Bytes
```

### 5.2 Header Authentication Proof
- In `chacha20_poly1305_encrypt(key, nonce, plaintext, aad=header_bytes)`, the Poly1305 MAC input is:
  $$\text{mac\_data} = \operatorname{pad}_{16}(\text{header\_bytes}) \parallel \operatorname{pad}_{16}(\text{ciphertext}) \parallel \operatorname{len}(\text{header}) \parallel \operatorname{len}(\text{ciphertext})$$
- Because `header_bytes` is the exact 64-byte AAD:
  1. Altering any header field (e.g., modifying `CredentialID`, changing `Version`, or tampering with `TriangleCount`) alters `mac_data`.
  2. The Poly1305 tag validation fails in constant time (`hmac.compare_digest`), throwing `AeadIntegrityError` before payload decryption or memory allocation occurs.
- **CRC32 Demotion:** Header CRC32 at offset 60 is strictly a transport-layer bit-flip detection check. Cryptographic integrity and access control decisions are governed **exclusively** by Poly1305.

### 5.3 Adversarial Tamper Matrix Verification

| Tamper Test Case | Injected Corruption | Detection Mechanism | Thrown Exception | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Header AAD: Magic Bytes** | Bit-flip at byte 0 | Poly1305 MAC / Framing | `BiometricServiceException` | **PASS** |
| **Header AAD: Version** | Bit-flip at byte 4 | Poly1305 MAC / Version Check | `BiometricServiceException` | **PASS** |
| **Header AAD: Flags** | Bit-flip at byte 6 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Header AAD: Credential UUID** | Bit-flip at byte 12 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Header AAD: Salt** | Bit-flip at byte 28 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Header AAD: Triangle Count $N$** | Bit-flip at byte 44 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Header AAD: Nonce** | Bit-flip at byte 48 | Poly1305 OTK Subkey Failure | `AeadIntegrityError` | **PASS** |
| **Header AAD: CRC32** | Bit-flip at byte 60 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Ciphertext Payload** | 1-bit flip at byte 70 | Poly1305 MAC Tag Failure | `AeadIntegrityError` | **PASS** |
| **Poly1305 Tag** | 1-bit flip at byte 943 | Poly1305 Tag Mismatch | `AeadIntegrityError` | **PASS** |
| **Wrong Storage Key** | Decrypt with key $K'$ | Poly1305 OTK Mismatch | `AeadIntegrityError` | **PASS** |
| **Payload Truncation** | Truncate raw bytes | Frame Size Validation | `BiometricServiceException` | **PASS** |

---

## 6. Mock HSM Trust Boundary Disclosure (Section 6.5)

In `src/services/key_service.py`, direct access to `_master_secret` is guarded by:
```python
if name in ("master_secret", "_master_secret", "_secret", "secret", "raw_secret"):
  frame = inspect.currentframe().f_back
  if frame.f_locals.get("self") is not self:
    raise MasterSecretAccessViolation(...)
```
**Formal Disclosure:**  
This Python stack-inspection mechanism is a **demonstrator guard** intended to enforce clean service-layer architectural separation. It does **not** provide hardware memory isolation or FIPS 140-3 boundary validation. A process running with equal privileges or native C-extensions can bypass Python stack checks. Production deployments require physical hardware security modules (HSM) using PKCS#11 interfaces, AWS CloudHSM, or Google Cloud KMS.

---

## 7. Linkability and Privacy Audit (Section 7)

### 7.1 Statistical Methodology: Replicate-Wise Joint Maximum Bootstrap
To evaluate system privacy across multiple attack surfaces without optimistic bias:
1. For each bootstrap replicate $b \in [1, B]$:
   - Resample subjects with replacement (subject-level clustering).
   - Recompute comparison scores for each threat model $m \in \{\text{Descriptor-Only}, \text{Topology-Only}, \text{Combined}\}$.
   - Compute system linkability $D_{\text{sys}}^{(m)}(b)$ using boundary-corrected Silverman KDE.
   - Record the replicate maximum:
     $$D_{\text{sys}}^{\text{max}}(b) = \max_{m} D_{\text{sys}}^{(m)}(b)$$
2. The joint point estimate and two-sided 95% confidence intervals are derived directly from the empirical distribution of $D_{\text{sys}}^{\text{max}}(b)$.
3. Under no circumstances should the joint upper bound be estimated by taking the maximum of independently calculated marginal upper confidence limits.

### 7.2 Results and Classification Policy

| Threat Model | Mated Distribution ($H_m$) | Non-Mated Distribution ($H_{nm}$) | Point Estimate ($D_{\text{sys}}$) | Clustered 95% CI | Engineering Assessment |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Descriptor-Only Attacker** | Same-Subject, Different-Key | Cross-Subject, Different-Key | 0.0064 | [0.0033, 0.1167] | Low Residual Linkability |
| **Topology-Only Attacker** | Same-Subject Graph Adjacency | Cross-Subject Graph Adjacency | 0.0375 | [0.0252, 0.1186] | Low Residual Linkability |
| **Combined Threat Model** | Weighted Joint Score | Cross-Subject Joint Score | 0.0015 | [0.0000, 0.0253] | Unlinkable ($D_{\text{sys}} \le 0.05$, Upper CI $\le 0.05$) |
| **Joint Maximum ($D_{\text{sys}}^{\text{max}}$)** | Worst-Case Across Models | Baseline Non-Mated | **0.0375** | **[0.0296, 0.1333]** | **Low Residual Linkability** |

> [!WARNING]
> **Project Acceptance Threshold vs. Normative Standards:**  
> A value of $D_{\text{sys}} \le 0.05$ represents an **internal project-defined engineering threshold**, not a universal pass/fail threshold defined in ISO/IEC 30136. While the point estimate $D_{\text{sys}}^{\text{max}} = 0.0375 \le 0.05$, the upper 95% confidence limit reaches $\mathbf{0.1333} > 0.05$. Therefore, the platform is officially designated as exhibiting **LOW RESIDUAL LINKABILITY**, not an unconditional pass.

---

## 8. Lifecycle Properties Disambiguation (Section 7.4)

Biometric template lifecycle properties must be evaluated independently:

1. **Registry Revocation:**  
   - Mechanism: Logical status update in `CredentialRepository` (`status = REVOKED`).  
   - Result: 100.0% block rate on subsequent authentication requests (0% bypass ASR).  
   - Scope: Access control enforcement; does *not* imply physical template destruction.
2. **Replay Resistance:**  
   - Mechanism: Re-submission of previously intercepted serialized credential envelopes is blocked by nonce freshness checks in `NonceManager`.
3. **Template Re-issuance:**  
   - Mechanism: Derivation of fresh projection matrix $\mathbf{W}'$ using a new presentation salt and epoch counter.  
   - Result: Verification of re-issued credential against fresh probe impressions achieves normal operational TAR ($59.07\%$ at $\tau_{\text{EER}}$).
4. **Cross-Epoch Statistical Unlinkability:**  
   - Metric: $D_{\text{sys}}^{\text{epoch}} = \mathbf{0.0689}$ with clustered 95% CI $[\mathbf{0.0046}, \mathbf{0.2014}]$.  
   - Interpretation: Old and new credentials exhibit measurable residual correlation ($D_{\text{sys}} > 0$). This is expected because physical minutiae topology is partially retained across impressions. It is not "robust isolation", but low residual correlation.
5. **Post-Compromise Security:**  
   - Compromise of an individual credential's storage key does not compromise other credentials.  
   - Compromise of $K_{\text{master}}$ requires a full root-of-trust key rollover.

---

## 9. Credential Storage Accounting Verification (Section 8.1)

### Exact $N=144$ Serialized Credential Byte Layout
For standard templates with $N=144$ Delaunay triangles and $M=32$ projection dimension:

$$\begin{aligned}
\text{Header Size} &= 64\text{ Bytes (Fixed framing)} \\
\text{Descriptors Size} &= 144 \times (32 / 8) = 144 \times 4 = 576\text{ Bytes} \\
\text{Weights Size} &= 144 \times 2 = 288\text{ Bytes (float16)} \\
\text{Poly1305 Tag Size} &= 16\text{ Bytes} \\
\hline
\mathbf{\text{Total Serialized Envelope}} &= \mathbf{64 + 576 + 288 + 16 = 944\text{ Bytes}}
\end{aligned}$$

Analytical specification matches physical measurements bit-for-bit (verified in `tests/test_wp5_wp6_crypto_privacy_audit.py`).
Compared to unquantized continuous baseline storage ($144 \times 9 \times 4\text{ B} + 144 \times 4\text{ B} = 5,760\text{ B} + 64\text{ B} = 5,824\text{ B}$), the protected credential achieves an **$83.8\%$ storage reduction** ($6.17\times$ compression).
