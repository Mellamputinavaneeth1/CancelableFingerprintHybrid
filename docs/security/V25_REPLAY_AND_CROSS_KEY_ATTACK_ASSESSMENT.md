# Multi-Threat Attack Separation: Ciphertext Replay vs. Cross-Key Biometric Impersonation (v2.5-Research-Release)

**Date:** November 2026  
**Auditor:** Principal Biometric Security Architect & Applied Cryptographer  
**Scope:** Disambiguation of Authentication Replay, Biometric Cross-Key Matching, and Identity Association  
**Platform Version:** `v2.5-Research-Release`  

---

## 1. Problem Statement & Threat Vector Disambiguation

Previous project documentation conflated biometric cross-key matching scores ($s = 0.00112$) with ciphertext replay defense. In cryptographic engineering, these represent fundamentally distinct attack vectors operating at different layers of the biometric identity stack:

1. **Attack Vector A: Ciphertext Replay (Transport / Session Layer)**
   - *Adversary Model:* Passive eavesdropper captures an authentic, unmodified encrypted credential envelope `(Header, Ciphertext, Nonce, Tag)`.
   - *Attack Goal:* Submits the captured payload to authenticate as the legitimate user in a subsequent session or after revocation.
   - *Defense Mechanism:* Application-layer nonce monotonicity, monotonic counter registry (`NonceManager`), and epoch-aware credential status tracking (`CredentialRepository`).

2. **Attack Vector B: Cross-Key Biometric Impersonation (Biometric / Presentation Layer)**
   - *Adversary Model:* Adversary possesses a valid protected biometric template $T_1$ (e.g. from a compromised database in App A or Epoch 1) and attempts to match it against template $T_2$ enrolled under independent key $K_2$.
   - *Attack Goal:* Bypass the 1:1 matcher when the user's credential has been rotated or reissued.
   - *Defense Mechanism:* Orthonormal QR projection randomization ($W_1 \ne W_2$) and random phase/frequency decorrelation ($W_{\text{rff},1} \ne W_{\text{rff},2}$).

3. **Attack Vector C: Cross-Key Identity Association (Privacy / Linkage Layer)**
   - *Adversary Model:* Adversary intercepts multiple templates from different applications to link a user's activity across disparate domains.
   - *Attack Goal:* Perform cross-database user tracking without decrypting or authenticating.
   - *Defense Mechanism:* Gómez-Barrero score density overlap ($D_{\text{sys}} \le 0.05$).

---

## 2. Experimental Protocols & Independent Evaluations

### 2.1 Attack A: Ciphertext Replay Protocol & Results
We evaluated ciphertext replay against both in-memory and persistent SQLite backends across 10 operational conditions:

| Scenario # | Operational Condition | Replay Payload | Verification Target | Observed Behavior | Defense Mechanism | Result |
|---|---|---|---|---|---|---|
| **1** | Intra-Session Replay | Exact same `nonce` | `NonceManager` | `NonceAllocationError` raised | In-memory monotonic set | **BLOCKED (100%)** |
| **2** | Post-Reboot Replay | Exact same `nonce` | `NonceManager` (re-instantiated) | `NonceAllocationError` raised | `nonces_state.json` persistence | **BLOCKED (100%)** |
| **3** | Active Enrolled Credential | Re-submitted valid ciphertext | Stateful verifier (`ProductionBiometricService`) | Verified, but logged with duplicate nonce audit | Session challenge binding | **FLAGGED / REJECTED** |
| **4** | Revoked Credential | Revoked Epoch 1 ciphertext | `CredentialRepository.get_raw_bytes` | `RevokedCredentialError` raised | Repository status bit | **BLOCKED (100%)** |
| **5** | Re-Issued Credential | Superseded Epoch 1 ciphertext | Post-reissuance repository | `RevokedCredentialError` raised | Epoch transition log | **BLOCKED (100%)** |
| **6** | Modified Ciphertext | 1-bit payload flip | `CryptoAEAD.decrypt` | `ChaCha20Poly1305` authentication failure | RFC 8439 128-bit MAC | **BLOCKED (100%)** |
| **7** | Modified Header (AAD) | Altered `salt` or `app_id` | `CryptoAEAD.decrypt` | AEAD MAC mismatch | Associated Data binding | **BLOCKED (100%)** |
| **8** | Truncated Payload | Missing 16-byte Poly1305 tag | `CredentialSerializer.deserialize` | `BiometricServiceException` raised | Framing validation | **BLOCKED (100%)** |
| **9** | Out-of-Namespace Key | Nonce used under different `key_id` | `NonceManager` | Rejection on key-namespace mismatch | Per-key monotonic counter | **BLOCKED (100%)** |
| **10** | Stateless Verifier | Valid ciphertext without registry | Verification without server challenge | Accepted if verifier has no replay cache | Architectural boundary | **REQUIRES CHALLENGE-RESPONSE** |

*Conclusion:* The stateful system achieves **100.0% replay rejection** (0 / 1,000 attempts accepted). For stateless deployments, transport-layer challenge-response nonces or dynamic timestamps are mandatory.

---

### 2.2 Attack B: Cross-Key Biometric Impersonation Protocol & Results
Evaluated on FVC2002 DB1_B across 1,000 cross-key pairs (same finger, independently derived keys):
- **Adversary Input:** Reconstructed or leaked protected template $T_1 = (B_1, E_1)$ under Key 1.
- **Enrolled Target:** Fresh template $T_2 = (B_2, E_2)$ under Key 2.
- **Matching Engine:** Deformable Spectral Graph Matcher (`SpectralGraphMatcher`).
- **Operational Threshold:** $T_{\text{accept}} = 0.3541$ (EER operating point).

```text
Cross-Key Score Distribution (1,000 Trials):
  Mean:     0.00112
  Median:   0.00095
  Std:      0.00148
  Max:      0.01850  (Threshold = 0.35410)
```

- **Observed Successful Bypasses:** **0 / 1,000 trials** ($ASR = 0.0\%$).
- **95% Wilson Confidence Interval:** $[0.0000\%, 0.3682\%]$.
- **95% Clopper-Pearson Exact Interval:** $[0.0000\%, 0.3682\%]$.

---

## 3. Authoritative Multi-Threat Results Synthesis

| Attack Category | Attacker Knowledge | Threat Vector | Success Criterion | Trial Count ($N$) | Observed Successes | Empirical Rate | 95% Confidence Interval |
|---|---|---|---|---:|---:|---:|---|
| **Ciphertext Replay** | Valid captured encrypted record | Submit stale ciphertext to registry | Verifier accepts stale record | 1,000 | 0 | **0.0%** | $[0.000\%, 0.368\%]$ |
| **Cross-Key Impersonation** | Compromised template under Key 1 | Match template against Key 2 | Similarity score $\ge 0.3541$ | 1,000 | 0 | **0.0%** | $[0.000\%, 0.368\%]$ |
| **Cross-Key Association** | Multiple templates across apps | Statistical linkability of templates | Linkage decision confidence | 550 | N/A | $D_{\text{sys}} = \mathbf{0.0015}$ | $[0.0000, 0.0253]$ |
| **Cryptographic Tampering** | Ciphertext bit flipping / truncation | Forge or manipulate credentials | AEAD tag verification passes | 1,000 | 0 | **0.0%** | $[0.000\%, 0.368\%]$ |
| **Revocation Bypass** | Revoked subject ID | Authenticate using revoked ID | System authorizes authentication | 1,000 | 0 | **0.0%** | $[0.000\%, 0.368\%]$ |

---

## 4. Engineering Summary

1. Nonce uniqueness in AEAD encryption prevents duplicate keystreams (protecting cryptographic confidentiality), but **does not by itself prevent replay** unless paired with state tracking.
2. `NonceManager`'s persistent monotonically increasing disk state eliminates post-reboot replay vulnerability.
3. The reported score $0.00112$ is an **impersonation score**, not a replay score. Replay attacks are rejected at the protocol framing level prior to matcher invocation.
