import sys
import os
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.minutiae.extractor import Minutia
from src.matching.bipartite_matcher import match_protected_templates
from src.revocation.registry import TemplateRegistry

def generate_biometric_impression(seed: int, rotation_deg: float = 0.0, 
                                  trans_x: float = 0.0, trans_y: float = 0.0,
                                  noise_std: float = 0.0, drop_ratio: float = 0.0, 
                                  spurious_count: int = 0):
    """
    Simulates a synthetic biometric fingerprint impression with non-linear
    displacement, rotation, and acquisition noise.
    
    Note:
        This simulation provides synthetic minutiae for pipeline validation.
        It is intended for preliminary algorithmic and security verification
        prior to full benchmark evaluation on real fingerprint datasets (e.g., FVC2002).
    """
    rng = np.random.default_rng(seed)
    n_pts = 45
    x = rng.uniform(40, 220, n_pts)
    y = rng.uniform(40, 220, n_pts)
    theta = rng.uniform(0, np.pi, n_pts)  # Axial orientation in [0, pi)
    m_type = rng.integers(0, 2, n_pts)
    qual = rng.uniform(0.75, 1.0, n_pts)
    
    rot = np.deg2rad(rotation_deg)
    c, s = np.cos(rot), np.sin(rot)
    
    keep_count = int(round(n_pts * (1.0 - drop_ratio)))
    keep_indices = rng.choice(n_pts, size=keep_count, replace=False)
    
    impression_minutiae = []
    for i in keep_indices:
        dx = x[i] - 128.0
        dy = y[i] - 128.0
        xr = 128.0 + dx * c - dy * s + trans_x + rng.normal(0, noise_std)
        yr = 128.0 + dx * s + dy * c + trans_y + rng.normal(0, noise_std)
        thr = (theta[i] + rot + rng.normal(0, noise_std * 0.04)) % np.pi
        impression_minutiae.append(Minutia(float(xr), float(yr), float(thr), int(m_type[i]), float(qual[i])))
        
    for _ in range(spurious_count):
        impression_minutiae.append(Minutia(
            float(rng.uniform(40, 220)), float(rng.uniform(40, 220)),
            float(rng.uniform(0, np.pi)), int(rng.integers(0, 2)),
            float(rng.uniform(0.75, 1.0))
        ))
        
    return impression_minutiae

def run_demo():
    print("=" * 80)
    print(" CANCELABLE FINGERPRINT BIOMETRICS DEMONSTRATOR")
    print(" Quality-Aware Keyed Minutiae-Graph Protection")
    print(" Evaluated using principles from ISO/IEC 24745:2022 & ISO/IEC 30136:2018")
    print("=" * 80)
    print("[Note] Running preliminary evaluation using simulated multi-impression minutiae.")
    
    pipeline = CancelableFingerprintPipeline(
        master_secret="CancelableBioMasterSecret_2026_256Bits_KMS!",
        in_dim=12,
        out_dim=32,
        dpi=500.0,
        hamming_threshold=0.22,
        operating_threshold=0.20
    )
    
    # -------------------------------------------------------------
    # 1. ENROLLMENT
    # -------------------------------------------------------------
    print("\n[Step 1] Enrolling User 'Alice' with Master Secret & App ID 'banking_portal'...")
    alice_impression_1 = generate_biometric_impression(seed=101)
    feat_a1 = pipeline.process_minutiae_list(alice_impression_1)
    
    salt_v1 = "alice_salt_v1"
    app_id = "banking_portal"
    enroll_res = pipeline.enroll_from_features(feat_a1["descriptors"], feat_a1["weights"], salt=salt_v1, app_id=app_id)
    
    template_v1 = enroll_res["template"]
    print(f" -> Enrolled successfully!")
    print(f" -> Extracted {enroll_res['num_triangles']} Delaunay triangle descriptors from {len(alice_impression_1)} minutiae.")
    print(f" -> Protected template bitstrings shape: {template_v1['bitstrings'].shape} (Bits per triangle: {pipeline.out_dim})")
    print(f" -> Salt: '{template_v1['salt']}', App ID: '{template_v1['app_id']}'")
    
    # -------------------------------------------------------------
    # 2. GENUINE VERIFICATION (Natural repeat impression with rotation & noise)
    # -------------------------------------------------------------
    print("\n[Step 2] Verifying Genuine Repeat Impression of Alice...")
    print("         (Simulating 6° rotation, 4px displacement, 10% minutiae dropout & noise)")
    alice_impression_2 = generate_biometric_impression(
        seed=101, rotation_deg=6.0, trans_x=4.0, trans_y=-2.0, noise_std=1.0, drop_ratio=0.10, spurious_count=3
    )
    feat_a2 = pipeline.process_minutiae_list(alice_impression_2)
    
    verif_res = pipeline.verify_from_features(feat_a2["descriptors"], feat_a2["weights"], template_v1)
    print(f" -> Decision: {verif_res['decision']}")
    print(f" -> Match Score: {verif_res['score']:.4f} (Operating Threshold: {verif_res['threshold']:.4f})")
    assert verif_res["decision"] == "ACCEPT", "Genuine repeat impression must be accepted!"
    
    # -------------------------------------------------------------
    # 3. IMPOSTOR ATTEMPT (Different subject 'Bob')
    # -------------------------------------------------------------
    print("\n[Step 3] Verifying Impostor 'Bob' against Alice's enrolled template...")
    bob_impression = generate_biometric_impression(seed=202)
    feat_b = pipeline.process_minutiae_list(bob_impression)
    
    impostor_res = pipeline.verify_from_features(feat_b["descriptors"], feat_b["weights"], template_v1)
    print(f" -> Decision: {impostor_res['decision']}")
    print(f" -> Match Score: {impostor_res['score']:.4f} (Operating Threshold: {impostor_res['threshold']:.4f})")
    assert impostor_res["decision"] == "REJECT", "Impostor attempt must be rejected!"
    
    # -------------------------------------------------------------
    # 4. UNLINKABILITY TEST (Same physical finger, different application key)
    # -------------------------------------------------------------
    print("\n[Step 4] Unlinkability Test: Same finger Alice, but independent application key...")
    template_healthcare = pipeline.enroll_from_features(
        feat_a1["descriptors"], feat_a1["weights"], salt="alice_health_salt", app_id="healthcare_portal"
    )["template"]
    
    cross_key_similarity = match_protected_templates(
        template_v1["bitstrings"], template_healthcare["bitstrings"],
        template_v1["weights"], template_healthcare["weights"],
        hamming_threshold=pipeline.hamming_threshold
    )
    print(f" -> Cross-Application Match Score: {cross_key_similarity:.4f} (Threshold: {pipeline.operating_threshold:.4f})")
    print(" -> UNLINKABILITY PROPERTY: Cross-key similarity is well below operating threshold.")
    assert cross_key_similarity < pipeline.operating_threshold, "Cross-key matching must be rejected!"
    
    # -------------------------------------------------------------
    # 5. REVOCATION AND REISSUANCE (Option A - 4-Step Lifecycle)
    # -------------------------------------------------------------
    print("\n[Step 5] Revocation Test: Key Compromise Detected -> Database Invalidation & Reissuance...")
    registry = TemplateRegistry()
    tid_v1 = registry.enroll_template("alice", app_id, template_v1)
    print(f" -> [Step 5a] Enrolled active credential ID: '{tid_v1}' (Status: {registry.get_template_status(tid_v1)})")
    
    # Invalidate in database
    registry.revoke_template(tid_v1, reason="key_compromise_detected")
    print(f" -> [Step 5b] Database Invalidation: Credential ID '{tid_v1}' status transitioned to: {registry.get_template_status(tid_v1)}")
    
    # Attempt verification using revoked credential ID
    blocked_req = registry.verify_request(feat_a2["descriptors"], feat_a2["weights"], tid_v1, pipeline)
    print(f" -> [Step 5c] Verification with revoked ID: Decision = {blocked_req['decision']} (Status: {blocked_req['status']})")
    assert blocked_req["decision"] == "REJECT", "Revoked credential ID must be rejected by registry!"
    
    # Fresh capture presentation and reissuance under new key
    alice_impression_3 = generate_biometric_impression(seed=101, rotation_deg=-3.0, trans_x=2.0)
    feat_a3 = pipeline.process_minutiae_list(alice_impression_3)
    new_salt = "alice_salt_v2_REVOKED_REISSUED"
    reissue_info = registry.reissue_from_fresh_capture(
        "alice", app_id, feat_a3["descriptors"], feat_a3["weights"], new_salt=new_salt, pipeline=pipeline
    )
    tid_v2 = reissue_info["new_template_id"]
    template_v2 = registry._database[tid_v2]["template"]
    print(f" -> [Step 5d] Fresh Capture Presentation -> Reissued Credential ID: '{tid_v2}' (Salt: '{new_salt}')")
    
    new_verif = registry.verify_request(feat_a2["descriptors"], feat_a2["weights"], tid_v2, pipeline)
    print(f" -> Genuine authentication with reissued template: {new_verif['decision']} (Score: {new_verif['score']:.4f})")
    assert new_verif["decision"] == "ACCEPT", "Genuine user must authenticate with reissued template!"
    
    revoked_score = match_protected_templates(
        template_v1["bitstrings"], template_v2["bitstrings"],
        template_v1["weights"], template_v2["weights"],
        hamming_threshold=pipeline.hamming_threshold
    )
    print(f" -> Match Score of revoked template v1 against reissued template v2: {revoked_score:.4f} << 0.20")
    print(" -> REVOCABILITY PROPERTY: Both database-level rejection and biometric cross-key decorrelation confirmed.")
    assert revoked_score < pipeline.operating_threshold, "Revoked template must not match reissued template!"
    
    print("\n" + "=" * 80)
    print(" DEMONSTRATION COMPLETE: Pipeline principles successfully verified.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
