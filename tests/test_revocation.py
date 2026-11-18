import numpy as np
import pytest
from src.pipeline import CancelableFingerprintPipeline
from src.revocation.registry import TemplateRegistry
from demo.app import generate_biometric_impression

def test_template_registry_lifecycle():
    pipeline = CancelableFingerprintPipeline()
    reg = TemplateRegistry()
    
    # 1. Enroll
    m1 = generate_biometric_impression(seed=301)
    f1 = pipeline.process_minutiae_list(m1)
    t1 = pipeline.enroll_from_features(f1["descriptors"], f1["weights"], salt="salt1", app_id="app")["template"]
    tid1 = reg.enroll_template("alice", "app", t1)
    
    assert reg.get_template_status(tid1) == "ACTIVE"
    
    # 2. Verify Active
    m2 = generate_biometric_impression(seed=301, rotation_deg=2.0)
    f2 = pipeline.process_minutiae_list(m2)
    v1 = reg.verify_request(f2["descriptors"], f2["weights"], tid1, pipeline)
    assert v1["decision"] == "ACCEPT"
    assert v1["status"] == "ACTIVE"
    
    # 3. Revoke
    assert reg.revoke_template(tid1, reason="compromise") is True
    assert reg.get_template_status(tid1) == "REVOKED"
    
    # 4. Blocked verification on revoked template
    v_rev = reg.verify_request(f2["descriptors"], f2["weights"], tid1, pipeline)
    assert v_rev["decision"] == "REJECT"
    assert v_rev["status"] == "REVOKED"
    
    # 5. Reissue from fresh capture
    m3 = generate_biometric_impression(seed=301, rotation_deg=-2.0)
    f3 = pipeline.process_minutiae_list(m3)
    reissue = reg.reissue_from_fresh_capture("alice", "app", f3["descriptors"], f3["weights"], new_salt="salt2", pipeline=pipeline)
    tid2 = reissue["new_template_id"]
    
    assert reg.get_template_status(tid2) == "ACTIVE"
    v2 = reg.verify_request(f2["descriptors"], f2["weights"], tid2, pipeline)
    assert v2["decision"] == "ACCEPT"
    assert v2["status"] == "ACTIVE"
