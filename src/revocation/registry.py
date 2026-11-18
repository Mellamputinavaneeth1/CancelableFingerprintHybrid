import time
import uuid
from typing import Dict, Any, Optional, List
import numpy as np
from src.matching.bipartite_matcher import match_protected_templates

class TemplateRegistry:
    """
    Database-level Biometric Template Registry managing credential lifecycles
    in compliance with ISO/IEC 24745:2022 revocation principles.
    
    4-Step Operational Revocation Workflow:
    --------------------------------------
    Step 1: Enrolls protected templates with unique reference IDs and 'ACTIVE' status.
    Step 2: Database Revocation: Marks compromised template ID as 'REVOKED'.
    Step 3: Verification Rejection: Any authentication request presenting a REVOKED
            template ID is immediately blocked at the database layer (STATUS_REVOKED).
    Step 4: Fresh Biometric Reissuance: User presents fresh biometric capture, generating
            a newly keyed template reference under updated salt/key, which authenticates
            successfully while the revoked template reference remains blocked.
    """
    def __init__(self):
        self._database: Dict[str, Dict[str, Any]] = {}
        self._subject_active_map: Dict[str, str] = {} # subject_id -> active template_id
        
    def enroll_template(self, subject_id: str, app_id: str, template: Dict[str, Any]) -> str:
        """Enrolls a protected biometric template into the database."""
        template_id = f"tmpl_{uuid.uuid4().hex[:12]}"
        record = {
            "template_id": template_id,
            "subject_id": subject_id,
            "app_id": app_id,
            "template": template,
            "status": "ACTIVE",
            "enrolled_at": time.time(),
            "revoked_at": None,
            "revocation_reason": None
        }
        self._database[template_id] = record
        self._subject_active_map[subject_id] = template_id
        return template_id

    def revoke_template(self, template_id: str, reason: str = "key_compromise") -> bool:
        """
        Step 2: Marks a compromised template ID as REVOKED in the database.
        """
        if template_id not in self._database:
            return False
            
        record = self._database[template_id]
        record["status"] = "REVOKED"
        record["revoked_at"] = time.time()
        record["revocation_reason"] = reason
        
        # Remove from active mapping if present
        subj = record["subject_id"]
        if self._subject_active_map.get(subj) == template_id:
            del self._subject_active_map[subj]
            
        return True

    def get_template_status(self, template_id: str) -> Optional[str]:
        """Returns the current lifecycle status ('ACTIVE', 'REVOKED', or None)."""
        record = self._database.get(template_id)
        return record["status"] if record else None

    def verify_request(self, query_descriptors: np.ndarray, query_weights: np.ndarray,
                       template_id: str, pipeline) -> Dict[str, Any]:
        """
        Step 3: Verification attempt against template ID.
        If the template is REVOKED in the database, requests are immediately blocked.
        """
        if template_id not in self._database:
            return {
                "decision": "REJECT",
                "status": "NOT_FOUND",
                "score": 0.0,
                "reason": f"Template ID '{template_id}' does not exist in registry."
            }
            
        record = self._database[template_id]
        if record["status"] == "REVOKED":
            return {
                "decision": "REJECT",
                "status": "REVOKED",
                "score": 0.0,
                "reason": f"Template ID '{template_id}' has been revoked ({record['revocation_reason']}). Access denied."
            }
            
        # Proceed to cryptographic biometric verification
        verif = pipeline.verify_from_features(query_descriptors, query_weights, record["template"])
        verif["status"] = "ACTIVE"
        verif["template_id"] = template_id
        return verif

    def reissue_from_fresh_capture(self, subject_id: str, app_id: str,
                                   fresh_descriptors: np.ndarray, fresh_weights: np.ndarray,
                                   new_salt: str, pipeline) -> Dict[str, Any]:
        """
        Step 4: Fresh Biometric Presentation & Reissuance.
        Revokes previous credential, generates new reference under new_salt, and activates it.
        """
        # Revoke previous active template if one exists
        old_tid = self._subject_active_map.get(subject_id)
        if old_tid:
            self.revoke_template(old_tid, reason="reissued_fresh_presentation")
            
        enroll_res = pipeline.enroll_from_features(
            fresh_descriptors, fresh_weights, salt=new_salt, app_id=app_id, subject_id=subject_id
        )
        if enroll_res["status"] != "success":
            return {"status": "error", "message": "Enrollment failed", "template_id": None}
            
        new_template = enroll_res["template"]
        new_template["reissued_from"] = old_tid
        new_tid = self.enroll_template(subject_id, app_id, new_template)
        
        return {
            "status": "success",
            "revoked_template_id": old_tid,
            "new_template_id": new_tid,
            "new_salt": new_salt
        }
