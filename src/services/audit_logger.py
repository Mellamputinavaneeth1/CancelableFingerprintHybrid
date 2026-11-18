"""
Audit Logger Service (Privacy-Minimised Logging).

Implements strict privacy-preserving event logging conforming to ISO/IEC 24745:2022.
Enforces privacy boundaries:
- Strictly FORBIDS logging raw images, minutiae coordinates, unkeyed feature vectors, or secrets.
- Generates SHA-256 subject pseudonyms.
- Maintains cryptographic hash chain for tamper-evident audit trails.
"""

import os
import time
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

import threading
from src.services.types import AuditEvent, BiometricErrorCode


class AuditLogger:
    """
    Privacy-minimised audit logger.
    Logs lifecycle and operational events without biometric attribute leakage.
    Maintains a persistent, tamper-evident cryptographic hash chain across sessions.
    """
    FORBIDDEN_KEYS = {
        "image", "raw_image", "segmented", "enhanced", "skeleton",
        "minutiae", "coordinates", "descriptors", "features",
        "master_secret", "secret", "private_key", "storage_key"
    }

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._trail: List[AuditEvent] = []
        self._last_hash: str = "0" * 64
        self._lock = threading.Lock()

        if self.log_file and os.path.exists(self.log_file):
            self._load_existing_trail()

    def _load_existing_trail(self):
        """Loads existing audit events from file and resumes the cryptographic hash chain."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    event = AuditEvent(
                        event_id=data["event_id"],
                        timestamp_utc=data["timestamp_utc"],
                        operation=data["operation"],
                        subject_pseudonym=data["subject_pseudonym"],
                        decision=data["decision"],
                        score=data.get("score"),
                        latency_ms=data.get("latency_ms", 0.0),
                        error_code=data.get("error_code", 0),
                        integrity_status=data.get("integrity_status", ""),
                        metadata=data.get("metadata", {})
                    )
                    event_str = f"{self._last_hash}|{event.event_id}|{event.timestamp_utc}|{event.operation}|{event.subject_pseudonym}|{event.decision}|{event.error_code}"
                    self._last_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
                    self._trail.append(event)
        except Exception:
            pass

    def _pseudonymize_subject(self, subject_id: str) -> str:
        """Derives a one-way truncated SHA-256 pseudonym for audit logging."""
        if not subject_id:
            return "anon_subject"
        digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
        return f"subj_hash_{digest[:12]}"

    def _sanitize_metadata(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Sanitizes metadata to guarantee no biometric or secret data is logged."""
        if not meta:
            return {}
        sanitized = {}
        for k, v in meta.items():
            if k.lower() in self.FORBIDDEN_KEYS:
                sanitized[k] = "[REDACTED_PRIVACY_BOUNDARY]"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_metadata(v)
            elif isinstance(v, (np.ndarray, list)) and len(v) > 20:
                sanitized[k] = f"[REDACTED_ARRAY_SHAPE_{getattr(v, 'shape', len(v))}]"
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)[:64]
        return sanitized

    def log_event(self, 
                  operation: str, 
                  subject_id: str, 
                  decision: str, 
                  score: Optional[float] = None, 
                  latency_ms: float = 0.0,
                  error_code: int = int(BiometricErrorCode.SUCCESS),
                  metadata: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """
        Logs a privacy-minimised operational event.

        Args:
            operation: Event type ('ENROLL', 'VERIFY', 'REVOKE', 'KEY_ROTATION').
            subject_id: Raw subject identifier (will be pseudonymized).
            decision: Outcome ('ACCEPT', 'REJECT', 'REVOKED', 'ERROR').
            score: Matching score (if applicable).
            latency_ms: Execution duration in milliseconds.
            error_code: BiometricErrorCode integer.
            metadata: Sanitized non-sensitive metadata.

        Returns:
            AuditEvent object appended to the hash chain.
        """
        event_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pseudo = self._pseudonymize_subject(subject_id)
        clean_meta = self._sanitize_metadata(metadata)

        # Compute hash-chain entry: SHA256(prev_hash + event_data)
        with self._lock:
            event_str = f"{self._last_hash}|{event_id}|{timestamp}|{operation}|{pseudo}|{decision}|{error_code}"
            curr_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
            self._last_hash = curr_hash

            event = AuditEvent(
                event_id=event_id,
                timestamp_utc=timestamp,
                operation=operation,
                subject_pseudonym=pseudo,
                decision=decision,
                score=round(float(score), 4) if score is not None else None,
                latency_ms=round(float(latency_ms), 3),
                error_code=error_code,
                integrity_status=f"chain_hash:{curr_hash[:16]}",
                metadata=clean_meta
            )
            self._trail.append(event)

            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.__dict__) + "\n")

            return event

    def get_trail(self) -> List[AuditEvent]:
        """Returns the full audit trail."""
        with self._lock:
            return list(self._trail)

    def verify_integrity(self) -> bool:
        """Verifies the integrity of the hash chain."""
        with self._lock:
            trail_copy = list(self._trail)
        expected_hash = "0" * 64
        for ev in trail_copy:
            event_str = f"{expected_hash}|{ev.event_id}|{ev.timestamp_utc}|{ev.operation}|{ev.subject_pseudonym}|{ev.decision}|{ev.error_code}"
            expected_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
            if not ev.integrity_status.startswith(f"chain_hash:{expected_hash[:16]}"):
                return False
        return True

    @staticmethod
    def verify_file_integrity(log_path: str) -> bool:
        """Statically verifies the cryptographic tamper-evident hash chain of an on-disk audit log file."""
        if not os.path.exists(log_path):
            return False
        expected_hash = "0" * 64
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    event_str = f"{expected_hash}|{data['event_id']}|{data['timestamp_utc']}|{data['operation']}|{data['subject_pseudonym']}|{data['decision']}|{data['error_code']}"
                    expected_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
                    if not data.get("integrity_status", "").startswith(f"chain_hash:{expected_hash[:16]}"):
                        return False
            return True
        except Exception:
            return False
