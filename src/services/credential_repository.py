"""
Credential Repository Service.

Secure repository abstraction for storing, fetching, updating, and revoking
serialized encrypted biometric credentials at rest.
Integrates with TemplateRegistry to enforce ISO/IEC 24745:2022 revocation policies.
"""

from typing import Dict, Any, Optional, List
import time

from src.services.types import (
    ProtectedCredentialDTO,
    SerializedCredentialEnvelope,
    BiometricErrorCode,
    BiometricServiceException,
    RevokedCredentialError
)
from src.services.credential_serializer import CredentialSerializer
from src.revocation.registry import TemplateRegistry
import threading


class CredentialRepository:
    """
    Production repository managing encrypted credential persistence and lifecycle status.
    Guarantees that revoked credentials cannot be retrieved or authenticated.
    Supports in-memory storage (default) or persistent ACID SQLite storage when db_path is provided.
    """
    def __init__(self, 
                 serializer: Optional[CredentialSerializer] = None, 
                 registry: Optional[TemplateRegistry] = None,
                 db_path: Optional[str] = None):
        self.serializer = serializer or CredentialSerializer()
        self.registry = registry or TemplateRegistry()
        self.db_path = db_path
        self._lock = threading.RLock()

        if self.db_path:
            from src.services.sqlite_repository import SQLiteCredentialRepository
            self._sqlite = SQLiteCredentialRepository(db_path=self.db_path, serializer=self.serializer, registry=self.registry)
        else:
            self._sqlite = None
            self._store: Dict[str, bytes] = {}  # credential_id -> raw_bytes
            self._subject_map: Dict[str, str] = {}  # subject_id -> credential_id
            self._revoked_subjects: set = set()
            self._revocation_log: Dict[str, Dict[str, Any]] = {}

    def store(self, subject_id: str, envelope: SerializedCredentialEnvelope, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores an encrypted credential envelope under a given subject ID.
        """
        if self._sqlite:
            return self._sqlite.store(subject_id, envelope, metadata)

        with self._lock:
            cid = envelope.credential_id
            self._store[cid] = envelope.raw_bytes
            self._subject_map[subject_id] = cid
            self._revoked_subjects.discard(subject_id)

            # Register active status in template registry
            self.registry.enroll_template(
                subject_id=subject_id,
                app_id=metadata.get("app_id", "default_app") if metadata else "default_app",
                template={"credential_id": cid}
            )
            return cid

    def get_raw_bytes(self, subject_id: str) -> bytes:
        """Retrieves raw encrypted bytes if active; raises error if revoked or missing."""
        if self._sqlite:
            return self._sqlite.get_raw_bytes(subject_id)

        with self._lock:
            if self.is_revoked(subject_id):
                raise RevokedCredentialError(
                    BiometricErrorCode.CREDENTIAL_REVOKED,
                    f"Subject '{subject_id}' credential has been revoked and cannot be retrieved"
                )

            cid = self._subject_map.get(subject_id)
            if not cid or cid not in self._store:
                raise BiometricServiceException(
                    BiometricErrorCode.CREDENTIAL_NOT_FOUND,
                    f"No active credential found for subject '{subject_id}'"
                )
            return self._store[cid]

    def get_credential(self, 
                        subject_id: str, 
                        storage_key: bytes, 
                        app_id: str = "") -> ProtectedCredentialDTO:
        """
        Retrieves and decrypts the active protected credential for a subject.
        """
        if self._sqlite:
            return self._sqlite.get_credential(subject_id, storage_key, app_id)

        raw_bytes = self.get_raw_bytes(subject_id)
        return self.serializer.deserialize_protected(
            raw_bytes=raw_bytes,
            storage_key=storage_key,
            subject_id=subject_id,
            app_id=app_id
        )

    def revoke(self, subject_id: str, reason: str = "security_compromise") -> bool:
        """
        Revokes the active credential for a subject.
        """
        if self._sqlite:
            return self._sqlite.revoke(subject_id, reason)

        with self._lock:
            cid = self._subject_map.get(subject_id)
            if not cid:
                return False

            self._revoked_subjects.add(subject_id)
            self._revocation_log[cid] = {
                "subject_id": subject_id,
                "revoked_at": time.time(),
                "reason": reason
            }
            del self._subject_map[subject_id]

            # Also revoke in TemplateRegistry
            for tmpl_id, rec in self.registry._database.items():
                if rec.get("subject_id") == subject_id and rec.get("status") == "ACTIVE":
                    self.registry.revoke_template(tmpl_id, reason=reason)

            return True

    def is_revoked(self, subject_id: str) -> bool:
        """Returns True if subject had a credential that was revoked and has not re-enrolled."""
        if self._sqlite:
            return self._sqlite.is_revoked(subject_id)

        with self._lock:
            return subject_id in self._revoked_subjects and subject_id not in self._subject_map

    def count(self) -> int:
        """Returns number of active stored credentials."""
        if self._sqlite:
            return self._sqlite.count()

        with self._lock:
            return len(self._subject_map)

    def close(self):
        """Closes the underlying SQLite repository if persistent."""
        if self._sqlite:
            self._sqlite.close()
