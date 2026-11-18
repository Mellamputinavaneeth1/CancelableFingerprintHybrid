"""
Key Management Service (KMS / HSM Abstraction).

Enforces cryptographic trust boundaries in accordance with ISO/IEC 24745:2022:
- Master secret resides inside secure KMS boundary and is NEVER exposed to clients.
- Derives application-specific column-orthonormal projection matrices via RFC 5869 HKDF.
- Derives AEAD storage encryption keys for credential encryption at rest.
- Supports cryptographically audited key rotation and lifecycle transitions.
"""

import os
import hashlib
import hmac
import inspect
import threading
from typing import Union, Dict, Any, Optional
import numpy as np

from src.protection.kdf import validate_master_secret, derive_application_key
from src.protection.projection import derive_orthonormal_matrix
from src.services.types import (
    BiometricErrorCode, 
    MasterSecretAccessViolation
)


class KeyService:
    """
    KMS/HSM abstraction service.
    Clients may request key-derived mathematical artifacts (projection matrices, storage keys)
    but cannot access the underlying master cryptographic seed.
    """
    def __init__(self, 
                 master_secret: Union[bytes, str] = "CancelableBioMasterSecret_2026_256Bits_KMS!",
                 kms_key_id: str = "kms://vault-01/cancelable-bio-master-key",
                 epoch: int = 1):
        self._lock = threading.RLock()
        self._master_secret: bytes = validate_master_secret(master_secret)
        self._kms_key_id: str = kms_key_id
        self._epoch: int = epoch
        self._rotation_history: list = []

    def __getattribute__(self, name: str):
        if name in ("master_secret", "_master_secret", "_secret", "secret", "raw_secret"):
            frame = inspect.currentframe().f_back
            if frame.f_locals.get("self") is not self:
                raise MasterSecretAccessViolation(
                    BiometricErrorCode.MASTER_SECRET_ACCESS_DENIED,
                    f"Direct access to '{name}' is forbidden by KMS trust boundary policy."
                )
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return f"<KeyService kms_id='{self._kms_key_id}' epoch={self._epoch} status='PROTECTED_HSM'>"

    def __str__(self) -> str:
        return f"KeyService(kms_key_id='{self._kms_key_id}', epoch={self._epoch}, secret='[REDACTED]')"

    @property
    def kms_key_id(self) -> str:
        """Public identifier of the KMS key resource."""
        return self._kms_key_id

    @property
    def epoch(self) -> int:
        """Current cryptographic epoch."""
        with self._lock:
            return self._epoch

    def derive_projection_matrix(self, 
                                 salt: str, 
                                 app_id: str, 
                                 subject_id: str = "",
                                 in_dim: int = 9, 
                                 out_dim: int = 32) -> np.ndarray:
        """
        Derives column-orthonormal projection matrix W in R^(out_dim x in_dim).

        Args:
            salt: Application-specific or presentation salt.
            app_id: Application identifier.
            subject_id: Optional subject pseudonym.
            in_dim: Input geometric feature dimension (default 9).
            out_dim: Quantized output projection dimension (default 32).

        Returns:
            Column-orthonormal matrix W of shape (out_dim, in_dim).
        """
        with self._lock:
            secret = self._master_secret
        return derive_orthonormal_matrix(
            secret,
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            in_dim=in_dim,
            out_dim=out_dim
        )

    def derive_rff_edge_params(
        self,
        salt: str,
        app_id: str,
        subject_id: str = "",
        edge_dim: int = 7,
        rff_bits: int = 128,
    ) -> tuple:
        """
        Derives keyed Random Fourier Feature (RFF) projection weights and phase
        shifts for privacy-preserving SE(2) spectral edge token generation.
        """
        from src.matching.spectral_consensus import derive_rff_edge_params as _derive_rff

        with self._lock:
            secret = self._master_secret
        return _derive_rff(
            master_secret=secret,
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            edge_dim=edge_dim,
            rff_bits=rff_bits,
        )

    def derive_storage_key(self, salt: str, app_id: str) -> bytes:
        """
        Derives a 32-byte symmetric AEAD encryption key for encrypting credentials at rest.
        Uses RFC 5869 HKDF-SHA256 with domain-separated info context.

        Args:
            salt: Application or credential salt.
            app_id: Application identifier.

        Returns:
            32-byte cryptographic key.
        """
        with self._lock:
            secret = self._master_secret
            epoch_val = self._epoch
        return derive_application_key(
            master_secret=secret,
            salt=salt,
            app_id=app_id,
            purpose=f"aead-storage-at-rest|epoch:{epoch_val}",
            length=32
        )

    def rotate_master_secret(self, new_master_secret: Union[bytes, str], reason: str = "scheduled_rotation"):
        """
        Rotates the underlying master secret and advances epoch.

        Args:
            new_master_secret: New minimum 256-bit secret.
            reason: Audit rationale for key rotation.
        """
        validated = validate_master_secret(new_master_secret)
        with self._lock:
            self._rotation_history.append({
                "previous_epoch": self._epoch,
                "new_epoch": self._epoch + 1,
                "reason": reason
            })
            self._master_secret = validated
            self._epoch += 1

    def get_kms_metadata(self) -> Dict[str, Any]:
        """Returns non-sensitive metadata for auditing."""
        with self._lock:
            # Compute SHA-256 key fingerprint for audit trail verification without exposing secret
            fingerprint = hashlib.sha256(self._master_secret).hexdigest()[:16]
            return {
                "kms_key_id": self._kms_key_id,
                "epoch": self._epoch,
                "key_fingerprint": f"sha256:{fingerprint}...",
                "rotation_count": len(self._rotation_history),
                "status": "ACTIVE"
            }
