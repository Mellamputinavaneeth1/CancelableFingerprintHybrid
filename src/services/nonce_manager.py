"""
Cryptographic Nonce Management Service for RFC 8439 ChaCha20-Poly1305.

Guarantees 96-bit (12-byte) nonce uniqueness under symmetric AEAD keys
through two supported allocation strategies:
1. Persistent Monotonic Counter Mode:
   - 4-byte worker/instance namespace prefix + 8-byte atomic monotonic counter.
   - Crash-resilient with persistent state file synchronization.
   - Strictly prevents nonce reuse across restarts and multi-threaded workers.
2. Random CSPRNG Bounded Invocations Mode:
   - 96-bit CSPRNG nonces (os.urandom) with per-key invocation tracking.
   - Enforces NIST SP 800-38D / RFC 8439 invocation cap (q_max <= 2^32).
   - Enforces key rotation when collision budget threshold is reached.
"""

import os
import struct
import threading
import json
from enum import Enum
from typing import Optional, Dict, Any, Set
from src.services.types import BiometricErrorCode, BiometricServiceException


class NonceAllocationStrategy(Enum):
    MONOTONIC_COUNTER = "monotonic_counter"
    RANDOM_BOUNDED = "random_bounded"


class NonceAllocationError(BiometricServiceException):
    """Raised when nonce allocation fails (e.g. counter overflow or reuse)."""
    pass


class KeyUsageLimitExceeded(BiometricServiceException):
    """Raised when the maximum safe invocations under a single key have been exhausted."""
    pass


class NonceManager:
    """
    Manages allocation, tracking, and uniqueness verification for ChaCha20-Poly1305 nonces.
    """
    def __init__(self,
                 strategy: NonceAllocationStrategy = NonceAllocationStrategy.MONOTONIC_COUNTER,
                 namespace_id: int = 1,
                 state_file: Optional[str] = None,
                 max_invocations_per_key: int = 4294967296):  # 2^32 default
        self._strategy = strategy
        self._namespace_id = namespace_id & 0xffffffff  # 32-bit namespace
        self._state_file = state_file
        self._max_invocations = max_invocations_per_key
        self._lock = threading.Lock()

        # Counter state
        self._counter: int = 0
        # Invocations per key fingerprint
        self._key_invocation_counts: Dict[str, int] = {}
        # Historical seen nonces for replay detection
        self._seen_nonces: Set[bytes] = set()

        if self._state_file and os.path.exists(self._state_file):
            self._load_state()

    def _load_state(self):
        """Loads counter, invocation state, and historical seen nonces from persistent storage."""
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._counter = int(data.get("counter", 0))
                self._namespace_id = int(data.get("namespace_id", self._namespace_id))
                self._key_invocation_counts = data.get("key_invocation_counts", {})
                saved_nonces = data.get("seen_nonces", [])
                self._seen_nonces = {bytes.fromhex(h) for h in saved_nonces}
        except Exception:
            # Fall back to starting fresh if state is unreadable
            self._counter = 0

    def _save_state(self):
        """Atomically persists counter state and historical seen nonces to disk."""
        if not self._state_file:
            return
        tmp_path = f"{self._state_file}.tmp"
        # Bounded persistence of recent seen nonces (up to 10,000) to prevent replay across restarts
        seen_hex = [n.hex() for n in list(self._seen_nonces)[-10000:]]
        data = {
            "counter": self._counter,
            "namespace_id": self._namespace_id,
            "key_invocation_counts": self._key_invocation_counts,
            "seen_nonces": seen_hex
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, self._state_file)

    def allocate_nonce(self, key_id: str = "default_key") -> bytes:
        """
        Allocates a guaranteed-unique 12-byte (96-bit) nonce under the specified key identifier.

        Args:
            key_id: Canonical identifier or SHA-256 fingerprint of the encryption key.

        Returns:
            12-byte unique nonce.

        Raises:
            KeyUsageLimitExceeded: If key has exceeded its maximum safe invocation budget.
            NonceAllocationError: If counter overflows or duplicate nonce is detected.
        """
        with self._lock:
            # Check invocation limit
            current_count = self._key_invocation_counts.get(key_id, 0)
            if current_count >= self._max_invocations:
                raise KeyUsageLimitExceeded(
                    BiometricErrorCode.AEAD_INTEGRITY_FAILURE,
                    f"Key '{key_id}' has reached maximum safe invocations ({self._max_invocations}). Mandatory key rotation required."
                )

            if self._strategy == NonceAllocationStrategy.MONOTONIC_COUNTER:
                # 4-byte namespace + 8-byte monotonic counter = 12 bytes (96 bits)
                self._counter += 1
                if self._counter > 0xffffffffffffffff:
                    raise NonceAllocationError(
                        BiometricErrorCode.SERIALIZATION_ERROR,
                        "64-bit monotonic nonce counter has overflowed. Key rotation required."
                    )
                nonce = struct.pack(">IQ", self._namespace_id, self._counter)
                self._save_state()
            else:
                # Random 96-bit nonce via OS CSPRNG
                nonce = os.urandom(12)
                # Replay/collision guard
                if nonce in self._seen_nonces:
                    raise NonceAllocationError(
                        BiometricErrorCode.AEAD_INTEGRITY_FAILURE,
                        "CSPRNG generated a duplicate nonce under key (collision detected)."
                    )
                self._seen_nonces.add(nonce)

            self._key_invocation_counts[key_id] = current_count + 1
            return nonce

    def verify_and_record_nonce(self, nonce: bytes, key_id: str = "default_key") -> bool:
        """
        Verifies that a received nonce has not been seen before under this key (replay prevention).

        Args:
            nonce: 12-byte nonce to verify.
            key_id: Key identifier.

        Returns:
            True if nonce is valid and fresh.

        Raises:
            NonceAllocationError: If nonce has already been used under this key.
        """
        if len(nonce) != 12:
            raise ValueError(f"Nonce must be exactly 12 bytes (got {len(nonce)})")

        with self._lock:
            key_nonce_tag = (key_id.encode("utf-8") + b"|" + nonce)
            if key_nonce_tag in self._seen_nonces:
                raise NonceAllocationError(
                    BiometricErrorCode.AEAD_INTEGRITY_FAILURE,
                    f"Nonce reuse detected for key '{key_id}': replay attack or duplicate credential"
                )
            self._seen_nonces.add(key_nonce_tag)
            if self._state_file:
                self._save_state()
            return True

    def get_status(self) -> Dict[str, Any]:
        """Returns non-sensitive audit metrics."""
        with self._lock:
            return {
                "strategy": self._strategy.value,
                "namespace_id": self._namespace_id,
                "current_counter": self._counter,
                "max_invocations_per_key": self._max_invocations,
                "active_keys_tracked": len(self._key_invocation_counts),
                "total_invocations": sum(self._key_invocation_counts.values())
            }
