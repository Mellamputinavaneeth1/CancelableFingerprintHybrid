"""
Credential Serializer Service.

Implements binary framing, symmetric 2-bit matrix packing, and AEAD encryption at rest
(ChaCha20-Poly1305 RFC 8439) for protected biometric credentials,
along with strictly like-for-like continuous baseline serializers.
"""

import os
import struct
import zlib
import uuid
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from src.matching.graph_consensus import pack_adjacency_matrix, unpack_adjacency_matrix
from src.services.crypto_aead import chacha20_poly1305_encrypt, chacha20_poly1305_decrypt
from src.services.types import (
    ProtectedCredentialDTO,
    SerializedCredentialEnvelope,
    BiometricErrorCode,
    BiometricServiceException,
    AeadIntegrityError
)

MAGIC_BYTES = b"CFPB"  # Cancelable Fingerprint Biometrics
SCHEMA_VERSION = 1
HEADER_FORMAT = "<4sHHBB2s16s16sI12sI"
# Layout:
# 4s : Magic (4B)
# H  : Version (2B)
# H  : Flags (2B)
# B  : In-dim (1B)
# B  : Reserved0 (1B)
# 2s : Reserved1 (2B)
# 16s: Credential ID bytes (16B)
# 16s: Salt bytes (16B)
# I  : Num triangles N (4B)
# 12s: Nonce (12B)
# I  : Header CRC32 (4B)
# Total = 4 + 2 + 2 + 1 + 1 + 2 + 16 + 16 + 4 + 12 + 4 = 64 bytes
HEADER_SIZE = 64
TAG_SIZE = 16


def _encode_uuid(u_str: str) -> bytes:
    try:
        return uuid.UUID(u_str).bytes
    except Exception:
        # Fallback to truncated SHA-256
        import hashlib
        return hashlib.sha256(u_str.encode("utf-8")).digest()[:16]


def _decode_uuid(u_bytes: bytes) -> str:
    try:
        return str(uuid.UUID(bytes=u_bytes))
    except Exception:
        return u_bytes.hex()


def normalize_salt(salt_str: str) -> str:
    """Normalizes salt string to a canonical 16-character ASCII string."""
    if len(salt_str) == 16 and all(c.isascii() for c in salt_str):
        return salt_str
    import hashlib
    return hashlib.sha256(salt_str.encode("utf-8")).hexdigest()[:16]


def _pad_or_hash_salt(salt_str: str) -> bytes:
    norm = normalize_salt(salt_str)
    return norm.encode("ascii")


class CredentialSerializer:
    """
    Production binary serializer and AEAD at rest encryptor.
    Provides byte-exact serialized formats for protected and unprotected credentials.
    """
    def __init__(self, default_cipher: str = "chacha20_poly1305"):
        self.default_cipher = default_cipher

    def serialize_protected(self, 
                            credential: ProtectedCredentialDTO, 
                            storage_key: bytes,
                            encrypt: bool = True,
                            nonce: Optional[bytes] = None,
                            compact_edges: bool = False) -> SerializedCredentialEnvelope:
        """
        Serializes a protected credential into binary framing and encrypts at rest with AEAD.

        Args:
            credential: ProtectedCredentialDTO containing bitstrings, weights, salt, etc.
            storage_key: 32-byte key derived by KeyService.
            encrypt: Whether to apply AEAD encryption (default True).
            nonce: Optional 12-byte nonce (e.g. allocated by NonceManager). If None, generated via CSPRNG.

        Returns:
            SerializedCredentialEnvelope with complete byte payload and metadata.
        """
        n = int(credential.num_triangles)
        bits_arr = np.asarray(credential.bitstrings, dtype=np.uint8)
        if bits_arr.ndim == 2 and bits_arr.shape[1] > 0:
            out_dim = int(bits_arr.shape[1])
        else:
            out_dim = 32
        if out_dim % 8 != 0 or out_dim <= 0 or out_dim > 4095:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Invalid out_dim {out_dim}: must be a positive multiple of 8 <= 4095"
            )

        in_dim = int(getattr(credential, "in_dim", 9))
        has_topology = credential.packed_topology is not None and len(credential.packed_topology) > 0
        has_edge_tokens = credential.edge_tokens is not None and credential.edge_tokens.size > 0

        edge_rff_bits = 0
        packed_edges = b""
        has_sparse_edges = False
        if has_edge_tokens:
            edges_arr = np.ascontiguousarray(credential.edge_tokens, dtype="<u8")
            if edges_arr.ndim != 3 or edges_arr.shape[0] != n or edges_arr.shape[1] != n:
                raise BiometricServiceException(
                    BiometricErrorCode.SERIALIZATION_ERROR,
                    f"Edge token shape {edges_arr.shape} must match ({n}, {n}, w64)"
                )
            edge_rff_bits = int(edges_arr.shape[2]) * 64
            edge_mask_raw = getattr(credential, "edge_mask", None)
            has_sparse_edges = bool(n > 0 and (compact_edges or edge_mask_raw is not None))
            if has_sparse_edges:
                if edge_mask_raw is not None:
                    mask = np.asarray(edge_mask_raw, dtype=bool)
                    if mask.shape != (n, n):
                        raise BiometricServiceException(
                            BiometricErrorCode.SERIALIZATION_ERROR,
                            f"edge_mask shape {mask.shape} must match ({n}, {n})"
                        )
                else:
                    nonzero_mask = np.any(edges_arr != 0, axis=-1)
                    if n <= 1:
                        mask = np.zeros((n, n), dtype=bool)
                    elif int(nonzero_mask.sum()) <= (n * n) // 2:
                        mask = nonzero_mask
                    else:
                        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                mask_bytes = np.packbits(mask.ravel()).tobytes()
                active_edges = np.ascontiguousarray(edges_arr[mask], dtype="<u8")
                packed_edges = mask_bytes + active_edges.tobytes()
            else:
                packed_edges = edges_arr.tobytes()

        # Construct Flags (uint16):
        # bit 0: has_topology
        # bit 1: cipher_suite (0 = chacha20_poly1305)
        # bit 2: is_encrypted (1 if encrypt else 0)
        # bit 4: has_edge_tokens
        # bit 5: has_sparse_edges (masked compact edge tokens)
        # bit 8..15: low 8 bits of out_dim (high 4 bits in res0[4..7] when >= 256)
        flags = 0
        if has_topology:
            flags |= (1 << 0)
        if encrypt:
            flags |= (1 << 2)
        if has_edge_tokens:
            flags |= (1 << 4)
        if has_sparse_edges:
            flags |= (1 << 5)
        flags |= ((out_dim & 0xff) << 8)

        res0 = ((in_dim >> 8) & 0x0f) | (((out_dim >> 8) & 0x0f) << 4)
        res1 = struct.pack("<H", edge_rff_bits) if has_edge_tokens else b"\x00\x00"

        cred_id_bytes = _encode_uuid(credential.credential_id)
        salt_bytes = _pad_or_hash_salt(credential.salt)
        if nonce is None:
            nonce = os.urandom(12) if encrypt else (b"\x00" * 12)
        else:
            if len(nonce) != 12:
                raise ValueError(f"Nonce must be exactly 12 bytes (got {len(nonce)})")

        # 1. Pack Plaintext Payload
        # (a) Bitstrings: N x out_dim bits -> N x (out_dim // 8) bytes
        if bits_arr.ndim == 2 and bits_arr.shape[1] == out_dim:
            packed_bits = np.packbits(bits_arr, axis=1).tobytes()
        elif bits_arr.ndim == 1:
            # Already packed uint32
            packed_bits = bits_arr.astype("<u4").tobytes()
        else:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Unexpected bitstring shape: {bits_arr.shape}"
            )

        # (b) Float16 weights: N x 2 bytes
        weights_arr = np.asarray(credential.weights, dtype=np.float16)
        packed_weights = weights_arr.tobytes()

        # (c) Topology: optional packed matrix A bytes
        packed_topo = credential.packed_topology if has_topology else b""

        plaintext_payload = packed_bits + packed_weights + packed_topo + packed_edges

        # 2. Build 64-byte Header (excluding CRC initially)
        header_pre = struct.pack(
            "<4sHHBB2s16s16sI12s",
            MAGIC_BYTES,
            SCHEMA_VERSION,
            flags,
            in_dim & 0xff,
            res0,
            res1,
            cred_id_bytes,
            salt_bytes,
            n,
            nonce
        )
        # Header CRC32 computed over first 60 bytes
        header_crc = zlib.crc32(header_pre) & 0xffffffff
        header_bytes = header_pre + struct.pack("<I", header_crc)
        assert len(header_bytes) == HEADER_SIZE, f"Header size must be {HEADER_SIZE}, got {len(header_bytes)}"

        # 3. AEAD Encryption
        if encrypt:
            ciphertext, auth_tag = chacha20_poly1305_encrypt(
                key=storage_key,
                nonce=nonce,
                plaintext=plaintext_payload,
                aad=header_bytes
            )
        else:
            ciphertext = plaintext_payload
            auth_tag = b"\x00" * TAG_SIZE

        full_raw_bytes = header_bytes + ciphertext + auth_tag

        return SerializedCredentialEnvelope(
            raw_bytes=full_raw_bytes,
            total_bytes=len(full_raw_bytes),
            credential_id=credential.credential_id,
            num_triangles=n,
            has_topology=has_topology,
            cipher_suite="ChaCha20-Poly1305" if encrypt else "Plaintext",
            nonce=nonce,
            auth_tag=auth_tag,
            is_encrypted=encrypt,
            has_edge_tokens=has_edge_tokens,
            has_sparse_edges=has_sparse_edges,
        )

    def deserialize_protected(self, 
                              raw_bytes: bytes, 
                              storage_key: bytes, 
                              subject_id: str = "",
                              app_id: str = "") -> ProtectedCredentialDTO:
        """
        Deserializes binary credential, validates magic, checks CRC, verifies AEAD tag,
        and unpacks into ProtectedCredentialDTO.

        Args:
            raw_bytes: Binary serialized credential envelope.
            storage_key: 32-byte key derived by KeyService.
            subject_id: Subject identifier.
            app_id: Application identifier.

        Returns:
            ProtectedCredentialDTO.

        Raises:
            BiometricServiceException / AeadIntegrityError: If tampered, corrupt, or invalid.
        """
        if len(raw_bytes) < HEADER_SIZE + TAG_SIZE:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Payload too short ({len(raw_bytes)} bytes) for valid credential"
            )

        header_bytes = raw_bytes[:HEADER_SIZE]
        (
            magic,
            version,
            flags,
            in_dim,
            res0,
            res1,
            cred_id_bytes,
            salt_bytes,
            n,
            nonce,
            stored_crc
        ) = struct.unpack(HEADER_FORMAT, header_bytes)

        if magic != MAGIC_BYTES:
            raise BiometricServiceException(
                BiometricErrorCode.INVALID_MAGIC_BYTES,
                f"Invalid magic bytes: expected {MAGIC_BYTES}, got {magic}"
            )

        if version != SCHEMA_VERSION:
            raise BiometricServiceException(
                BiometricErrorCode.UNSUPPORTED_VERSION,
                f"Unsupported schema version: {version}"
            )

        # Check header CRC32
        computed_crc = zlib.crc32(header_bytes[:60]) & 0xffffffff
        if computed_crc != stored_crc:
            raise AeadIntegrityError(
                BiometricErrorCode.AEAD_INTEGRITY_FAILURE,
                "Header checksum mismatch: header framing corrupted or tampered"
            )

        has_topology = bool(flags & (1 << 0))
        is_encrypted = bool(flags & (1 << 2))
        has_edge_tokens = bool(flags & (1 << 4))
        has_sparse_edges = bool(flags & (1 << 5))
        full_in_dim = ((res0 & 0x0f) << 8) | in_dim
        out_dim = (((res0 >> 4) & 0x0f) << 8) | ((flags >> 8) & 0xff)
        if out_dim == 0:
            out_dim = 32

        if out_dim % 8 != 0:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Invalid out_dim {out_dim}: must be a multiple of 8"
            )

        edge_rff_bits = struct.unpack("<H", res1)[0] if has_edge_tokens else 0
        if has_edge_tokens and (edge_rff_bits <= 0 or edge_rff_bits % 64 != 0):
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Invalid edge_rff_bits {edge_rff_bits}: must be a positive multiple of 64"
            )

        desc_bytes_len = n * (out_dim // 8)
        w_bytes_len = n * 2  # float16
        expected_topo_len = ((n * (n - 1) // 2) + 3) // 4 if (has_topology and n > 1) else 0
        prefix_len = desc_bytes_len + w_bytes_len + expected_topo_len
        w_edge_bytes = edge_rff_bits // 8
        mask_bytes_len = ((n * n + 7) // 8) if (has_edge_tokens and has_sparse_edges and n > 0) else 0

        ciphertext = raw_bytes[HEADER_SIZE:-TAG_SIZE]
        auth_tag = raw_bytes[-TAG_SIZE:]

        if has_edge_tokens and has_sparse_edges:
            min_sparse_len = prefix_len + mask_bytes_len
            if len(ciphertext) < min_sparse_len or ((len(ciphertext) - min_sparse_len) % w_edge_bytes != 0):
                raise BiometricServiceException(
                    BiometricErrorCode.SERIALIZATION_ERROR,
                    f"Sparse payload size mismatch: got {len(ciphertext)} bytes for N={n} (min={min_sparse_len}, edge_bytes={w_edge_bytes})"
                )
        else:
            expected_edge_len = (n * n * w_edge_bytes) if has_edge_tokens else 0
            expected_payload_len = prefix_len + expected_edge_len
            if len(ciphertext) != expected_payload_len:
                raise BiometricServiceException(
                    BiometricErrorCode.SERIALIZATION_ERROR,
                    f"Payload size mismatch: expected {expected_payload_len} bytes for N={n} (desc={desc_bytes_len}, w={w_bytes_len}, topo={expected_topo_len}, edge={expected_edge_len}), got {len(ciphertext)} bytes"
                )

        # Decrypt payload
        if is_encrypted:
            plaintext = chacha20_poly1305_decrypt(
                key=storage_key,
                nonce=nonce,
                ciphertext=ciphertext,
                tag=auth_tag,
                aad=header_bytes
            )
        else:
            plaintext = ciphertext

        if not (has_edge_tokens and has_sparse_edges) and len(plaintext) != expected_payload_len:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Decrypted plaintext size mismatch: expected {expected_payload_len} bytes, got {len(plaintext)} bytes"
            )

        # Unpack components from plaintext
        try:
            off_w = desc_bytes_len
            off_topo = off_w + w_bytes_len
            off_edge = off_topo + expected_topo_len

            packed_bits = plaintext[:off_w]
            packed_weights = plaintext[off_w:off_topo]
            packed_topo = plaintext[off_topo:off_edge] if has_topology else None
            edge_tokens = None
            edge_mask = None
            if has_edge_tokens and n > 0:
                w64 = edge_rff_bits // 64
                if has_sparse_edges:
                    if len(plaintext) < off_edge + mask_bytes_len:
                        raise ValueError("Truncated sparse edge mask")
                    mask_buf = np.frombuffer(plaintext[off_edge:off_edge + mask_bytes_len], dtype=np.uint8)
                    edge_mask = np.unpackbits(mask_buf)[: n * n].reshape((n, n)).astype(bool)
                    k_edges = int(edge_mask.sum())
                    expected_total = off_edge + mask_bytes_len + k_edges * w_edge_bytes
                    if len(plaintext) != expected_total:
                        raise ValueError(f"Sparse edge count mismatch: expected {expected_total} bytes, got {len(plaintext)}")
                    edge_tokens = np.zeros((n, n, w64), dtype=np.uint64)
                    if k_edges > 0:
                        active_edges = (
                            np.frombuffer(plaintext[off_edge + mask_bytes_len:], dtype="<u8")
                            .reshape((k_edges, w64))
                            .copy()
                            .astype(np.uint64)
                        )
                        edge_tokens[edge_mask] = active_edges
                else:
                    edge_tokens = (
                        np.frombuffer(plaintext[off_edge:], dtype="<u8")
                        .reshape((n, n, w64))
                        .copy()
                        .astype(np.uint64)
                    )

            # Convert bitstrings back to (N, out_dim) uint8 array
            bit_arr_packed = np.frombuffer(packed_bits, dtype=np.uint8).reshape((n, out_dim // 8))
            bitstrings = np.unpackbits(bit_arr_packed, axis=1)[:, :out_dim]

            # Convert weights back to float32
            weights = np.frombuffer(packed_weights, dtype=np.float16).astype(np.float32)
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.SERIALIZATION_ERROR,
                f"Failed to unpack credential payload: {str(e)}"
            )

        cred_id_str = _decode_uuid(cred_id_bytes)
        salt_str = salt_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")

        return ProtectedCredentialDTO(
            credential_id=cred_id_str,
            subject_id=subject_id,
            salt=salt_str,
            app_id=app_id,
            bitstrings=bitstrings,
            weights=weights,
            num_triangles=n,
            packed_topology=packed_topo,
            version=str(version),
            edge_tokens=edge_tokens,
            in_dim=full_in_dim,
            edge_mask=edge_mask,
        )

    def serialize_unprotected_baseline(self, 
                                       continuous_features: np.ndarray, 
                                       weights: np.ndarray,
                                       packed_topology: Optional[bytes] = None,
                                       credential_id: Optional[str] = None) -> bytes:
        """
        Serializes continuous unprotected baseline credential with strictly equivalent functionality:
        - 64-byte Header (Magic CFPB, flags with baseline indicator)
        - Continuous geometric descriptors: N x 9 x 4 bytes (float32)
        - Continuous reliability weights: N x 4 bytes (float32)
        - Total per triangle = 40 bytes/triangle + header + optional topology
        """
        n = len(continuous_features)
        has_topology = packed_topology is not None and len(packed_topology) > 0
        cid = credential_id or str(uuid.uuid4())
        cred_id_bytes = _encode_uuid(cid)
        salt_bytes = b"unprotected_base"

        flags = (1 << 3)  # baseline flag
        if has_topology:
            flags |= (1 << 0)

        header_pre = struct.pack(
            "<4sHHBB2s16s16sI12s",
            MAGIC_BYTES,
            SCHEMA_VERSION,
            flags,
            9,
            0,
            b"\x00\x00",
            cred_id_bytes,
            salt_bytes,
            n,
            b"\x00" * 12
        )
        header_crc = zlib.crc32(header_pre) & 0xffffffff
        header_bytes = header_pre + struct.pack("<I", header_crc)

        desc_bytes = continuous_features.astype(np.float32).tobytes()
        w_bytes = weights.astype(np.float32).tobytes()
        topo_bytes = packed_topology if has_topology else b""

        return header_bytes + desc_bytes + w_bytes + topo_bytes
