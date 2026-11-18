import hashlib
import hmac
import os
import secrets
from typing import Union

def generate_master_secret() -> bytes:
    """
    Generates a cryptographically strong 32-byte (256-bit) master secret
    using the operating system CSPRNG (secrets module).
    """
    return secrets.token_bytes(32)

def validate_master_secret(master_secret: Union[bytes, str]) -> bytes:
    """
    Validates that the master secret satisfies cryptographic key standards (>= 256 bits of entropy).
    
    Accepts:
        master_secret: 32-byte raw bytes, or 64-character hexadecimal string,
                       or high-entropy secret string of at least 32 bytes.
                       
    Returns:
        secret_bytes: Normalized 32+ byte cryptographic secret.
        
    Raises:
        ValueError if the secret fails cryptographic entropy requirements.
    """
    if isinstance(master_secret, str):
        # Check if hex encoded
        try:
            if len(master_secret) == 64:
                secret_bytes = bytes.fromhex(master_secret)
            else:
                secret_bytes = master_secret.encode('utf-8')
        except ValueError:
            secret_bytes = master_secret.encode('utf-8')
    elif isinstance(master_secret, bytes):
        secret_bytes = master_secret
    else:
        raise TypeError("Master secret must be bytes or string.")
        
    if len(secret_bytes) < 32:
        raise ValueError(
            f"Master secret length ({len(secret_bytes)} bytes) is below the required 32 bytes (256 bits). "
            "Biometric master secrets must be generated from an OS CSPRNG."
        )
        
    return secret_bytes

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract(salt, IKM) -> PRK (RFC 5869) using HMAC-SHA256."""
    if len(salt) == 0:
        salt = bytes([0] * 32)
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand(PRK, info, L) -> OKM (RFC 5869).
    Enforces maximum expansion limit L <= 255 * HashLen (8160 bytes for SHA-256).
    """
    hash_len = 32
    if length > 255 * hash_len:
        raise ValueError(f"Requested length {length} exceeds RFC 5869 limit of 255 * {hash_len} = {255 * hash_len} bytes.")
        
    t = b""
    okm = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]

def derive_application_key(master_secret: Union[bytes, str], salt: str, app_id: str,
                           subject_id: str = "",
                           purpose: str = "template_projection",
                           version: str = "1.0",
                           length: int = 32) -> bytes:
    """
    Derives an application-specific cryptographic key using HKDF-SHA256 (RFC 5869).
    
    Context Binding:
        Uses a structured info string containing app_id, purpose, and version
        to ensure cryptographic domain separation across applications.
        
    Args:
        master_secret: 32-byte master cryptographic secret.
        salt: Per-enrollment or per-application salt string.
        app_id: Application identifier.
        purpose: Cryptographic purpose identifier.
        version: Protocol version.
        length: Desired key length in bytes.
        
    Returns:
        Derived pseudorandom key bytes.
    """
    ikm = validate_master_secret(master_secret)
    salt_bytes = salt.encode('utf-8')
    
    if subject_id:
        info_str = f"CancelableBio|subject={subject_id}|app={app_id}|purpose={purpose}|ver={version}"
    else:
        info_str = f"CancelableBio|app={app_id}|purpose={purpose}|ver={version}"
    info = info_str.encode('utf-8')
    
    prk = hkdf_extract(salt_bytes, ikm)
    return hkdf_expand(prk, info, length)
