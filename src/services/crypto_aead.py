"""
Cryptographic Authenticated Encryption with Associated Data (AEAD) Service.

Implements pure-Python RFC 8439 ChaCha20-Poly1305 AEAD cipher.
Provides AEAD at rest protection for serialized cancelable biometric credentials,
guaranteeing confidentiality, authenticity, and tamper detection with zero external dependencies.
"""

import struct
import hmac
from typing import Tuple, Optional, Union
from src.services.types import BiometricErrorCode, AeadIntegrityError


def _rol(v: int, c: int) -> int:
    """32-bit left bitwise rotation."""
    return ((v << c) & 0xffffffff) | (v >> (32 - c))


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """Generates a 64-byte ChaCha20 keystream block as defined in RFC 8439 Section 2.3."""
    constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
    key_words = list(struct.unpack("<8I", key))
    nonce_words = list(struct.unpack("<3I", nonce))
    state = constants + key_words + [counter & 0xffffffff] + nonce_words
    working = list(state)

    def _quarter_round(a: int, b: int, c: int, d: int):
        working[a] = (working[a] + working[b]) & 0xffffffff
        working[d] = _rol(working[d] ^ working[a], 16)
        working[c] = (working[c] + working[d]) & 0xffffffff
        working[b] = _rol(working[b] ^ working[c], 12)
        working[a] = (working[a] + working[b]) & 0xffffffff
        working[d] = _rol(working[d] ^ working[a], 8)
        working[c] = (working[c] + working[d]) & 0xffffffff
        working[b] = _rol(working[b] ^ working[c], 7)

    # 10 double rounds (20 rounds total)
    for _ in range(10):
        # Column round
        _quarter_round(0, 4, 8, 12)
        _quarter_round(1, 5, 9, 13)
        _quarter_round(2, 6, 10, 14)
        _quarter_round(3, 7, 11, 15)
        # Diagonal round
        _quarter_round(0, 5, 10, 15)
        _quarter_round(1, 6, 11, 12)
        _quarter_round(2, 7, 8, 13)
        _quarter_round(3, 4, 9, 14)

    out = [(working[i] + state[i]) & 0xffffffff for i in range(16)]
    return struct.pack("<16I", *out)


def _chacha20_xor(key: bytes, counter: int, nonce: bytes, data: bytes) -> bytes:
    """Encrypts/decrypts data by XORing with ChaCha20 keystream starting at counter."""
    res = bytearray()
    curr_counter = counter
    for i in range(0, len(data), 64):
        block = _chacha20_block(key, curr_counter, nonce)
        chunk = data[i:i + 64]
        res.extend(bytes(a ^ b for a, b in zip(chunk, block[:len(chunk)])))
        curr_counter = (curr_counter + 1) & 0xffffffff
    return bytes(res)


def _poly1305_mac(msg: bytes, key: bytes) -> bytes:
    """
    Computes 16-byte Poly1305 one-time authenticator as defined in RFC 8439 Section 2.5.
    Prime p = 2^130 - 5.
    """
    # Clamp r according to RFC 8439 Section 2.5
    r = int.from_bytes(key[:16], "little") & 0x0ffffffc0ffffffc0ffffffc0fffffff
    s = int.from_bytes(key[16:32], "little")
    p = (1 << 130) - 5
    a = 0

    for i in range(0, len(msg), 16):
        block = msg[i:i + 16]
        # Append 0x01 byte to block as little-endian integer
        n = int.from_bytes(block + b"\x01", "little")
        a = ((a + n) * r) % p

    a = (a + s) % (1 << 128)
    return a.to_bytes(16, "little")


def chacha20_poly1305_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
    """
    Encrypts plaintext and authenticates with Associated Authenticated Data (AAD)
    under RFC 8439 ChaCha20-Poly1305.

    Args:
        key: 32-byte (256-bit) encryption key.
        nonce: 12-byte (96-bit) unique nonce.
        plaintext: Bytes to encrypt.
        aad: Optional associated authenticated data bytes (e.g. credential header).

    Returns:
        (ciphertext, auth_tag): 
            ciphertext: Bytes of equal length to plaintext.
            auth_tag: 16-byte Poly1305 MAC tag.
    """
    if len(key) != 32:
        raise ValueError(f"ChaCha20 key must be exactly 32 bytes (got {len(key)})")
    if len(nonce) != 12:
        raise ValueError(f"ChaCha20 nonce must be exactly 12 bytes (got {len(nonce)})")

    # Generate one-time Poly1305 subkey from block 0
    otk = _chacha20_block(key, 0, nonce)[:32]

    # Encrypt plaintext starting at block counter 1
    ciphertext = _chacha20_xor(key, 1, nonce, plaintext)

    # Construct authenticated payload: pad16(AAD) + pad16(Ciphertext) + len(AAD) + len(Ciphertext)
    pad16 = lambda b: b + (b"\x00" * ((16 - (len(b) % 16)) % 16))
    mac_data = (
        pad16(aad) +
        pad16(ciphertext) +
        struct.pack("<QQ", len(aad), len(ciphertext))
    )

    tag = _poly1305_mac(mac_data, otk)
    return ciphertext, tag


def chacha20_poly1305_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"") -> bytes:
    """
    Verifies authentication tag and decrypts ciphertext under RFC 8439 ChaCha20-Poly1305.

    Args:
        key: 32-byte encryption key.
        nonce: 12-byte nonce.
        ciphertext: Ciphertext bytes.
        tag: 16-byte Poly1305 authentication tag.
        aad: Associated authenticated data bytes.

    Returns:
        plaintext: Decrypted bytes.

    Raises:
        AeadIntegrityError: If authentication tag validation fails (tamper detection).
    """
    if len(key) != 32:
        raise ValueError(f"ChaCha20 key must be exactly 32 bytes (got {len(key)})")
    if len(nonce) != 12:
        raise ValueError(f"ChaCha20 nonce must be exactly 12 bytes (got {len(nonce)})")
    if len(tag) != 16:
        raise ValueError(f"Poly1305 tag must be exactly 16 bytes (got {len(tag)})")

    otk = _chacha20_block(key, 0, nonce)[:32]

    pad16 = lambda b: b + (b"\x00" * ((16 - (len(b) % 16)) % 16))
    mac_data = (
        pad16(aad) +
        pad16(ciphertext) +
        struct.pack("<QQ", len(aad), len(ciphertext))
    )

    computed_tag = _poly1305_mac(mac_data, otk)

    # Constant-time tag comparison to prevent timing attacks
    if not hmac.compare_digest(computed_tag, tag):
        raise AeadIntegrityError(
            BiometricErrorCode.AEAD_INTEGRITY_FAILURE,
            "AEAD authentication tag mismatch: credential payload or header was tampered with"
        )

    return _chacha20_xor(key, 1, nonce, ciphertext)


class ChaCha20Poly1305Cipher:
    """Convenience class providing symmetric AEAD operations for a pre-configured key."""
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError(f"Key must be exactly 32 bytes (got {len(key)})")
        self._key = key

    def encrypt(self, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
        return chacha20_poly1305_encrypt(self._key, nonce, plaintext, aad)

    def decrypt(self, nonce: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"") -> bytes:
        return chacha20_poly1305_decrypt(self._key, nonce, ciphertext, tag, aad)
