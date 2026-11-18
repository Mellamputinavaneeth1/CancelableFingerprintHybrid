import hmac
import hashlib
import math
from typing import Optional, Union
import numpy as np
from src.protection.kdf import derive_application_key
from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD

def crypto_gaussian_matrix(key_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    """
    Cryptographic counter-mode PRF Gaussian matrix generator.
    
    Generates deterministic standard Gaussian samples N(0, 1) directly from
    cryptographic key bytes using a custom HMAC-SHA256 counter-mode PRF
    combined with the Box-Muller transform:
        z1 = sqrt(-2 ln u1) cos(2 pi u2)
        z2 = sqrt(-2 ln u1) sin(2 pi u2)
        
    Design & Mathematical Guarantees:
    ---------------------------------
    1. Generator Classification:
       This is a stateless, deterministic counter-mode PRF using HMAC-SHA256 (RFC 2104).
       It is intentionally NOT a full stateful NIST SP 800-90A HMAC_DRBG (which requires
       internal Key/V reseed cycles and prediction-resistance state tracking).
       
    2. Numerical Range & Singularity Prevention:
       Each 32-bit big-endian integer v in [0, 2^32 - 1] is mapped to:
           u = (v + 0.5) / 2^32
       This guarantees:
           u_min = 0.5 / 2^32 ≈ 1.164 x 10^-10 > 0
           u_max = (2^32 - 0.5) / 2^32 ≈ 1 - 1.164 x 10^-10 < 1
       Hence u is strictly inside the open interval (0, 1), mathematically preventing
       log(0) singularities and sqrt(negative) values.
       
    3. Counter Encoding & Domain Separation:
       Each HMAC block inputs:
       b"GaussianPRF|v1.0|rows=" + rows.to_bytes(2, 'big') + b"|cols=" + cols.to_bytes(2, 'big') + b"|" + counter.to_bytes(4, 'big').
       Domain separation string with explicit versioning and matrix dimensions ensures
       outputs are cryptographically distinct from other HKDF or HMAC applications
       and cannot be transposed across different dimension requests.
    """
    n_needed = rows * cols
    if n_needed % 2 != 0:
        n_needed += 1
    gaussians = []
    counter = 0
    buffer = bytearray()
    prefix = b"GaussianPRF|v1.0|rows=" + rows.to_bytes(2, 'big') + b"|cols=" + cols.to_bytes(2, 'big') + b"|"
    while len(gaussians) < n_needed:
        if len(buffer) < 8:
            block = hmac.new(key_bytes, prefix + counter.to_bytes(4, 'big'), hashlib.sha256).digest()
            buffer.extend(block)
            counter += 1
        u1_int = int.from_bytes(buffer[:4], 'big')
        u2_int = int.from_bytes(buffer[4:8], 'big')
        del buffer[:8]
        u1 = (u1_int + 0.5) / 4294967296.0
        u2 = (u2_int + 0.5) / 4294967296.0
        mag = math.sqrt(-2.0 * math.log(u1))
        z1 = mag * math.cos(2.0 * math.pi * u2)
        z2 = mag * math.sin(2.0 * math.pi * u2)
        gaussians.extend([z1, z2])
    A = np.array(gaussians[:rows * cols], dtype=np.float32).reshape(rows, cols)
    return A

def derive_orthonormal_matrix(master_secret: Union[bytes, str], salt: str, app_id: str, 
                              in_dim: int = 12, out_dim: int = 32,
                              subject_id: str = "",
                              purpose: str = "template_projection",
                              version: str = "1.0") -> np.ndarray:
    """
    Derives a column-orthonormal embedding matrix W of shape (out_dim, in_dim) 
    using HKDF-derived cryptographic PRF Box-Muller Gaussian generation followed by QR decomposition.
    
    Cryptographic Derivation Chain:
        K_master -> HKDF-Extract -> PRK -> HKDF-Expand -> K_app -> HMAC-SHA256 Counter PRF / Box-Muller -> A -> QR -> W
        
    Mathematical Dimensionality & Orthonormality:
    --------------------------------------------
    When out_dim >= in_dim (e.g., 32 >= 9 or 32 >= 12):
    - W has shape (out_dim, in_dim), i.e., (32, 9) or (32, 12).
    - W is strictly COLUMN-ORTHONORMAL (the in_dim columns in R^32 are mutually orthonormal).
    - Gram Matrix:
          W.T @ W = I_{in_dim}
      with numerical tolerance ||W.T @ W - I_{in_dim}||_max < 1e-6.
    - Note: Because out_dim > in_dim, W is NOT row-orthonormal (rank <= in_dim < out_dim).
        
    Sign Stabilization:
        Q' = Q * diag(sign(diag(R))) removes column-sign ambiguity from the QR decomposition.
        
    Quantization Definition:
        Continuous projection: p = W @ z = (z @ W.T).
        Binary bitstrings are obtained via the explicit binary indicator:
            b_j = I(p_j >= 0) = { 1 if p_j >= 0, else 0 }
        
    Args:
        master_secret: 32-byte master cryptographic secret.
        salt: Application or enrollment salt string.
        app_id: Application identifier.
        in_dim: Input descriptor dimension (default: 12, primary candidate: 9).
        out_dim: Projection dimension (default: 32).
        subject_id: Optional subject identifier for context binding.
        purpose: Domain separation identifier.
        version: Version string.
        
    Returns:
        W: np.ndarray of shape (out_dim, in_dim), float32, with orthonormal columns (W.T @ W ≈ I).
    """
    key_bytes = derive_application_key(master_secret, salt, app_id, subject_id=subject_id, 
                                       purpose=purpose, version=version, length=32)
    
    if out_dim >= in_dim:
        A = crypto_gaussian_matrix(key_bytes, out_dim, in_dim)
        Q, R = np.linalg.qr(A)
        r_diag_signs = np.sign(np.diag(R))
        r_diag_signs[r_diag_signs == 0] = 1.0
        Q = Q * r_diag_signs  # Canonical positive diagonal convention
        W = Q[:, :in_dim]
    else:
        A = crypto_gaussian_matrix(key_bytes, in_dim, out_dim)
        Q, R = np.linalg.qr(A)
        r_diag_signs = np.sign(np.diag(R))
        r_diag_signs[r_diag_signs == 0] = 1.0
        Q = Q * r_diag_signs
        W = Q.T[:out_dim, :]
        
    return W.astype(np.float32)

def project_descriptors(descriptors: np.ndarray, W: np.ndarray, 
                        mean: Optional[np.ndarray] = DEFAULT_DESCRIPTOR_MEAN,
                        std: Optional[np.ndarray] = DEFAULT_DESCRIPTOR_STD) -> np.ndarray:
    """
    Applies the keyed orthonormal embedding W to dev-set z-score standardized descriptors.
    
    Standardization:
        z = (d - mu_dev) / (sigma_dev + eps)
        proj = z @ W.T
    """
    if descriptors.size == 0:
        return np.empty((0, W.shape[0]), dtype=np.float32)
        
    d_dim = descriptors.shape[1]

    if mean is None or (mean is DEFAULT_DESCRIPTOR_MEAN and d_dim not in (9, 12)):
        m = None
    elif len(mean) == d_dim:
        m = mean
    elif len(mean) > d_dim:
        m = mean[:d_dim]
    else:
        raise ValueError(f"Mean dimension {len(mean)} incompatible with descriptor dimension {d_dim}")

    if std is None or (std is DEFAULT_DESCRIPTOR_STD and d_dim not in (9, 12)):
        s = None
    elif len(std) == d_dim:
        s = std
    elif len(std) > d_dim:
        s = std[:d_dim]
    else:
        raise ValueError(f"Std dimension {len(std)} incompatible with descriptor dimension {d_dim}")
    
    if m is not None and s is not None:
        standardized = (descriptors - m) / (s + 1e-8)
    elif m is not None:
        standardized = descriptors - m
    else:
        standardized = descriptors
        
    return np.matmul(standardized, W.T).astype(np.float32)
