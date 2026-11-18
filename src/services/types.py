"""
Data Transfer Objects (DTOs), Strict Schemas, and Standard Error Codes for
Cancelable Fingerprint Production Services.

Compliant with ISO/IEC 24745:2022 (Biometric Information Protection) and
ISO/IEC 30136:2018 (Performance Evaluation of Biometric Template Protection).
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class BiometricErrorCode(IntEnum):
    """Standardized biometric error codes across service trust boundaries."""
    SUCCESS = 0
    
    # 1000s: Capture & Preprocessing
    INVALID_IMAGE = 1001
    IMAGE_TOO_SMALL = 1002
    INVALID_BIT_DEPTH = 1003
    SEGMENTATION_FAILED = 1004
    
    # 2000s: Minutiae & Triangulation
    INSUFFICIENT_MINUTIAE = 2001
    TRIANGULATION_FAILED = 2002
    DEGENERATE_TRIANGLE_SET = 2003
    
    # 3000s: Lifecycle & Revocation
    CREDENTIAL_REVOKED = 3001
    CREDENTIAL_EXPIRED = 3002
    CREDENTIAL_NOT_FOUND = 3003
    
    # 4000s: Cryptography & Serialization
    AEAD_INTEGRITY_FAILURE = 4001
    DECRYPTION_FAILED = 4002
    SERIALIZATION_ERROR = 4003
    INVALID_MAGIC_BYTES = 4004
    UNSUPPORTED_VERSION = 4005
    INVALID_TAG = 4006
    
    # 5000s: Key Management (KMS / HSM)
    KEY_DERIVATION_FAILED = 5001
    MASTER_SECRET_ACCESS_DENIED = 5002
    KMS_UNAVAILABLE = 5003
    
    # 6000s: Thresholds & Decisions
    THRESHOLD_NOT_FOUND = 6001
    INVALID_OPERATING_POINT = 6002
    
    # 7000s: Matching & Consensus
    MATCHING_ERROR = 7001
    DIMENSION_MISMATCH = 7002


class BiometricServiceException(Exception):
    """Base exception for all biometric service errors."""
    def __init__(self, code: BiometricErrorCode, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code.name}:{int(code)}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


class InvalidCaptureError(BiometricServiceException):
    """Raised when an ingested capture is malformed, corrupt, or out of spec."""
    pass


class InsufficientMinutiaeError(BiometricServiceException):
    """Raised when minutiae yield is below the minimum operational threshold (<3)."""
    pass


class TriangulationError(BiometricServiceException):
    """Raised when Delaunay triangulation cannot be constructed."""
    pass


class AeadIntegrityError(BiometricServiceException):
    """Raised when an encrypted credential payload fails AEAD authentication tag validation."""
    pass


class RevokedCredentialError(BiometricServiceException):
    """Raised when an authentication attempt is made against a revoked credential."""
    pass


class MasterSecretAccessViolation(BiometricServiceException):
    """Raised when client code attempts to bypass KMS trust boundary and access master secret."""
    pass


class ThresholdNotFoundError(BiometricServiceException):
    """Raised when requested sensor/database operating point is not calibrated."""
    pass


@dataclass
class RawCapture:
    """Validated raw biometric image container."""
    image: np.ndarray
    dpi: float
    width: int
    height: int
    sensor_id: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedImage:
    """Output of foreground segmentation, orientation field, and Gabor enhancement."""
    segmented: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    enhanced: np.ndarray
    orientations: np.ndarray
    coherence: np.ndarray


@dataclass
class ExtractedMinutiae:
    """Output of minutiae ridge skeleton analysis."""
    minutiae: List[Any]  # List[Minutia]
    skeleton: np.ndarray
    count: int


@dataclass
class BuiltTriangles:
    """Output of Delaunay or k-NN graph construction."""
    triangles: List[Tuple[Any, Any, Any]]
    count: int
    adjacency_matrix: Optional[np.ndarray] = None


@dataclass
class BuiltDescriptors:
    """Extracted 9-D invariant descriptors and directional quality weights."""
    descriptors: np.ndarray  # Shape: (N, 9) float64
    weights: np.ndarray      # Shape: (N,) float64
    in_dim: int = 9
    count: int = 0

    def __post_init__(self):
        self.count = len(self.descriptors)


@dataclass
class ProtectedCredentialDTO:
    """
    In-memory representation of protected biometric credential.
    Contains quantized bitstrings, float16 weights, optional 2-bit packed topology,
    and optional keyed RFF edge tokens (zero plaintext coordinates).
    """
    credential_id: str
    subject_id: str
    salt: str
    app_id: str
    bitstrings: np.ndarray  # Shape: (N, out_dim) uint8 or (N,) uint32
    weights: np.ndarray     # Shape: (N,) float16 or float64
    num_triangles: int
    packed_topology: Optional[bytes] = None
    version: str = "1.0"
    created_at_utc: Optional[str] = None
    edge_tokens: Optional[np.ndarray] = None  # Shape: (N, N, w64) uint64
    in_dim: int = 9
    edge_mask: Optional[np.ndarray] = None  # Shape: (N, N) bool for sparse edge subsets


@dataclass
class SerializedCredentialEnvelope:
    """Binary framed and AEAD-encrypted credential at rest."""
    raw_bytes: bytes
    total_bytes: int
    credential_id: str
    num_triangles: int
    has_topology: bool
    cipher_suite: str
    nonce: bytes
    auth_tag: bytes
    is_encrypted: bool
    has_edge_tokens: bool = False
    has_sparse_edges: bool = False


@dataclass
class MatchResult:
    """Standardized verification response across all matching strategies."""
    decision: str  # "ACCEPT" or "REJECT"
    score: float
    threshold: float
    operating_point: str
    algorithm: str
    matched_triangles: int
    total_query_triangles: int
    total_enrolled_triangles: int
    error_code: BiometricErrorCode = BiometricErrorCode.SUCCESS
    latency_ms: float = 0.0


@dataclass
class AuditEvent:
    """
    Privacy-minimised audit event conforming to ISO/IEC 24745:2022.
    Strictly forbids raw image arrays, coordinates, continuous descriptors, or secret keys.
    """
    event_id: str
    timestamp_utc: str
    operation: str  # "ENROLL", "VERIFY", "REVOKE", "ROTATE_KEY"
    subject_pseudonym: str  # SHA-256 hash or pseudonym of subject ID
    decision: str
    score: Optional[float]
    latency_ms: float
    error_code: int
    integrity_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
