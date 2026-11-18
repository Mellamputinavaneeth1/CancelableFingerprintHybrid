"""
Modular Production Biometric Services Package.

Compliant with ISO/IEC 24745:2022 and ISO/IEC 30136:2018.
"""

from src.services.types import (
    BiometricErrorCode,
    BiometricServiceException,
    InvalidCaptureError,
    InsufficientMinutiaeError,
    TriangulationError,
    AeadIntegrityError,
    RevokedCredentialError,
    MasterSecretAccessViolation,
    ThresholdNotFoundError,
    RawCapture,
    PreprocessedImage,
    ExtractedMinutiae,
    BuiltTriangles,
    BuiltDescriptors,
    ProtectedCredentialDTO,
    SerializedCredentialEnvelope,
    MatchResult,
    AuditEvent
)
from src.services.crypto_aead import (
    chacha20_poly1305_encrypt,
    chacha20_poly1305_decrypt,
    ChaCha20Poly1305Cipher
)
from src.services.capture_adapter import CaptureAdapter
from src.services.preprocessor import Preprocessor
from src.services.minutiae_extractor import MinutiaeExtractor
from src.services.triangle_builder import TriangleBuilder
from src.services.descriptor_builder import DescriptorBuilder
from src.services.scaler_registry import ScalerRegistry
from src.services.key_service import KeyService
from src.services.projection_engine import ProjectionEngine
from src.services.credential_serializer import CredentialSerializer
from src.services.credential_repository import CredentialRepository
from src.services.matcher import Matcher
from src.services.threshold_registry import ThresholdRegistry
from src.services.audit_logger import AuditLogger
from src.services.orchestrator import ProductionBiometricService

__all__ = [
    "BiometricErrorCode",
    "BiometricServiceException",
    "InvalidCaptureError",
    "InsufficientMinutiaeError",
    "TriangulationError",
    "AeadIntegrityError",
    "RevokedCredentialError",
    "MasterSecretAccessViolation",
    "ThresholdNotFoundError",
    "RawCapture",
    "PreprocessedImage",
    "ExtractedMinutiae",
    "BuiltTriangles",
    "BuiltDescriptors",
    "ProtectedCredentialDTO",
    "SerializedCredentialEnvelope",
    "MatchResult",
    "AuditEvent",
    "chacha20_poly1305_encrypt",
    "chacha20_poly1305_decrypt",
    "ChaCha20Poly1305Cipher",
    "CaptureAdapter",
    "Preprocessor",
    "MinutiaeExtractor",
    "TriangleBuilder",
    "DescriptorBuilder",
    "ScalerRegistry",
    "KeyService",
    "ProjectionEngine",
    "CredentialSerializer",
    "CredentialRepository",
    "Matcher",
    "ThresholdRegistry",
    "AuditLogger",
    "ProductionBiometricService"
]
