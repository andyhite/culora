"""Domain models for CuLoRA.

Business entity models and data structures.
"""

from .config import (
    CuLoRAConfig,
    DeviceConfig,
    FaceAnalysisConfig,
    ImageConfig,
    LoggingConfig,
)
from .device import Device
from .face import BatchFaceAnalysisResult, FaceAnalysisResult, FaceDetection
from .face_reference import (
    ReferenceEmbedding,
    ReferenceImage,
    ReferenceMatchResult,
    ReferenceProcessingResult,
    ReferenceSet,
    SimilarityMatch,
)
from .image import BatchLoadResult, DirectoryScanResult, ImageLoadResult, ImageMetadata
from .memory import Memory

__all__ = [
    "BatchFaceAnalysisResult",
    "BatchLoadResult",
    "CuLoRAConfig",
    "Device",
    "DeviceConfig",
    "DirectoryScanResult",
    "FaceAnalysisConfig",
    "FaceAnalysisResult",
    "FaceDetection",
    "ImageConfig",
    "ImageLoadResult",
    "ImageMetadata",
    "LoggingConfig",
    "Memory",
    "ReferenceEmbedding",
    "ReferenceImage",
    "ReferenceMatchResult",
    "ReferenceProcessingResult",
    "ReferenceSet",
    "SimilarityMatch",
]
