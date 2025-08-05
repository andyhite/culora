"""Domain models for CuLoRA.

Business entity models and data structures.
"""

from .composition import (
    BatchCompositionResult,
    CompositionAnalysis,
    CompositionResult,
)
from .config import (
    CompositionConfig,
    CuLoRAConfig,
    DeviceConfig,
    FaceAnalysisConfig,
    ImageConfig,
    LoggingConfig,
    QualityConfig,
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
from .quality import (
    BatchQualityResult,
    ImageQualityResult,
    QualityScore,
    TechnicalQualityMetrics,
)

__all__ = [
    "BatchCompositionResult",
    "BatchFaceAnalysisResult",
    "BatchLoadResult",
    "BatchQualityResult",
    "CompositionAnalysis",
    "CompositionConfig",
    "CompositionResult",
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
    "ImageQualityResult",
    "LoggingConfig",
    "Memory",
    "QualityConfig",
    "QualityScore",
    "ReferenceEmbedding",
    "ReferenceImage",
    "ReferenceMatchResult",
    "ReferenceProcessingResult",
    "ReferenceSet",
    "SimilarityMatch",
    "TechnicalQualityMetrics",
]
