"""Domain layer for CuLoRA.

Business entities and models with minimal dependencies.
Only depends on the core layer.
"""

from .enums import DeviceType
from .models import (
    BatchFaceAnalysisResult,
    BatchLoadResult,
    CuLoRAConfig,
    Device,
    DirectoryScanResult,
    FaceAnalysisConfig,
    FaceAnalysisResult,
    FaceDetection,
    ImageConfig,
    ImageLoadResult,
    ImageMetadata,
    Memory,
)

__all__ = [
    "BatchFaceAnalysisResult",
    "BatchLoadResult",
    "CuLoRAConfig",
    "Device",
    "DeviceType",
    "DirectoryScanResult",
    "FaceAnalysisConfig",
    "FaceAnalysisResult",
    "FaceDetection",
    "ImageConfig",
    "ImageLoadResult",
    "ImageMetadata",
    "Memory",
]
