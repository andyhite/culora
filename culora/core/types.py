"""Core type definitions and enums for CuLoRA.

This module defines common types, enums, and constants used across the application
to ensure type safety and consistency.
"""

import pathlib
from enum import Enum
from typing import Final

from pydantic import BaseModel


class DeviceType(str, Enum):
    """Supported device types for AI model execution."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceCapability(str, Enum):
    """Device capability tiers for performance optimization."""

    HIGH_END = "high_end"
    MID_RANGE = "mid_range"
    LOW_END = "low_end"
    CPU_ONLY = "cpu_only"


class LogLevel(str, Enum):
    """Logging levels for structured logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProcessingStatus(str, Enum):
    """Processing status for images and operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ShotType(str, Enum):
    """Image composition shot types."""

    PORTRAIT = "portrait"
    MEDIUM_SHOT = "medium_shot"
    FULL_BODY = "full_body"
    CLOSE_UP = "close_up"
    UNKNOWN = "unknown"


class SceneType(str, Enum):
    """Scene environment types."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    STUDIO = "studio"
    UNKNOWN = "unknown"


class QualityThreshold(BaseModel):
    """Quality threshold configuration."""

    min_value: float
    max_value: float = 1.0
    enabled: bool = True


# Type aliases for common types
FilePath = str | pathlib.Path
ImageTensor = "torch.Tensor"  # Forward reference to avoid torch import
NumpyArray = "numpy.ndarray"  # Forward reference to avoid numpy import

# Constants
DEFAULT_LOG_LEVEL: Final[LogLevel] = LogLevel.INFO
DEFAULT_DEVICE_TYPE: Final[DeviceType] = DeviceType.CPU
MIN_IMAGE_SIZE: Final[int] = 64
MAX_IMAGE_SIZE: Final[int] = 4096
DEFAULT_BATCH_SIZE: Final[int] = 32
