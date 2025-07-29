"""Device-related domain enums."""

from enum import Enum


class DeviceType(str, Enum):
    """Supported device types for AI model execution."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
