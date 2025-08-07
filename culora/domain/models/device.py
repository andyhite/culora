"""Device domain model for CuLoRA."""

from dataclasses import dataclass
from enum import Enum

from .memory import Memory


class DeviceType(str, Enum):
    """Supported device types for AI model execution."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class Device:
    """Represents a computing device available for AI model execution.

    Attributes:
        device_type: The type of device (CUDA, MPS, CPU)
        name: Human-readable device name
        memory: Memory information for the device
        is_available: Whether the device is available for use
        error_message: Error message if device detection failed
    """

    device_type: DeviceType
    name: str
    memory: Memory | None = None
    is_available: bool = True
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Initialize default memory for CPU devices."""
        if self.memory is None:
            if self.device_type == DeviceType.CPU:
                # CPU has unlimited system memory
                self.memory = Memory(total_mb=None, available_mb=None)
            else:
                # Other devices should have memory info provided
                self.memory = Memory(total_mb=0, available_mb=0)

    @property
    def has_sufficient_memory(self) -> bool:
        """Check if device has sufficient memory for AI models.

        Returns True for CPU (unlimited) or devices with >2GB available.
        For MPS, assumes sufficient memory since we can't query it directly.
        """
        if self.device_type == DeviceType.CPU:
            return True

        if not self.is_available:
            return False

        # For MPS, assume sufficient memory if available (can't query directly)
        if self.device_type == DeviceType.MPS:
            return True

        # For other devices, check actual memory (require at least 2GB)
        if self.memory is None:
            return False

        return self.memory.has_sufficient_memory(2048)  # 2GB minimum

    def check_memory_requirement(self, required_mb: int) -> bool:
        """Check if device can meet specific memory requirement.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if device can meet the requirement
        """
        if not self.is_available:
            return False

        if self.device_type == DeviceType.CPU:
            return True  # CPU uses system memory

        if self.memory is None:
            return False

        return self.memory.has_sufficient_memory(required_mb)

    def __str__(self) -> str:
        """Human-readable device description."""
        if not self.is_available:
            return f"{self.name} (unavailable: {self.error_message})"

        if self.memory is not None and self.memory.is_limited:
            return f"{self.name} ({self.memory})"

        return self.name
