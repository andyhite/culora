"""Device information models for CuLoRA.

This module defines data structures for representing device information,
including availability, memory, and error states.
"""

from dataclasses import dataclass

from .types import DeviceType


@dataclass
class DeviceInfo:
    """Information about a detected device.

    Attributes:
        device_type: The type of device (CUDA, MPS, CPU)
        name: Human-readable device name
        memory_total: Total device memory in MB (None for CPU)
        memory_available: Available device memory in MB (None for CPU)
        is_available: Whether the device is available for use
        error_message: Error message if device detection failed
    """

    device_type: DeviceType
    name: str
    memory_total: int | None = None
    memory_available: int | None = None
    is_available: bool = True
    error_message: str | None = None

    @property
    def memory_usage_percent(self) -> float | None:
        """Calculate memory usage percentage."""
        if self.memory_total is None or self.memory_available is None:
            return None

        if self.memory_total == 0:
            return 100.0

        used = self.memory_total - self.memory_available
        return (used / self.memory_total) * 100.0

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

        # For CUDA, check actual memory
        if self.memory_available is None:
            return False

        # Require at least 2GB for AI models
        return self.memory_available >= 2048

    def __str__(self) -> str:
        """Human-readable device description."""
        if not self.is_available:
            return f"{self.name} (unavailable: {self.error_message})"

        if self.memory_total is not None:
            return f"{self.name} ({self.memory_available}/{self.memory_total} MB)"

        return self.name
