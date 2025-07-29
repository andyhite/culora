"""Device detection functionality for CuLoRA.

This module handles detection of available devices (CUDA GPUs, Apple Silicon MPS, CPU)
and gathering device information including memory and availability status.
"""

from .device_info import DeviceInfo
from .types import DeviceType


class DeviceDetector:
    """Detects and analyzes available computing devices."""

    def __init__(self) -> None:
        """Initialize device detector."""
        self._torch_available: bool | None = None

    def detect_all_devices(self) -> list[DeviceInfo]:
        """Detect all available devices.

        Returns:
            List of DeviceInfo objects for all detected devices

        Raises:
            DeviceError: If critical device detection fails
        """
        devices: list[DeviceInfo] = []

        # Always add CPU as fallback
        devices.append(self._detect_cpu())

        # Try to detect CUDA devices
        cuda_devices = self._detect_cuda_devices()
        devices.extend(cuda_devices)

        # Try to detect MPS
        mps_device = self._detect_mps()
        if mps_device:
            devices.append(mps_device)

        return devices

    def get_optimal_device(self) -> DeviceInfo:
        """Get the optimal device for AI model execution.

        Priority: CUDA > MPS > CPU

        Returns:
            DeviceInfo for the best available device
        """
        devices = self.detect_all_devices()

        # Filter to available devices with sufficient memory
        available_devices = [
            d for d in devices if d.is_available and d.has_sufficient_memory
        ]

        if not available_devices:
            # Return CPU as ultimate fallback
            return next(
                (d for d in devices if d.device_type == DeviceType.CPU),
                self._detect_cpu(),
            )

        # Priority order: CUDA, MPS, CPU
        for device_type in [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]:
            for device in available_devices:
                if device.device_type == device_type:
                    return device

        # Should never reach here, but return first available as fallback
        return available_devices[0]

    def _detect_cpu(self) -> DeviceInfo:
        """Detect CPU device (always available)."""
        return DeviceInfo(device_type=DeviceType.CPU, name="CPU", is_available=True)

    def _detect_cuda_devices(self) -> list[DeviceInfo]:
        """Detect CUDA GPU devices."""
        devices: list[DeviceInfo] = []

        if not self.torch_available:
            return devices

        try:
            import torch

            if not torch.cuda.is_available():
                return devices

            device_count = torch.cuda.device_count()

            for i in range(device_count):
                try:
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)

                    # Get memory information
                    torch.cuda.set_device(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory // (
                        1024 * 1024
                    )  # MB
                    memory_free = torch.cuda.memory_reserved(i) // (1024 * 1024)  # MB
                    memory_available = memory_total - memory_free

                    device_info = DeviceInfo(
                        device_type=DeviceType.CUDA,
                        name=f"CUDA:{i} ({props.name})",
                        memory_total=memory_total,
                        memory_available=memory_available,
                        is_available=True,
                    )
                    devices.append(device_info)

                except Exception as e:
                    # Individual device failed, mark as unavailable
                    device_info = DeviceInfo(
                        device_type=DeviceType.CUDA,
                        name=f"CUDA:{i}",
                        is_available=False,
                        error_message=str(e),
                    )
                    devices.append(device_info)

        except Exception:
            # CUDA detection failed entirely, but don't fail - just log
            pass

        return devices

    def _detect_mps(self) -> DeviceInfo | None:
        """Detect Apple Silicon MPS device."""
        if not self.torch_available:
            return None

        try:
            import torch

            if not torch.backends.mps.is_available():
                return None

            # MPS doesn't provide memory information like CUDA
            return DeviceInfo(
                device_type=DeviceType.MPS, name="Apple Silicon MPS", is_available=True
            )

        except Exception as e:
            # MPS detection failed
            return DeviceInfo(
                device_type=DeviceType.MPS,
                name="Apple Silicon MPS",
                is_available=False,
                error_message=str(e),
            )

    @property
    def torch_available(self) -> bool:
        """Check if PyTorch is available for device detection."""
        if self._torch_available is None:
            try:
                __import__("torch")
                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available
