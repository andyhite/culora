"""Tests for device information models."""

from culora.core.device_info import DeviceInfo
from culora.core.types import DeviceType


class TestDeviceInfo:
    """Test DeviceInfo model functionality."""

    def test_basic_device_info(self) -> None:
        """Test creating basic device info."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA, name="CUDA:0 (GeForce RTX 3080)"
        )

        assert device.device_type == DeviceType.CUDA
        assert device.name == "CUDA:0 (GeForce RTX 3080)"
        assert device.is_available is True
        assert device.memory_total is None
        assert device.memory_available is None
        assert device.error_message is None

    def test_device_with_memory(self) -> None:
        """Test device info with memory information."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory_total=8192,
            memory_available=6144,
        )

        assert device.memory_total == 8192
        assert device.memory_available == 6144

    def test_unavailable_device(self) -> None:
        """Test unavailable device with error message."""
        device = DeviceInfo(
            device_type=DeviceType.MPS,
            name="Apple Silicon MPS",
            is_available=False,
            error_message="MPS not supported on this system",
        )

        assert device.is_available is False
        assert device.error_message == "MPS not supported on this system"

    def test_memory_usage_percent(self) -> None:
        """Test memory usage percentage calculation."""
        # Device with memory info
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory_total=8192,
            memory_available=6144,  # 2048 used = 25%
        )

        assert device.memory_usage_percent == 25.0

    def test_memory_usage_percent_no_memory(self) -> None:
        """Test memory usage percent when no memory info."""
        device = DeviceInfo(device_type=DeviceType.CPU, name="CPU")

        assert device.memory_usage_percent is None

    def test_memory_usage_percent_zero_total(self) -> None:
        """Test memory usage percent with zero total memory."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory_total=0,
            memory_available=0,
        )

        assert device.memory_usage_percent == 100.0

    def test_has_sufficient_memory_cpu(self) -> None:
        """Test sufficient memory check for CPU (always True)."""
        device = DeviceInfo(device_type=DeviceType.CPU, name="CPU")

        assert device.has_sufficient_memory is True

    def test_has_sufficient_memory_gpu_sufficient(self) -> None:
        """Test sufficient memory check for GPU with enough memory."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory_total=8192,
            memory_available=4096,  # 4GB available
        )

        assert device.has_sufficient_memory is True

    def test_has_sufficient_memory_gpu_insufficient(self) -> None:
        """Test sufficient memory check for GPU with insufficient memory."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory_total=4096,
            memory_available=1024,  # Only 1GB available
        )

        assert device.has_sufficient_memory is False

    def test_has_sufficient_memory_unavailable(self) -> None:
        """Test sufficient memory check for unavailable device."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            is_available=False,
            memory_total=8192,
            memory_available=4096,
        )

        assert device.has_sufficient_memory is False

    def test_has_sufficient_memory_cuda_no_memory_info(self) -> None:
        """Test sufficient memory check for CUDA when no memory info available."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA, name="CUDA:0", memory_available=None
        )

        assert device.has_sufficient_memory is False

    def test_has_sufficient_memory_mps_available(self) -> None:
        """Test sufficient memory check for MPS (always True when available)."""
        device = DeviceInfo(
            device_type=DeviceType.MPS, name="Apple Silicon MPS", is_available=True
        )

        assert device.has_sufficient_memory is True

    def test_str_representation_available(self) -> None:
        """Test string representation for available device."""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            name="CUDA:0 (RTX 3080)",
            memory_total=8192,
            memory_available=6144,
        )

        assert str(device) == "CUDA:0 (RTX 3080) (6144/8192 MB)"

    def test_str_representation_no_memory(self) -> None:
        """Test string representation for device without memory info."""
        device = DeviceInfo(device_type=DeviceType.CPU, name="CPU")

        assert str(device) == "CPU"

    def test_str_representation_unavailable(self) -> None:
        """Test string representation for unavailable device."""
        device = DeviceInfo(
            device_type=DeviceType.MPS,
            name="Apple Silicon MPS",
            is_available=False,
            error_message="Not supported",
        )

        assert str(device) == "Apple Silicon MPS (unavailable: Not supported)"
