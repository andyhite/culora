"""Tests for Device model."""

import pytest

from culora.domain.models.device import Device, DeviceType
from culora.domain.models.memory import Memory


class TestDevice:
    """Test cases for Device model."""

    def test_device_basic_initialization(self) -> None:
        """Test Device basic initialization."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0")
        assert device.device_type == DeviceType.CUDA
        assert device.name == "CUDA:0"
        assert device.is_available is True
        assert device.error_message is None

    def test_device_cpu_post_init(self) -> None:
        """Test Device post-init for CPU device."""
        device = Device(device_type=DeviceType.CPU, name="CPU")
        assert device.memory is not None
        assert device.memory.total_mb is None
        assert device.memory.available_mb is None
        assert not device.memory.is_limited

    def test_device_gpu_post_init(self) -> None:
        """Test Device post-init for GPU device."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0")
        assert device.memory is not None
        assert device.memory.total_mb == 0
        assert device.memory.available_mb == 0
        assert device.memory.is_limited

    def test_device_with_custom_memory(self) -> None:
        """Test Device with custom memory information."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        assert device.memory == memory
        assert device.memory is not None
        assert device.memory.total_mb == 8192
        assert device.memory.available_mb == 4096

    def test_device_unavailable(self) -> None:
        """Test Device marked as unavailable."""
        device = Device(
            device_type=DeviceType.MPS,
            name="MPS",
            is_available=False,
            error_message="Device not found",
        )
        assert device.is_available is False
        assert device.error_message == "Device not found"

    def test_device_has_sufficient_memory_cpu(self) -> None:
        """Test has_sufficient_memory for CPU device."""
        device = Device(device_type=DeviceType.CPU, name="CPU")
        assert device.has_sufficient_memory is True

    def test_device_has_sufficient_memory_mps_available(self) -> None:
        """Test has_sufficient_memory for available MPS device."""
        device = Device(device_type=DeviceType.MPS, name="MPS", is_available=True)
        assert device.has_sufficient_memory is True

    def test_device_has_sufficient_memory_mps_unavailable(self) -> None:
        """Test has_sufficient_memory for unavailable MPS device."""
        device = Device(device_type=DeviceType.MPS, name="MPS", is_available=False)
        assert device.has_sufficient_memory is False

    def test_device_has_sufficient_memory_cuda_sufficient(self) -> None:
        """Test has_sufficient_memory for CUDA device with sufficient memory."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        assert device.has_sufficient_memory is True

    def test_device_has_sufficient_memory_cuda_insufficient(self) -> None:
        """Test has_sufficient_memory for CUDA device with insufficient memory."""
        memory = Memory(total_mb=2048, available_mb=1024)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        assert device.has_sufficient_memory is False

    def test_device_has_sufficient_memory_cuda_no_memory(self) -> None:
        """Test has_sufficient_memory for CUDA device with no memory info."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=None)
        assert device.has_sufficient_memory is False

    def test_device_check_memory_requirement_cpu(self) -> None:
        """Test check_memory_requirement for CPU device."""
        device = Device(device_type=DeviceType.CPU, name="CPU")
        assert device.check_memory_requirement(1000) is True
        assert device.check_memory_requirement(1000000) is True

    def test_device_check_memory_requirement_unavailable(self) -> None:
        """Test check_memory_requirement for unavailable device."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", is_available=False)
        assert device.check_memory_requirement(1000) is False

    def test_device_check_memory_requirement_sufficient(self) -> None:
        """Test check_memory_requirement with sufficient memory."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        assert device.check_memory_requirement(2048) is True
        assert device.check_memory_requirement(4096) is True

    def test_device_check_memory_requirement_insufficient(self) -> None:
        """Test check_memory_requirement with insufficient memory."""
        memory = Memory(total_mb=4096, available_mb=2048)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        assert device.check_memory_requirement(4096) is False
        assert device.check_memory_requirement(8192) is False

    def test_device_check_memory_requirement_no_memory(self) -> None:
        """Test check_memory_requirement with no memory info."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=None)
        assert device.check_memory_requirement(1000) is False

    def test_device_str_available_unlimited(self) -> None:
        """Test string representation for available device with unlimited memory."""
        device = Device(device_type=DeviceType.CPU, name="CPU")
        assert str(device) == "CPU"

    def test_device_str_available_with_memory(self) -> None:
        """Test string representation for available device with memory info."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        expected = "CUDA:0 (4,096/8,192 MB (50.0% used))"
        assert str(device) == expected

    def test_device_str_unavailable(self) -> None:
        """Test string representation for unavailable device."""
        device = Device(
            device_type=DeviceType.MPS,
            name="MPS",
            is_available=False,
            error_message="Device initialization failed",
        )
        expected = "MPS (unavailable: Device initialization failed)"
        assert str(device) == expected

    def test_device_str_unavailable_no_error_message(self) -> None:
        """Test string representation for unavailable device without error message."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0", is_available=False)
        expected = "CUDA:0 (unavailable: None)"
        assert str(device) == expected

    def test_device_equality(self) -> None:
        """Test Device equality comparison."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device1 = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        device2 = Device(device_type=DeviceType.CUDA, name="CUDA:0", memory=memory)
        device3 = Device(device_type=DeviceType.MPS, name="MPS")

        assert device1 == device2
        assert device1 != device3

    def test_device_repr(self) -> None:
        """Test Device string representation."""
        device = Device(device_type=DeviceType.CUDA, name="CUDA:0")
        repr_str = repr(device)
        assert "Device" in repr_str
        assert "device_type" in repr_str
        assert "name" in repr_str

    @pytest.mark.parametrize(
        "device_type,expected_cpu_behavior",
        [
            (DeviceType.CPU, True),
            (DeviceType.CUDA, False),
            (DeviceType.MPS, False),
        ],
    )
    def test_device_cpu_specific_behavior(
        self, device_type: DeviceType, expected_cpu_behavior: bool
    ) -> None:
        """Parametrized test for CPU-specific behavior."""
        device = Device(device_type=device_type, name="Test Device")

        if expected_cpu_behavior:
            assert device.has_sufficient_memory is True
            assert device.check_memory_requirement(1000000) is True
            assert device.memory is not None
            assert device.memory.total_mb is None
            assert device.memory.available_mb is None
        else:
            # Non-CPU devices have different behavior
            if device_type == DeviceType.MPS and device.is_available:
                assert device.has_sufficient_memory is True
            else:
                # CUDA devices need actual memory info
                pass

    @pytest.mark.parametrize(
        "is_available,has_memory,expected_sufficient",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_device_memory_availability_combinations(
        self, is_available: bool, has_memory: bool, expected_sufficient: bool
    ) -> None:
        """Parametrized test for device availability and memory combinations."""
        memory = Memory(total_mb=8192, available_mb=4096) if has_memory else None
        device = Device(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory=memory,
            is_available=is_available,
        )

        if is_available and has_memory:
            assert device.has_sufficient_memory is True
        else:
            assert device.has_sufficient_memory is False

    def test_device_mps_special_behavior(self) -> None:
        """Test MPS device special behavior (assumes sufficient memory when available)."""
        device = Device(device_type=DeviceType.MPS, name="MPS", is_available=True)
        assert device.has_sufficient_memory is True

        # Even with limited memory info, MPS should return True when available
        limited_memory = Memory(total_mb=1024, available_mb=512)  # Less than 2GB
        device_with_memory = Device(
            device_type=DeviceType.MPS,
            name="MPS",
            memory=limited_memory,
            is_available=True,
        )
        assert device_with_memory.has_sufficient_memory is True

    def test_device_cuda_memory_threshold(self) -> None:
        """Test CUDA device memory threshold (2GB minimum)."""
        # Just above threshold
        memory_sufficient = Memory(total_mb=4096, available_mb=2048)
        device_sufficient = Device(
            device_type=DeviceType.CUDA, name="CUDA:0", memory=memory_sufficient
        )
        assert device_sufficient.has_sufficient_memory is True

        # Just below threshold
        memory_insufficient = Memory(total_mb=4096, available_mb=2047)
        device_insufficient = Device(
            device_type=DeviceType.CUDA, name="CUDA:0", memory=memory_insufficient
        )
        assert device_insufficient.has_sufficient_memory is False
