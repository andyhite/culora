"""Tests for device detection functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

from culora.core.device_detector import DeviceDetector
from culora.core.device_info import DeviceInfo
from culora.core.types import DeviceType


class TestDeviceDetector:
    """Test device detection functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = DeviceDetector()

    def test_detect_cpu_always_available(self) -> None:
        """Test that CPU is always detected as available."""
        cpu_device = self.detector._detect_cpu()

        assert cpu_device.device_type == DeviceType.CPU
        assert cpu_device.name == "CPU"
        assert cpu_device.is_available is True
        assert cpu_device.memory_total is None
        assert cpu_device.memory_available is None

    @patch("torch.cuda.is_available", return_value=False)
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_cuda_not_available(self, mock_cuda_available: Any) -> None:
        """Test CUDA detection when CUDA is not available."""
        devices = self.detector._detect_cuda_devices()

        assert len(devices) == 0

    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024)  # 1GB
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_cuda_single_device(self, *mocks: Any) -> None:
        """Test CUDA detection with single device."""
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "GeForce RTX 3080"
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB

        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            devices = self.detector._detect_cuda_devices()

        assert len(devices) == 1
        device = devices[0]
        assert device.device_type == DeviceType.CUDA
        assert device.name == "CUDA:0 (GeForce RTX 3080)"
        assert device.is_available is True
        assert device.memory_total == 8192  # 8GB in MB
        assert device.memory_available == 7168  # 8GB - 1GB reserved

    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_cuda_multiple_devices(self, *mocks: Any) -> None:
        """Test CUDA detection with multiple devices."""

        # Mock device properties for two devices
        def mock_get_props(device_id: int) -> Any:
            mock_props = MagicMock()
            if device_id == 0:
                mock_props.name = "GeForce RTX 3080"
                mock_props.total_memory = 8 * 1024 * 1024 * 1024
            else:
                mock_props.name = "GeForce RTX 3070"
                mock_props.total_memory = 6 * 1024 * 1024 * 1024
            return mock_props

        with patch("torch.cuda.get_device_properties", side_effect=mock_get_props):
            devices = self.detector._detect_cuda_devices()

        assert len(devices) == 2
        assert devices[0].name == "CUDA:0 (GeForce RTX 3080)"
        assert devices[1].name == "CUDA:1 (GeForce RTX 3070)"

    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device", side_effect=RuntimeError("Device error"))
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_cuda_device_error(self, *mocks: Any) -> None:
        """Test CUDA detection when individual device fails."""
        devices = self.detector._detect_cuda_devices()

        assert len(devices) == 1
        device = devices[0]
        assert device.device_type == DeviceType.CUDA
        assert device.name == "CUDA:0"
        assert device.is_available is False
        assert device.error_message is not None

    @patch("culora.core.device_detector.DeviceDetector.torch_available", False)
    def test_detect_cuda_no_torch(self) -> None:
        """Test CUDA detection when PyTorch not available."""
        devices = self.detector._detect_cuda_devices()

        assert len(devices) == 0

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_mps_available(self, mock_mps_available: Any) -> None:
        """Test MPS detection when available."""
        device = self.detector._detect_mps()

        assert device is not None
        assert device.device_type == DeviceType.MPS
        assert device.name == "Apple Silicon MPS"
        assert device.is_available is True

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_mps_not_available(self, mock_mps_available: Any) -> None:
        """Test MPS detection when not available."""
        device = self.detector._detect_mps()

        assert device is None

    @patch("torch.backends.mps.is_available", side_effect=RuntimeError("MPS error"))
    @patch("culora.core.device_detector.DeviceDetector.torch_available", True)
    def test_detect_mps_error(self, mock_mps_available: Any) -> None:
        """Test MPS detection when error occurs."""
        device = self.detector._detect_mps()

        assert device is not None
        assert device.device_type == DeviceType.MPS
        assert device.is_available is False
        assert device.error_message is not None and "MPS error" in device.error_message

    @patch("culora.core.device_detector.DeviceDetector.torch_available", False)
    def test_detect_mps_no_torch(self) -> None:
        """Test MPS detection when PyTorch not available."""
        device = self.detector._detect_mps()

        assert device is None

    @patch("culora.core.device_detector.DeviceDetector._detect_cuda_devices")
    @patch("culora.core.device_detector.DeviceDetector._detect_mps")
    def test_detect_all_devices(self, mock_mps: Any, mock_cuda: Any) -> None:
        """Test detecting all devices."""
        # Mock return values
        cuda_device = DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True)
        mps_device = DeviceInfo(DeviceType.MPS, "MPS", is_available=True)

        mock_cuda.return_value = [cuda_device]
        mock_mps.return_value = mps_device

        devices = self.detector.detect_all_devices()

        # Should include CPU, CUDA, and MPS
        assert len(devices) == 3
        device_types = [d.device_type for d in devices]
        assert DeviceType.CPU in device_types
        assert DeviceType.CUDA in device_types
        assert DeviceType.MPS in device_types

    @patch("culora.core.device_detector.DeviceDetector.detect_all_devices")
    def test_get_optimal_device_cuda_preferred(self, mock_detect: Any) -> None:
        """Test optimal device selection with CUDA available."""
        # Mock devices: CPU, CUDA, MPS all available
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=8192,
                memory_available=4096,
                is_available=True,
            ),
            DeviceInfo(DeviceType.MPS, "MPS", is_available=True),
        ]
        mock_detect.return_value = devices

        optimal = self.detector.get_optimal_device()

        assert optimal.device_type == DeviceType.CUDA

    @patch("culora.core.device_detector.DeviceDetector.detect_all_devices")
    def test_get_optimal_device_mps_fallback(self, mock_detect: Any) -> None:
        """Test optimal device selection with MPS when CUDA unavailable."""
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=False),
            DeviceInfo(
                DeviceType.MPS,
                "MPS",
                is_available=True,
                memory_total=None,
                memory_available=None,
            ),
        ]
        mock_detect.return_value = devices

        optimal = self.detector.get_optimal_device()

        assert optimal.device_type == DeviceType.MPS

    @patch("culora.core.device_detector.DeviceDetector.detect_all_devices")
    def test_get_optimal_device_cpu_fallback(self, mock_detect: Any) -> None:
        """Test optimal device selection falls back to CPU."""
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=False),
            DeviceInfo(DeviceType.MPS, "MPS", is_available=False),
        ]
        mock_detect.return_value = devices

        optimal = self.detector.get_optimal_device()

        assert optimal.device_type == DeviceType.CPU

    @patch("culora.core.device_detector.DeviceDetector.detect_all_devices")
    def test_get_optimal_device_insufficient_memory(self, mock_detect: Any) -> None:
        """Test optimal device selection skips devices with insufficient memory."""
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=2048,
                memory_available=1024,
                is_available=True,
            ),  # Only 1GB available
        ]
        mock_detect.return_value = devices

        optimal = self.detector.get_optimal_device()

        # Should skip CUDA due to insufficient memory, use CPU
        assert optimal.device_type == DeviceType.CPU

    def test_torch_available_property(self) -> None:
        """Test torch_available property caching."""
        # Create a new detector to avoid interference from other tests
        detector = DeviceDetector()
        detector._torch_available = None

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'torch'")
        ):
            # First call should cache the result
            assert detector.torch_available is False

            # Second call should use cached result
            assert detector.torch_available is False
