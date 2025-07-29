"""Tests for device manager functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from culora.core.config import CuLoRAConfig
from culora.core.device_info import DeviceInfo
from culora.core.device_manager import DeviceManager
from culora.core.exceptions import DeviceError
from culora.core.logging import CuLoRALogger
from culora.core.types import DeviceType


class TestDeviceManager:
    """Test device manager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = CuLoRAConfig()
        self.logger = CuLoRALogger("test")
        self.manager = DeviceManager(self.config, self.logger)

    @patch("culora.core.device_manager.DeviceDetector.detect_all_devices")
    @patch("culora.core.device_manager.DeviceDetector.get_optimal_device")
    def test_initialize_success(self, mock_optimal: Any, mock_detect: Any) -> None:
        """Test successful device manager initialization."""
        # Mock detected devices
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=8192,
                memory_available=4096,
                is_available=True,
            ),
        ]
        cuda_device = devices[1]

        mock_detect.return_value = devices
        mock_optimal.return_value = cuda_device

        selected_device = self.manager.initialize()

        assert selected_device == cuda_device
        assert self.manager._selected_device == cuda_device
        assert self.manager._all_devices == devices

    @patch("culora.core.device_manager.DeviceDetector.detect_all_devices")
    def test_initialize_failure(self, mock_detect: Any) -> None:
        """Test device manager initialization failure."""
        mock_detect.side_effect = RuntimeError("Detection failed")

        with pytest.raises(DeviceError, match="Failed to initialize device management"):
            self.manager.initialize()

    def test_get_selected_device_not_initialized(self) -> None:
        """Test getting selected device before initialization."""
        with pytest.raises(DeviceError, match="No device selected"):
            self.manager.get_selected_device()

    def test_get_selected_device_success(self) -> None:
        """Test getting selected device after initialization."""
        device = DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True)
        self.manager._selected_device = device

        result = self.manager.get_selected_device()

        assert result == device

    def test_get_all_devices_not_initialized(self) -> None:
        """Test getting all devices before initialization."""
        with pytest.raises(DeviceError, match="Device detection not run"):
            self.manager.get_all_devices()

    def test_get_all_devices_success(self) -> None:
        """Test getting all devices after initialization."""
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True),
        ]
        self.manager._all_devices = devices

        result = self.manager.get_all_devices()

        assert result == devices

    def test_display_device_status_not_initialized(self) -> None:
        """Test displaying device status before initialization."""
        console = Console(file=MagicMock(), width=80)

        self.manager.display_device_status(console)

        # Should print error message (we can't easily test Rich output)
        # This mainly tests that it doesn't crash

    def test_display_device_status_success(self) -> None:
        """Test displaying device status after initialization."""
        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=8192,
                memory_available=4096,
                is_available=True,
            ),
            DeviceInfo(
                DeviceType.MPS, "MPS", is_available=False, error_message="Not supported"
            ),
        ]
        self.manager._all_devices = devices
        self.manager._selected_device = devices[1]  # Select CUDA

        console = Console(file=MagicMock(), width=80)

        # Should not raise exception
        self.manager.display_device_status(console)

    def test_estimate_memory_usage_known_model(self) -> None:
        """Test memory estimation for known models."""
        assert self.manager.estimate_memory_usage("insightface") == 500
        assert self.manager.estimate_memory_usage("clip") == 800
        assert self.manager.estimate_memory_usage("moondream") == 1200

    def test_estimate_memory_usage_unknown_model(self) -> None:
        """Test memory estimation for unknown model."""
        assert self.manager.estimate_memory_usage("unknown_model") is None

    def test_estimate_memory_usage_case_insensitive(self) -> None:
        """Test memory estimation is case insensitive."""
        assert self.manager.estimate_memory_usage("INSIGHTFACE") == 500
        assert self.manager.estimate_memory_usage("ClIp") == 800

    def test_check_memory_availability_no_device(self) -> None:
        """Test memory check when no device selected."""
        assert self.manager.check_memory_availability(1000) is False

    def test_check_memory_availability_cpu(self) -> None:
        """Test memory check for CPU (always sufficient)."""
        cpu_device = DeviceInfo(DeviceType.CPU, "CPU", is_available=True)
        self.manager._selected_device = cpu_device

        assert self.manager.check_memory_availability(10000) is True

    def test_check_memory_availability_gpu_sufficient(self) -> None:
        """Test memory check for GPU with sufficient memory."""
        gpu_device = DeviceInfo(
            DeviceType.CUDA,
            "CUDA:0",
            memory_total=8192,
            memory_available=4096,
            is_available=True,
        )
        self.manager._selected_device = gpu_device

        assert self.manager.check_memory_availability(2000) is True

    def test_check_memory_availability_gpu_insufficient(self) -> None:
        """Test memory check for GPU with insufficient memory."""
        gpu_device = DeviceInfo(
            DeviceType.CUDA,
            "CUDA:0",
            memory_total=8192,
            memory_available=1000,
            is_available=True,
        )
        self.manager._selected_device = gpu_device

        assert self.manager.check_memory_availability(2000) is False

    def test_check_memory_availability_mps_no_memory_info(self) -> None:
        """Test memory check for MPS without memory info (assume sufficient)."""
        mps_device = DeviceInfo(DeviceType.MPS, "MPS", is_available=True)
        self.manager._selected_device = mps_device

        assert self.manager.check_memory_availability(2000) is True

    def test_check_memory_availability_cuda_no_memory_info(self) -> None:
        """Test memory check for CUDA without memory info (assume insufficient)."""
        cuda_device = DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True)
        self.manager._selected_device = cuda_device

        assert self.manager.check_memory_availability(2000) is False

    def test_select_device_preferred_available(self) -> None:
        """Test device selection with preferred device available."""
        # Set preferred device to CUDA
        self.config.device.preferred_device = DeviceType.CUDA

        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=8192,
                memory_available=4096,
                is_available=True,
            ),
        ]
        self.manager._all_devices = devices

        with patch.object(self.manager.detector, "get_optimal_device") as mock_optimal:
            selected = self.manager._select_device()

            # Should select preferred CUDA device, not call optimal
            assert selected.device_type == DeviceType.CUDA
            mock_optimal.assert_not_called()

    def test_select_device_preferred_unavailable(self) -> None:
        """Test device selection when preferred device unavailable."""
        # Set preferred device to CUDA
        self.config.device.preferred_device = DeviceType.CUDA

        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=False),
            DeviceInfo(DeviceType.MPS, "MPS", is_available=True),
        ]
        self.manager._all_devices = devices

        with patch.object(self.manager.detector, "get_optimal_device") as mock_optimal:
            mock_optimal.return_value = devices[2]  # Return MPS

            selected = self.manager._select_device()

            # Should fall back to optimal selection
            assert selected.device_type == DeviceType.MPS
            mock_optimal.assert_called_once()

    def test_select_device_preferred_insufficient_memory(self) -> None:
        """Test device selection when preferred device has insufficient memory."""
        # Set preferred device to CUDA
        self.config.device.preferred_device = DeviceType.CUDA

        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(
                DeviceType.CUDA,
                "CUDA:0",
                memory_total=2048,
                memory_available=1024,
                is_available=True,
            ),  # Insufficient
        ]
        self.manager._all_devices = devices

        with patch.object(self.manager.detector, "get_optimal_device") as mock_optimal:
            mock_optimal.return_value = devices[0]  # Return CPU

            selected = self.manager._select_device()

            # Should fall back to optimal selection
            assert selected.device_type == DeviceType.CPU
            mock_optimal.assert_called_once()

    def test_select_device_cpu_preferred(self) -> None:
        """Test device selection with CPU preferred (skip preference logic)."""
        # Set preferred device to CPU
        self.config.device.preferred_device = DeviceType.CPU

        devices = [
            DeviceInfo(DeviceType.CPU, "CPU", is_available=True),
            DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True),
        ]
        self.manager._all_devices = devices

        with patch.object(self.manager.detector, "get_optimal_device") as mock_optimal:
            mock_optimal.return_value = devices[1]  # Return CUDA (optimal)

            selected = self.manager._select_device()

            # Should use optimal selection, not CPU preference
            assert selected.device_type == DeviceType.CUDA
            mock_optimal.assert_called_once()

    def test_get_execution_provider_no_device(self) -> None:
        """Test execution provider when no device selected."""
        provider = self.manager.get_execution_provider()

        assert provider == "cpu"

    def test_get_execution_provider_cuda(self) -> None:
        """Test execution provider for CUDA device."""
        cuda_device = DeviceInfo(DeviceType.CUDA, "CUDA:0", is_available=True)
        self.manager._selected_device = cuda_device

        provider = self.manager.get_execution_provider()

        assert provider == "cuda"

    def test_get_execution_provider_mps(self) -> None:
        """Test execution provider for MPS device."""
        mps_device = DeviceInfo(DeviceType.MPS, "MPS", is_available=True)
        self.manager._selected_device = mps_device

        provider = self.manager.get_execution_provider()

        assert provider == "mps"

    def test_get_execution_provider_cpu(self) -> None:
        """Test execution provider for CPU device."""
        cpu_device = DeviceInfo(DeviceType.CPU, "CPU", is_available=True)
        self.manager._selected_device = cpu_device

        provider = self.manager.get_execution_provider()

        assert provider == "cpu"
