"""Tests for DeviceService."""

from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from culora.core.exceptions import DeviceError
from culora.domain.models import CuLoRAConfig
from culora.domain.models.device import Device, DeviceType
from culora.domain.models.memory import Memory
from culora.services.device_service import DeviceService

from ...helpers import AssertionHelpers
from ...mocks.mock_torch import MockContext


class TestDeviceService:
    """Test cases for DeviceService."""

    def test_device_service_initialization(self, default_config: CuLoRAConfig) -> None:
        """Test DeviceService initialization."""
        service = DeviceService(default_config)

        assert service.config == default_config
        assert service._selected_device is None
        assert service._all_devices is None
        assert service._torch_available is None

    @patch("culora.services.device_service.DeviceService._detect_all_devices")
    @patch("culora.services.device_service.DeviceService._select_optimal_device")
    def test_initialize_success(
        self, mock_select: Mock, mock_detect: Mock, device_service: DeviceService
    ) -> None:
        """Test successful device initialization."""
        # Mock detected devices
        cpu_device = Device(device_type=DeviceType.CPU, name="CPU")
        cuda_device = Device(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory=Memory(total_mb=8192, available_mb=4096),
        )
        mock_devices = [cpu_device, cuda_device]

        mock_detect.return_value = mock_devices
        mock_select.return_value = cuda_device

        result = device_service.initialize()

        assert result == cuda_device
        assert device_service._selected_device == cuda_device
        assert device_service._all_devices == mock_devices
        mock_detect.assert_called_once()
        mock_select.assert_called_once_with(mock_devices)

    @patch("culora.services.device_service.DeviceService._detect_all_devices")
    def test_initialize_failure(
        self, mock_detect: Mock, device_service: DeviceService
    ) -> None:
        """Test device initialization failure."""
        mock_detect.side_effect = Exception("Detection failed")

        with pytest.raises(DeviceError) as exc_info:
            device_service.initialize()

        assert "Failed to initialize device management" in str(exc_info.value)

    def test_get_selected_device_success(self, device_service: DeviceService) -> None:
        """Test getting selected device when available."""
        device = Device(device_type=DeviceType.CPU, name="CPU")
        device_service._selected_device = device

        result = device_service.get_selected_device()
        assert result == device

    def test_get_selected_device_not_initialized(
        self, device_service: DeviceService
    ) -> None:
        """Test getting selected device when not initialized."""
        with pytest.raises(DeviceError) as exc_info:
            device_service.get_selected_device()

        assert "No device selected" in str(exc_info.value)

    def test_get_all_devices_success(self, device_service: DeviceService) -> None:
        """Test getting all devices when available."""
        devices = [Device(device_type=DeviceType.CPU, name="CPU")]
        device_service._all_devices = devices

        result = device_service.get_all_devices()
        assert result == devices

    def test_get_all_devices_not_detected(self, device_service: DeviceService) -> None:
        """Test getting all devices when not detected."""
        with pytest.raises(DeviceError) as exc_info:
            device_service.get_all_devices()

        assert "Device detection not run" in str(exc_info.value)

    def test_display_device_status_not_detected(
        self, device_service: DeviceService
    ) -> None:
        """Test displaying device status when not detected."""
        console = Mock(spec=Console)

        device_service.display_device_status(console)

        console.print.assert_called_once_with("[red]Device detection not run[/red]")

    @patch("culora.services.device_service.DeviceService._display_device_status")
    def test_display_device_status_success(
        self, mock_display: Mock, device_service: DeviceService
    ) -> None:
        """Test displaying device status when devices are detected."""
        devices = [Device(device_type=DeviceType.CPU, name="CPU")]
        selected_device = devices[0]
        device_service._all_devices = devices
        device_service._selected_device = selected_device

        console = Mock(spec=Console)
        device_service.display_device_status(console)

        mock_display.assert_called_once_with(console, devices, selected_device)

    def test_get_execution_provider_no_device(
        self, device_service: DeviceService
    ) -> None:
        """Test getting execution provider when no device selected."""
        result = device_service.get_execution_provider()
        assert result == "cpu"

    def test_get_execution_provider_with_devices(
        self, device_service: DeviceService
    ) -> None:
        """Test getting execution provider for different device types."""
        # Test CUDA
        cuda_device = Device(device_type=DeviceType.CUDA, name="CUDA:0")
        device_service._selected_device = cuda_device
        assert device_service.get_execution_provider() == "cuda"

        # Test MPS
        mps_device = Device(device_type=DeviceType.MPS, name="MPS")
        device_service._selected_device = mps_device
        assert device_service.get_execution_provider() == "mps"

        # Test CPU
        cpu_device = Device(device_type=DeviceType.CPU, name="CPU")
        device_service._selected_device = cpu_device
        assert device_service.get_execution_provider() == "cpu"

    def test_get_device_summary_not_detected(
        self, device_service: DeviceService
    ) -> None:
        """Test getting device summary when not detected."""
        result = device_service.get_device_summary()
        assert result == {"error": "Device detection not run"}

    @patch("culora.services.device_service.DeviceService._create_device_summary")
    def test_get_device_summary_success(
        self, mock_create: Mock, device_service: DeviceService
    ) -> None:
        """Test getting device summary when devices detected."""
        devices = [Device(device_type=DeviceType.CPU, name="CPU")]
        selected_device = devices[0]
        device_service._all_devices = devices
        device_service._selected_device = selected_device

        expected_summary: dict[str, list[dict[str, str]] | None] = {
            "devices": [],
            "selected_device": None,
        }
        mock_create.return_value = expected_summary

        result = device_service.get_device_summary()

        assert result == expected_summary
        mock_create.assert_called_once_with(devices, selected_device)

    def test_detect_cpu(self, device_service: DeviceService) -> None:
        """Test CPU device detection."""
        cpu_device = device_service._detect_cpu()

        assert cpu_device.device_type == DeviceType.CPU
        assert cpu_device.name == "CPU"
        assert cpu_device.is_available is True
        assert cpu_device.memory is not None
        assert cpu_device.memory.total_mb is None
        assert cpu_device.memory.available_mb is None

    def test_detect_cuda_devices_not_available(
        self, device_service: DeviceService
    ) -> None:
        """Test CUDA device detection when CUDA not available."""
        with MockContext().mock_cuda_devices(count=0):
            device_service._torch_available = True
            devices = device_service._detect_cuda_devices()
            assert devices == []

    def test_detect_cuda_devices_torch_not_available(
        self, device_service: DeviceService
    ) -> None:
        """Test CUDA device detection when torch not available."""
        with MockContext().mock_torch_available(False):
            devices = device_service._detect_cuda_devices()
            assert devices == []

    def test_detect_cuda_devices_success(self, device_service: DeviceService) -> None:
        """Test successful CUDA device detection."""
        with MockContext().mock_cuda_devices(count=2, memory_mb=8192):
            device_service._torch_available = True
            devices = device_service._detect_cuda_devices()

            assert len(devices) == 2
            for i, device in enumerate(devices):
                assert device.device_type == DeviceType.CUDA
                assert device.name == f"CUDA:{i} (GeForce RTX 3080)"
                assert device.is_available is True
                assert device.memory is not None
                assert device.memory.total_mb == 8192
                assert device.memory.available_mb == 6144  # 8192 - 2048

    def test_detect_mps_not_available(self, device_service: DeviceService) -> None:
        """Test MPS device detection when MPS not available."""
        with MockContext().mock_torch_available(True):
            device_service._torch_available = True
            result = device_service._detect_mps()
            assert result is None

    def test_detect_mps_torch_not_available(
        self, device_service: DeviceService
    ) -> None:
        """Test MPS device detection when torch not available."""
        with MockContext().mock_torch_available(False):
            result = device_service._detect_mps()
            assert result is None

    def test_detect_mps_success(self, device_service: DeviceService) -> None:
        """Test successful MPS device detection."""
        mock_torch = Mock()
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device_service._torch_available = True
            device = device_service._detect_mps()

            assert device is not None
            assert device.device_type == DeviceType.MPS
            assert device.name == "Apple Silicon MPS"
            assert device.is_available is True
            assert device.memory is not None
            assert device.memory.total_mb is None
            assert device.memory.available_mb is None

    def test_detect_mps_failure(self, device_service: DeviceService) -> None:
        """Test MPS device detection failure."""
        mock_torch = Mock()
        mock_torch.backends.mps.is_available.side_effect = Exception("MPS error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device_service._torch_available = True
            device = device_service._detect_mps()

            assert device is not None
            assert device.device_type == DeviceType.MPS
            assert device.name == "Apple Silicon MPS"
            assert device.is_available is False
            assert device.error_message is not None
            assert "MPS error" in device.error_message

    def test_torch_available_true(self, device_service: DeviceService) -> None:
        """Test torch availability detection when available."""
        # Reset cached value
        device_service._torch_available = None

        with MockContext().mock_torch_available(True):
            assert device_service.torch_available is True
            assert device_service._torch_available is True

    def test_torch_available_false(self, device_service: DeviceService) -> None:
        """Test torch availability detection when not available."""
        # Reset cached value
        device_service._torch_available = None

        with MockContext().mock_torch_available(False):
            assert device_service.torch_available is False
            assert device_service._torch_available is False

    def test_torch_available_cached(self, device_service: DeviceService) -> None:
        """Test that torch availability is cached."""
        device_service._torch_available = True

        # Should return cached value without trying to import
        assert device_service.torch_available is True

    @patch("culora.services.device_service.DeviceService._detect_all_devices")
    def test_select_optimal_device_preferred_available(
        self, mock_detect: Mock, cuda_config: CuLoRAConfig
    ) -> None:
        """Test device selection with preferred device available."""
        # Create devices
        cpu_device = Device(device_type=DeviceType.CPU, name="CPU")
        cuda_device = Device(
            device_type=DeviceType.CUDA,
            name="CUDA:0",
            memory=Memory(total_mb=8192, available_mb=4096),
            is_available=True,
        )
        mock_devices = [cpu_device, cuda_device]
        mock_detect.return_value = mock_devices

        service = DeviceService(cuda_config)
        result = service._select_optimal_device(mock_devices)

        assert result == cuda_device

    @patch("culora.services.device_service.DeviceService._get_optimal_device")
    def test_select_optimal_device_preferred_not_available(
        self, mock_get_optimal: Mock, cuda_config: CuLoRAConfig
    ) -> None:
        """Test device selection when preferred device not available."""
        # Create devices with CUDA unavailable
        cpu_device = Device(device_type=DeviceType.CPU, name="CPU")
        cuda_device = Device(
            device_type=DeviceType.CUDA, name="CUDA:0", is_available=False
        )
        mock_devices = [cpu_device, cuda_device]
        mock_get_optimal.return_value = cpu_device

        service = DeviceService(cuda_config)
        result = service._select_optimal_device(mock_devices)

        assert result == cpu_device
        mock_get_optimal.assert_called_once()

    def test_create_device_summary(self, device_service: DeviceService) -> None:
        """Test creating device summary."""
        memory = Memory(total_mb=8192, available_mb=4096)
        devices = [
            Device(device_type=DeviceType.CPU, name="CPU"),
            Device(
                device_type=DeviceType.CUDA,
                name="CUDA:0",
                memory=memory,
                is_available=True,
            ),
            Device(
                device_type=DeviceType.MPS,
                name="MPS",
                is_available=False,
                error_message="Not available",
            ),
        ]
        selected_device = devices[1]  # CUDA device

        summary = device_service._create_device_summary(devices, selected_device)

        # Validate summary structure using helper
        AssertionHelpers.assert_device_summary_valid(summary)

        # Check specific values
        assert len(summary["devices"]) == 3
        assert summary["total_devices"] == 3
        assert summary["available_devices"] == 2
        assert summary["selected_device"]["name"] == "CUDA:0"
        assert summary["selected_device"]["type"] == "cuda"

        # Check device details
        cuda_device_info = summary["devices"][1]
        assert cuda_device_info["selected"] is True
        assert cuda_device_info["memory"]["total_mb"] == 8192
        assert cuda_device_info["memory"]["available_mb"] == 4096

    @pytest.mark.parametrize(
        "device_type,expected_provider",
        [
            (DeviceType.CUDA, "cuda"),
            (DeviceType.MPS, "mps"),
            (DeviceType.CPU, "cpu"),
        ],
    )
    def test_execution_provider_mapping(
        self,
        device_type: DeviceType,
        expected_provider: str,
        device_service: DeviceService,
    ) -> None:
        """Parametrized test for execution provider mapping."""
        device = Device(device_type=device_type, name="Test Device")
        device_service._selected_device = device

        assert device_service.get_execution_provider() == expected_provider
