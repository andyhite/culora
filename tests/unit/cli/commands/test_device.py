"""Tests for CLI device commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from culora.cli.commands.device import device_app
from culora.core import DeviceError
from culora.domain import Device, DeviceType, Memory


class TestDeviceCommands:
    """Test device CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_devices(self) -> list[Device]:
        """Create mock devices for testing."""
        return [
            Device(
                device_type=DeviceType.CPU,
                name="CPU",
                memory=Memory(total_mb=None, available_mb=None),
                is_available=True,
            ),
            Device(
                device_type=DeviceType.CUDA,
                name="NVIDIA GeForce RTX 3080",
                memory=Memory(total_mb=10240, available_mb=8192),
                is_available=True,
            ),
            Device(
                device_type=DeviceType.MPS,
                name="Apple Silicon MPS",
                memory=Memory(total_mb=None, available_mb=None),
                is_available=True,
            ),
        ]

    def test_device_info_success(
        self, runner: CliRunner, mock_devices: list[Device]
    ) -> None:
        """Test successful device info command."""
        selected_device = mock_devices[1]  # CUDA device
        available_devices = [d for d in mock_devices if d.is_available]

        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.return_value = mock_devices
            mock_service.get_selected_device.return_value = selected_device
            mock_get_service.return_value = mock_service

            with patch(
                "culora.cli.commands.device.display_device_table"
            ) as mock_display:
                result = runner.invoke(device_app, ["info"])

                assert result.exit_code == 0
                mock_service.get_all_devices.assert_called_once()
                mock_service.get_selected_device.assert_called_once()
                mock_display.assert_called_once_with(available_devices)
                assert "Selected device:" in result.stdout

    def test_device_info_no_available_devices(self, runner: CliRunner) -> None:
        """Test device info command with no available devices."""
        unavailable_devices = [
            Device(
                device_type=DeviceType.CUDA,
                name="Unavailable GPU",
                memory=None,
                is_available=False,
                error_message="Driver not found",
            )
        ]

        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.return_value = unavailable_devices
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["info"])

            assert result.exit_code == 0
            assert "No available devices detected" in result.stdout

    def test_device_info_device_error(self, runner: CliRunner) -> None:
        """Test device info command with device detection error."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.side_effect = DeviceError(
                "Detection failed", "TEST_ERROR"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["info"])

            assert result.exit_code == 1
            assert "Device detection error" in result.stdout

    def test_device_info_unexpected_error(self, runner: CliRunner) -> None:
        """Test device info command with unexpected error."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.side_effect = RuntimeError("Unexpected error")
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["info"])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout

    def test_device_list_success(
        self, runner: CliRunner, mock_devices: list[Device]
    ) -> None:
        """Test successful device list command."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.return_value = mock_devices
            mock_get_service.return_value = mock_service

            with patch(
                "culora.cli.commands.device.display_device_table"
            ) as mock_display:
                result = runner.invoke(device_app, ["list"])

                assert result.exit_code == 0
                mock_service.get_all_devices.assert_called_once()
                mock_display.assert_called_once_with(mock_devices)
                assert "Available devices: 3/3" in result.stdout

    def test_device_list_mixed_availability(self, runner: CliRunner) -> None:
        """Test device list command with mixed device availability."""
        mixed_devices = [
            Device(
                device_type=DeviceType.CPU,
                name="CPU",
                memory=Memory(total_mb=None, available_mb=None),
                is_available=True,
            ),
            Device(
                device_type=DeviceType.CUDA,
                name="Unavailable GPU",
                memory=None,
                is_available=False,
                error_message="Driver error",
            ),
        ]

        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.return_value = mixed_devices
            mock_get_service.return_value = mock_service

            with patch("culora.cli.commands.device.display_device_table"):
                result = runner.invoke(device_app, ["list"])

                assert result.exit_code == 0
                assert "Available devices: 1/2" in result.stdout

    def test_device_list_no_devices(self, runner: CliRunner) -> None:
        """Test device list command with no devices found."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.return_value = []
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["list"])

            assert result.exit_code == 0
            assert "No devices found" in result.stdout

    def test_device_list_device_error(self, runner: CliRunner) -> None:
        """Test device list command with device detection error."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_all_devices.side_effect = DeviceError(
                "Detection failed", "TEST_ERROR"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["list"])

            assert result.exit_code == 1
            assert "Device detection error" in result.stdout

    def test_device_memory_success(
        self, runner: CliRunner, mock_devices: list[Device]
    ) -> None:
        """Test successful device memory command."""
        selected_device = mock_devices[1]  # CUDA device with memory info

        with (
            patch("culora.cli.commands.device.get_device_service") as mock_get_service,
            patch("culora.cli.commands.device.get_memory_service") as mock_get_memory,
            patch("culora.cli.commands.device.display_memory_table") as mock_display,
        ):
            mock_device_service = MagicMock()
            mock_device_service.get_selected_device.return_value = selected_device
            mock_get_service.return_value = mock_device_service

            mock_memory_service = MagicMock()
            mock_get_memory.return_value = mock_memory_service

            result = runner.invoke(device_app, ["memory"])

            assert result.exit_code == 0
            mock_device_service.get_selected_device.assert_called_once()
            mock_display.assert_called_once_with(selected_device.memory)
            assert "Memory Information" in result.stdout

    def test_device_memory_no_memory_info(self, runner: CliRunner) -> None:
        """Test device memory command with device that has basic memory info."""
        device_basic_memory = Device(
            device_type=DeviceType.CPU,
            name="CPU",
            memory=None,  # Will be auto-initialized to Memory(total_mb=None, available_mb=None)
            is_available=True,
        )

        with (
            patch("culora.cli.commands.device.get_device_service") as mock_get_service,
            patch("culora.cli.commands.device.get_memory_service") as mock_get_memory,
            patch("culora.cli.commands.device.display_memory_table") as mock_display,
        ):
            mock_device_service = MagicMock()
            mock_device_service.get_selected_device.return_value = device_basic_memory
            mock_get_service.return_value = mock_device_service

            mock_memory_service = MagicMock()
            mock_memory_service.get_memory_details.return_value = {}
            mock_get_memory.return_value = mock_memory_service

            result = runner.invoke(device_app, ["memory"])

            assert result.exit_code == 0
            assert "Memory Information - CPU" in result.output
            # Memory table should be displayed since device.memory is automatically created
            mock_display.assert_called_once()

    def test_device_memory_with_additional_details(
        self, runner: CliRunner, mock_devices: list[Device]
    ) -> None:
        """Test device memory command with additional memory details."""
        selected_device = mock_devices[1]  # CUDA device

        with (
            patch("culora.cli.commands.device.get_device_service") as mock_get_service,
            patch("culora.cli.commands.device.get_memory_service") as mock_get_memory,
            patch("culora.cli.commands.device.display_memory_table"),
        ):
            mock_device_service = MagicMock()
            mock_device_service.get_selected_device.return_value = selected_device
            mock_get_service.return_value = mock_device_service

            mock_memory_service = MagicMock()
            mock_memory_service.get_memory_details.return_value = {
                "driver_version": "12.0",
                "compute_capability": "8.6",
            }
            mock_get_memory.return_value = mock_memory_service

            result = runner.invoke(device_app, ["memory"])

            assert result.exit_code == 0
            assert "Additional memory details" in result.stdout

    def test_device_memory_device_error(self, runner: CliRunner) -> None:
        """Test device memory command with device error."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_selected_device.side_effect = DeviceError(
                "Device error", "TEST_ERROR"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["memory"])

            assert result.exit_code == 1
            assert "Device or memory error" in result.stdout

    def test_device_memory_unexpected_error(self, runner: CliRunner) -> None:
        """Test device memory command with unexpected error."""
        with patch("culora.cli.commands.device.get_device_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_selected_device.side_effect = RuntimeError(
                "Unexpected error"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(device_app, ["memory"])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout
