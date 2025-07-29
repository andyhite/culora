"""Integration tests demonstrating fixture usage."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.domain.models import CuLoRAConfig
from culora.domain.models.device import Device
from culora.domain.models.memory import Memory
from culora.services.config_service import ConfigService
from culora.services.device_service import DeviceService

from ..helpers import AssertionHelpers, ConfigBuilder
from ..mocks.mock_torch import MockContext


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    def test_default_configuration_setup(self, default_config: CuLoRAConfig) -> None:
        """Test that default configuration is properly set up."""
        assert default_config.device.preferred_device == DeviceType.CPU
        assert default_config.logging.log_level == LogLevel.INFO

    def test_cuda_configuration_setup(self, cuda_config: CuLoRAConfig) -> None:
        """Test that CUDA configuration is properly set up."""
        assert cuda_config.device.preferred_device == DeviceType.CUDA
        assert cuda_config.logging.log_level == LogLevel.INFO

    def test_debug_configuration_setup(self, debug_config: CuLoRAConfig) -> None:
        """Test that debug configuration is properly set up."""
        assert debug_config.device.preferred_device == DeviceType.CPU
        assert debug_config.logging.log_level == LogLevel.DEBUG


@pytest.mark.integration
class TestDeviceIntegration:
    """Integration tests for device management."""

    def test_device_list_fixture(self, device_list: list[Device]) -> None:
        """Test that device list fixture provides expected devices."""
        assert len(device_list) == 3

        device_types = [device.device_type for device in device_list]
        assert DeviceType.CPU in device_types
        assert DeviceType.CUDA in device_types
        assert DeviceType.MPS in device_types

    def test_device_service_with_fixtures(
        self,
        device_service: DeviceService,
        cuda_config: CuLoRAConfig,
        mock_logger: Mock,
    ) -> None:
        """Test device service integration with fixtures."""
        # Create a new service with CUDA config
        cuda_service = DeviceService(cuda_config, mock_logger)

        assert cuda_service.config.device.preferred_device == DeviceType.CUDA
        assert cuda_service.logger == mock_logger


@pytest.mark.integration
class TestBuilderPatterns:
    """Tests demonstrating builder pattern usage."""

    def test_config_builder_basic(self) -> None:
        """Test basic configuration builder usage."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.DEBUG

    def test_config_builder_fluent_interface(self) -> None:
        """Test fluent interface configuration building."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.MPS)
            .with_log_level(LogLevel.WARNING)
            .build()
        )

        AssertionHelpers.assert_config_equal(
            config,
            CuLoRAConfig(
                device=CuLoRAConfig().device.__class__(preferred_device=DeviceType.MPS),
                logging=CuLoRAConfig().logging.__class__(log_level=LogLevel.WARNING),
            ),
        )


@pytest.mark.integration
class TestMockContext:
    """Tests demonstrating mock context usage."""

    def test_mock_context_torch_available(self) -> None:
        """Test mock context for torch availability."""
        with MockContext().mock_torch_available(True):
            # Import torch within the context
            import torch

            assert torch.cuda.is_available() is True

    def test_mock_context_torch_unavailable(self) -> None:
        """Test mock context for torch unavailability."""
        with (
            MockContext().mock_torch_available(False),
            pytest.raises(ImportError),
        ):
            # This should raise ImportError if torch is not available
            import torch  # noqa: F401

    def test_mock_context_cuda_devices(self) -> None:
        """Test mock context for CUDA devices."""
        with MockContext().mock_cuda_devices(count=2, memory_mb=16384):
            import torch

            assert torch.cuda.is_available() is True
            assert torch.cuda.device_count() == 2

            props = torch.cuda.get_device_properties(0)
            assert props.name == "GeForce RTX 3080"
            assert props.total_memory == 16384 * 1024 * 1024


@pytest.mark.integration
class TestFixtureComposition:
    """Tests demonstrating fixture composition."""

    def test_temp_directory_with_config(
        self,
        temp_dir: Path,
        sample_config_dict: dict[str, Any],
        config_service: ConfigService,
    ) -> None:
        """Test combining temporary directory and configuration fixtures."""
        # Create a config file in the temp directory
        config_file = temp_dir / "test_config.json"

        import json

        with open(config_file, "w") as f:
            json.dump(sample_config_dict, f, indent=2)

        # Load config from file
        config = config_service.load_config(config_file=config_file)

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.DEBUG

    def test_memory_fixtures_composition(
        self, unlimited_memory: Memory, limited_memory: Memory, low_memory: Memory
    ) -> None:
        """Test memory fixture composition."""
        # Test unlimited memory
        assert unlimited_memory.is_limited is False
        assert unlimited_memory.has_sufficient_memory(1000000) is True

        # Test limited memory
        assert limited_memory.is_limited is True
        assert limited_memory.has_sufficient_memory(2048) is True
        assert limited_memory.has_sufficient_memory(8192) is False

        # Test low memory
        assert low_memory.is_limited is True
        assert low_memory.has_sufficient_memory(256) is True
        assert low_memory.has_sufficient_memory(1024) is False


@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests marked as slow."""

    def test_large_config_processing(self, config_service: ConfigService) -> None:
        """Test processing of large configuration (marked as slow)."""
        # This would be a slow test in a real scenario
        large_config = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "debug"},
        }

        config = config_service.load_config(cli_overrides=large_config)
        assert config.device.preferred_device == DeviceType.CUDA
