"""Tests for CuLoRAConfig model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.domain.models.config import CuLoRAConfig, DeviceConfig, LoggingConfig

from .....helpers import AssertionHelpers, ConfigBuilder


class TestCuLoRAConfig:
    """Test cases for CuLoRAConfig model."""

    def test_culora_config_default_values(self) -> None:
        """Test CuLoRAConfig default initialization."""
        config = CuLoRAConfig()
        assert isinstance(config.device, DeviceConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.device.preferred_device == DeviceType.CPU
        assert config.logging.log_level == LogLevel.INFO

    def test_culora_config_with_custom_values(self) -> None:
        """Test CuLoRAConfig with custom values."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.DEBUG

    def test_culora_config_from_dict(self) -> None:
        """Test CuLoRAConfig creation from dictionary."""
        config_dict = {
            "device": {"preferred_device": "mps"},
            "logging": {"log_level": "warning"},
        }
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.MPS
        assert config.logging.log_level == LogLevel.WARNING

    def test_culora_config_from_dict_partial(self) -> None:
        """Test CuLoRAConfig creation from partial dictionary."""
        config_dict = {"device": {"preferred_device": "cuda"}}
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.INFO  # Default

    def test_culora_config_from_dict_empty(self) -> None:
        """Test CuLoRAConfig creation from empty dictionary."""
        config = CuLoRAConfig.from_dict({})

        assert config.device.preferred_device == DeviceType.CPU
        assert config.logging.log_level == LogLevel.INFO

    def test_culora_config_model_dump(self) -> None:
        """Test CuLoRAConfig serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.ERROR)
            .build()
        )

        dumped = config.model_dump(mode="json")
        expected = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "error"},
            "images": {
                "supported_formats": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".tiff",
                    ".tif",
                ],
                "max_batch_size": 32,
                "max_image_size": [4096, 4096],  # tuple becomes list in JSON mode
                "max_file_size": 52428800,
                "recursive_scan": True,
                "max_scan_depth": 10,
                "skip_hidden_files": True,
                "progress_update_interval": 10,
            },
        }
        assert dumped == expected

    def test_culora_config_model_dump_json(self) -> None:
        """Test CuLoRAConfig JSON serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.MPS)
            .with_log_level(LogLevel.WARNING)
            .build()
        )

        json_str = config.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["device"]["preferred_device"] == "mps"
        assert parsed["logging"]["log_level"] == "warning"

        # Check formatting (should be indented)
        assert "\n" in json_str

    def test_culora_config_get_section(self) -> None:
        """Test CuLoRAConfig get_section method."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        device_section = config.get_section("device")
        logging_section = config.get_section("logging")

        assert isinstance(device_section, DeviceConfig)
        assert isinstance(logging_section, LoggingConfig)
        assert device_section.preferred_device == DeviceType.CUDA
        assert logging_section.log_level == LogLevel.DEBUG

    def test_culora_config_get_section_invalid(self) -> None:
        """Test CuLoRAConfig get_section with invalid section name."""
        config = CuLoRAConfig()

        with pytest.raises(AttributeError):
            config.get_section("invalid_section")

    def test_culora_config_validate_assignment(self) -> None:
        """Test that CuLoRAConfig validates assignments."""
        config = CuLoRAConfig()

        # Valid assignment
        config.device = DeviceConfig(preferred_device=DeviceType.MPS)
        assert config.device.preferred_device == DeviceType.MPS

        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            config.device = "invalid"  # type: ignore[assignment]

    def test_culora_config_nested_validation(self) -> None:
        """Test CuLoRAConfig nested validation."""
        with pytest.raises(ValidationError) as exc_info:
            CuLoRAConfig(
                device={"preferred_device": "invalid_device"},  # type: ignore[arg-type]
                logging={"log_level": "debug"},  # type: ignore[arg-type]
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "device" in str(errors[0]["loc"])
        assert "preferred_device" in str(errors[0]["loc"])

    def test_culora_config_equality(self) -> None:
        """Test CuLoRAConfig equality comparison."""
        config1 = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )
        config2 = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )
        config3 = (
            ConfigBuilder()
            .with_device(DeviceType.MPS)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        AssertionHelpers.assert_config_equal(config1, config2)
        assert config1 != config3

    def test_culora_config_repr(self) -> None:
        """Test CuLoRAConfig string representation."""
        config = CuLoRAConfig()
        repr_str = repr(config)
        assert "CuLoRAConfig" in repr_str

    def test_culora_config_json_encoders(self) -> None:
        """Test that Path objects are properly encoded."""
        # Create a config that would use Path encoding
        config = CuLoRAConfig()

        # The json_encoders should handle Path objects
        # This is tested indirectly through the ConfigDict
        config_dict = config.model_config
        if "json_encoders" in config_dict and config_dict["json_encoders"] is not None:
            assert Path in config_dict["json_encoders"]
            assert config_dict["json_encoders"][Path] is str

    def test_culora_config_use_enum_values(self) -> None:
        """Test that enum values are used in serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.ERROR)
            .build()
        )

        # When serialized, should use enum values, not enum names
        dumped = config.model_dump()
        assert dumped["device"]["preferred_device"] == "cuda"  # value, not "CUDA"
        assert dumped["logging"]["log_level"] == "error"  # value, not "ERROR"

    @pytest.mark.parametrize(
        "device_type,log_level",
        [
            (DeviceType.CUDA, LogLevel.DEBUG),
            (DeviceType.MPS, LogLevel.INFO),
            (DeviceType.CPU, LogLevel.WARNING),
            (DeviceType.CUDA, LogLevel.ERROR),
            (DeviceType.MPS, LogLevel.CRITICAL),
        ],
    )
    def test_culora_config_combinations(
        self, device_type: DeviceType, log_level: LogLevel
    ) -> None:
        """Parametrized test for various device/log level combinations."""
        config = (
            ConfigBuilder().with_device(device_type).with_log_level(log_level).build()
        )

        assert config.device.preferred_device == device_type
        assert config.logging.log_level == log_level
