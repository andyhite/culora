"""Tests for ConfigService."""

import json
from pathlib import Path

import pytest

from culora.core.exceptions import InvalidConfigError, MissingConfigError
from culora.domain import CuLoRAConfig
from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.services.config_service import ConfigService, get_config_service

from ...helpers import TempFileHelper, patch_environment


class TestConfigService:
    """Test cases for ConfigService."""

    def test_config_service_initialization(self) -> None:
        """Test ConfigService initialization."""
        service = ConfigService()
        assert service._config is None
        assert service._config_sources == {}

    def test_load_config_defaults_only(self) -> None:
        """Test loading configuration with defaults only."""
        service = ConfigService()
        config = service.load_config()

        assert isinstance(config, CuLoRAConfig)
        assert config.device.preferred_device == DeviceType.CPU
        assert config.logging.log_level == LogLevel.INFO
        assert "defaults" in service._config_sources

    def test_load_config_from_json_file(self) -> None:
        """Test loading configuration from JSON file."""
        config_data = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "debug"},
        }

        with TempFileHelper.create_config_file(config_data, ".json") as config_file:
            service = ConfigService()
            config = service.load_config(config_file=config_file)

            assert config.device.preferred_device == DeviceType.CUDA
            assert config.logging.log_level == LogLevel.DEBUG
            assert "file" in service._config_sources
            assert str(config_file) in service._config_sources["file"]

    def test_load_config_from_yaml_file(self) -> None:
        """Test loading configuration from YAML file."""
        config_data = {
            "device": {"preferred_device": "mps"},
            "logging": {"log_level": "warning"},
        }

        with TempFileHelper.create_config_file(config_data, ".yaml") as config_file:
            service = ConfigService()
            config = service.load_config(config_file=config_file)

            assert config.device.preferred_device == DeviceType.MPS
            assert config.logging.log_level == LogLevel.WARNING
            assert "file" in service._config_sources

    def test_load_config_nonexistent_file(self) -> None:
        """Test loading configuration with nonexistent file."""
        service = ConfigService()
        nonexistent_file = Path("/nonexistent/config.json")

        # Should not raise error, just skip file loading
        config = service.load_config(config_file=nonexistent_file)
        assert config.device.preferred_device == DeviceType.CPU
        assert "file" not in service._config_sources

    def test_load_config_from_environment(self) -> None:
        """Test loading configuration from environment variables."""
        env_vars = {"CULORA_DEVICE_PREFERRED": "cuda", "CULORA_LOG_LEVEL": "error"}

        with patch_environment(**env_vars):
            service = ConfigService()
            config = service.load_config()

            assert config.device.preferred_device == DeviceType.CUDA
            assert config.logging.log_level == LogLevel.ERROR
            assert "environment" in service._config_sources

    def test_load_config_cli_overrides(self) -> None:
        """Test loading configuration with CLI overrides."""
        cli_overrides = {
            "device": {"preferred_device": "mps"},
            "logging": {"log_level": "critical"},
        }

        service = ConfigService()
        config = service.load_config(cli_overrides=cli_overrides)

        assert config.device.preferred_device == DeviceType.MPS
        assert config.logging.log_level == LogLevel.CRITICAL
        assert "cli" in service._config_sources

    def test_load_config_precedence(self) -> None:
        """Test configuration source precedence (CLI > env > file > defaults)."""
        # Create config file
        file_config = {
            "device": {"preferred_device": "cpu"},
            "logging": {"log_level": "info"},
        }

        with TempFileHelper.create_config_file(file_config, ".json") as config_file:
            # Set environment variables
            env_vars = {
                "CULORA_DEVICE_PREFERRED": "cuda",
                "CULORA_LOG_LEVEL": "warning",
            }

            # Set CLI overrides (should have highest precedence)
            cli_overrides = {"logging": {"log_level": "error"}}

            with patch_environment(**env_vars):
                service = ConfigService()
                config = service.load_config(
                    config_file=config_file, cli_overrides=cli_overrides
                )

                # CLI should override logging, env should override device
                assert config.device.preferred_device == DeviceType.CUDA  # from env
                assert config.logging.log_level == LogLevel.ERROR  # from CLI

    def test_get_config_success(self) -> None:
        """Test getting configuration after loading."""
        service = ConfigService()
        original_config = service.load_config()
        retrieved_config = service.get_config()

        assert retrieved_config == original_config

    def test_get_config_not_loaded(self) -> None:
        """Test getting configuration before loading."""
        service = ConfigService()

        with pytest.raises(MissingConfigError) as exc_info:
            service.get_config()

        assert "configuration" in str(exc_info.value)
        assert exc_info.value.error_code == "CONFIG_NOT_LOADED"

    def test_export_config(self) -> None:
        """Test exporting configuration to file."""
        service = ConfigService()
        service.load_config()

        with TempFileHelper.create_temp_file(".json") as output_path:
            service.export_config(output_path)

            with open(output_path) as f:
                exported_data = json.load(f)

            assert "device" in exported_data
            assert "logging" in exported_data
            assert exported_data["device"]["preferred_device"] == "cpu"
            assert exported_data["logging"]["log_level"] == "info"

    def test_export_config_not_loaded(self) -> None:
        """Test exporting configuration before loading."""
        service = ConfigService()
        output_path = Path("/tmp/test_config.json")

        with pytest.raises(MissingConfigError):
            service.export_config(output_path)

    def test_validate_config_valid(self) -> None:
        """Test validating valid configuration."""
        service = ConfigService()
        config_dict = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "debug"},
        }

        errors = service.validate_config(config_dict)
        assert errors == {}

    def test_validate_config_invalid(self) -> None:
        """Test validating invalid configuration."""
        service = ConfigService()
        config_dict = {
            "device": {"preferred_device": "invalid_device"},
            "logging": {"log_level": "invalid_level"},
        }

        errors = service.validate_config(config_dict)
        assert len(errors) > 0

    def test_get_config_summary(self) -> None:
        """Test getting configuration summary."""
        service = ConfigService()
        service.load_config()

        summary = service.get_config_summary()

        assert "sources" in summary
        assert "device" in summary
        assert "defaults" in summary["sources"]
        assert summary["device"]["preferred"] == DeviceType.CPU

    def test_load_from_file_invalid_json(self) -> None:
        """Test loading from invalid JSON file."""
        with TempFileHelper.create_temp_file(".json") as config_file:
            with open(config_file, "w") as f:
                f.write("invalid json content")

            service = ConfigService()
            with pytest.raises(InvalidConfigError) as exc_info:
                service._load_from_file(config_file)

            assert "CONFIG_FILE_PARSE_ERROR" in str(exc_info.value.error_code)

    def test_load_from_file_invalid_yaml(self) -> None:
        """Test loading from invalid YAML file."""
        with TempFileHelper.create_temp_file(".yaml") as config_file:
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content: [")

            service = ConfigService()
            with pytest.raises(InvalidConfigError) as exc_info:
                service._load_from_file(config_file)

            assert "CONFIG_FILE_PARSE_ERROR" in str(exc_info.value.error_code)

    def test_convert_env_value_boolean(self) -> None:
        """Test converting environment variable boolean values."""
        service = ConfigService()

        assert service._convert_env_value("true") is True
        assert service._convert_env_value("True") is True
        assert service._convert_env_value("1") is True
        assert service._convert_env_value("yes") is True
        assert service._convert_env_value("on") is True

        assert service._convert_env_value("false") is False
        assert service._convert_env_value("False") is False
        assert service._convert_env_value("0") is False
        assert service._convert_env_value("no") is False
        assert service._convert_env_value("off") is False

    def test_convert_env_value_numbers(self) -> None:
        """Test converting environment variable numeric values."""
        service = ConfigService()

        assert service._convert_env_value("42") == 42
        assert service._convert_env_value("3.14") == 3.14
        assert service._convert_env_value("not_a_number") == "not_a_number"

    def test_deep_merge(self) -> None:
        """Test deep merge functionality."""
        service = ConfigService()

        base = {"device": {"preferred_device": "cpu"}, "logging": {"log_level": "info"}}

        update = {
            "device": {"preferred_device": "cuda"},
            "new_section": {"new_key": "new_value"},
        }

        result = service._deep_merge(base, update)

        assert result["device"]["preferred_device"] == "cuda"
        assert result["logging"]["log_level"] == "info"
        assert result["new_section"]["new_key"] == "new_value"

    def test_global_config_service(self) -> None:
        """Test global configuration service instance."""
        service1 = get_config_service()
        service2 = get_config_service()

        assert service1 is service2  # Should be the same instance

    @pytest.mark.parametrize(
        "file_extension,content_type",
        [
            (".json", "json"),
            (".yaml", "yaml"),
            (".yml", "yaml"),
        ],
    )
    def test_supported_file_formats(
        self, file_extension: str, content_type: str
    ) -> None:
        """Parametrized test for supported configuration file formats."""
        config_data = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "debug"},
        }

        with TempFileHelper.create_config_file(
            config_data, file_extension
        ) as config_file:
            service = ConfigService()
            config = service.load_config(config_file=config_file)

            assert config.device.preferred_device == DeviceType.CUDA
            assert config.logging.log_level == LogLevel.DEBUG

    def test_load_config_validation_error(self) -> None:
        """Test load_config with validation errors."""
        cli_overrides = {"device": {"preferred_device": "invalid_device"}}

        service = ConfigService()
        with pytest.raises(InvalidConfigError) as exc_info:
            service.load_config(cli_overrides=cli_overrides)

        assert exc_info.value.error_code == "INVALID_CONFIG"
