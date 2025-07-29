"""Tests for configuration manager."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from culora.core.config import CuLoRAConfig
from culora.core.config_manager import ConfigManager, get_config_manager
from culora.core.exceptions import (
    ConfigurationError,
    InvalidConfigError,
    MissingConfigError,
)


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_config_manager_creation(self) -> None:
        """Test creating config manager instance."""
        manager = ConfigManager()

        assert manager._config is None
        assert manager._config_sources == {}

    def test_load_config_defaults_only(self) -> None:
        """Test loading configuration with defaults only."""
        manager = ConfigManager()
        config = manager.load_config()

        assert isinstance(config, CuLoRAConfig)
        assert "defaults" in manager._config_sources

    def test_load_config_from_json_file(self, temp_config_file: Path) -> None:
        """Test loading configuration from JSON file."""
        manager = ConfigManager()
        config = manager.load_config(config_file=temp_config_file)

        assert config.device.batch_size == 16  # From fixture
        assert config.face_analysis.detection_threshold == 0.6
        assert "file" in manager._config_sources

    def test_load_config_from_yaml_file(self, temp_yaml_config_file: Path) -> None:
        """Test loading configuration from YAML file."""
        manager = ConfigManager()
        config = manager.load_config(config_file=temp_yaml_config_file)

        assert config.device.batch_size == 16  # From fixture
        assert "file" in manager._config_sources

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test loading configuration from non-existent file."""
        manager = ConfigManager()
        nonexistent_file = tmp_path / "nonexistent.json"

        # Should load defaults when file doesn't exist
        config = manager.load_config(config_file=nonexistent_file)
        assert isinstance(config, CuLoRAConfig)
        assert "file" not in manager._config_sources

    def test_load_config_invalid_file_format(self, tmp_path: Path) -> None:
        """Test loading configuration from invalid file format."""
        manager = ConfigManager()
        invalid_file = tmp_path / "config.txt"
        invalid_file.write_text("invalid config")

        with pytest.raises(ConfigurationError):
            manager.load_config(config_file=invalid_file)

    def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Test loading configuration from invalid JSON."""
        manager = ConfigManager()
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        with pytest.raises(InvalidConfigError):
            manager.load_config(config_file=invalid_json)

    def test_load_config_with_env_vars(self) -> None:
        """Test loading configuration from environment variables."""
        manager = ConfigManager()

        env_vars = {
            "CULORA_DEVICE_BATCH_SIZE": "64",
            "CULORA_FACE_ANALYSIS_DETECTION_THRESHOLD": "0.8",
            "CULORA_LOGGING_LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = manager.load_config()

        assert config.device.batch_size == 64
        assert config.face_analysis.detection_threshold == 0.8
        assert config.logging.log_level == "debug"
        assert "environment" in manager._config_sources

    def test_load_config_with_cli_overrides(self) -> None:
        """Test loading configuration with CLI overrides."""
        manager = ConfigManager()

        cli_overrides = {
            "device": {"batch_size": 128},
            "selection": {"target_count": 200},
        }

        config = manager.load_config(cli_overrides=cli_overrides)

        assert config.device.batch_size == 128
        assert config.selection.target_count == 200
        assert "cli" in manager._config_sources

    def test_config_precedence(self, temp_config_file: Path) -> None:
        """Test configuration source precedence (CLI > env > file > defaults)."""
        manager = ConfigManager()

        # File sets batch_size to 16 (from fixture)
        # Env var sets it to 32
        # CLI override sets it to 64
        env_vars = {"CULORA_DEVICE_BATCH_SIZE": "32"}
        cli_overrides = {"device": {"batch_size": 64}}

        with patch.dict(os.environ, env_vars):
            config = manager.load_config(
                config_file=temp_config_file,
                cli_overrides=cli_overrides,
            )

        # CLI should win
        assert config.device.batch_size == 64

    def test_get_config_before_loading(self) -> None:
        """Test getting config before loading raises error."""
        manager = ConfigManager()

        with pytest.raises(MissingConfigError):
            manager.get_config()

    def test_get_config_after_loading(self) -> None:
        """Test getting config after loading returns same instance."""
        manager = ConfigManager()
        loaded_config = manager.load_config()
        retrieved_config = manager.get_config()

        assert loaded_config is retrieved_config

    def test_export_config_json(self, tmp_path: Path) -> None:
        """Test exporting configuration to JSON file."""
        manager = ConfigManager()
        manager.load_config()

        export_path = tmp_path / "exported_config.json"
        manager.export_config(export_path)

        assert export_path.exists()

        # Verify exported content
        with open(export_path) as f:
            exported_data = json.load(f)

        assert "device" in exported_data
        assert "face_analysis" in exported_data

    def test_export_config_yaml(self, tmp_path: Path) -> None:
        """Test exporting configuration to YAML file."""
        manager = ConfigManager()
        manager.load_config()

        export_path = tmp_path / "exported_config.yaml"
        manager.export_config(export_path)

        assert export_path.exists()

        # Verify exported content
        with open(export_path) as f:
            exported_data = yaml.safe_load(f)

        assert "device" in exported_data
        assert "face_analysis" in exported_data

    def test_export_config_unsupported_format(self, tmp_path: Path) -> None:
        """Test exporting configuration to unsupported format."""
        manager = ConfigManager()
        manager.load_config()

        export_path = tmp_path / "config.txt"

        with pytest.raises(ConfigurationError):
            manager.export_config(export_path)

    def test_validate_config_valid(self, valid_config_dict: dict[str, Any]) -> None:
        """Test validating valid configuration dictionary."""
        manager = ConfigManager()
        errors = manager.validate_config(valid_config_dict)

        assert len(errors) == 0

    def test_validate_config_invalid(self, invalid_config_dict: dict[str, Any]) -> None:
        """Test validating invalid configuration dictionary."""
        manager = ConfigManager()
        errors = manager.validate_config(invalid_config_dict)

        assert len(errors) > 0
        # Should have errors for various invalid fields
        assert any("batch_size" in key for key in errors)

    def test_get_config_summary(self) -> None:
        """Test getting configuration summary."""
        manager = ConfigManager()
        manager.load_config()

        summary = manager.get_config_summary()

        assert "sources" in summary
        assert "device" in summary
        assert "quality" in summary
        assert "selection" in summary
        assert "export" in summary

    def test_deep_merge(self) -> None:
        """Test deep merging of configuration dictionaries."""
        manager = ConfigManager()

        base = {
            "device": {"batch_size": 16, "auto_detect": True},
            "logging": {"log_level": "info"},
        }

        update = {
            "device": {"batch_size": 32},  # Override
            "face_analysis": {"detection_threshold": 0.7},  # New section
        }

        result = manager._deep_merge(base, update)

        assert result["device"]["batch_size"] == 32  # Overridden
        assert result["device"]["auto_detect"] is True  # Preserved
        assert result["logging"]["log_level"] == "info"  # Preserved
        assert result["face_analysis"]["detection_threshold"] == 0.7  # Added

    def test_convert_env_value(self) -> None:
        """Test converting environment variable string values."""
        manager = ConfigManager()

        assert manager._convert_env_value("true") is True
        assert manager._convert_env_value("false") is False
        assert manager._convert_env_value("42") == 42
        assert manager._convert_env_value("3.14") == 3.14
        assert manager._convert_env_value("string_value") == "string_value"


class TestGlobalConfigManager:
    """Test global configuration manager functions."""

    def test_get_config_manager_singleton(self) -> None:
        """Test that global config manager is singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2
