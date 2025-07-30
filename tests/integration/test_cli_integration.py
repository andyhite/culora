"""Integration tests for CLI functionality."""

import json

import pytest
from typer.testing import CliRunner

from culora.cli.app import app
from tests.helpers import TempFileHelper


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner for testing."""
        return CliRunner()

    def test_config_workflow_integration(self, runner: CliRunner) -> None:
        """Test complete config workflow: show -> set -> get -> export."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            config_file = temp_dir / "test_config.yaml"
            export_file = temp_dir / "exported_config.json"

            # Step 1: Set a configuration value (creates config file)
            result = runner.invoke(
                app,
                [
                    "config",
                    "set",
                    "device.preferred_device",
                    "cuda",
                    "--config",
                    str(config_file),
                ],
            )
            assert result.exit_code == 0
            assert "Set device.preferred_device = cuda" in result.stdout
            assert config_file.exists()

            # Step 2: Get the configuration value
            result = runner.invoke(
                app,
                [
                    "config",
                    "get",
                    "device.preferred_device",
                    "--config",
                    str(config_file),
                ],
            )
            assert result.exit_code == 0
            assert "device.preferred_device: DeviceType.CUDA" in result.stdout

            # Step 3: Show full configuration
            result = runner.invoke(
                app, ["config", "show", "--config", str(config_file)]
            )
            assert result.exit_code == 0
            assert "CuLoRA Configuration" in result.stdout
            assert "DeviceType.CUDA" in result.stdout

            # Step 4: Export configuration
            result = runner.invoke(
                app,
                ["config", "export", str(export_file), "--config", str(config_file)],
            )
            assert result.exit_code == 0
            assert "Configuration exported" in result.stdout
            assert export_file.exists()

            # Verify exported content
            with open(export_file) as f:
                exported_data = json.load(f)
            assert exported_data["device"]["preferred_device"] == "cuda"
            assert exported_data["logging"]["log_level"] == "info"

    def test_config_validation_integration(self, runner: CliRunner) -> None:
        """Test config validation with valid and invalid files."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            # Test valid config file
            valid_config = temp_dir / "valid_config.yaml"
            valid_config.write_text(
                """
device:
  preferred_device: cpu
logging:
  log_level: debug
"""
            )

            result = runner.invoke(
                app, ["config", "validate", "--config", str(valid_config)]
            )
            assert result.exit_code == 0
            assert "Configuration is valid" in result.stdout

            # Test invalid config file (bad YAML)
            invalid_config = temp_dir / "invalid_config.yaml"
            invalid_config.write_text("invalid: yaml: content: [")

            result = runner.invoke(
                app, ["config", "validate", "--config", str(invalid_config)]
            )
            assert result.exit_code == 1
            assert "Configuration validation failed" in result.stdout

    def test_device_info_integration(self, runner: CliRunner) -> None:
        """Test device info command integration."""
        result = runner.invoke(app, ["device", "info"])

        # Should succeed (may show no devices or available devices)
        assert result.exit_code == 0

        # Should contain device information header or warning about no devices
        assert (
            "Device Information" in result.stdout
            or "No available devices detected" in result.stdout
        )

    def test_device_list_integration(self, runner: CliRunner) -> None:
        """Test device list command integration."""
        result = runner.invoke(app, ["device", "list"])

        # Should succeed
        assert result.exit_code == 0

        # Should contain device list or warning about no devices
        assert "All Devices" in result.stdout or "No devices found" in result.stdout

        # Should show device count
        assert "Available devices:" in result.stdout

    def test_device_memory_integration(self, runner: CliRunner) -> None:
        """Test device memory command integration."""
        result = runner.invoke(app, ["device", "memory"])

        # Should succeed
        assert result.exit_code == 0

        # Should contain memory information or warning about no memory info
        assert (
            "Memory Information" in result.stdout
            or "No memory information available" in result.stdout
        )

    def test_error_handling_integration(self, runner: CliRunner) -> None:
        """Test error handling in integrated scenarios."""
        # Test invalid config file path
        result = runner.invoke(
            app, ["config", "show", "--config", "/nonexistent/config.yaml"]
        )
        assert result.exit_code == 2  # Validation error
        assert (
            "Configuration file" in result.output and "does not exist" in result.output
        )

        # Test invalid config key
        result = runner.invoke(app, ["config", "get", "invalid.section.key"])
        assert result.exit_code == 2  # Validation error
        assert "Invalid configuration section" in result.output

        # Test invalid command
        result = runner.invoke(app, ["nonexistent"])
        assert result.exit_code == 2
        assert "No such command" in result.output

    def test_help_system_integration(self, runner: CliRunner) -> None:
        """Test help system integration."""
        # Main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CuLoRA - Advanced LoRA Dataset Curation Utility" in result.stdout
        assert "config" in result.stdout
        assert "device" in result.stdout

        # Config help
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.stdout
        assert "show" in result.stdout
        assert "get" in result.stdout
        assert "set" in result.stdout

        # Device help
        result = runner.invoke(app, ["device", "--help"])
        assert result.exit_code == 0
        assert "Device information and management" in result.stdout
        assert "info" in result.stdout
        assert "list" in result.stdout
        assert "memory" in result.stdout

    def test_version_integration(self, runner: CliRunner) -> None:
        """Test version display integration."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "CuLoRA v0.1.0" in result.stdout

        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "CuLoRA v0.1.0" in result.stdout

    def test_config_type_conversion_integration(self, runner: CliRunner) -> None:
        """Test configuration type conversion in real scenarios."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            config_file = temp_dir / "type_test_config.yaml"

            # Test boolean conversion
            result = runner.invoke(
                app,
                [
                    "config",
                    "set",
                    "logging.log_level",
                    "debug",
                    "--config",
                    str(config_file),
                ],
            )
            assert result.exit_code == 0

            result = runner.invoke(
                app,
                ["config", "get", "logging.log_level", "--config", str(config_file)],
            )
            assert result.exit_code == 0
            assert "LogLevel.DEBUG" in result.stdout

    def test_config_file_formats_integration(self, runner: CliRunner) -> None:
        """Test different config file formats."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            # Test YAML format
            yaml_config = temp_dir / "config.yaml"
            result = runner.invoke(
                app,
                [
                    "config",
                    "set",
                    "device.preferred_device",
                    "mps",
                    "--config",
                    str(yaml_config),
                ],
            )
            assert result.exit_code == 0
            assert yaml_config.exists()

            # Test JSON export
            json_export = temp_dir / "exported.json"
            result = runner.invoke(
                app,
                ["config", "export", str(json_export), "--config", str(yaml_config)],
            )
            assert result.exit_code == 0
            assert json_export.exists()

            # Verify JSON content
            with open(json_export) as f:
                data = json.load(f)
            assert data["device"]["preferred_device"] == "mps"
