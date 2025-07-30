"""Tests for CLI config commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from culora.cli.commands.config import config_app
from culora.core import ConfigError, InvalidConfigError, MissingConfigError
from tests.helpers import ConfigBuilder, TempFileHelper


class TestConfigCommands:
    """Test configuration CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_config_service(self) -> MagicMock:
        """Create mock config service."""
        return MagicMock()

    def test_show_config_success(self, runner: CliRunner) -> None:
        """Test successful config show command."""
        config = ConfigBuilder().build()

        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.load_config.return_value = config
            mock_service.get_config_summary.return_value = {
                "sources": {"defaults": "Built-in defaults"}
            }
            mock_get_service.return_value = mock_service

            with patch("culora.cli.commands.config.display_config_table"):
                result = runner.invoke(config_app, ["show"])

                assert result.exit_code == 0
                mock_service.load_config.assert_called_once_with(config_file=None)
                mock_service.get_config_summary.assert_called_once()

    def test_show_config_with_file(self, runner: CliRunner) -> None:
        """Test config show command with specific file."""
        config = ConfigBuilder().build()

        with TempFileHelper.create_temp_file(".yaml") as config_file:
            config_file.write_text("device:\n  preferred_device: cpu\n")

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.load_config.return_value = config
                mock_service.get_config_summary.return_value = {"sources": {}}
                mock_get_service.return_value = mock_service

                with patch("culora.cli.commands.config.display_config_table"):
                    result = runner.invoke(
                        config_app, ["show", "--config", str(config_file)]
                    )

                    assert result.exit_code == 0
                    mock_service.load_config.assert_called_once_with(
                        config_file=config_file
                    )

    def test_show_config_missing_error(self, runner: CliRunner) -> None:
        """Test config show command with missing configuration."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.load_config.side_effect = MissingConfigError("test")
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["show"])

            assert result.exit_code == 1
            assert "No configuration loaded" in result.stdout

    def test_show_config_error(self, runner: CliRunner) -> None:
        """Test config show command with configuration error."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.load_config.side_effect = ConfigError(
                "Test error", "TEST_ERROR"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["show"])

            assert result.exit_code == 1
            assert "Configuration error" in result.stdout

    def test_get_config_value_success(self, runner: CliRunner) -> None:
        """Test successful config get command."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_config.return_value = MagicMock()
            mock_service.get_config_value.return_value = "cpu"
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["get", "device.preferred_device"])

            assert result.exit_code == 0
            mock_service.get_config_value.assert_called_once_with(
                "device.preferred_device"
            )

    def test_get_config_value_with_file(self, runner: CliRunner) -> None:
        """Test config get command with specific file."""
        with TempFileHelper.create_temp_file(".yaml") as config_file:
            config_file.write_text("device:\n  preferred_device: cpu\n")

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.get_config.side_effect = [
                    MissingConfigError("test"),
                    MagicMock(),
                ]
                mock_service.get_config_value.return_value = "cpu"
                mock_get_service.return_value = mock_service

                result = runner.invoke(
                    config_app,
                    ["get", "device.preferred_device", "--config", str(config_file)],
                )

                assert result.exit_code == 0
                mock_service.load_config.assert_called_once_with(
                    config_file=config_file
                )

    def test_get_config_value_key_error(self, runner: CliRunner) -> None:
        """Test config get command with invalid key."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_config.return_value = MagicMock()
            mock_service.get_config_value.side_effect = KeyError("Invalid key")
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["get", "device.nonexistent_key"])

            assert result.exit_code == 1
            assert "Configuration key not found" in result.stdout

    def test_set_config_value_success(self, runner: CliRunner) -> None:
        """Test successful config set command."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_config.return_value = MagicMock()
            mock_service.get_config_file.return_value = Path("/test/config.yaml")
            mock_get_service.return_value = mock_service

            result = runner.invoke(
                config_app, ["set", "device.preferred_device", "cuda"]
            )

            assert result.exit_code == 0
            mock_service.set_config_value.assert_called_once_with(
                "device.preferred_device", "cuda", None
            )
            assert "Set device.preferred_device = cuda" in result.stdout

    def test_set_config_value_with_file(self, runner: CliRunner) -> None:
        """Test config set command with specific file."""
        with TempFileHelper.create_temp_file(".yaml") as config_file:
            config_file.write_text("device:\n  preferred_device: cpu\n")

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.get_config.side_effect = [
                    MissingConfigError("test"),
                    MagicMock(),
                ]
                mock_service.get_config_file.return_value = config_file
                mock_get_service.return_value = mock_service

                result = runner.invoke(
                    config_app,
                    [
                        "set",
                        "device.preferred_device",
                        "cuda",
                        "--config",
                        str(config_file),
                    ],
                )

                assert result.exit_code == 0
                mock_service.load_config.assert_called_once_with(
                    config_file=config_file
                )
                mock_service.set_config_value.assert_called_once_with(
                    "device.preferred_device", "cuda", config_file
                )

    def test_set_config_value_invalid_value(self, runner: CliRunner) -> None:
        """Test config set command with invalid value."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_config.return_value = MagicMock()
            mock_service.set_config_value.side_effect = InvalidConfigError(
                "test", "invalid", "valid"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(
                config_app, ["set", "device.preferred_device", "invalid"]
            )

            assert result.exit_code == 1
            assert "Invalid configuration value" in result.stdout

    def test_set_config_value_type_conversion(self, runner: CliRunner) -> None:
        """Test config set command with type conversion."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_config.return_value = MagicMock()
            mock_service.get_config_file.return_value = Path("/test/config.yaml")
            mock_get_service.return_value = mock_service

            # Test boolean conversion
            result = runner.invoke(config_app, ["set", "logging.log_level", "debug"])
            assert result.exit_code == 0
            mock_service.set_config_value.assert_called_with(
                "logging.log_level", "debug", None
            )

    def test_validate_config_success(self, runner: CliRunner) -> None:
        """Test successful config validate command."""
        config = ConfigBuilder().build()

        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.load_config.return_value = config
            mock_service.get_config_file.return_value = Path("/test/config.yaml")
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["validate"])

            assert result.exit_code == 0
            assert "Configuration is valid" in result.stdout

    def test_validate_config_error(self, runner: CliRunner) -> None:
        """Test config validate command with validation error."""
        with patch("culora.cli.commands.config.get_config_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.load_config.side_effect = ConfigError(
                "Validation failed", "TEST_ERROR"
            )
            mock_get_service.return_value = mock_service

            result = runner.invoke(config_app, ["validate"])

            assert result.exit_code == 1
            assert "Configuration validation failed" in result.stdout

    def test_export_config_success(self, runner: CliRunner) -> None:
        """Test successful config export command."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            output_file = temp_dir / "exported.yaml"

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.get_config.return_value = MagicMock()
                mock_get_service.return_value = mock_service

                result = runner.invoke(config_app, ["export", str(output_file)])

                assert result.exit_code == 0
                mock_service.export_config.assert_called_once_with(
                    output_file, include_defaults=True
                )
                assert "Configuration exported" in result.stdout

    def test_export_config_exclude_defaults(self, runner: CliRunner) -> None:
        """Test config export command excluding defaults."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            output_file = temp_dir / "exported.yaml"

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.get_config.return_value = MagicMock()
                mock_get_service.return_value = mock_service

                result = runner.invoke(
                    config_app, ["export", str(output_file), "--exclude-defaults"]
                )

                assert result.exit_code == 0
                mock_service.export_config.assert_called_once_with(
                    output_file, include_defaults=False
                )

    def test_export_config_error(self, runner: CliRunner) -> None:
        """Test config export command with export error."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            output_file = temp_dir / "exported.yaml"

            with patch(
                "culora.cli.commands.config.get_config_service"
            ) as mock_get_service:
                mock_service = MagicMock()
                mock_service.get_config.return_value = MagicMock()
                mock_service.export_config.side_effect = ConfigError(
                    "Export failed", "TEST_ERROR"
                )
                mock_get_service.return_value = mock_service

                result = runner.invoke(config_app, ["export", str(output_file)])

                assert result.exit_code == 1
                assert "Export failed" in result.stdout

    def test_invalid_config_file_validation(self, runner: CliRunner) -> None:
        """Test validation of invalid config file path."""
        result = runner.invoke(
            config_app, ["show", "--config", "/nonexistent/file.yaml"]
        )

        assert result.exit_code == 2  # Typer validation error
        assert (
            "Configuration file" in result.output and "does not exist" in result.output
        )

    def test_invalid_config_key_validation(self, runner: CliRunner) -> None:
        """Test validation of invalid config key."""
        result = runner.invoke(config_app, ["get", "invalid.section.key"])

        assert result.exit_code == 2  # Typer validation error
        assert "Invalid configuration section" in result.output
