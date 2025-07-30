"""Tests for CLI application."""

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from culora.cli.app import app, cli_main
from culora.core import ConfigError, CuLoRAError


class TestCLIApp:
    """Test main CLI application."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner for testing."""
        return CliRunner()

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "CuLoRA - Advanced LoRA Dataset Curation Utility" in result.stdout
        assert "config" in result.stdout
        assert "device" in result.stdout

    def test_version_flag(self, runner: CliRunner) -> None:
        """Test version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "CuLoRA v0.1.0" in result.stdout

    def test_version_flag_short(self, runner: CliRunner) -> None:
        """Test short version flag."""
        result = runner.invoke(app, ["-v"])

        assert result.exit_code == 0
        assert "CuLoRA v0.1.0" in result.stdout

    def test_config_command_available(self, runner: CliRunner) -> None:
        """Test that config command is available."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "Configuration management" in result.stdout

    def test_device_command_available(self, runner: CliRunner) -> None:
        """Test that device command is available."""
        result = runner.invoke(app, ["device", "--help"])

        assert result.exit_code == 0
        assert "Device information and management" in result.stdout

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test handling of invalid command."""
        result = runner.invoke(app, ["invalid"])

        assert result.exit_code == 2  # Typer "No such command" error
        assert "No such command" in result.output or "Usage:" in result.output


class TestCLIMain:
    """Test CLI main entry point with error handling."""

    def test_cli_main_success(self) -> None:
        """Test successful CLI main execution."""
        with patch("culora.cli.app.app") as mock_app:
            cli_main()
            mock_app.assert_called_once()

    def test_cli_main_culora_error(self) -> None:
        """Test CLI main with CuLoRA error."""
        with patch("culora.cli.app.app") as mock_app:
            error = CuLoRAError("Test CuLoRA error", "TEST_ERROR")
            mock_app.side_effect = error

            with patch("culora.cli.app.console") as mock_console:
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                assert exc_info.value.exit_code == 1
                mock_console.error.assert_called_with("CuLoRA Error: Test CuLoRA error")
                mock_console.info.assert_called_with("Error Code: TEST_ERROR")

    def test_cli_main_culora_error_no_code(self) -> None:
        """Test CLI main with CuLoRA error without error code."""
        with patch("culora.cli.app.app") as mock_app:
            # Create error and then override error_code to be falsy
            error = CuLoRAError("Test error")
            error.error_code = ""  # Override to empty string after construction
            mock_app.side_effect = error

            with patch("culora.cli.app.console") as mock_console:
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                assert exc_info.value.exit_code == 1
                mock_console.error.assert_called_with("CuLoRA Error: Test error")
                # Should not call info for error code since it's empty
                assert not any(
                    call[0][0].startswith("Error Code:")
                    for call in mock_console.info.call_args_list
                )

    def test_cli_main_config_error(self) -> None:
        """Test CLI main with configuration error."""
        with patch("culora.cli.app.app") as mock_app:
            error = ConfigError("Test config error", "CONFIG_ERROR")
            mock_app.side_effect = error

            with patch("culora.cli.app.console") as mock_console:
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                assert exc_info.value.exit_code == 1
                mock_console.error.assert_called_with(
                    "Configuration Error: Test config error"
                )
                mock_console.info.assert_called_with("Error Code: CONFIG_ERROR")

    def test_cli_main_keyboard_interrupt(self) -> None:
        """Test CLI main with keyboard interrupt."""
        with patch("culora.cli.app.app") as mock_app:
            mock_app.side_effect = KeyboardInterrupt()

            with patch("culora.cli.app.console") as mock_console:
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                assert exc_info.value.exit_code == 130
                mock_console.warning.assert_called_with("Operation cancelled by user")

    def test_cli_main_unexpected_error(self) -> None:
        """Test CLI main with unexpected error."""
        with patch("culora.cli.app.app") as mock_app:
            mock_app.side_effect = RuntimeError("Unexpected error")

            with patch("culora.cli.app.console") as mock_console:
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                assert exc_info.value.exit_code == 1
                mock_console.error.assert_called_with(
                    "Unexpected error: Unexpected error"
                )
                mock_console.info.assert_called_with(
                    "This may be a bug. Please report it at: https://github.com/andyhite/culora/issues"
                )

    def test_cli_main_exception_chaining(self) -> None:
        """Test that CLI main properly chains exceptions."""
        with patch("culora.cli.app.app") as mock_app:
            original_error = ValueError("Original error")
            mock_app.side_effect = original_error

            with patch("culora.cli.app.console"):
                with pytest.raises(typer.Exit) as exc_info:
                    cli_main()

                # Verify exception is properly chained
                assert exc_info.value.__cause__ is original_error
