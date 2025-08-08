"""Tests for main CLI module."""

from typer.testing import CliRunner

from culora.main import app


def test_version_command():
    """Test the version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "CuLoRA version: 0.1.0" in result.stdout


def test_no_arguments_shows_help():
    """Test that running with no arguments shows help."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    # Typer shows help when no_args_is_help=True
    assert "Usage:" in result.stdout
    assert (
        "A command-line tool for intelligently curating image datasets" in result.stdout
    )


def test_missing_required_arguments():
    """Test commands with missing required arguments."""
    runner = CliRunner()

    # Analyze without directory should fail
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code != 0

    # Select without output directory should fail
    result = runner.invoke(app, ["select"])
    assert result.exit_code != 0
