"""Tests for main CLI module."""

from culora.cli.main import app
from typer.testing import CliRunner


def test_version_command():
    """Test the version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "CuLoRA version: 0.1.0" in result.stdout


def test_analyze_command():
    """Test the analyze command with a non-existent directory."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "/nonexistent"])
    assert result.exit_code == 0
    assert "Analyzing images in:" in result.stdout
    assert "Analysis not yet implemented" in result.stdout


def test_select_command():
    """Test the select command."""
    runner = CliRunner()
    result = runner.invoke(app, ["select", "/tmp/output"])
    assert result.exit_code == 0
    assert "Selecting images from:" in result.stdout
    assert "Selection not yet implemented" in result.stdout
