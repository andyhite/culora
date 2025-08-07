"""Tests for analyze command."""

import tempfile

from typer.testing import CliRunner

from culora.cli.main import app


def test_analyze_command():
    """Test the analyze command with a non-existent directory."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "/nonexistent"])
    assert result.exit_code == 1
    assert "Directory not found" in result.stdout


def test_analyze_with_stage_flags():
    """Test analyze command with stage disable flags."""
    runner = CliRunner()

    # Test with all stages disabled
    result = runner.invoke(
        app, ["analyze", "/tmp/test", "--no-dedupe", "--no-quality", "--no-face"]
    )
    assert result.exit_code == 0
    assert "No analysis stages enabled" in result.stdout

    # Test with some stages disabled - this will fail because /tmp/test doesn't exist
    # Let's use a temp directory instead
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir, "--no-dedupe"])
        assert result.exit_code == 0
        assert "quality assessment" in result.stdout
        assert "face detection" in result.stdout
        assert "deduplication" not in result.stdout


def test_analyze_command_options():
    """Test analyze command shows enabled stages correctly."""
    runner = CliRunner()

    # Test with actual temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test default (all stages enabled)
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        assert "deduplication" in result.stdout
        assert "quality assessment" in result.stdout
        assert "face detection" in result.stdout

        # Test individual stage disabling
        result = runner.invoke(app, ["analyze", temp_dir, "--no-quality"])
        assert result.exit_code == 0
        assert "deduplication" in result.stdout
        assert "face detection" in result.stdout
        assert "quality assessment" not in result.stdout


def test_analyze_with_real_directory():
    """Test analyze command with a real temporary directory."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        # Rich may format output with newlines, so check for directory path anywhere in output
        assert temp_dir in result.stdout
        assert "Analyzing images in:" in result.stdout
        assert "Analysis Complete!" in result.stdout
