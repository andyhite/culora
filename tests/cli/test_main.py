"""Tests for main CLI module."""

import tempfile

from typer.testing import CliRunner

from culora.cli.main import app


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
    assert result.exit_code == 1
    assert "Directory not found" in result.stdout


def test_select_command():
    """Test the select command."""
    runner = CliRunner()
    result = runner.invoke(app, ["select", "/tmp/output"])
    assert result.exit_code == 0
    assert "Selecting images from:" in result.stdout
    assert "Selection not yet implemented" in result.stdout


def test_no_arguments_shows_help():
    """Test that running with no arguments shows help."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    # Typer shows help when no_args_is_help=True
    assert "Usage:" in result.stdout
    assert (
        "A command-line tool for intelligently curating image datasets" in result.stdout
    )


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
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir, "--no-dedupe"])
        assert result.exit_code == 0
        assert "quality assessment" in result.stdout
        assert "face detection" in result.stdout
        assert "deduplication" not in result.stdout


def test_select_with_options():
    """Test select command with options."""
    runner = CliRunner()

    # Test with custom input directory
    result = runner.invoke(app, ["select", "/tmp/output", "--input-dir", "/tmp/custom"])
    assert result.exit_code == 0
    assert "Selecting images from: /tmp/custom" in result.stdout
    assert "Output directory: /tmp/output" in result.stdout

    # Test with dry run
    result = runner.invoke(app, ["select", "/tmp/output", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run mode" in result.stdout
    assert "no files will be copied" in result.stdout


def test_analyze_command_options():
    """Test analyze command shows enabled stages correctly."""
    runner = CliRunner()

    # Test with actual temp directories
    import tempfile

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


def test_missing_required_arguments():
    """Test commands with missing required arguments."""
    runner = CliRunner()

    # Analyze without directory should fail
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code != 0

    # Select without output directory should fail
    result = runner.invoke(app, ["select"])
    assert result.exit_code != 0
