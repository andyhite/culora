"""Tests for analyze command."""

import tempfile

from typer.testing import CliRunner

from culora.main import app


def test_analyze_command():
    """Test the analyze command with a non-existent directory."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "/nonexistent"])
    assert result.exit_code == 1
    assert "Directory not found" in result.stdout


def test_analyze_with_stage_flags():
    """Test analyze command with stage disable flags."""
    runner = CliRunner()

    # Test with some stages disabled - use a temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir, "--no-dedupe"])
        assert result.exit_code == 0
        # Check that analysis ran successfully even with some stages disabled
        assert "Image Analysis Results" in result.stdout


def test_analyze_command_options():
    """Test analyze command runs successfully with different options."""
    runner = CliRunner()

    # Test with actual temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test default (all stages enabled)
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        assert "Image Analysis Results" in result.stdout

        # Test individual stage disabling
        result = runner.invoke(app, ["analyze", temp_dir, "--no-quality"])
        assert result.exit_code == 0
        assert "Image Analysis Results" in result.stdout


def test_analyze_with_real_directory():
    """Test analyze command with a real temporary directory."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        # Analysis results should be shown
        assert "Image Analysis Results" in result.stdout


def test_analyze_with_output_option():
    """Test analyze command with --output option shows correct behavior."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_input_dir:
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # Test with output directory
            result = runner.invoke(
                app, ["analyze", temp_input_dir, "--output", temp_output_dir]
            )
            # For debugging: print output if test fails
            if result.exit_code != 0:
                print(f"Command output: {result.stdout}")
                print(f"Exit code: {result.exit_code}")
            assert result.exit_code == 0
            assert "Image Analysis Results" in result.stdout
            assert "Selecting curated images..." in result.stdout


def test_analyze_with_draw_boxes_without_output():
    """Test that --draw-boxes without --output shows an error."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir, "--draw-boxes"])
        assert result.exit_code == 1
        assert "--draw-boxes can only be used with --output" in result.stdout
