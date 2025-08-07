"""Tests for analyze command."""

import tempfile

from typer.testing import CliRunner

from culora.cli.analyze import format_face_result
from culora.cli.main import app
from culora.models.analysis import AnalysisResult, AnalysisStage, StageResult


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
        assert "Analyzing images in:" in result.stdout


def test_analyze_command_options():
    """Test analyze command runs successfully with different options."""
    runner = CliRunner()

    # Test with actual temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test default (all stages enabled)
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        assert "Analyzing images in:" in result.stdout

        # Test individual stage disabling
        result = runner.invoke(app, ["analyze", temp_dir, "--no-quality"])
        assert result.exit_code == 0
        assert "Analyzing images in:" in result.stdout


def test_analyze_with_real_directory():
    """Test analyze command with a real temporary directory."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir])
        assert result.exit_code == 0
        # Rich may format output with newlines, so check for directory path anywhere in output
        assert temp_dir in result.stdout
        assert "Analyzing images in:" in result.stdout


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
            assert "Analyzing images in:" in result.stdout
            assert "Selecting curated images..." in result.stdout


def test_analyze_with_draw_boxes_without_output():
    """Test that --draw-boxes without --output shows an error."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["analyze", temp_dir, "--draw-boxes"])
        assert result.exit_code == 1
        assert "--draw-boxes can only be used with --output" in result.stdout


def test_format_face_result_pass_single():
    """Test face result formatting for single face detection (pass)."""
    result = StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.PASS,
        reason="Detected 1 face with confidence 0.850",
        metadata={
            "face_count": "1",
            "highest_confidence": "0.850",
            "confidence_threshold": "0.500",
        },
    )
    formatted = format_face_result(result)
    assert formatted == "[green]1 (85.00%)[/green]"


def test_format_face_result_pass_multiple():
    """Test face result formatting for multiple face detection (pass)."""
    result = StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.PASS,
        reason="Detected 2 faces (best: 0.920, avg: 0.885)",
        metadata={
            "face_count": "2",
            "highest_confidence": "0.920",
            "confidence_threshold": "0.500",
        },
    )
    formatted = format_face_result(result)
    assert formatted == "[green]2 (92.00%)[/green]"


def test_format_face_result_fail_no_faces():
    """Test face result formatting for no faces detected (fail)."""
    result = StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.FAIL,
        reason="No faces detected in image",
        metadata={
            "face_count": "0",
            "confidence_scores": "",
            "average_confidence": "0.000",
        },
    )
    formatted = format_face_result(result)
    assert formatted == "[red]none[/red]"


def test_format_face_result_fail_low_confidence():
    """Test face result formatting for faces detected but confidence too low (fail)."""
    result = StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.FAIL,
        reason="Detected 2 face(s) but highest confidence 0.400 below threshold 0.500",
        metadata={
            "face_count": "2",
            "highest_confidence": "0.400",
            "confidence_threshold": "0.500",
        },
    )
    formatted = format_face_result(result)
    assert formatted == "[red]2 (40.00%)[/red]"


def test_format_face_result_skip():
    """Test face result formatting for skipped analysis."""
    result = StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.SKIP,
        reason="Analysis skipped",
        metadata={},
    )
    formatted = format_face_result(result)
    assert formatted == "[yellow]skip[/yellow]"
