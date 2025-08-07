"""Tests for select command."""

import tempfile
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from culora.cli.main import app
from culora.models.analysis import (
    AnalysisResult,
    AnalysisStage,
    DirectoryAnalysis,
    ImageAnalysis,
    StageResult,
)
from culora.utils.cache import save_analysis_cache


def test_select_command():
    """Test the select command with no analysis cache."""
    runner = CliRunner()
    result = runner.invoke(app, ["select", "/tmp/output"])
    assert result.exit_code == 1
    assert "Selecting images from:" in result.stdout
    assert "No analysis results found" in result.stdout


def test_select_with_options():
    """Test select command with options."""
    runner = CliRunner()

    # Test with custom input directory (will fail - no analysis cache)
    result = runner.invoke(app, ["select", "/tmp/output", "--input-dir", "/tmp/custom"])
    assert result.exit_code == 1
    assert "Selecting images from:" in result.stdout
    assert "Output directory:" in result.stdout
    assert "No analysis results found" in result.stdout

    # Test with dry run (will also fail - no analysis cache)
    result = runner.invoke(app, ["select", "/tmp/output", "--dry-run"])
    assert result.exit_code == 1
    assert "Dry run mode" in result.stdout
    assert "No analysis results found" in result.stdout


def test_select_with_analysis_data():
    """Test select command with actual analysis data."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"

        # Create input directory and some fake image files
        input_dir.mkdir()
        image1 = input_dir / "image1.jpg"
        image2 = input_dir / "image2.jpg"
        image3 = input_dir / "image3.jpg"

        # Create fake image files
        image1.write_text("fake image 1")
        image2.write_text("fake image 2")
        image3.write_text("fake image 3")

        # Create mock analysis data
        analysis = DirectoryAnalysis(
            input_directory=str(input_dir),
            analysis_time=datetime.now(),
            enabled_stages=[AnalysisStage.DEDUPLICATION, AnalysisStage.QUALITY],
            images=[
                # Image that passes all stages
                ImageAnalysis(
                    file_path=str(image1),
                    file_size=100,
                    modified_time=datetime.now(),
                    stage_results=[
                        StageResult(
                            stage=AnalysisStage.DEDUPLICATION,
                            result=AnalysisResult.PASS,
                        ),
                        StageResult(
                            stage=AnalysisStage.QUALITY, result=AnalysisResult.PASS
                        ),
                    ],
                ),
                # Image that fails quality stage
                ImageAnalysis(
                    file_path=str(image2),
                    file_size=100,
                    modified_time=datetime.now(),
                    stage_results=[
                        StageResult(
                            stage=AnalysisStage.DEDUPLICATION,
                            result=AnalysisResult.PASS,
                        ),
                        StageResult(
                            stage=AnalysisStage.QUALITY,
                            result=AnalysisResult.FAIL,
                            reason="Too blurry",
                        ),
                    ],
                ),
                # Image that passes all stages
                ImageAnalysis(
                    file_path=str(image3),
                    file_size=100,
                    modified_time=datetime.now(),
                    stage_results=[
                        StageResult(
                            stage=AnalysisStage.DEDUPLICATION,
                            result=AnalysisResult.PASS,
                        ),
                        StageResult(
                            stage=AnalysisStage.QUALITY, result=AnalysisResult.PASS
                        ),
                    ],
                ),
            ],
        )

        # Save analysis cache
        save_analysis_cache(analysis)

        # Test dry run
        result = runner.invoke(
            app, ["select", str(output_dir), "--input-dir", str(input_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Dry run mode" in result.stdout
        assert "Would copy: image1.jpg → 001.jpg" in result.stdout
        assert "Would copy: image3.jpg → 002.jpg" in result.stdout
        assert "2 selected for training" in result.stdout
        assert "1 skipped" in result.stdout
        assert "Selection Complete!" in result.stdout

        # Test actual copy
        result = runner.invoke(
            app, ["select", str(output_dir), "--input-dir", str(input_dir)]
        )
        assert result.exit_code == 0
        assert "Successfully copied 2 images" in result.stdout
        assert "Selection Complete!" in result.stdout

        # Verify files were copied and renamed
        copied_files = list(output_dir.glob("*.jpg"))
        assert len(copied_files) == 2
        assert (output_dir / "001.jpg").exists()
        assert (output_dir / "002.jpg").exists()

        # Verify content was copied correctly (image1 -> 001.jpg, image3 -> 002.jpg)
        # Since images are sorted alphabetically, image1.jpg comes before image3.jpg
        assert (output_dir / "001.jpg").read_text() == "fake image 1"
        assert (output_dir / "002.jpg").read_text() == "fake image 3"


def test_missing_required_output_directory():
    """Test select command with missing required output directory argument."""
    runner = CliRunner()

    # Select without output directory should fail
    result = runner.invoke(app, ["select"])
    assert result.exit_code != 0
