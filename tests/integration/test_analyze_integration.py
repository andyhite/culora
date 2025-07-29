"""Simplified integration tests for the analyze command."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from culora.main import app


class TestAnalyzeIntegration:
    """Simplified integration tests for the analyze command."""

    def _create_mock_directory_analysis(self, temp_dir: Path):
        """Helper to create a proper DirectoryAnalysis mock."""
        from datetime import datetime

        from culora.config import CuLoRAConfig
        from culora.models.directory_analysis import DirectoryAnalysis

        config = CuLoRAConfig()
        return DirectoryAnalysis(
            input_directory=str(temp_dir),
            analysis_time=datetime.now(),
            analysis_config=config,
            images=[],
        )

    def test_analyze_basic_success(
        self,
        cli_runner: CliRunner,
        temp_image_dir: Path,
    ):
        """Test basic analyze command with default settings."""
        with patch("culora.commands.analyze.ImageAnalyzer") as mock_analyzer_class:
            mock_analyzer_instance = mock_analyzer_class.return_value
            mock_analysis_result = self._create_mock_directory_analysis(temp_image_dir)
            mock_analyzer_instance.analyze_directory.return_value = mock_analysis_result

            result = cli_runner.invoke(app, ["analyze", str(temp_image_dir)])
            assert result.exit_code == 0

    def test_analyze_with_output_directory(
        self,
        cli_runner: CliRunner,
        temp_image_dir: Path,
        temp_output_dir: Path,
    ):
        """Test analyze command with automatic output selection."""
        with (
            patch("culora.commands.analyze.ImageAnalyzer") as mock_analyzer_class,
            patch("culora.commands.analyze.ImageCurator") as mock_curator_class,
        ):

            mock_analyzer_instance = mock_analyzer_class.return_value
            mock_curator_instance = mock_curator_class.return_value

            mock_analysis_result = self._create_mock_directory_analysis(temp_image_dir)
            mock_analyzer_instance.analyze_directory.return_value = mock_analysis_result
            mock_curator_instance.select_images.return_value = (0, 0)

            result = cli_runner.invoke(
                app, ["analyze", str(temp_image_dir), "--output", str(temp_output_dir)]
            )

            assert result.exit_code == 0
            mock_analyzer_instance.analyze_directory.assert_called_once()
            mock_curator_instance.select_images.assert_called_once()

    def test_analyze_with_disabled_stages(
        self,
        cli_runner: CliRunner,
        temp_image_dir: Path,
    ):
        """Test analyze command with individual stages disabled."""
        with patch("culora.commands.analyze.ImageAnalyzer") as mock_analyzer_class:
            mock_analyzer_instance = mock_analyzer_class.return_value
            mock_analysis_result = self._create_mock_directory_analysis(temp_image_dir)
            mock_analyzer_instance.analyze_directory.return_value = mock_analysis_result

            # Test with deduplication disabled
            result = cli_runner.invoke(
                app, ["analyze", str(temp_image_dir), "--no-dedupe"]
            )
            assert result.exit_code == 0

    def test_analyze_invalid_directory(self, cli_runner: CliRunner):
        """Test analyze command with non-existent directory."""
        result = cli_runner.invoke(app, ["analyze", "/nonexistent/directory"])
        assert result.exit_code != 0

    def test_analyze_empty_directory(self, cli_runner: CliRunner):
        """Test analyze command with directory containing no images."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with patch("culora.commands.analyze.ImageAnalyzer") as mock_analyzer_class:
                mock_analyzer_instance = mock_analyzer_class.return_value
                mock_analysis_result = self._create_mock_directory_analysis(
                    Path(empty_dir)
                )
                mock_analyzer_instance.analyze_directory.return_value = (
                    mock_analysis_result
                )

                result = cli_runner.invoke(app, ["analyze", empty_dir])
                assert result.exit_code == 0

    def test_analyze_error_handling(self, cli_runner: CliRunner, temp_image_dir: Path):
        """Test error handling in analysis pipeline."""
        with patch("culora.commands.analyze.ImageAnalyzer") as mock_analyzer_class:
            mock_analyzer_instance = mock_analyzer_class.return_value
            mock_analyzer_instance.analyze_directory.side_effect = RuntimeError(
                "Mock error"
            )

            result = cli_runner.invoke(app, ["analyze", str(temp_image_dir)])
            assert result.exit_code != 0

    # Test CLI help and version commands
    def test_version_command(self, cli_runner: CliRunner):
        """Test version command works."""
        result = cli_runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_help_command(self, cli_runner: CliRunner):
        """Test help command works."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output

    def test_analyze_help(self, cli_runner: CliRunner):
        """Test analyze help command works."""
        result = cli_runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "input-dir" in result.output or "Directory" in result.output
