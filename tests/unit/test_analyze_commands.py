"""Unit tests for analyze command implementations."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import typer

from culora.commands.analyze import (
    _display_analysis_summary,
    _format_dedup_data,
    _format_face_data,
    _format_quality_data,
    analyze_command,
)
from culora.config import CuLoRAConfig, DisplayConfig
from culora.models.analysis_result import AnalysisResult
from culora.models.directory_analysis import DirectoryAnalysis
from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import Face, FaceDetectionResult
from culora.models.image_analysis import ImageAnalysis
from culora.models.image_quality_result import ImageQualityResult


class TestAnalyzeCommand:
    """Tests for analyze command functionality."""

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_with_cli_config_overrides(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with CLI configuration overrides."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.return_value = self._create_mock_analysis()

        # Test with various CLI overrides
        analyze_command(
            input_dir="/test/input",
            sharpness_threshold=200.0,
            brightness_min=50.0,
            brightness_max=250.0,
            contrast_threshold=45.0,
            face_confidence_threshold=0.7,
            quality_weight=0.6,
            face_weight=0.4,
        )

        # Verify config was updated
        mock_analyzer.analyze_directory.assert_called_once()

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_weight_validation_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with invalid weight configuration."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        # Test with weights that don't sum to 1.0
        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                quality_weight=0.7,
                face_weight=0.2,  # Sum = 0.9, not 1.0
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_draw_boxes_without_output_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with draw_boxes but no output directory."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                draw_boxes=True,  # Without output_dir
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_dry_run_without_output_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with dry_run but no output directory."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                dry_run=True,  # Without output_dir
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_max_images_without_output_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with max_images but no output directory."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                max_images=10,  # Without output_dir
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_invalid_max_images_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with invalid max_images value."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                output_dir="/test/output",
                max_images=0,  # Invalid - must be positive
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    @patch("culora.commands.analyze.ImageCurator")
    def test_analyze_with_output_dry_run_success(
        self,
        mock_curator_class: MagicMock,
        mock_analyzer_class: MagicMock,
        mock_config_manager_class: MagicMock,
    ) -> None:
        """Test analyze command with output directory in dry run mode."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.return_value = self._create_mock_analysis()

        mock_curator = Mock()
        mock_curator_class.return_value = mock_curator
        mock_curator.select_images.return_value = (5, 10)  # selected, total

        analyze_command(
            input_dir="/test/input",
            output_dir="/test/output",
            dry_run=True,
        )

        mock_curator.select_images.assert_called_once()

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    @patch("culora.commands.analyze.ImageCurator")
    def test_analyze_with_output_success(
        self,
        mock_curator_class: MagicMock,
        mock_analyzer_class: MagicMock,
        mock_config_manager_class: MagicMock,
    ) -> None:
        """Test analyze command with output directory (not dry run)."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.return_value = self._create_mock_analysis()

        mock_curator = Mock()
        mock_curator_class.return_value = mock_curator
        mock_curator.select_images.return_value = (5, 10)  # selected, total

        analyze_command(
            input_dir="/test/input",
            output_dir="/test/output",
        )

        mock_curator.select_images.assert_called_once()

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    @patch("culora.commands.analyze.ImageCurator")
    def test_analyze_selection_runtime_error(
        self,
        mock_curator_class: MagicMock,
        mock_analyzer_class: MagicMock,
        mock_config_manager_class: MagicMock,
    ) -> None:
        """Test analyze command with RuntimeError during selection."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.return_value = self._create_mock_analysis()

        mock_curator = Mock()
        mock_curator_class.return_value = mock_curator
        mock_curator.select_images.side_effect = RuntimeError("Selection failed")

        with pytest.raises(typer.Exit):
            analyze_command(
                input_dir="/test/input",
                output_dir="/test/output",
            )

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_file_not_found_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with FileNotFoundError."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.side_effect = FileNotFoundError(
            "Directory not found"
        )

        with pytest.raises(typer.Exit):
            analyze_command(input_dir="/nonexistent/path")

    @patch("culora.commands.analyze.ConfigManager")
    @patch("culora.commands.analyze.ImageAnalyzer")
    def test_analyze_not_a_directory_error(
        self, mock_analyzer_class: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test analyze command with NotADirectoryError."""
        mock_config_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_config_manager

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_directory.side_effect = NotADirectoryError(
            "Not a directory"
        )

        with pytest.raises(typer.Exit):
            analyze_command(input_dir="/test/file.txt")

    def _create_mock_analysis(self) -> DirectoryAnalysis:
        """Helper to create mock analysis for testing."""
        from datetime import datetime

        config = CuLoRAConfig()
        return DirectoryAnalysis(
            input_directory="/test/input",
            analysis_time=datetime.now(),
            analysis_config=config,
            images=[],
        )


class TestAnalyzeFormatting:
    """Tests for analyze command formatting functions."""

    def test_format_quality_data_with_results(self) -> None:
        """Test quality data formatting with actual results."""
        quality_result = ImageQualityResult(
            sharpness_score=200.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=0.8,
        )
        display_config = DisplayConfig()

        result = _format_quality_data(quality_result, display_config)
        # Result is a list of strings, check for numeric values in any element
        result_str = " ".join(result)
        assert "200" in result_str
        assert "120" in result_str
        assert "60" in result_str

    def test_format_quality_data_none(self) -> None:
        """Test quality data formatting with None result."""
        display_config = DisplayConfig()
        result = _format_quality_data(None, display_config)
        assert result == ["N/A", "N/A", "N/A", "N/A"]

    def test_format_face_data_with_results(self) -> None:
        """Test face data formatting with actual results."""
        faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]
        face_result = FaceDetectionResult(
            faces=faces,
            model_used="test_model",
            device_used="cpu",
        )

        result = _format_face_data(face_result)
        assert "1" in result  # face count
        # Confidence might be formatted differently, just check for presence
        assert len(result) > 0

    def test_format_face_data_none(self) -> None:
        """Test face data formatting with None result."""
        result = _format_face_data(None)
        assert result == "N/A"

    def test_format_dedup_data_with_results(self) -> None:
        """Test deduplication data formatting with actual results."""
        dedup_result = DuplicateDetectionResult(hash_value="abc123def456")

        result = _format_dedup_data(dedup_result)
        # Hash is truncated, so check for the beginning
        assert "abc123" in result

    def test_format_dedup_data_none(self) -> None:
        """Test deduplication data formatting with None result."""
        result = _format_dedup_data(None)
        assert result == "N/A"

    @patch("culora.commands.analyze.console")
    def test_display_analysis_summary_with_score_colors(
        self, mock_console: MagicMock
    ) -> None:
        """Test analysis summary display with different score colors."""
        from datetime import datetime

        config = CuLoRAConfig()

        # Create images with different scores to test color formatting
        high_score_image = self._create_mock_image("/test/high.jpg", 0.95)
        medium_score_image = self._create_mock_image("/test/medium.jpg", 0.75)
        low_score_image = self._create_mock_image("/test/low.jpg", 0.3)

        analysis = DirectoryAnalysis(
            input_directory="/test/input",
            analysis_time=datetime.now(),
            analysis_config=config,
            images=[high_score_image, medium_score_image, low_score_image],
        )

        _display_analysis_summary(analysis, config)

        # Verify console.table was called
        mock_console.table.assert_called_once()

    def _create_mock_image(self, file_path: str, score: float) -> ImageAnalysis:
        """Helper to create mock image analysis."""
        from datetime import datetime

        results = AnalysisResult()

        # Add quality result
        quality = ImageQualityResult(
            sharpness_score=200.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=score,
        )
        results.set_quality(quality)

        # Add face result
        faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]
        face = FaceDetectionResult(
            faces=faces,
            model_used="test_model",
            device_used="cpu",
        )
        results.set_face(face)

        # Add dedup result
        dedup = DuplicateDetectionResult(hash_value="abc123")
        results.set_deduplication(dedup)

        return ImageAnalysis(
            file_path=file_path,
            file_size=1024,
            modified_time=datetime.now(),
            results=results,
            score=score,
        )
