"""Integration tests for the analysis pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from culora.config import AnalysisConfig
from culora.managers.config_manager import ConfigManager
from culora.models.directory_analysis import DirectoryAnalysis
from culora.orchestrators.image_analyzer import ImageAnalyzer
from culora.orchestrators.image_curator import ImageCurator


class TestAnalysisPipelineIntegration:
    """Integration tests for the analysis pipeline orchestration."""

    def test_analyzer_initialization(self) -> None:
        """Test that the ImageAnalyzer initializes correctly."""
        config = AnalysisConfig()
        config_manager = ConfigManager.get_instance()
        config_manager._analysis_config = config  # type: ignore[attr-defined]

        analyzer = ImageAnalyzer(config_manager)

        # Check that the analyzer was created
        assert analyzer is not None

    def test_curator_initialization(self) -> None:
        """Test that the ImageCurator initializes correctly."""
        config = AnalysisConfig()
        config_manager = ConfigManager.get_instance()
        config_manager._analysis_config = config  # type: ignore[attr-defined]

        curator = ImageCurator(config_manager)

        # Check that the curator was created
        assert curator is not None

    @patch("culora.orchestrators.image_analyzer.ImageAnalyzer.analyze_directory")
    def test_analyzer_directory_analysis(
        self, mock_analyze: Any, tmp_path: Path
    ) -> None:
        """Test directory analysis through ImageAnalyzer."""
        # Create temporary directory
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()

        # Create config
        config = AnalysisConfig()
        config_manager = ConfigManager.get_instance()
        config_manager._analysis_config = config  # type: ignore[attr-defined]

        analyzer = ImageAnalyzer(config_manager)

        # Mock the analyzer response
        mock_analysis = DirectoryAnalysis(
            input_directory=str(test_dir),
            analysis_time=datetime.now(),
            analysis_config=config,
            images=[],
        )
        mock_analyze.return_value = mock_analysis

        # Test
        result = analyzer.analyze_directory(test_dir)

        # Assertions
        assert result is not None
        assert result.input_directory == str(test_dir)
        mock_analyze.assert_called_once_with(test_dir)
