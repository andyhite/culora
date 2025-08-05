"""Tests for quality CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from culora.cli.commands.quality import quality_app
from culora.domain.models.quality import (
    BatchQualityResult,
    ImageQualityResult,
    QualityScore,
    TechnicalQualityMetrics,
)
from tests.helpers import ConfigBuilder, ImageFixtures, TempFileHelper


class TestQualityCLI:
    """Test quality CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_single_image_success(self, mock_get_services: MagicMock) -> None:
        """Test analyzing a single image successfully."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock successful image loading
        mock_load_result = MagicMock()
        mock_load_result.success = True
        mock_load_result.image = ImageFixtures.create_test_image(256, 256)
        mock_image_service.load_image.return_value = mock_load_result

        # Mock successful quality analysis
        mock_quality_result = ImageQualityResult(
            path=Path("test.jpg"),
            success=True,
            metrics=TechnicalQualityMetrics(
                sharpness=0.8,
                brightness_score=0.7,
                contrast_score=0.9,
                color_quality=0.6,
                noise_score=0.5,
                laplacian_variance=1000.0,
                mean_brightness=0.5,
                contrast_value=0.4,
                mean_saturation=0.3,
                noise_level=25.0,
                analysis_width=256,
                analysis_height=256,
                was_resized=False,
            ),
            score=QualityScore(
                technical_score=0.75,
                overall_score=0.75,
                sharpness_contribution=0.24,
                brightness_contribution=0.14,
                contrast_contribution=0.18,
                color_contribution=0.12,
                noise_contribution=0.05,
                passes_threshold=True,
            ),
            analysis_duration=0.5,
        )
        mock_quality_service.analyze_image.return_value = mock_quality_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            test_file = temp_dir / "test.jpg"
            test_file.touch()

            result = self.runner.invoke(quality_app, ["analyze", str(test_file)])

            assert result.exit_code == 0
            assert "Quality Analysis Results" in result.stdout
            assert "0.750" in result.stdout  # Overall score
            mock_image_service.load_image.assert_called_once()
            mock_quality_service.analyze_image.assert_called_once()

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_single_image_with_details(
        self, mock_get_services: MagicMock
    ) -> None:
        """Test analyzing a single image with detailed output."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock successful responses
        mock_load_result = MagicMock()
        mock_load_result.success = True
        mock_load_result.image = ImageFixtures.create_test_image(256, 256)
        mock_image_service.load_image.return_value = mock_load_result

        mock_quality_result = ImageQualityResult(
            path=Path("test.jpg"),
            success=True,
            metrics=TechnicalQualityMetrics(
                sharpness=0.8,
                brightness_score=0.7,
                contrast_score=0.9,
                color_quality=0.6,
                noise_score=0.5,
                laplacian_variance=1000.0,
                mean_brightness=0.5,
                contrast_value=0.4,
                mean_saturation=0.3,
                noise_level=25.0,
                analysis_width=256,
                analysis_height=256,
                was_resized=False,
            ),
            score=QualityScore(
                technical_score=0.75,
                overall_score=0.75,
                sharpness_contribution=0.24,
                brightness_contribution=0.14,
                contrast_contribution=0.18,
                color_contribution=0.12,
                noise_contribution=0.05,
                passes_threshold=True,
            ),
            analysis_duration=0.5,
        )
        mock_quality_service.analyze_image.return_value = mock_quality_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            test_file = temp_dir / "test.jpg"
            test_file.touch()

            result = self.runner.invoke(
                quality_app, ["analyze", str(test_file), "--details"]
            )

            assert result.exit_code == 0
            assert "Sharpness" in result.stdout
            assert "Brightness" in result.stdout
            assert "Contrast" in result.stdout
            assert "Color Quality" in result.stdout
            assert "Noise Score" in result.stdout
            assert "Analysis time" in result.stdout

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_single_image_load_failure(
        self, mock_get_services: MagicMock
    ) -> None:
        """Test handling image load failure."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock failed image loading
        mock_load_result = MagicMock()
        mock_load_result.success = False
        mock_load_result.error = "Invalid image format"
        mock_image_service.load_image.return_value = mock_load_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            test_file = temp_dir / "test.jpg"
            test_file.touch()

            result = self.runner.invoke(quality_app, ["analyze", str(test_file)])

            assert result.exit_code == 0  # CLI doesn't exit on individual failures
            assert "Failed to load image" in result.stdout
            mock_quality_service.analyze_image.assert_not_called()

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_directory_success(self, mock_get_services: MagicMock) -> None:
        """Test analyzing a directory successfully."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock directory scan
        from culora.domain.models.image import DirectoryScanResult

        mock_scan_result = DirectoryScanResult(
            total_files=2,
            valid_images=2,
            invalid_images=0,
            supported_formats={".jpg": 2},
            total_size=1024,
            scan_duration=0.1,
            errors=[],
            image_paths=[Path("image1.jpg"), Path("image2.jpg")],
        )
        mock_image_service.scan_directory.return_value = mock_scan_result

        # Mock image loading
        mock_load_result = MagicMock()
        mock_load_result.success = True
        mock_load_result.image = ImageFixtures.create_test_image(256, 256)
        mock_image_service.load_image.return_value = mock_load_result

        # Mock batch quality analysis
        mock_batch_result = BatchQualityResult(
            results=[
                ImageQualityResult(
                    path=Path("image1.jpg"),
                    success=True,
                    score=QualityScore(
                        technical_score=0.8,
                        overall_score=0.8,
                        sharpness_contribution=0.24,
                        brightness_contribution=0.16,
                        contrast_contribution=0.2,
                        color_contribution=0.12,
                        noise_contribution=0.08,
                        passes_threshold=True,
                        quality_percentile=75.0,
                    ),
                ),
                ImageQualityResult(
                    path=Path("image2.jpg"),
                    success=True,
                    score=QualityScore(
                        technical_score=0.6,
                        overall_score=0.6,
                        sharpness_contribution=0.18,
                        brightness_contribution=0.12,
                        contrast_contribution=0.15,
                        color_contribution=0.09,
                        noise_contribution=0.06,
                        passes_threshold=True,
                        quality_percentile=25.0,
                    ),
                ),
            ],
            successful_analyses=2,
            failed_analyses=0,
            mean_quality_score=0.7,
            median_quality_score=0.7,
            quality_score_std=0.1,
            quality_score_range=(0.6, 0.8),
            total_duration=1.0,
            images_per_second=2.0,
            scores_by_percentile={25: 0.6, 50: 0.7, 75: 0.8},
            passing_threshold_count=2,
        )
        mock_quality_service.analyze_batch.return_value = mock_batch_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            result = self.runner.invoke(quality_app, ["analyze", str(temp_dir)])

            assert result.exit_code == 0
            assert "Quality analysis completed" in result.stdout
            assert "Successful: 2" in result.stdout
            assert "Mean quality: 0.700" in result.stdout
            assert "Quality Results" in result.stdout

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_directory_no_images(self, mock_get_services: MagicMock) -> None:
        """Test analyzing directory with no images."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock directory scan with no images
        from culora.domain.models.image import DirectoryScanResult

        mock_scan_result = DirectoryScanResult(
            total_files=0,
            valid_images=0,
            invalid_images=0,
            supported_formats={},
            total_size=0,
            scan_duration=0.1,
            errors=[],
            image_paths=[],
        )
        mock_image_service.scan_directory.return_value = mock_scan_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            result = self.runner.invoke(quality_app, ["analyze", str(temp_dir)])

            assert result.exit_code == 0
            assert "No valid images found" in result.stdout
            mock_quality_service.analyze_batch.assert_not_called()

    def test_analyze_nonexistent_path(self) -> None:
        """Test analyzing non-existent path."""
        result = self.runner.invoke(quality_app, ["analyze", "/nonexistent/path"])

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_with_min_score_filter(self, mock_get_services: MagicMock) -> None:
        """Test analyzing with minimum score filter."""
        # Mock services with batch result containing various scores
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock successful directory processing
        from culora.domain.models.image import DirectoryScanResult

        mock_scan_result = DirectoryScanResult(
            total_files=3,
            valid_images=3,
            invalid_images=0,
            supported_formats={".jpg": 3},
            total_size=2048,
            scan_duration=0.1,
            errors=[],
            image_paths=[Path("image1.jpg"), Path("image2.jpg"), Path("image3.jpg")],
        )
        mock_image_service.scan_directory.return_value = mock_scan_result

        mock_load_result = MagicMock()
        mock_load_result.success = True
        mock_load_result.image = ImageFixtures.create_test_image(256, 256)
        mock_image_service.load_image.return_value = mock_load_result

        # Create batch result with scores: 0.8, 0.5, 0.3
        mock_batch_result = BatchQualityResult(
            results=[
                ImageQualityResult(
                    path=Path("image1.jpg"),
                    success=True,
                    score=QualityScore(
                        technical_score=0.8,
                        overall_score=0.8,
                        sharpness_contribution=0.24,
                        brightness_contribution=0.16,
                        contrast_contribution=0.2,
                        color_contribution=0.12,
                        noise_contribution=0.08,
                        passes_threshold=True,
                    ),
                    metrics=TechnicalQualityMetrics(
                        sharpness=0.8,
                        brightness_score=0.7,
                        contrast_score=0.9,
                        color_quality=0.6,
                        noise_score=0.5,
                        laplacian_variance=1000.0,
                        mean_brightness=0.5,
                        contrast_value=0.4,
                        mean_saturation=0.3,
                        noise_level=25.0,
                        analysis_width=256,
                        analysis_height=256,
                        was_resized=False,
                    ),
                ),
                ImageQualityResult(
                    path=Path("image2.jpg"),
                    success=True,
                    score=QualityScore(
                        technical_score=0.5,
                        overall_score=0.5,
                        sharpness_contribution=0.15,
                        brightness_contribution=0.1,
                        contrast_contribution=0.125,
                        color_contribution=0.075,
                        noise_contribution=0.05,
                        passes_threshold=False,
                    ),
                    metrics=TechnicalQualityMetrics(
                        sharpness=0.5,
                        brightness_score=0.4,
                        contrast_score=0.6,
                        color_quality=0.3,
                        noise_score=0.2,
                        laplacian_variance=500.0,
                        mean_brightness=0.4,
                        contrast_value=0.3,
                        mean_saturation=0.2,
                        noise_level=50.0,
                        analysis_width=256,
                        analysis_height=256,
                        was_resized=False,
                    ),
                ),
                ImageQualityResult(
                    path=Path("image3.jpg"),
                    success=True,
                    score=QualityScore(
                        technical_score=0.3,
                        overall_score=0.3,
                        sharpness_contribution=0.09,
                        brightness_contribution=0.06,
                        contrast_contribution=0.075,
                        color_contribution=0.045,
                        noise_contribution=0.03,
                        passes_threshold=False,
                    ),
                    metrics=TechnicalQualityMetrics(
                        sharpness=0.3,
                        brightness_score=0.2,
                        contrast_score=0.4,
                        color_quality=0.1,
                        noise_score=0.1,
                        laplacian_variance=200.0,
                        mean_brightness=0.2,
                        contrast_value=0.2,
                        mean_saturation=0.1,
                        noise_level=75.0,
                        analysis_width=256,
                        analysis_height=256,
                        was_resized=False,
                    ),
                ),
            ],
            successful_analyses=3,
            failed_analyses=0,
            mean_quality_score=0.533,
            median_quality_score=0.5,
            quality_score_std=0.2,
            quality_score_range=(0.3, 0.8),
            total_duration=1.0,
            images_per_second=3.0,
            scores_by_percentile={25: 0.3, 50: 0.5, 75: 0.8},
            passing_threshold_count=1,
        )
        mock_quality_service.analyze_batch.return_value = mock_batch_result

        with TempFileHelper.create_temp_dir() as temp_dir:
            result = self.runner.invoke(
                quality_app, ["analyze", str(temp_dir), "--min-score", "0.6"]
            )

            assert result.exit_code == 0
            assert "minimum score: 0.600" in result.stdout
            # Should only show image1.jpg (score 0.8 >= 0.6)
            assert "image1.jpg" in result.stdout

    @patch("culora.services.config_service.ConfigService")
    def test_show_quality_config(self, mock_config_service_class: MagicMock) -> None:
        """Test showing quality configuration."""
        # Mock config service
        mock_config_service = MagicMock()
        mock_config_service_class.return_value = mock_config_service

        mock_config = ConfigBuilder().build()
        mock_config_service.load_config.return_value = mock_config

        result = self.runner.invoke(quality_app, ["config"])

        assert result.exit_code == 0
        assert "Quality Assessment Configuration" in result.stdout
        assert "Sharpness Weight" in result.stdout
        assert "Brightness Weight" in result.stdout
        assert "Min Quality Score" in result.stdout

    @patch("culora.cli.commands.quality.ConfigService")
    def test_show_quality_config_failure(
        self, mock_config_service_class: MagicMock
    ) -> None:
        """Test showing quality configuration with service failure."""
        # Mock config service failure
        mock_config_service_class.side_effect = Exception("Config load failed")

        result = self.runner.invoke(quality_app, ["config"])

        assert result.exit_code == 1
        assert "Failed to load configuration" in result.stdout

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_keyboard_interrupt(self, mock_get_services: MagicMock) -> None:
        """Test handling keyboard interrupt during analysis."""
        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock keyboard interrupt during directory scan
        mock_image_service.scan_directory.side_effect = KeyboardInterrupt()

        with TempFileHelper.create_temp_dir() as temp_dir:
            result = self.runner.invoke(quality_app, ["analyze", str(temp_dir)])

            assert result.exit_code == 130
            assert "interrupted by user" in result.stdout

    @patch("culora.cli.commands.quality._get_services")
    def test_analyze_service_error(self, mock_get_services: MagicMock) -> None:
        """Test handling service errors during analysis."""
        from culora.services.quality_service import QualityServiceError

        # Mock services
        mock_image_service = MagicMock()
        mock_quality_service = MagicMock()
        mock_get_services.return_value = (mock_image_service, mock_quality_service)

        # Mock service error
        mock_image_service.scan_directory.side_effect = QualityServiceError(
            "Service failed"
        )

        with TempFileHelper.create_temp_dir() as temp_dir:
            result = self.runner.invoke(quality_app, ["analyze", str(temp_dir)])

            assert result.exit_code == 1
            assert "Quality analysis failed" in result.stdout
