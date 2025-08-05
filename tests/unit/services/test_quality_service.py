"""Tests for QualityService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from culora.domain.models.config.quality import QualityConfig
from culora.services.quality_service import (
    QualityService,
    get_quality_service,
)
from tests.helpers import ConfigBuilder, ImageFixtures, TempFileHelper


class TestQualityService:
    """Test QualityService functionality."""

    def test_quality_service_initialization(self) -> None:
        """Test QualityService initialization."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        assert service.config == config
        assert service.quality_config == config.quality
        assert isinstance(service.quality_config, QualityConfig)

    def test_get_quality_service(self) -> None:
        """Test getting quality service singleton."""
        # Reset singleton
        import culora.services.quality_service

        culora.services.quality_service._quality_service = None

        config = ConfigBuilder().build()
        service1 = get_quality_service(config)
        service2 = get_quality_service()

        assert isinstance(service1, QualityService)
        assert service1 is service2  # Should be singleton

    @patch("cv2.Laplacian")
    @patch("cv2.cvtColor")
    def test_analyze_image_success(
        self, mock_cvtcolor: MagicMock, mock_laplacian: MagicMock
    ) -> None:
        """Test successful image quality analysis."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Mock CV2 operations
        mock_laplacian.return_value = np.array([[100, 200], [150, 180]])
        mock_cvtcolor.return_value = np.array([[[128, 100, 50]]])

        with TempFileHelper.create_temp_dir() as temp_dir:
            image = ImageFixtures.create_test_image(256, 256)
            image_path = temp_dir / "test.jpg"

            result = service.analyze_image(image, image_path)

            assert result.success is True
            assert result.path == image_path
            assert result.metrics is not None
            assert result.score is not None
            assert result.analysis_duration > 0

            # Check metrics are in valid ranges
            metrics = result.metrics
            assert 0.0 <= metrics.sharpness <= 1.0
            assert 0.0 <= metrics.brightness_score <= 1.0
            assert 0.0 <= metrics.contrast_score <= 1.0
            assert 0.0 <= metrics.color_quality <= 1.0
            assert 0.0 <= metrics.noise_score <= 1.0

            # Check score components
            score = result.score
            assert 0.0 <= score.technical_score <= 1.0
            assert 0.0 <= score.overall_score <= 1.0
            # passes_threshold should be a boolean
            assert score.passes_threshold in [True, False]

    def test_analyze_image_failure(self) -> None:
        """Test image quality analysis failure handling."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            image_path = temp_dir / "test.jpg"

            # Create invalid image that will cause processing to fail
            with patch.object(
                service,
                "_prepare_image_for_analysis",
                side_effect=Exception("Mock error"),
            ):
                result = service.analyze_image(Image.new("RGB", (100, 100)), image_path)

            assert result.success is False
            assert result.error is not None
            assert "Mock error" in result.error
            assert result.error_code == "QUALITY_ANALYSIS_FAILED"
            assert result.metrics is None
            assert result.score is None

    @patch("cv2.Laplacian")
    @patch("cv2.cvtColor")
    def test_analyze_batch(
        self, mock_cvtcolor: MagicMock, mock_laplacian: MagicMock
    ) -> None:
        """Test batch quality analysis."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Mock CV2 operations with varying results
        mock_laplacian.return_value = np.array([[100, 200], [150, 180]])
        mock_cvtcolor.return_value = np.array([[[128, 100, 50]]])

        with TempFileHelper.create_temp_dir() as temp_dir:
            # Create test images with paths
            images_and_paths = []
            for i in range(3):
                image = ImageFixtures.create_test_image(256, 256)
                path = temp_dir / f"test_{i}.jpg"
                images_and_paths.append((image, path))

            result = service.analyze_batch(images_and_paths)

            assert len(result.results) == 3
            assert result.successful_analyses == 3
            assert result.failed_analyses == 0
            assert result.mean_quality_score > 0
            assert result.median_quality_score > 0
            assert result.total_duration > 0
            assert result.images_per_second > 0
            assert len(result.scores_by_percentile) > 0

    def test_analyze_batch_empty(self) -> None:
        """Test batch analysis with empty input."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        result = service.analyze_batch([])

        assert len(result.results) == 0
        assert result.successful_analyses == 0
        assert result.failed_analyses == 0
        assert result.mean_quality_score == 0.0
        assert result.median_quality_score == 0.0

    def test_prepare_image_for_analysis_no_resize(self) -> None:
        """Test image preparation without resizing."""
        config = (
            ConfigBuilder()
            .with_quality_config(QualityConfig(resize_for_analysis=False))
            .build()
        )
        service = QualityService(config)

        image = ImageFixtures.create_test_image(256, 256)
        prepared_image, was_resized = service._prepare_image_for_analysis(image)

        assert prepared_image is image
        assert was_resized is False

    def test_prepare_image_for_analysis_with_resize(self) -> None:
        """Test image preparation with resizing."""
        config = (
            ConfigBuilder()
            .with_quality_config(
                QualityConfig(resize_for_analysis=True, max_analysis_size=(128, 128))
            )
            .build()
        )
        service = QualityService(config)

        # Create large image that needs resizing
        image = ImageFixtures.create_test_image(512, 512)
        prepared_image, was_resized = service._prepare_image_for_analysis(image)

        assert prepared_image is not image
        assert was_resized is True
        assert max(prepared_image.size) <= 128

    def test_prepare_image_for_analysis_no_resize_needed(self) -> None:
        """Test image preparation when no resize is needed."""
        config = (
            ConfigBuilder()
            .with_quality_config(
                QualityConfig(resize_for_analysis=True, max_analysis_size=(512, 512))
            )
            .build()
        )
        service = QualityService(config)

        # Create small image that doesn't need resizing
        image = ImageFixtures.create_test_image(256, 256)
        prepared_image, was_resized = service._prepare_image_for_analysis(image)

        assert prepared_image is image
        assert was_resized is False

    @patch("cv2.Laplacian")
    def test_calculate_sharpness(self, mock_laplacian: MagicMock) -> None:
        """Test sharpness calculation."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Mock Laplacian with known variance
        mock_laplacian.return_value = np.array(
            [[100, 200], [150, 180]], dtype=np.float64
        )

        gray_array = np.array([[128, 129], [127, 130]], dtype=np.uint8)
        sharpness = service._calculate_sharpness(gray_array)

        # Check that sharpness is calculated (exact value depends on mock return)
        assert isinstance(sharpness, float)
        assert sharpness > 0
        mock_laplacian.assert_called_once()

    def test_normalize_sharpness(self) -> None:
        """Test sharpness normalization."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Test various sharpness values
        assert service._normalize_sharpness(0) == 0.0  # Below minimum
        assert service._normalize_sharpness(50) == 0.0  # At minimum
        assert service._normalize_sharpness(1025) == 0.5  # Middle
        assert service._normalize_sharpness(2000) == 1.0  # At maximum
        assert service._normalize_sharpness(3000) == 1.0  # Above maximum (clamped)

    def test_score_brightness(self) -> None:
        """Test brightness scoring."""
        config = (
            ConfigBuilder()
            .with_quality_config(QualityConfig(optimal_brightness_range=(0.3, 0.7)))
            .build()
        )
        service = QualityService(config)

        # Test optimal range
        assert service._score_brightness(0.5) == 1.0
        assert service._score_brightness(0.3) == 1.0
        assert service._score_brightness(0.7) == 1.0

        # Test too dark
        assert service._score_brightness(0.15) == 0.5
        assert service._score_brightness(0.0) == 0.0

        # Test too bright
        assert service._score_brightness(0.85) == 0.5
        assert service._score_brightness(1.0) == 0.0

    def test_score_contrast(self) -> None:
        """Test contrast scoring."""
        config = (
            ConfigBuilder()
            .with_quality_config(QualityConfig(high_contrast_threshold=0.4))
            .build()
        )
        service = QualityService(config)

        assert service._score_contrast(0.0) == 0.0
        assert service._score_contrast(0.2) == 0.5
        assert service._score_contrast(0.4) == 1.0
        assert service._score_contrast(0.8) == 1.0  # Clamped at 1.0

    @patch("cv2.cvtColor")
    def test_calculate_color_quality(self, mock_cvtcolor: MagicMock) -> None:
        """Test color quality calculation."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Mock HSV conversion - saturation channel (index 1)
        mock_cvtcolor.return_value = np.array([[[0, 128, 255], [0, 64, 255]]])

        rgb_array = np.array([])  # Unused due to mock
        mean_saturation = service._calculate_color_quality(rgb_array)

        # Mean of [128, 64] / 255 = 96/255 ≈ 0.376
        expected = (128 + 64) / 2 / 255
        assert abs(mean_saturation - expected) < 0.001

    def test_score_color_quality(self) -> None:
        """Test color quality scoring."""
        config = (
            ConfigBuilder()
            .with_quality_config(QualityConfig(min_saturation=0.1, max_saturation=0.8))
            .build()
        )
        service = QualityService(config)

        # Test good range
        assert service._score_color_quality(0.5) == 1.0
        assert service._score_color_quality(0.1) == 1.0
        assert service._score_color_quality(0.8) == 1.0

        # Test too low
        assert service._score_color_quality(0.05) == 0.5
        assert service._score_color_quality(0.0) == 0.0

        # Test too high (oversaturated)
        assert service._score_color_quality(0.9) == 0.5
        assert service._score_color_quality(1.0) == 0.0

    @patch("cv2.Laplacian")
    def test_calculate_noise_level(self, mock_laplacian: MagicMock) -> None:
        """Test noise level calculation."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Mock Laplacian with known standard deviation
        test_data = np.array([[10, 20], [15, 25]], dtype=np.float64)
        mock_laplacian.return_value = test_data

        gray_array = np.array([])  # Unused due to mock
        noise_level = service._calculate_noise_level(gray_array)

        expected_std = float(np.std(test_data))
        assert abs(noise_level - expected_std) < 0.001

    def test_score_noise_level(self) -> None:
        """Test noise level scoring."""
        config = (
            ConfigBuilder()
            .with_quality_config(QualityConfig(noise_threshold=50.0))
            .build()
        )
        service = QualityService(config)

        # Lower noise = higher score
        assert service._score_noise_level(0.0) == 1.0
        assert service._score_noise_level(25.0) == 0.5
        assert service._score_noise_level(50.0) == 0.0
        assert service._score_noise_level(100.0) == 0.0  # Clamped at 0.0

    @patch("cv2.Laplacian")
    @patch("cv2.cvtColor")
    def test_calculate_technical_metrics(
        self, mock_cvtcolor: MagicMock, mock_laplacian: MagicMock
    ) -> None:
        """Test complete technical metrics calculation."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Setup mocks
        mock_laplacian.side_effect = [
            np.array([[100, 200]], dtype=np.float64),  # For sharpness
            np.array([[10, 20]], dtype=np.float64),  # For noise
        ]
        mock_cvtcolor.return_value = np.array([[[0, 128, 255]]])

        image = ImageFixtures.create_test_image(100, 100)
        metrics = service._calculate_technical_metrics(image, was_resized=False)

        assert isinstance(metrics.sharpness, float)
        assert isinstance(metrics.brightness_score, float)
        assert isinstance(metrics.contrast_score, float)
        assert isinstance(metrics.color_quality, float)
        assert isinstance(metrics.noise_score, float)
        assert metrics.analysis_width == 100
        assert metrics.analysis_height == 100
        assert metrics.was_resized is False

    def test_calculate_quality_score(self) -> None:
        """Test quality score calculation."""
        config = (
            ConfigBuilder()
            .with_quality_config(
                QualityConfig(
                    sharpness_weight=0.3,
                    brightness_weight=0.2,
                    contrast_weight=0.2,
                    color_weight=0.2,
                    noise_weight=0.1,
                    min_quality_score=0.5,
                )
            )
            .build()
        )
        service = QualityService(config)

        # Create mock metrics
        from culora.domain.models.quality import TechnicalQualityMetrics

        metrics = TechnicalQualityMetrics(
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
        )

        score = service._calculate_quality_score(metrics)

        # Calculate expected score: 0.8*0.3 + 0.7*0.2 + 0.9*0.2 + 0.6*0.2 + 0.5*0.1
        expected = 0.8 * 0.3 + 0.7 * 0.2 + 0.9 * 0.2 + 0.6 * 0.2 + 0.5 * 0.1
        assert abs(score.technical_score - expected) < 0.001
        assert score.overall_score == score.technical_score
        assert score.passes_threshold == (expected >= 0.5)

    def test_calculate_batch_statistics_empty(self) -> None:
        """Test batch statistics calculation with no successful results."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Create failed results
        from culora.domain.models.quality import ImageQualityResult

        failed_results = [
            ImageQualityResult(
                path=Path("test1.jpg"),
                success=False,
                error="Test error",
            ),
            ImageQualityResult(
                path=Path("test2.jpg"),
                success=False,
                error="Test error",
            ),
        ]

        batch_result = service._calculate_batch_statistics(
            failed_results, [], total_duration=1.0
        )

        assert batch_result.successful_analyses == 0
        assert batch_result.failed_analyses == 2
        assert batch_result.mean_quality_score == 0.0
        assert batch_result.median_quality_score == 0.0
        assert batch_result.quality_score_std == 0.0
        assert batch_result.quality_score_range == (0.0, 0.0)
        assert batch_result.passing_threshold_count == 0

    def test_calculate_batch_statistics_with_results(self) -> None:
        """Test batch statistics calculation with successful results."""
        config = ConfigBuilder().build()
        service = QualityService(config)

        # Create mock successful results
        from culora.domain.models.quality import ImageQualityResult, QualityScore

        scores = [0.8, 0.6, 0.9, 0.5, 0.7]
        successful_results = []

        for i, score_val in enumerate(scores):
            score = QualityScore(
                technical_score=score_val,
                overall_score=score_val,
                sharpness_contribution=0.2,
                brightness_contribution=0.1,
                contrast_contribution=0.1,
                color_contribution=0.1,
                noise_contribution=0.05,
                passes_threshold=score_val >= 0.6,
            )
            result = ImageQualityResult(
                path=Path(f"test{i}.jpg"),
                success=True,
                score=score,
            )
            successful_results.append(result)

        batch_result = service._calculate_batch_statistics(
            successful_results, successful_results, total_duration=2.0
        )

        assert batch_result.successful_analyses == 5
        assert batch_result.failed_analyses == 0
        assert abs(batch_result.mean_quality_score - 0.7) < 0.001  # Mean of scores
        assert abs(batch_result.median_quality_score - 0.7) < 0.001  # Median of scores
        assert batch_result.quality_score_range == (0.5, 0.9)
        assert batch_result.passing_threshold_count == 4  # 0.8, 0.6, 0.9, 0.7 >= 0.6
        assert batch_result.images_per_second == 2.5  # 5 images / 2.0 seconds


class TestQualityConfig:
    """Test QualityConfig validation and functionality."""

    def test_quality_config_defaults(self) -> None:
        """Test QualityConfig default values."""
        config = QualityConfig()

        assert config.sharpness_kernel_size == 3
        assert config.optimal_brightness_range == (0.3, 0.7)
        assert config.high_contrast_threshold == 0.4
        assert config.min_saturation == 0.1
        assert config.max_saturation == 0.8
        assert config.noise_threshold == 50.0
        assert config.min_quality_score == 0.3
        assert config.resize_for_analysis is True
        assert config.max_analysis_size == (1024, 1024)

    def test_quality_config_weights_validation_valid(self) -> None:
        """Test valid weight configuration."""
        config = QualityConfig(
            sharpness_weight=0.4,
            brightness_weight=0.2,
            contrast_weight=0.2,
            color_weight=0.1,
            noise_weight=0.1,
        )
        # Should not raise exception
        assert config.sharpness_weight == 0.4

    def test_quality_config_weights_validation_invalid(self) -> None:
        """Test invalid weight configuration."""
        with pytest.raises(ValueError, match="Quality weights must sum to 1.0"):
            QualityConfig(
                sharpness_weight=0.5,
                brightness_weight=0.2,
                contrast_weight=0.2,
                color_weight=0.2,
                noise_weight=0.2,  # Total = 1.3
            )

    def test_quality_config_brightness_range_validation(self) -> None:
        """Test brightness range validation."""
        with pytest.raises(
            ValueError, match="Minimum brightness must be less than maximum"
        ):
            QualityConfig(optimal_brightness_range=(0.7, 0.3))

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            QualityConfig(optimal_brightness_range=(-0.1, 0.5))

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            QualityConfig(optimal_brightness_range=(0.3, 1.1))

    def test_quality_config_saturation_validation(self) -> None:
        """Test saturation validation."""
        with pytest.raises(
            ValueError, match="Maximum saturation must be greater than minimum"
        ):
            QualityConfig(min_saturation=0.8, max_saturation=0.5)

    def test_quality_config_analysis_size_validation(self) -> None:
        """Test analysis size validation."""
        with pytest.raises(ValueError, match="Analysis dimensions must be positive"):
            QualityConfig(max_analysis_size=(0, 100))

        with pytest.raises(ValueError, match="Analysis dimensions too small"):
            QualityConfig(max_analysis_size=(32, 32))

    def test_quality_config_from_dict(self) -> None:
        """Test creating QualityConfig from dictionary."""
        data = {
            "sharpness_weight": 0.4,
            "brightness_weight": 0.2,
            "contrast_weight": 0.2,
            "color_weight": 0.1,
            "noise_weight": 0.1,
            "min_quality_score": 0.5,
        }

        config = QualityConfig.from_dict(data)
        assert config.sharpness_weight == 0.4
        assert config.min_quality_score == 0.5
