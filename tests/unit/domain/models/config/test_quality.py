"""Tests for QualityConfig model."""

import pytest

from culora.domain.models.config.quality import QualityConfig


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

        # Check default weights
        assert config.sharpness_weight == 0.35
        assert config.brightness_weight == 0.2
        assert config.contrast_weight == 0.25
        assert config.color_weight == 0.15
        assert config.noise_weight == 0.05

    def test_quality_config_custom_values(self) -> None:
        """Test QualityConfig with custom values."""
        config = QualityConfig(
            sharpness_kernel_size=5,
            optimal_brightness_range=(0.2, 0.8),
            high_contrast_threshold=0.5,
            min_saturation=0.2,
            max_saturation=0.9,
            noise_threshold=100.0,
            sharpness_weight=0.4,
            brightness_weight=0.2,
            contrast_weight=0.2,
            color_weight=0.1,
            noise_weight=0.1,
            min_quality_score=0.5,
            resize_for_analysis=False,
            max_analysis_size=(512, 512),
        )

        assert config.sharpness_kernel_size == 5
        assert config.optimal_brightness_range == (0.2, 0.8)
        assert config.high_contrast_threshold == 0.5
        assert config.min_saturation == 0.2
        assert config.max_saturation == 0.9
        assert config.noise_threshold == 100.0
        assert config.sharpness_weight == 0.4
        assert config.min_quality_score == 0.5
        assert config.resize_for_analysis is False
        assert config.max_analysis_size == (512, 512)

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

    def test_quality_config_weights_validation_invalid_sum(self) -> None:
        """Test invalid weight configuration - doesn't sum to 1.0."""
        with pytest.raises(ValueError, match="Quality weights must sum to 1.0"):
            QualityConfig(
                sharpness_weight=0.5,
                brightness_weight=0.2,
                contrast_weight=0.2,
                color_weight=0.2,
                noise_weight=0.2,  # Total = 1.3
            )

    def test_quality_config_weights_validation_close_to_one(self) -> None:
        """Test weight validation allows small floating point errors."""
        # This should work (sum = 1.001, within tolerance)
        config = QualityConfig(
            sharpness_weight=0.3501,
            brightness_weight=0.2001,
            contrast_weight=0.2501,
            color_weight=0.1501,
            noise_weight=0.0496,  # Total ≈ 1.0
        )
        assert config.sharpness_weight == 0.3501

    def test_quality_config_brightness_range_validation_invalid_order(self) -> None:
        """Test brightness range validation - invalid order."""
        with pytest.raises(
            ValueError, match="Minimum brightness must be less than maximum"
        ):
            QualityConfig(optimal_brightness_range=(0.7, 0.3))

    def test_quality_config_brightness_range_validation_out_of_bounds(self) -> None:
        """Test brightness range validation - out of bounds."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            QualityConfig(optimal_brightness_range=(-0.1, 0.5))

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            QualityConfig(optimal_brightness_range=(0.3, 1.1))

    def test_quality_config_saturation_validation_invalid_order(self) -> None:
        """Test saturation validation - invalid order."""
        with pytest.raises(
            ValueError, match="Maximum saturation must be greater than minimum"
        ):
            QualityConfig(min_saturation=0.8, max_saturation=0.5)

    def test_quality_config_saturation_validation_equal_values(self) -> None:
        """Test saturation validation - equal values."""
        with pytest.raises(
            ValueError, match="Maximum saturation must be greater than minimum"
        ):
            QualityConfig(min_saturation=0.5, max_saturation=0.5)

    def test_quality_config_analysis_size_validation_zero_dimensions(self) -> None:
        """Test analysis size validation - zero dimensions."""
        with pytest.raises(ValueError, match="Analysis dimensions must be positive"):
            QualityConfig(max_analysis_size=(0, 100))

        with pytest.raises(ValueError, match="Analysis dimensions must be positive"):
            QualityConfig(max_analysis_size=(100, 0))

    def test_quality_config_analysis_size_validation_negative_dimensions(self) -> None:
        """Test analysis size validation - negative dimensions."""
        with pytest.raises(ValueError, match="Analysis dimensions must be positive"):
            QualityConfig(max_analysis_size=(-100, 100))

    def test_quality_config_analysis_size_validation_too_small(self) -> None:
        """Test analysis size validation - dimensions too small."""
        with pytest.raises(ValueError, match="Analysis dimensions too small"):
            QualityConfig(max_analysis_size=(32, 32))

        with pytest.raises(ValueError, match="Analysis dimensions too small"):
            QualityConfig(max_analysis_size=(63, 64))

    def test_quality_config_kernel_size_validation(self) -> None:
        """Test kernel size validation."""
        # Valid odd kernel sizes
        config = QualityConfig(sharpness_kernel_size=3)
        assert config.sharpness_kernel_size == 3

        config = QualityConfig(sharpness_kernel_size=15)
        assert config.sharpness_kernel_size == 15

        # Invalid kernel sizes (too small)
        with pytest.raises(ValueError):
            QualityConfig(sharpness_kernel_size=1)

        # Invalid kernel sizes (too large)
        with pytest.raises(ValueError):
            QualityConfig(sharpness_kernel_size=17)

    def test_quality_config_threshold_validation(self) -> None:
        """Test threshold validation."""
        # Valid thresholds
        config = QualityConfig(
            high_contrast_threshold=0.5,
            min_quality_score=0.7,
        )
        assert config.high_contrast_threshold == 0.5
        assert config.min_quality_score == 0.7

        # Invalid contrast threshold (negative)
        with pytest.raises(ValueError):
            QualityConfig(high_contrast_threshold=-0.1)

        # Invalid quality score (out of range)
        with pytest.raises(ValueError):
            QualityConfig(min_quality_score=1.5)

    def test_quality_config_from_dict(self) -> None:
        """Test creating QualityConfig from dictionary."""
        data = {
            "sharpness_weight": 0.4,
            "brightness_weight": 0.2,
            "contrast_weight": 0.2,
            "color_weight": 0.1,
            "noise_weight": 0.1,
            "min_quality_score": 0.5,
            "resize_for_analysis": False,
            "max_analysis_size": (512, 512),
        }

        config = QualityConfig.from_dict(data)
        assert config.sharpness_weight == 0.4
        assert config.min_quality_score == 0.5
        assert config.resize_for_analysis is False
        assert config.max_analysis_size == (512, 512)

    def test_quality_config_from_dict_partial(self) -> None:
        """Test creating QualityConfig from partial dictionary."""
        data = {
            "min_quality_score": 0.6,
            "resize_for_analysis": False,
        }

        config = QualityConfig.from_dict(data)
        assert config.min_quality_score == 0.6
        assert config.resize_for_analysis is False
        # Other values should be defaults
        assert config.sharpness_weight == 0.35
        assert config.max_analysis_size == (1024, 1024)

    def test_quality_config_validate_weights_sum_method(self) -> None:
        """Test validate_weights_sum method directly."""
        config = QualityConfig(
            sharpness_weight=0.3,
            brightness_weight=0.2,
            contrast_weight=0.2,
            color_weight=0.2,
            noise_weight=0.1,
        )

        # Should not raise exception
        config.validate_weights_sum()

        # Manually change a weight to make sum invalid
        config.__dict__["sharpness_weight"] = 0.5  # Now sum = 1.2

        with pytest.raises(ValueError, match="Quality weights must sum to 1.0"):
            config.validate_weights_sum()

    def test_quality_config_serialization(self) -> None:
        """Test that QualityConfig can be serialized and deserialized."""
        original_config = QualityConfig(
            sharpness_weight=0.4,
            brightness_weight=0.2,
            contrast_weight=0.2,
            color_weight=0.1,
            noise_weight=0.1,
            min_quality_score=0.6,
            resize_for_analysis=False,
        )

        # Convert to dict and back
        config_dict = original_config.model_dump()
        restored_config = QualityConfig(**config_dict)

        assert restored_config.sharpness_weight == original_config.sharpness_weight
        assert restored_config.min_quality_score == original_config.min_quality_score
        assert (
            restored_config.resize_for_analysis == original_config.resize_for_analysis
        )
