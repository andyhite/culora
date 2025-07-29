"""Tests for configuration models and validation."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from culora.core.config import (
    CuLoRAConfig,
    DeviceConfig,
    FaceAnalysisConfig,
    QualityAssessmentConfig,
    SelectionConfig,
)
from culora.core.types import DeviceType


class TestDeviceConfig:
    """Test DeviceConfig validation and behavior."""

    def test_valid_device_config(self) -> None:
        """Test creating valid device configuration."""
        config = DeviceConfig(
            preferred_device=DeviceType.CUDA,
            fallback_device=DeviceType.CPU,
            batch_size=32,
        )

        assert config.preferred_device == DeviceType.CUDA
        assert config.fallback_device == DeviceType.CPU
        assert config.batch_size == 32
        assert config.auto_detect is True

    def test_fallback_device_validation(self) -> None:
        """Test that fallback device is set to CPU if same as preferred."""
        config = DeviceConfig(
            preferred_device=DeviceType.CUDA,
            fallback_device=DeviceType.CUDA,  # Same as preferred
        )

        assert config.fallback_device == DeviceType.CPU

    def test_invalid_batch_size(self) -> None:
        """Test validation of batch size bounds."""
        with pytest.raises(ValidationError):
            DeviceConfig(batch_size=0)  # Below minimum

        with pytest.raises(ValidationError):
            DeviceConfig(batch_size=1000)  # Above maximum

    def test_invalid_memory_limit(self) -> None:
        """Test validation of memory limit bounds."""
        with pytest.raises(ValidationError):
            DeviceConfig(memory_limit_mb=100)  # Below minimum

        with pytest.raises(ValidationError):
            DeviceConfig(memory_limit_mb=100000)  # Above maximum


class TestFaceAnalysisConfig:
    """Test FaceAnalysisConfig validation and behavior."""

    def test_valid_face_config(self) -> None:
        """Test creating valid face analysis configuration."""
        config = FaceAnalysisConfig(
            detection_threshold=0.7,
            similarity_threshold=0.8,
            max_faces_per_image=5,
            min_face_size=64,
        )

        assert config.detection_threshold == 0.7
        assert config.similarity_threshold == 0.8
        assert config.max_faces_per_image == 5
        assert config.min_face_size == 64

    def test_threshold_bounds(self) -> None:
        """Test validation of threshold bounds."""
        with pytest.raises(ValidationError):
            FaceAnalysisConfig(detection_threshold=1.5)  # Above maximum

        with pytest.raises(ValidationError):
            FaceAnalysisConfig(similarity_threshold=-0.1)  # Below minimum

    def test_reference_image_validation(self, tmp_path: Path) -> None:
        """Test reference image path validation."""
        # Create a valid reference image
        valid_ref = tmp_path / "reference.jpg"
        valid_ref.write_text("mock image")

        # This should work
        config = FaceAnalysisConfig(reference_images=[valid_ref])
        assert len(config.reference_images) == 1

        # Non-existent file should fail
        invalid_ref = tmp_path / "nonexistent.jpg"
        with pytest.raises(ValidationError):
            FaceAnalysisConfig(reference_images=[invalid_ref])


class TestQualityAssessmentConfig:
    """Test QualityAssessmentConfig validation and behavior."""

    def test_valid_quality_config(self) -> None:
        """Test creating valid quality assessment configuration."""
        config = QualityAssessmentConfig(
            enable_brisque=True,
            enable_technical_metrics=True,
            quality_weights={
                "brisque": 0.4,
                "sharpness": 0.2,
                "brightness": 0.1,
                "contrast": 0.1,
                "face_quality": 0.2,
            },
        )

        assert config.enable_brisque is True
        assert config.enable_technical_metrics is True
        assert sum(config.quality_weights.values()) == pytest.approx(1.0)

    def test_quality_weights_validation(self) -> None:
        """Test that quality weights sum to 1.0."""
        # Weights that don't sum to 1.0 should fail
        with pytest.raises(ValidationError):
            QualityAssessmentConfig(
                quality_weights={
                    "brisque": 0.5,
                    "sharpness": 0.8,  # Sum > 1.0
                }
            )

        # Weights that sum to approximately 1.0 should pass
        config = QualityAssessmentConfig(
            quality_weights={
                "brisque": 0.333,
                "sharpness": 0.333,
                "brightness": 0.334,  # Sum = 1.0 (with rounding)
            }
        )
        assert abs(sum(config.quality_weights.values()) - 1.0) < 0.05


class TestSelectionConfig:
    """Test SelectionConfig validation and behavior."""

    def test_valid_selection_config(self) -> None:
        """Test creating valid selection configuration."""
        config = SelectionConfig(
            target_count=100,
            quality_weight=0.6,
            diversity_weight=0.4,
            min_quality_score=0.5,
        )

        assert config.target_count == 100
        assert config.quality_weight == 0.6
        assert config.diversity_weight == 0.4

    def test_weight_sum_validation(self) -> None:
        """Test that quality and diversity weights sum to 1.0."""
        with pytest.raises(ValidationError):
            SelectionConfig(
                quality_weight=0.8,
                diversity_weight=0.4,  # Sum > 1.0
            )

    def test_clustering_method_validation(self) -> None:
        """Test validation of clustering method."""
        # Valid method
        config = SelectionConfig(clustering_method="kmeans")
        assert config.clustering_method == "kmeans"

        # Invalid method
        with pytest.raises(ValidationError):
            SelectionConfig(clustering_method="invalid_method")


class TestCuLoRAConfig:
    """Test main CuLoRA configuration."""

    def test_default_config_creation(self) -> None:
        """Test creating configuration with all defaults."""
        config = CuLoRAConfig()

        assert config.device.preferred_device == DeviceType.CPU
        assert config.face_analysis.detection_threshold == 0.5
        assert config.quality_assessment.enable_brisque is True
        assert config.selection.enable_duplicate_removal is True

    def test_config_from_dict(self, valid_config_dict: dict[str, Any]) -> None:
        """Test creating configuration from dictionary."""
        config = CuLoRAConfig.from_dict(valid_config_dict)

        assert config.device.preferred_device == DeviceType.CPU
        assert config.face_analysis.detection_threshold == 0.6
        assert config.selection.target_count == 100

    def test_invalid_config_from_dict(
        self, invalid_config_dict: dict[str, Any]
    ) -> None:
        """Test validation errors when creating from invalid dictionary."""
        with pytest.raises(ValidationError):
            CuLoRAConfig.from_dict(invalid_config_dict)

    def test_config_json_export(self, sample_config: CuLoRAConfig) -> None:
        """Test exporting configuration as JSON."""
        json_str = sample_config.model_dump_json()

        assert isinstance(json_str, str)
        assert "device" in json_str
        assert "face_analysis" in json_str

    def test_get_section(self, sample_config: CuLoRAConfig) -> None:
        """Test getting specific configuration sections."""
        device_config = sample_config.get_section("device")
        assert isinstance(device_config, DeviceConfig)

        face_config = sample_config.get_section("face_analysis")
        assert isinstance(face_config, FaceAnalysisConfig)
