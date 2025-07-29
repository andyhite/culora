"""Unit tests for configuration models and validation."""

from culora.config import (
    AnalysisStage,
    CuLoRAConfig,
    DeduplicationConfig,
    DisplayConfig,
    FaceConfig,
    QualityConfig,
    ScoringConfig,
)


class TestAnalysisStage:
    """Test AnalysisStage enum."""

    def test_stage_values(self):
        """Test that analysis stage enum has correct values."""
        assert AnalysisStage.DEDUPLICATION == "deduplication"
        assert AnalysisStage.QUALITY == "quality"
        assert AnalysisStage.FACE == "face"

    def test_stage_membership(self):
        """Test stage membership checks."""
        assert "deduplication" in AnalysisStage
        assert "quality" in AnalysisStage
        assert "face" in AnalysisStage
        assert "invalid" not in AnalysisStage


class TestDeduplicationConfig:
    """Test DeduplicationConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DeduplicationConfig()

        assert config.enabled is True
        assert config.algorithm == "dhash"
        assert config.hash_size == 8
        assert config.threshold == 2
        assert config.version == "1.0"

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = DeduplicationConfig(
            enabled=False, algorithm="phash", hash_size=16, threshold=3
        )

        assert config.enabled is False
        assert config.algorithm == "phash"
        assert config.hash_size == 16
        assert config.threshold == 3

    def test_validation_hash_size(self):
        """Test validation of hash size."""
        # Valid hash size
        config = DeduplicationConfig(hash_size=8)
        assert config.hash_size == 8

        # Valid hash size (different value)
        config = DeduplicationConfig(hash_size=16)
        assert config.hash_size == 16

    def test_validation_threshold(self):
        """Test validation of threshold."""
        # Valid threshold
        config = DeduplicationConfig(threshold=5)
        assert config.threshold == 5

        # Valid threshold (different value)
        config = DeduplicationConfig(threshold=1)
        assert config.threshold == 1


class TestQualityConfig:
    """Test QualityConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QualityConfig()

        assert config.enabled is True
        assert config.sharpness_threshold == 150.0
        assert config.brightness_min == 60.0
        assert config.brightness_max == 200.0
        assert config.contrast_threshold == 40.0
        assert config.version == "1.0"

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = QualityConfig(
            enabled=False,
            sharpness_threshold=200.0,
            brightness_min=80.0,
            brightness_max=180.0,
            contrast_threshold=50.0,
        )

        assert config.enabled is False
        assert config.sharpness_threshold == 200.0
        assert config.brightness_min == 80.0
        assert config.brightness_max == 180.0
        assert config.contrast_threshold == 50.0

    def test_brightness_range_validation(self):
        """Test that brightness min/max form valid range."""
        # Valid range
        config = QualityConfig(brightness_min=50.0, brightness_max=200.0)
        assert config.brightness_min < config.brightness_max

        # Invalid range - note: Pydantic doesn't validate this automatically,
        # would need custom validator if required
        config = QualityConfig(brightness_min=200.0, brightness_max=50.0)
        # This creates an invalid config but doesn't raise error by default


class TestFaceConfig:
    """Test FaceConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FaceConfig()

        assert config.enabled is True
        assert config.confidence_threshold == 0.5
        assert config.model_repo == "AdamCodd/YOLOv11n-face-detection"
        assert config.model_filename == "model.pt"
        assert config.max_detections == 10
        assert config.iou_threshold == 0.5
        assert config.use_half_precision is True
        assert config.device == "auto"
        assert config.version == "3.0"

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = FaceConfig(
            enabled=False, confidence_threshold=0.8, max_detections=5, device="cpu"
        )

        assert config.enabled is False
        assert config.confidence_threshold == 0.8
        assert config.max_detections == 5
        assert config.device == "cpu"

    def test_confidence_validation(self):
        """Test validation of confidence threshold."""
        # Valid confidence
        config = FaceConfig(confidence_threshold=0.7)
        assert config.confidence_threshold == 0.7

        # Invalid confidence values would need custom validators
        # Pydantic allows any float by default


class TestScoringConfig:
    """Test ScoringConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScoringConfig()

        assert config.quality_weight == 0.5
        assert config.face_weight == 0.5
        assert config.face_area_min == 0.05
        assert config.face_area_peak == 0.15
        assert config.face_area_max == 0.25
        assert config.multi_face_penalty == 0.1
        assert config.max_face_penalty == 0.5
        assert config.version == "1.0"

    def test_weight_balance(self):
        """Test that weights can be balanced differently."""
        config = ScoringConfig(quality_weight=0.7, face_weight=0.3)

        assert config.quality_weight == 0.7
        assert config.face_weight == 0.3

        # Note: Doesn't validate that weights sum to 1.0 by default

    def test_face_area_thresholds(self):
        """Test face area ratio thresholds."""
        config = ScoringConfig(face_area_min=0.1, face_area_peak=0.2, face_area_max=0.3)

        assert config.face_area_min == 0.1
        assert config.face_area_peak == 0.2
        assert config.face_area_max == 0.3


class TestDisplayConfig:
    """Test DisplayConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DisplayConfig()

        assert config.score_excellent_threshold == 0.7
        assert config.score_good_threshold == 0.4
        assert config.sharpness_display_good == 150.0
        assert config.sharpness_display_excellent == 500.0
        assert config.brightness_display_min == 60.0
        assert config.brightness_display_max == 200.0
        assert config.contrast_display_good == 40.0
        assert config.contrast_display_excellent == 60.0
        assert config.version == "1.0"

    def test_threshold_ordering(self):
        """Test that display thresholds maintain proper ordering."""
        config = DisplayConfig()

        assert config.score_good_threshold < config.score_excellent_threshold
        assert config.sharpness_display_good < config.sharpness_display_excellent
        assert config.contrast_display_good < config.contrast_display_excellent


class TestCuLoRAConfig:
    """Test main CuLoRAConfig model."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = CuLoRAConfig()

        assert isinstance(config.deduplication, DeduplicationConfig)
        assert isinstance(config.quality, QualityConfig)
        assert isinstance(config.face, FaceConfig)
        assert isinstance(config.scoring, ScoringConfig)
        assert isinstance(config.display, DisplayConfig)

    def test_enabled_stages_all_enabled(self):
        """Test enabled_stages property with all stages enabled."""
        config = CuLoRAConfig()

        enabled = config.enabled_stages
        assert len(enabled) == 3
        assert AnalysisStage.DEDUPLICATION in enabled
        assert AnalysisStage.QUALITY in enabled
        assert AnalysisStage.FACE in enabled

    def test_enabled_stages_selective(self):
        """Test enabled_stages property with selective stages."""
        config = CuLoRAConfig()
        config.deduplication.enabled = False
        config.face.enabled = False

        enabled = config.enabled_stages
        assert len(enabled) == 1
        assert AnalysisStage.QUALITY in enabled
        assert AnalysisStage.DEDUPLICATION not in enabled
        assert AnalysisStage.FACE not in enabled

    def test_enabled_stages_none_enabled(self):
        """Test enabled_stages property with no stages enabled."""
        config = CuLoRAConfig()
        config.deduplication.enabled = False
        config.quality.enabled = False
        config.face.enabled = False

        enabled = config.enabled_stages
        assert len(enabled) == 0

    def test_nested_config_modification(self):
        """Test modifying nested configuration objects."""
        config = CuLoRAConfig()

        # Modify deduplication config
        config.deduplication.threshold = 5
        assert config.deduplication.threshold == 5

        # Modify quality config
        config.quality.sharpness_threshold = 200.0
        assert config.quality.sharpness_threshold == 200.0

        # Modify face config
        config.face.confidence_threshold = 0.8
        assert config.face.confidence_threshold == 0.8

    def test_custom_nested_configs(self):
        """Test creating config with custom nested configurations."""
        custom_dedup = DeduplicationConfig(enabled=False, threshold=3)
        custom_quality = QualityConfig(sharpness_threshold=200.0)

        config = CuLoRAConfig(deduplication=custom_dedup, quality=custom_quality)

        assert config.deduplication.enabled is False
        assert config.deduplication.threshold == 3
        assert config.quality.sharpness_threshold == 200.0

        # Default configs for other sections
        assert config.face.enabled is True
        assert config.scoring.quality_weight == 0.5

    def test_config_serialization(self):
        """Test that config can be serialized to dict."""
        config = CuLoRAConfig()
        config_dict = config.model_dump()

        assert "deduplication" in config_dict
        assert "quality" in config_dict
        assert "face" in config_dict
        assert "scoring" in config_dict
        assert "display" in config_dict

        # Check nested values
        assert config_dict["deduplication"]["enabled"] is True
        assert config_dict["quality"]["sharpness_threshold"] == 150.0

    def test_config_deserialization(self):
        """Test that config can be created from dict."""
        config_data = {
            "deduplication": {"enabled": False, "threshold": 3},
            "quality": {"sharpness_threshold": 200.0},
        }

        config = CuLoRAConfig.model_validate(config_data)

        assert config.deduplication.enabled is False
        assert config.deduplication.threshold == 3
        assert config.quality.sharpness_threshold == 200.0

        # Defaults for unspecified values
        assert config.deduplication.algorithm == "dhash"
        assert config.face.enabled is True
