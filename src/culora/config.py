"""Analysis stage configuration definitions for CuLoRA."""

from enum import Enum

from pydantic import BaseModel, Field


class AnalysisStage(str, Enum):
    """Analysis stages available in CuLoRA."""

    DEDUPLICATION = "deduplication"
    QUALITY = "quality"
    FACE = "face"


class DeduplicationConfig(BaseModel):
    """Configuration for image deduplication analysis."""

    enabled: bool = True
    algorithm: str = "dhash"
    hash_size: int = 8
    threshold: int = 2
    version: str = "1.0"


class QualityConfig(BaseModel):
    """Configuration for image quality analysis."""

    enabled: bool = True
    sharpness_threshold: float = 150.0
    brightness_min: float = 60.0
    brightness_max: float = 200.0
    contrast_threshold: float = 40.0
    version: str = "1.0"


class FaceConfig(BaseModel):
    """Configuration for face detection analysis."""

    enabled: bool = True
    confidence_threshold: float = 0.5
    model_repo: str = "AdamCodd/YOLOv11n-face-detection"
    model_filename: str = "model.pt"
    max_detections: int = 10
    iou_threshold: float = 0.5
    use_half_precision: bool = True
    device: str = "auto"
    version: str = "3.0"


class ScoringConfig(BaseModel):
    """Configuration for composite scoring algorithm."""

    quality_weight: float = 0.5  # Quality component weight (0.0 to 1.0)
    face_weight: float = 0.5  # Face component weight (0.0 to 1.0)

    # Face area ratio thresholds for sigmoid scoring
    face_area_min: float = 0.05  # Minimum good face area ratio (5%)
    face_area_peak: float = 0.15  # Peak face area ratio (15%)
    face_area_max: float = 0.25  # Maximum good face area ratio (25%)

    # Multi-face penalties
    multi_face_penalty: float = 0.1  # Penalty per additional face (10%)
    max_face_penalty: float = 0.5  # Maximum total face penalty (50%)

    version: str = "1.0"


class DisplayConfig(BaseModel):
    """Configuration for output display formatting."""

    # Score color thresholds for Rich display
    score_excellent_threshold: float = 0.7  # Green color threshold
    score_good_threshold: float = 0.4  # Yellow color threshold
    # Below good_threshold = red

    # Quality display thresholds
    sharpness_display_good: float = 150.0
    sharpness_display_excellent: float = 500.0
    brightness_display_min: float = 60.0
    brightness_display_max: float = 200.0
    contrast_display_good: float = 40.0
    contrast_display_excellent: float = 60.0

    version: str = "1.0"


class CuLoRAConfig(BaseModel):
    """Main configuration for CuLoRA."""

    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)

    @property
    def enabled_stages(self) -> list[AnalysisStage]:
        """Return list of enabled analysis stages based on individual stage configs."""
        stages: list[AnalysisStage] = []
        if self.deduplication.enabled:
            stages.append(AnalysisStage.DEDUPLICATION)
        if self.quality.enabled:
            stages.append(AnalysisStage.QUALITY)
        if self.face.enabled:
            stages.append(AnalysisStage.FACE)
        return stages
