"""Analysis stage configuration definitions for CuLoRA."""

from enum import Enum
from typing import Any

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


class AnalysisConfig(BaseModel):
    """Main configuration for analysis pipeline."""

    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)

    @property
    def enabled_stages(self) -> list[AnalysisStage]:
        """Get list of enabled stages."""
        stages: list[AnalysisStage] = []
        if self.deduplication.enabled:
            stages.append(AnalysisStage.DEDUPLICATION)
        if self.quality.enabled:
            stages.append(AnalysisStage.QUALITY)
        if self.face.enabled:
            stages.append(AnalysisStage.FACE)
        return stages

    def get_stage_config(self, stage: AnalysisStage) -> Any:
        """Get configuration for a specific stage."""
        if stage == AnalysisStage.DEDUPLICATION:
            return self.deduplication
        elif stage == AnalysisStage.QUALITY:
            return self.quality
        elif stage == AnalysisStage.FACE:
            return self.face
        else:
            raise ValueError(f"Unknown stage: {stage}")
