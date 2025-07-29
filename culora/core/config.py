"""Pydantic configuration models for CuLoRA.

This module defines type-safe configuration models with comprehensive validation
for all CuLoRA components.
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, validator

from .types import DeviceType, LogLevel, QualityThreshold, ShotType


class DeviceConfig(BaseModel):
    """Hardware device configuration.

    Configure device preferences, memory limits, and fallback strategies
    for AI model execution.
    """

    preferred_device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Preferred device type for AI model execution",
    )

    fallback_device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Fallback device when preferred device is unavailable",
    )

    memory_limit_mb: Optional[int] = Field(
        default=None,
        ge=512,
        le=65536,
        description="Maximum memory usage in MB (None for auto-detection)",
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for model inference",
    )

    auto_detect: bool = Field(
        default=True,
        description="Automatically detect optimal device configuration",
    )

    @validator("fallback_device")
    def validate_fallback_device(cls, v: DeviceType, values: dict) -> DeviceType:
        """Ensure fallback device is different from preferred device."""
        if "preferred_device" in values and v == values["preferred_device"]:
            return DeviceType.CPU
        return v


class FaceAnalysisConfig(BaseModel):
    """Face detection and analysis configuration.

    Configure InsightFace settings, similarity thresholds, and face
    detection parameters.
    """

    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for face detection",
    )

    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for reference matching",
    )

    max_faces_per_image: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of faces to detect per image",
    )

    min_face_size: int = Field(
        default=32,
        ge=16,
        le=512,
        description="Minimum face size in pixels",
    )

    enable_age_detection: bool = Field(
        default=False,
        description="Enable age estimation (requires additional models)",
    )

    enable_gender_detection: bool = Field(
        default=False,
        description="Enable gender detection (requires additional models)",
    )

    reference_images: list[Path] = Field(
        default_factory=list,
        description="Paths to reference images for identity matching",
    )

    @validator("reference_images", each_item=True)
    def validate_reference_paths(cls, v: Path) -> Path:
        """Validate that reference image paths exist."""
        if not v.exists():
            raise ValueError(f"Reference image not found: {v}")
        return v


class QualityAssessmentConfig(BaseModel):
    """Image quality assessment configuration.

    Configure BRISQUE settings, technical metric weights, and quality thresholds.
    """

    enable_brisque: bool = Field(
        default=True,
        description="Enable BRISQUE perceptual quality assessment",
    )

    brisque_threshold: QualityThreshold = Field(
        default=QualityThreshold(min_value=0.0, max_value=60.0),
        description="BRISQUE score thresholds (lower is better)",
    )

    enable_technical_metrics: bool = Field(
        default=True,
        description="Enable technical quality metrics (sharpness, contrast, etc.)",
    )

    sharpness_threshold: QualityThreshold = Field(
        default=QualityThreshold(min_value=100.0),
        description="Laplacian variance sharpness threshold",
    )

    brightness_threshold: QualityThreshold = Field(
        default=QualityThreshold(min_value=20.0, max_value=240.0),
        description="Image brightness threshold (0-255 scale)",
    )

    contrast_threshold: QualityThreshold = Field(
        default=QualityThreshold(min_value=20.0),
        description="Image contrast threshold",
    )

    quality_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "brisque": 0.4,
            "sharpness": 0.2,
            "brightness": 0.1,
            "contrast": 0.1,
            "face_quality": 0.2,
        },
        description="Weights for combining quality metrics",
    )

    @validator("quality_weights")
    def validate_weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that quality weights sum to 1.0."""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):  # Allow small floating point errors
            raise ValueError(f"Quality weights must sum to 1.0, got {total}")
        return v


class CompositionConfig(BaseModel):
    """Composition analysis configuration.

    Configure vision-language model settings and composition classification.
    """

    enable_shot_classification: bool = Field(
        default=True,
        description="Enable shot type classification (portrait, full body, etc.)",
    )

    enable_scene_classification: bool = Field(
        default=True,
        description="Enable scene type classification (indoor, outdoor, etc.)",
    )

    enable_clip_embeddings: bool = Field(
        default=True,
        description="Enable CLIP semantic embeddings for diversity analysis",
    )

    enable_pose_analysis: bool = Field(
        default=True,
        description="Enable MediaPipe pose analysis",
    )

    model_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for vision-language model inference",
    )

    max_tokens: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum tokens for model responses",
    )


class SelectionConfig(BaseModel):
    """Image selection algorithm configuration.

    Configure target distributions, diversity weights, and selection parameters.
    """

    target_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Target number of images to select (None for ratio-based)",
    )

    target_distributions: dict[str, float] = Field(
        default_factory=lambda: {
            ShotType.PORTRAIT.value: 0.4,
            ShotType.MEDIUM_SHOT.value: 0.3,
            ShotType.FULL_BODY.value: 0.2,
            ShotType.CLOSE_UP.value: 0.1,
        },
        description="Target distribution ratios for shot types",
    )

    quality_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for quality in selection algorithm",
    )

    diversity_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in selection algorithm",
    )

    min_quality_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for image selection",
    )

    enable_duplicate_removal: bool = Field(
        default=True,
        description="Enable perceptual duplicate detection and removal",
    )

    duplicate_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection",
    )

    clustering_method: str = Field(
        default="kmeans",
        pattern="^(kmeans|dbscan|hierarchical)$",
        description="Clustering method for diversity optimization",
    )

    @validator("quality_weight", "diversity_weight")
    def validate_weight_sum(cls, v: float, values: dict) -> float:
        """Validate that quality and diversity weights sum to 1.0."""
        if "quality_weight" in values:
            total = v + values["quality_weight"]
            if not (0.95 <= total <= 1.05):
                raise ValueError("Quality and diversity weights must sum to 1.0")
        return v


class ExportConfig(BaseModel):
    """Export system configuration.

    Configure output formats, naming schemes, and metadata options.
    """

    output_directory: Path = Field(
        default=Path("output"),
        description="Output directory for curated dataset",
    )

    sequential_naming: bool = Field(
        default=True,
        description="Use sequential naming (01.jpg, 02.jpg, etc.)",
    )

    preserve_original_names: bool = Field(
        default=True,
        description="Maintain mapping to original filenames",
    )

    export_metadata: bool = Field(
        default=True,
        description="Export comprehensive JSON metadata",
    )

    export_face_boxes: bool = Field(
        default=False,
        description="Export images with face bounding box overlays",
    )

    image_format: str = Field(
        default="jpg",
        pattern="^(jpg|jpeg|png|webp)$",
        description="Output image format",
    )

    image_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality (1-100, only for JPEG format)",
    )

    copy_mode: str = Field(
        default="selected_only",
        pattern="^(selected_only|all_with_status)$",
        description="Copy mode: selected images only or all with selection status",
    )

    @validator("output_directory")
    def validate_output_directory(cls, v: Path) -> Path:
        """Ensure output directory is writable."""
        v.mkdir(parents=True, exist_ok=True)
        if not v.is_dir():
            raise ValueError(f"Output directory is not accessible: {v}")
        return v


class LoggingConfig(BaseModel):
    """Logging system configuration."""

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level to capture",
    )

    log_directory: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    enable_console_output: bool = Field(
        default=False,
        description="Enable console output for development (separate from Rich UI)",
    )

    max_log_files: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of log files to retain",
    )

    max_log_size_mb: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum log file size in MB before rotation",
    )


class CuLoRAConfig(BaseModel):
    """Main CuLoRA configuration.

    Combines all configuration sections into a single validated model.
    """

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    face_analysis: FaceAnalysisConfig = Field(default_factory=FaceAnalysisConfig)
    quality_assessment: QualityAssessmentConfig = Field(
        default_factory=QualityAssessmentConfig
    )
    composition: CompositionConfig = Field(default_factory=CompositionConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        """Pydantic model configuration."""

        validate_assignment: ClassVar[bool] = True
        use_enum_values: ClassVar[bool] = True
        json_encoders: ClassVar[dict] = {
            Path: str,
        }

    def model_dump_json(self, **kwargs: Any) -> str:
        """Export configuration as JSON string."""
        return super().model_dump_json(indent=2, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CuLoRAConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def get_section(self, section_name: str) -> Any:
        """Get a specific configuration section."""
        return getattr(self, section_name)
