"""Quality assessment configuration models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QualityConfig(BaseModel):
    """Configuration for image quality assessment.

    Controls technical quality metrics calculation, scoring weights,
    and thresholds for quality filtering.
    """

    # Sharpness calculation
    sharpness_kernel_size: int = Field(
        default=3,
        ge=3,
        le=15,
        description="Kernel size for Laplacian variance sharpness calculation (must be odd)",
    )

    # Brightness and contrast thresholds
    optimal_brightness_range: tuple[float, float] = Field(
        default=(0.3, 0.7),
        description="Optimal brightness range (0.0-1.0) for highest scores",
    )

    high_contrast_threshold: float = Field(
        default=0.4,
        ge=0.1,
        le=1.0,
        description="Minimum contrast for high quality score",
    )

    # Color quality settings
    min_saturation: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum average saturation for good color quality",
    )

    max_saturation: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Maximum average saturation before oversaturation penalty",
    )

    # Noise detection
    noise_threshold: float = Field(
        default=50.0,
        ge=0.0,
        description="Threshold for noise detection in standard deviation",
    )

    # Scoring weights (must sum to 1.0)
    sharpness_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for sharpness in composite quality score",
    )

    brightness_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for brightness in composite quality score",
    )

    contrast_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for contrast in composite quality score",
    )

    color_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for color quality in composite quality score",
    )

    noise_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for noise (inverted) in composite quality score",
    )

    # Quality filtering
    min_quality_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for image selection",
    )

    # BRISQUE perceptual quality settings
    enable_brisque: bool = Field(
        default=True,
        description="Whether to calculate BRISQUE perceptual quality scores",
    )

    brisque_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BRISQUE score in composite quality assessment",
    )

    brisque_lower_better: bool = Field(
        default=True,
        description="Whether lower BRISQUE scores indicate better quality (default: True)",
    )

    brisque_score_range: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Expected BRISQUE score range for normalization",
    )

    # Performance optimization
    resize_for_analysis: bool = Field(
        default=True,
        description="Whether to resize large images for faster quality analysis",
    )

    max_analysis_size: tuple[int, int] = Field(
        default=(1024, 1024),
        description="Maximum image size for quality analysis when resizing enabled",
    )

    @field_validator("optimal_brightness_range")
    @classmethod
    def validate_brightness_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate brightness range."""
        min_bright, max_bright = v
        if min_bright < 0.0 or max_bright > 1.0:
            raise ValueError("Brightness range must be between 0.0 and 1.0")
        if min_bright >= max_bright:
            raise ValueError("Minimum brightness must be less than maximum brightness")
        return v

    @field_validator("max_saturation")
    @classmethod
    def validate_saturation_order(cls, v: float, info: Any) -> float:
        """Validate saturation order."""
        if "min_saturation" in info.data and v <= info.data["min_saturation"]:
            raise ValueError(
                "Maximum saturation must be greater than minimum saturation"
            )
        return v

    @field_validator("brisque_score_range")
    @classmethod
    def validate_brisque_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate BRISQUE score range."""
        min_score, max_score = v
        if min_score >= max_score:
            raise ValueError("BRISQUE minimum score must be less than maximum score")
        if min_score < 0.0:
            raise ValueError("BRISQUE minimum score must be non-negative")
        return v

    @field_validator("max_analysis_size")
    @classmethod
    def validate_analysis_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate analysis dimensions."""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("Analysis dimensions must be positive")
        if width < 64 or height < 64:
            raise ValueError("Analysis dimensions too small (minimum 64x64)")
        return v

    def validate_technical_weights_sum(self) -> None:
        """Validate that technical quality weights sum to approximately 1.0."""
        technical_weight = (
            self.sharpness_weight
            + self.brightness_weight
            + self.contrast_weight
            + self.color_weight
            + self.noise_weight
        )
        if abs(technical_weight - 1.0) > 0.001:
            raise ValueError(
                f"Technical quality weights must sum to 1.0, got {technical_weight}"
            )

    def model_post_init(self, _context: Any) -> None:
        """Post-initialization validation."""
        self.validate_technical_weights_sum()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityConfig":
        """Create QualityConfig from dictionary."""
        return cls(**data)
