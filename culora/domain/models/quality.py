"""Quality assessment domain models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TechnicalQualityMetrics:
    """Technical quality metrics for an image.

    Contains individual metric scores and measurements used
    for composite quality assessment.
    """

    # Core metrics (0.0 - 1.0)
    sharpness: float
    brightness_score: float
    contrast_score: float
    color_quality: float
    noise_score: float  # Higher is better (inverted from noise level)

    # Raw measurements
    laplacian_variance: float  # Raw sharpness measurement
    mean_brightness: float  # 0.0 - 1.0
    contrast_value: float  # Standard deviation based
    mean_saturation: float  # 0.0 - 1.0
    noise_level: float  # Standard deviation of noise

    # Analysis metadata
    analysis_width: int
    analysis_height: int
    was_resized: bool


@dataclass(frozen=True)
class QualityScore:
    """Composite quality score for an image.

    Combines technical metrics with configurable weights
    to produce overall quality assessment.
    """

    # Composite scores
    technical_score: float  # Weighted combination of technical metrics
    overall_score: (
        float  # Currently same as technical_score, future: includes perceptual
    )

    # Individual metric contributions
    sharpness_contribution: float
    brightness_contribution: float
    contrast_contribution: float
    color_contribution: float
    noise_contribution: float

    # Quality assessment
    passes_threshold: bool
    quality_percentile: float | None = None  # Set during batch analysis


@dataclass(frozen=True)
class ImageQualityResult:
    """Complete quality assessment result for an image.

    Contains all quality metrics, scores, and analysis metadata.
    """

    path: Path
    success: bool

    # Quality data (None if analysis failed)
    metrics: TechnicalQualityMetrics | None = None
    score: QualityScore | None = None

    # Error information
    error: str | None = None
    error_code: str | None = None
    analysis_duration: float = 0.0


@dataclass(frozen=True)
class BatchQualityResult:
    """Result of quality assessment for a batch of images.

    Contains individual results and batch-level statistics.
    """

    results: list[ImageQualityResult]
    successful_analyses: int
    failed_analyses: int

    # Quality statistics
    mean_quality_score: float
    median_quality_score: float
    quality_score_std: float
    quality_score_range: tuple[float, float]

    # Processing statistics
    total_duration: float
    images_per_second: float

    # Quality distribution
    scores_by_percentile: dict[int, float]  # percentile -> score
    passing_threshold_count: int
