"""Composition analysis domain models."""

from dataclasses import dataclass
from pathlib import Path

from culora.domain.enums.composition import (
    BackgroundComplexity,
    CameraAngle,
    FacialExpression,
    LightingQuality,
    SceneType,
    ShotType,
)


@dataclass(frozen=True)
class CompositionAnalysis:
    """Complete composition analysis results for an image."""

    shot_type: ShotType
    scene_type: SceneType
    lighting_quality: LightingQuality
    background_complexity: BackgroundComplexity
    facial_expression: FacialExpression | None = None
    camera_angle: CameraAngle | None = None
    confidence_score: float | None = None  # 0.0-1.0 confidence in analysis
    raw_description: str | None = None  # Raw model output for debugging


@dataclass(frozen=True)
class CompositionResult:
    """Result of composition analysis for a single image."""

    path: Path
    success: bool
    analysis: CompositionAnalysis | None = None
    error: str | None = None
    error_code: str | None = None
    analysis_duration: float | None = None
    model_response: str | None = None  # Raw model response for debugging


@dataclass(frozen=True)
class BatchCompositionResult:
    """Results of batch composition analysis."""

    results: list[CompositionResult]
    successful_analyses: int
    failed_analyses: int
    total_duration: float
    images_per_second: float

    # Distribution statistics
    shot_type_distribution: dict[ShotType, int]
    scene_type_distribution: dict[SceneType, int]
    lighting_distribution: dict[LightingQuality, int]
    background_distribution: dict[BackgroundComplexity, int]
    expression_distribution: dict[FacialExpression, int]
    angle_distribution: dict[CameraAngle, int]

    # Quality metrics
    mean_confidence: float
    confidence_distribution: dict[str, float]  # percentiles
