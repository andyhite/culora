"""Composition analysis domain models."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ShotType(str, Enum):
    """Shot type classifications based on subject framing."""

    EXTREME_CLOSEUP = "extreme_closeup"
    CLOSEUP = "closeup"
    MEDIUM_CLOSEUP = "medium_closeup"
    MEDIUM_SHOT = "medium_shot"
    MEDIUM_LONG_SHOT = "medium_long_shot"
    LONG_SHOT = "long_shot"
    EXTREME_LONG_SHOT = "extreme_long_shot"
    PORTRAIT = "portrait"
    HEADSHOT = "headshot"
    FULL_BODY = "full_body"
    UNKNOWN = "unknown"


class SceneType(str, Enum):
    """Scene type classifications."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    STUDIO = "studio"
    NATURAL = "natural"
    URBAN = "urban"
    INTERIOR = "interior"
    UNKNOWN = "unknown"


class LightingQuality(str, Enum):
    """Lighting quality assessment."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    HARSH = "harsh"
    SOFT = "soft"
    DRAMATIC = "dramatic"
    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    UNKNOWN = "unknown"


class BackgroundComplexity(str, Enum):
    """Background complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CLUTTERED = "cluttered"
    CLEAN = "clean"
    BOKEH = "bokeh"
    UNKNOWN = "unknown"


class FacialExpression(str, Enum):
    """Facial expression classifications."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SERIOUS = "serious"
    SMILING = "smiling"
    LAUGHING = "laughing"
    CONTEMPLATIVE = "contemplative"
    CONFIDENT = "confident"
    RELAXED = "relaxed"
    INTENSE = "intense"
    UNKNOWN = "unknown"


class CameraAngle(str, Enum):
    """Camera angle classifications."""

    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH_ANGLE = "dutch_angle"
    OVERHEAD = "overhead"
    WORMS_EYE = "worms_eye"
    BIRDS_EYE = "birds_eye"
    STRAIGHT_ON = "straight_on"
    UNKNOWN = "unknown"


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
