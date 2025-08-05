"""Composition analysis enumerations."""

from enum import Enum


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
