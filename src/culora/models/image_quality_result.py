from pydantic import BaseModel


class ImageQualityResult(BaseModel):
    """Image quality analysis result."""

    sharpness_score: float
    brightness_score: float
    contrast_score: float
    composite_score: float
