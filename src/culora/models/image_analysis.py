from datetime import datetime

from pydantic import BaseModel, Field

from culora.models.analysis_result import AnalysisResult


class ImageAnalysis(BaseModel):
    """Analysis results for a single image."""

    file_path: str
    file_size: int
    modified_time: datetime
    results: AnalysisResult = Field(default_factory=AnalysisResult)
    score: float = 0.0  # Composite score for selection ranking
