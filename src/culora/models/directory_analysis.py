from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from culora.config import AnalysisStage
from culora.models.image_analysis import ImageAnalysis


class DirectoryAnalysis(BaseModel):
    """Analysis results for an entire directory."""

    input_directory: str
    analysis_time: datetime
    analysis_config: Any
    images: list[ImageAnalysis] = Field(default_factory=lambda: [])

    @property
    def total_images(self) -> int:
        """Total number of images analyzed."""
        return len(self.images)

    def get_stage_config(self, stage: AnalysisStage) -> Any:
        """Get configuration for a specific stage."""
        return self.analysis_config.get_stage_config(stage)
