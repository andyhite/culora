"""Analysis result models for CuLoRA."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class AnalysisStage(str, Enum):
    """Analysis stages available in CuLoRA."""

    DEDUPLICATION = "deduplication"
    QUALITY = "quality"
    FACE = "face"


class AnalysisResult(str, Enum):
    """Result of an analysis stage."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class StageResult(BaseModel):
    """Result of a single analysis stage for an image."""

    stage: AnalysisStage
    result: AnalysisResult
    reason: str | None = None
    metadata: dict[str, str] = {}


class ImageAnalysis(BaseModel):
    """Analysis results for a single image."""

    file_path: str
    file_size: int
    modified_time: datetime
    stage_results: list[StageResult] = []

    @property
    def passed_stages(self) -> list[AnalysisStage]:
        """Get list of stages that passed."""
        return [
            result.stage
            for result in self.stage_results
            if result.result == AnalysisResult.PASS
        ]

    @property
    def failed_stages(self) -> list[AnalysisStage]:
        """Get list of stages that failed."""
        return [
            result.stage
            for result in self.stage_results
            if result.result == AnalysisResult.FAIL
        ]

    @property
    def overall_result(self) -> AnalysisResult:
        """Get overall result - pass only if all enabled stages passed."""
        if not self.stage_results:
            return AnalysisResult.SKIP

        if any(result.result == AnalysisResult.FAIL for result in self.stage_results):
            return AnalysisResult.FAIL

        return AnalysisResult.PASS


class DirectoryAnalysis(BaseModel):
    """Analysis results for an entire directory."""

    input_directory: str
    analysis_time: datetime
    enabled_stages: list[AnalysisStage]
    images: list[ImageAnalysis] = []

    @property
    def total_images(self) -> int:
        """Total number of images analyzed."""
        return len(self.images)

    @property
    def passed_images(self) -> list[ImageAnalysis]:
        """Images that passed all enabled stages."""
        return [img for img in self.images if img.overall_result == AnalysisResult.PASS]

    @property
    def failed_images(self) -> list[ImageAnalysis]:
        """Images that failed at least one enabled stage."""
        return [img for img in self.images if img.overall_result == AnalysisResult.FAIL]

    @property
    def skipped_images(self) -> list[ImageAnalysis]:
        """Images that were skipped."""
        return [img for img in self.images if img.overall_result == AnalysisResult.SKIP]
