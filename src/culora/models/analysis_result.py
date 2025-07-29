from pydantic import BaseModel

from culora.config import AnalysisStage
from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import FaceDetectionResult
from culora.models.image_quality_result import ImageQualityResult

StageResult = DuplicateDetectionResult | FaceDetectionResult | ImageQualityResult


class AnalysisResult(BaseModel):
    """Container for all stage results, keyed by AnalysisStage."""

    quality: ImageQualityResult | None = None
    face: FaceDetectionResult | None = None
    deduplication: DuplicateDetectionResult | None = None

    def get_quality(self) -> ImageQualityResult | None:
        """Get quality analysis result."""
        return self.quality

    def get_face(self) -> FaceDetectionResult | None:
        """Get face detection result."""
        return self.face

    def get_deduplication(self) -> DuplicateDetectionResult | None:
        """Get deduplication result."""
        return self.deduplication

    def set_quality(self, result: ImageQualityResult) -> None:
        """Set quality analysis result."""
        self.quality = result

    def set_face(self, result: FaceDetectionResult) -> None:
        """Set face detection result."""
        self.face = result

    def set_deduplication(self, result: DuplicateDetectionResult) -> None:
        """Set deduplication result."""
        self.deduplication = result

    def get(self, stage: AnalysisStage) -> StageResult | None:
        """Get result for a specific stage."""
        if stage == AnalysisStage.QUALITY:
            return self.quality
        elif stage == AnalysisStage.FACE:
            return self.face
        elif stage == AnalysisStage.DEDUPLICATION:
            return self.deduplication
        return None

    def set(self, stage: AnalysisStage, result: StageResult) -> None:
        """Set result for a specific stage."""
        if stage == AnalysisStage.QUALITY and isinstance(result, ImageQualityResult):
            self.quality = result
        elif stage == AnalysisStage.FACE and isinstance(result, FaceDetectionResult):
            self.face = result
        elif stage == AnalysisStage.DEDUPLICATION and isinstance(
            result, DuplicateDetectionResult
        ):
            self.deduplication = result

    def has_stage(self, stage: AnalysisStage) -> bool:
        """Check if a stage has results."""
        return self.get(stage) is not None

    def __contains__(self, stage: AnalysisStage) -> bool:
        """Check if a stage has results."""
        return self.has_stage(stage)
