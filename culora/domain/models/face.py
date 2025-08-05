"""Face detection and analysis domain models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """Represents a single detected face with all associated data.

    This model contains comprehensive information about a detected face including
    spatial information, confidence metrics, facial landmarks, and embeddings.
    """

    # Spatial information
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) coordinates
    confidence: float  # Detection confidence score (0.0 to 1.0)

    # Face characteristics
    landmarks: np.ndarray | None  # Facial landmark points (5 or 68 points)
    embedding: np.ndarray | None  # Face embedding vector for similarity

    # Relative metrics
    face_area_ratio: float  # Face area relative to total image area

    # Optional attributes (when available from model)
    age: int | None = None  # Estimated age
    gender: str | None = None  # Estimated gender

    def __post_init__(self) -> None:
        """Validate face detection data after initialization."""
        # Validate bounding box coordinates
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"Invalid bounding box coordinates: {self.bbox}")

        # Validate confidence score
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got: {self.confidence}"
            )

        # Validate face area ratio
        if not (0.0 <= self.face_area_ratio <= 1.0):
            raise ValueError(
                f"Face area ratio must be between 0.0 and 1.0, got: {self.face_area_ratio}"
            )

    @property
    def bbox_width(self) -> float:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_center(self) -> tuple[float, float]:
        """Get bounding box center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def has_embedding(self) -> bool:
        """Check if face has an embedding vector."""
        return self.embedding is not None

    @property
    def has_landmarks(self) -> bool:
        """Check if face has landmark points."""
        return self.landmarks is not None

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.bbox_width * self.bbox_height


@dataclass(frozen=True)
class FaceAnalysisResult:
    """Result of face analysis on a single image.

    Contains all detected faces, success/failure status, and comprehensive
    error information for debugging and user feedback.
    """

    # Core result data
    image_path: Path
    success: bool
    faces: list[FaceDetection]  # All detected faces

    # Timing and metadata
    processing_duration: float  # Analysis time in seconds
    processed_at: datetime

    # Image information
    image_width: int
    image_height: int

    # Error information
    error: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        """Validate face analysis result after initialization."""
        if self.success and not self.faces:
            # Successful analysis with no faces is valid (image with no faces)
            pass
        elif not self.success and self.error is None:
            raise ValueError("Failed analysis must have an error message")

        if self.processing_duration < 0:
            raise ValueError(
                f"Processing duration must be non-negative, got: {self.processing_duration}"
            )

    @property
    def face_count(self) -> int:
        """Get number of detected faces."""
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        """Check if any faces were detected."""
        return len(self.faces) > 0

    @property
    def primary_face(self) -> FaceDetection | None:
        """Get the primary face (highest confidence) if any faces detected."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.confidence)

    @property
    def average_confidence(self) -> float:
        """Get average confidence of all detected faces."""
        if not self.faces:
            return 0.0
        return sum(face.confidence for face in self.faces) / len(self.faces)

    @property
    def max_confidence(self) -> float:
        """Get maximum confidence of detected faces."""
        if not self.faces:
            return 0.0
        return max(face.confidence for face in self.faces)

    @property
    def total_face_area_ratio(self) -> float:
        """Get total face area as ratio of image area."""
        return sum(face.face_area_ratio for face in self.faces)


@dataclass(frozen=True)
class BatchFaceAnalysisResult:
    """Result of batch face analysis on multiple images.

    Aggregates results from multiple images with performance metrics
    and summary statistics for reporting and monitoring.
    """

    # Core results
    results: list[FaceAnalysisResult]

    # Performance metrics
    processing_duration: float  # Total batch processing time
    successful_analyses: int
    failed_analyses: int

    # Face statistics
    total_faces_detected: int
    images_with_faces: int
    images_without_faces: int

    def __post_init__(self) -> None:
        """Validate batch analysis result after initialization."""
        expected_total = self.successful_analyses + self.failed_analyses
        if len(self.results) != expected_total:
            raise ValueError(
                f"Result count {len(self.results)} doesn't match expected {expected_total}"
            )

        if self.processing_duration < 0:
            raise ValueError(
                f"Processing duration must be non-negative, got: {self.processing_duration}"
            )

    @property
    def total_images(self) -> int:
        """Get total number of images processed."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.successful_analyses / self.total_images) * 100

    @property
    def face_detection_rate(self) -> float:
        """Get percentage of images with at least one face."""
        if self.successful_analyses == 0:
            return 0.0
        return (self.images_with_faces / self.successful_analyses) * 100

    @property
    def average_faces_per_image(self) -> float:
        """Get average number of faces per successfully analyzed image."""
        if self.successful_analyses == 0:
            return 0.0
        return self.total_faces_detected / self.successful_analyses

    @property
    def average_processing_time_per_image(self) -> float:
        """Get average processing time per image in seconds."""
        if self.total_images == 0:
            return 0.0
        return self.processing_duration / self.total_images
