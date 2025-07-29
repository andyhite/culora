from pydantic import BaseModel, Field


class Face(BaseModel):
    """Individual face detection result."""

    confidence: float
    bounding_box: tuple[float, float, float, float]  # (x1, y1, x2, y2)


class FaceDetectionResult(BaseModel):
    """Face detection analysis result."""

    faces: list[Face] = Field(default_factory=lambda: [])
    model_used: str = ""
    device_used: str = ""

    @property
    def face_count(self) -> int:
        """Number of faces detected."""
        return len(self.faces)

    @property
    def highest_confidence(self) -> float:
        """Highest confidence score among all faces."""
        if not self.faces:
            return 0.0
        return max(face.confidence for face in self.faces)

    @property
    def average_confidence(self) -> float:
        """Average confidence score among all faces."""
        if not self.faces:
            return 0.0
        return sum(face.confidence for face in self.faces) / len(self.faces)

    @property
    def confidence_scores(self) -> list[float]:
        """List of all confidence scores (for backward compatibility)."""
        return [face.confidence for face in self.faces]

    @property
    def bounding_boxes(self) -> list[tuple[float, float, float, float]]:
        """List of all bounding boxes (for backward compatibility)."""
        return [face.bounding_box for face in self.faces]
