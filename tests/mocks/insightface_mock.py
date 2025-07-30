"""Mock implementations for InsightFace for testing."""

from pathlib import Path
from typing import Any

import numpy as np


class MockFaceData:
    """Mock face detection data from InsightFace."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        confidence: float,
        embedding: np.ndarray | None = None,
        landmarks: np.ndarray | None = None,
        age: int | None = None,
        gender: int | None = None,
    ) -> None:
        """Initialize mock face data.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            embedding: Face embedding vector
            landmarks: Facial landmark points
            age: Estimated age
            gender: Gender (0=female, 1=male)
        """
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = confidence

        if embedding is not None:
            self.embedding = embedding

        if landmarks is not None:
            self.kps = landmarks

        if age is not None:
            self.age = age

        if gender is not None:
            self.gender = gender


class MockFaceAnalysis:
    """Mock InsightFace FaceAnalysis model."""

    def __init__(
        self,
        name: str = "buffalo_l",
        root: str = ".",
        providers: list[str] | None = None,
    ) -> None:
        """Initialize mock face analysis model.

        Args:
            name: Model name
            root: Model root directory
            providers: Execution providers
        """
        self.name = name
        self.root = Path(root)
        self.providers = providers or ["CPUExecutionProvider"]
        self.prepared = False

        # Mock faces to return - can be customized per test
        self.mock_faces: list[MockFaceData] = []
        self.should_raise_error = False
        self.error_message = "Mock error"

    def prepare(self, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)) -> None:
        """Mock preparation of the model.

        Args:
            ctx_id: Context ID for device
            det_size: Detection size
        """
        if self.should_raise_error:
            raise RuntimeError(self.error_message)

        self.prepared = True
        self.ctx_id = ctx_id
        self.det_size = det_size

    def get(self, img: np.ndarray) -> list[MockFaceData]:
        """Mock face detection on an image.

        Args:
            img: Input image as numpy array

        Returns:
            List of detected faces
        """
        if not self.prepared:
            raise RuntimeError("Model not prepared")

        if self.should_raise_error:
            raise RuntimeError(self.error_message)

        # Return mock faces (can be customized for each test)
        return self.mock_faces

    def set_mock_faces(self, faces: list[MockFaceData]) -> None:
        """Set faces to return in get() method.

        Args:
            faces: List of mock face data to return
        """
        self.mock_faces = faces

    def set_error(self, should_raise: bool, message: str = "Mock error") -> None:
        """Configure error behavior.

        Args:
            should_raise: Whether to raise errors
            message: Error message to use
        """
        self.should_raise_error = should_raise
        self.error_message = message


class MockInsightFaceApp:
    """Mock InsightFace app module."""

    @staticmethod
    def FaceAnalysis(**kwargs: Any) -> MockFaceAnalysis:
        """Create mock FaceAnalysis instance.

        Args:
            **kwargs: Arguments passed to FaceAnalysis

        Returns:
            Mock FaceAnalysis instance
        """
        return MockFaceAnalysis(**kwargs)


class MockInsightFace:
    """Mock InsightFace module."""

    app = MockInsightFaceApp()


def create_mock_face_data(
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 100.0, 120.0),
    confidence: float = 0.85,
    with_embedding: bool = True,
    with_landmarks: bool = True,
    with_attributes: bool = False,
    embedding_size: int = 512,
) -> MockFaceData:
    """Create mock face data with specified features.

    Args:
        bbox: Bounding box coordinates
        confidence: Detection confidence
        with_embedding: Whether to include face embedding
        with_landmarks: Whether to include facial landmarks
        with_attributes: Whether to include age/gender attributes
        embedding_size: Size of embedding vector

    Returns:
        Mock face data
    """
    embedding = None
    if with_embedding:
        # Create normalized random embedding
        embedding = np.random.rand(embedding_size).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

    landmarks = None
    if with_landmarks:
        # Create mock 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
        landmarks = np.array(
            [
                [30.0, 40.0],  # left eye
                [70.0, 40.0],  # right eye
                [50.0, 60.0],  # nose
                [35.0, 80.0],  # left mouth
                [65.0, 80.0],  # right mouth
            ],
            dtype=np.float32,
        )

    age = 25 if with_attributes else None
    gender = 1 if with_attributes else None  # 1 = male

    return MockFaceData(
        bbox=bbox,
        confidence=confidence,
        embedding=embedding,
        landmarks=landmarks,
        age=age,
        gender=gender,
    )


def create_test_image_array(
    width: int = 640,
    height: int = 480,
    channels: int = 3,
) -> np.ndarray:
    """Create test image array for face analysis.

    Args:
        width: Image width
        height: Image height
        channels: Number of channels (3 for RGB)

    Returns:
        Random image array
    """
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
