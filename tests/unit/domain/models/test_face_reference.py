"""Tests for face reference domain models."""

from pathlib import Path

import numpy as np

from culora.domain.models.face import FaceDetection
from culora.domain.models.face_reference import (
    ReferenceEmbedding,
    ReferenceImage,
    ReferenceMatchResult,
    ReferenceProcessingResult,
    ReferenceSet,
    SimilarityMatch,
)


class TestReferenceEmbedding:
    """Tests for ReferenceEmbedding dataclass."""

    def test_reference_embedding_creation(self) -> None:
        """Test creating a reference embedding."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        ref_embedding = ReferenceEmbedding(
            embedding=embedding,
            face_detection=face_detection,
            source_image=Path("/test/image.jpg"),
            confidence_score=0.95,
        )

        assert np.array_equal(ref_embedding.embedding, embedding)
        assert ref_embedding.face_detection == face_detection
        assert ref_embedding.source_image == Path("/test/image.jpg")
        assert ref_embedding.confidence_score == 0.95


class TestReferenceImage:
    """Tests for ReferenceImage model."""

    def test_empty_reference_image(self) -> None:
        """Test creating an empty reference image."""
        ref_img = ReferenceImage(image_path=Path("/test/image.jpg"))

        assert ref_img.image_path == Path("/test/image.jpg")
        assert ref_img.embeddings == []
        assert ref_img.processing_success is True
        assert ref_img.error_message is None
        assert not ref_img.has_embeddings

    def test_reference_image_with_embeddings(self) -> None:
        """Test reference image with embeddings."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        ref_embedding = ReferenceEmbedding(
            embedding=embedding,
            face_detection=face_detection,
            source_image=Path("/test/image.jpg"),
            confidence_score=0.95,
        )

        ref_img = ReferenceImage(
            image_path=Path("/test/image.jpg"),
            embeddings=[ref_embedding],
        )

        assert ref_img.has_embeddings
        assert len(ref_img.embeddings) == 1
        assert ref_img.primary_embedding == ref_embedding


class TestReferenceSet:
    """Tests for ReferenceSet model."""

    def test_empty_reference_set(self) -> None:
        """Test creating an empty reference set."""
        ref_set = ReferenceSet()

        assert ref_set.images == []
        assert ref_set.total_embeddings == 0
        assert ref_set.valid_images == []
        assert ref_set.get_all_embeddings() == []

    def test_reference_set_with_images(self) -> None:
        """Test reference set with valid images."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        ref_embedding = ReferenceEmbedding(
            embedding=embedding,
            face_detection=face_detection,
            source_image=Path("/test/image.jpg"),
            confidence_score=0.95,
        )

        ref_img = ReferenceImage(
            image_path=Path("/test/image.jpg"),
            embeddings=[ref_embedding],
        )

        ref_set = ReferenceSet()
        ref_set.add_reference_image(ref_img)

        assert len(ref_set.images) == 1
        assert ref_set.total_embeddings == 1
        assert len(ref_set.valid_images) == 1
        assert len(ref_set.get_all_embeddings()) == 1


class TestSimilarityMatch:
    """Tests for SimilarityMatch model."""

    def test_similarity_match_creation(self) -> None:
        """Test creating a similarity match."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        match = SimilarityMatch(
            face_detection=face_detection,
            similarities=[0.8, 0.9, 0.7],
            best_similarity=0.9,
            best_reference_index=1,
            meets_threshold=True,
        )

        assert match.face_detection == face_detection
        assert match.similarities == [0.8, 0.9, 0.7]
        assert match.best_similarity == 0.9
        assert match.best_reference_index == 1
        assert match.meets_threshold is True


class TestReferenceProcessingResult:
    """Tests for ReferenceProcessingResult model."""

    def test_successful_processing_result(self) -> None:
        """Test successful processing result."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        ref_embedding = ReferenceEmbedding(
            embedding=embedding,
            face_detection=face_detection,
            source_image=Path("/test/image.jpg"),
            confidence_score=0.95,
        )

        result = ReferenceProcessingResult(
            success=True,
            embeddings=[ref_embedding],
        )

        assert result.success is True
        assert len(result.embeddings) == 1
        assert result.error is None

    def test_failed_processing_result(self) -> None:
        """Test failed processing result."""
        result = ReferenceProcessingResult(
            success=False,
            error="No faces detected",
        )

        assert result.success is False
        assert result.embeddings == []
        assert result.error == "No faces detected"


class TestReferenceMatchResult:
    """Tests for ReferenceMatchResult model."""

    def test_successful_match_result(self) -> None:
        """Test successful match result."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=embedding,
            face_area_ratio=0.1,
        )

        match = SimilarityMatch(
            face_detection=face_detection,
            similarities=[0.9],
            best_similarity=0.9,
            best_reference_index=0,
            meets_threshold=True,
        )

        result = ReferenceMatchResult(
            image_path=Path("/test/image.jpg"),
            success=True,
            matches=[match],
            primary_face_index=0,
            processing_duration=0.5,
        )

        assert result.image_path == Path("/test/image.jpg")
        assert result.success is True
        assert len(result.matches) == 1
        assert result.primary_face_index == 0
        assert result.processing_duration == 0.5
        assert result.has_matches is True
        assert result.primary_match == match

    def test_failed_match_result(self) -> None:
        """Test failed match result."""
        result = ReferenceMatchResult(
            image_path=Path("/test/image.jpg"),
            success=False,
            error="Analysis failed",
            processing_duration=0.1,
        )

        assert result.success is False
        assert result.error == "Analysis failed"
        assert not result.has_matches
        assert result.primary_match is None
