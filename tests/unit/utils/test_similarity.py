"""Tests for similarity utilities."""

import numpy as np
import pytest

from culora.utils.similarity import average_embeddings, cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_embeddings(self) -> None:
        """Test cosine similarity between identical embeddings."""
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(embedding, embedding)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_embeddings(self) -> None:
        """Test cosine similarity between orthogonal embeddings."""
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(embedding1, embedding2)
        assert abs(similarity - 0.0) < 1e-6

    def test_opposite_embeddings(self) -> None:
        """Test cosine similarity between opposite embeddings (clamped to 0)."""
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(embedding1, embedding2)
        # Similarity is clamped to [0, 1] range for face embeddings
        assert abs(similarity - 0.0) < 1e-6

    def test_normalized_result(self) -> None:
        """Test that cosine similarity result is normalized."""
        embedding1 = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # magnitude = 5
        embedding2 = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # magnitude = 1
        similarity = cosine_similarity(embedding1, embedding2)
        assert 0.0 <= similarity <= 1.0


class TestAverageEmbeddings:
    """Tests for embedding averaging."""

    def test_empty_embeddings_raises_error(self) -> None:
        """Test that empty embeddings list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot average empty list"):
            average_embeddings([])

    def test_single_embedding(self) -> None:
        """Test averaging single embedding returns copy."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = average_embeddings([embedding])

        assert np.array_equal(result, embedding)
        assert result is not embedding  # Should be a copy

    def test_two_embeddings(self) -> None:
        """Test averaging two embeddings."""
        embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        result = average_embeddings([embedding1, embedding2])

        assert np.allclose(result, expected)

    def test_multiple_embeddings(self) -> None:
        """Test averaging multiple embeddings."""
        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

        result = average_embeddings(embeddings)

        assert np.allclose(result, expected)

    def test_inconsistent_dimensions_raises_error(self) -> None:
        """Test that inconsistent embedding dimensions raise ValueError."""
        embedding1 = np.array([1.0, 2.0], dtype=np.float32)
        embedding2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(ValueError, match="same shape"):
            average_embeddings([embedding1, embedding2])

    def test_result_dtype(self) -> None:
        """Test that result has correct dtype."""
        embeddings = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ]

        result = average_embeddings(embeddings)

        assert result.dtype == np.float32
