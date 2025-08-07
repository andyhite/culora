"""Utilities for calculating face embedding similarities."""

import numpy as np


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two face embeddings.

    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector

    Returns:
        Cosine similarity score between 0 and 1 (1 = identical, 0 = completely different)

    Raises:
        ValueError: If embeddings have different dimensions or are invalid
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embedding dimensions must match: {embedding1.shape} vs {embedding2.shape}"
        )

    if embedding1.ndim != 1:
        raise ValueError(f"Embeddings must be 1-dimensional, got {embedding1.ndim}D")

    if len(embedding1) == 0:
        raise ValueError("Embeddings cannot be empty")

    # Calculate dot product
    dot_product = np.dot(embedding1, embedding2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    # Handle zero magnitude (shouldn't happen with valid face embeddings)
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    # Clamp to [0, 1] range (cosine similarity is normally [-1, 1])
    # Face embeddings should typically have positive similarity
    similarity = max(0.0, min(1.0, similarity))

    return float(similarity)


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Calculate the average of multiple face embeddings.

    Args:
        embeddings: List of face embedding vectors

    Returns:
        Averaged embedding

    Raises:
        ValueError: If embeddings are empty or have inconsistent dimensions
    """
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")

    if len(embeddings) == 1:
        return embeddings[0].copy()

    # Validate all embeddings have same shape
    first_shape = embeddings[0].shape
    for i, embedding in enumerate(embeddings[1:], 1):
        if embedding.shape != first_shape:
            raise ValueError(
                f"All embeddings must have same shape. "
                f"Embedding 0: {first_shape}, Embedding {i}: {embedding.shape}"
            )

    # Calculate mean across all embeddings
    stacked = np.stack(embeddings)
    averaged = np.mean(stacked, axis=0)

    result: np.ndarray = averaged.copy().astype(np.float32)
    return result


def similarity_to_references(
    face_embedding: np.ndarray,
    reference_embeddings: list[np.ndarray],
    method: str = "average",
) -> float:
    """
    Calculate similarity between a face and multiple reference embeddings.

    Args:
        face_embedding: Face embedding to compare
        reference_embeddings: List of reference face embeddings
        method: Comparison method - "average" (default), "max", or "min"

    Returns:
        Similarity score based on the specified method

    Raises:
        ValueError: If reference embeddings are empty or method is invalid
    """
    if not reference_embeddings:
        raise ValueError("Reference embeddings cannot be empty")

    if method not in ("average", "max", "min"):
        raise ValueError(
            f"Invalid method '{method}'. Must be 'average', 'max', or 'min'"
        )

    # Calculate similarity to each reference
    similarities = [
        cosine_similarity(face_embedding, ref_embedding)
        for ref_embedding in reference_embeddings
    ]

    # Apply the specified method
    if method == "average":
        result = float(np.mean(similarities))
    elif method == "max":
        result = float(np.max(similarities))
    elif method == "min":
        result = float(np.min(similarities))

    return result


def is_similarity_match(similarity: float, threshold: float) -> bool:
    """
    Determine if a similarity score indicates a match.

    Args:
        similarity: Similarity score (0-1)
        threshold: Minimum similarity for a match

    Returns:
        True if similarity meets or exceeds threshold
    """
    return similarity >= threshold


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding to unit length.

    Args:
        embedding: Face embedding vector

    Returns:
        L2-normalized embedding

    Raises:
        ValueError: If embedding has zero magnitude
    """
    magnitude = np.linalg.norm(embedding)

    if magnitude == 0.0:
        raise ValueError("Cannot normalize zero-magnitude embedding")

    normalized: np.ndarray = embedding / magnitude
    return normalized.copy()
