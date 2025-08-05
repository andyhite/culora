"""Mock implementations for CLIP testing."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from culora.domain.enums.clip import ClusteringMethod, SimilarityMetric
from culora.domain.models.clip import (
    BatchSemanticResult,
    ClusteringResult,
    DiversityAnalysis,
    EmbeddingSimilarity,
    SemanticAnalysisResult,
    SemanticCluster,
    SemanticEmbedding,
)


def create_mock_clip_model() -> MagicMock:
    """Create a mock CLIP model for testing."""
    model = MagicMock()

    # Mock get_image_features to return diverse embeddings for clustering
    def mock_get_image_features(**kwargs: Any) -> torch.Tensor:
        # Create different embeddings based on input to simulate different images
        batch_size = kwargs.get("pixel_values", torch.tensor([[[[1.0]]]])).shape[0]
        embedding_dim = 512
        embeddings = []

        for i in range(batch_size):
            # Create clearly distinct embeddings that can be properly clustered
            if i < 2:  # First cluster group
                base_embedding = np.zeros(embedding_dim)
                base_embedding[0] = 0.8 + (i * 0.1)  # 0.8, 0.9
                base_embedding[1] = 0.6 - (i * 0.1)  # 0.6, 0.5
            elif i < 4:  # Second cluster group
                base_embedding = np.zeros(embedding_dim)
                base_embedding[2] = 0.8 + ((i - 2) * 0.1)  # 0.8, 0.9
                base_embedding[3] = 0.6 - ((i - 2) * 0.1)  # 0.6, 0.5
            else:  # Third cluster group
                base_embedding = np.zeros(embedding_dim)
                base_embedding[4] = 0.8 + ((i - 4) * 0.1)
                base_embedding[5] = 0.6 - ((i - 4) * 0.1)

            # Add some random noise to other dimensions
            noise = np.random.RandomState(42 + i).normal(0, 0.05, embedding_dim)
            base_embedding += noise

            # Normalize to unit vector
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            embeddings.append(base_embedding)

        embedding_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
        return embedding_tensor

    model.get_image_features.side_effect = mock_get_image_features
    model.eval.return_value = model
    model.to.return_value = model

    return model


def create_mock_clip_processor() -> MagicMock:
    """Create a mock CLIP processor for testing."""
    processor = MagicMock()

    def mock_process(images: Any = None, **_kwargs: Any) -> dict[str, torch.Tensor]:
        # Return mock inputs that look like real CLIP inputs
        if isinstance(images, list | tuple):
            batch_size = len(images)
        elif isinstance(images, Image.Image):
            batch_size = 1
        else:
            batch_size = 1

        return {
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
            "attention_mask": torch.ones(batch_size, 77, dtype=torch.long),
        }

    processor.side_effect = mock_process
    return processor


def create_mock_semantic_embedding(
    path: Path,
    embedding_dim: int = 512,
    model_name: str = "openai/clip-vit-base-patch32",
    _seed: int = 42,
) -> SemanticEmbedding:
    """Create a mock semantic embedding for testing."""
    # Generate deterministic embedding based on path
    rng = np.random.RandomState(hash(str(path)) % 2**32)
    embedding = rng.normal(0, 1, embedding_dim)
    # Normalize to unit vector
    embedding = embedding / np.linalg.norm(embedding)

    return SemanticEmbedding(
        path=path,
        embedding=embedding.tolist(),
        model_name=model_name,
        embedding_dimension=embedding_dim,
        extraction_time=0.1,
        confidence_score=0.85,
    )


def create_mock_semantic_analysis_result(
    path: Path,
    success: bool = True,
    embedding: SemanticEmbedding | None = None,
) -> SemanticAnalysisResult:
    """Create a mock semantic analysis result for testing."""
    if success and embedding is None:
        embedding = create_mock_semantic_embedding(path)

    return SemanticAnalysisResult(
        path=path,
        success=success,
        embedding=embedding,
        error=None if success else "Mock error",
        error_code=None if success else "MOCK_ERROR",
        analysis_duration=0.1,
    )


def create_mock_batch_semantic_result(
    paths: list[Path],
    success_rate: float = 1.0,
) -> BatchSemanticResult:
    """Create a mock batch semantic result for testing."""
    results = []
    successful_count = int(len(paths) * success_rate)

    for i, path in enumerate(paths):
        success = i < successful_count
        result = create_mock_semantic_analysis_result(path, success)
        results.append(result)

    embedding_stats = {
        "mean_norm": 1.0,
        "std_norm": 0.1,
        "dimension": 512,
        "sparsity": 0.0,
    }

    return BatchSemanticResult(
        results=results,
        successful_analyses=successful_count,
        failed_analyses=len(paths) - successful_count,
        total_duration=len(paths) * 0.1,
        embeddings_per_second=10.0,
        mean_similarity=0.5,
        embedding_statistics=embedding_stats,
    )


def create_mock_embedding_similarity(
    path1: Path,
    path2: Path,
    similarity_score: float = 0.7,
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> EmbeddingSimilarity:
    """Create a mock embedding similarity for testing."""
    distance = (
        1.0 - similarity_score
        if metric == SimilarityMetric.COSINE
        else similarity_score
    )

    return EmbeddingSimilarity(
        path1=path1,
        path2=path2,
        similarity_score=similarity_score,
        distance=distance,
        metric=metric,
    )


def create_mock_diversity_analysis(
    embeddings: list[SemanticEmbedding],
    mean_similarity: float = 0.5,
) -> DiversityAnalysis:
    """Create a mock diversity analysis for testing."""
    diversity_score = 1.0 - mean_similarity

    # Create mock similarity pairs
    most_similar = []
    most_diverse = []

    if len(embeddings) >= 2:
        # Create a few mock similarity pairs
        most_similar = [
            create_mock_embedding_similarity(
                embeddings[0].path, embeddings[1].path, 0.9
            )
        ]
        most_diverse = [
            create_mock_embedding_similarity(
                embeddings[0].path, embeddings[-1].path, 0.1
            )
        ]

    return DiversityAnalysis(
        total_images=len(embeddings),
        mean_pairwise_similarity=mean_similarity,
        diversity_score=diversity_score,
        similarity_distribution={
            "p10": 0.1,
            "p25": 0.3,
            "p50": 0.5,
            "p75": 0.7,
            "p90": 0.9,
        },
        most_similar_pairs=most_similar,
        most_diverse_pairs=most_diverse,
    )


def create_mock_semantic_cluster(
    cluster_id: int,
    image_paths: list[Path],
    embedding_dim: int = 512,
) -> SemanticCluster:
    """Create a mock semantic cluster for testing."""
    # Generate mock centroid
    rng = np.random.RandomState(cluster_id)
    centroid = rng.normal(0, 1, embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    return SemanticCluster(
        cluster_id=cluster_id,
        image_paths=image_paths,
        centroid=centroid.tolist(),
        intra_cluster_similarity=0.8,
        size=len(image_paths),
    )


def create_mock_clustering_result(
    embeddings: list[SemanticEmbedding],
    num_clusters: int = 3,
    method: ClusteringMethod = ClusteringMethod.KMEANS,
) -> ClusteringResult:
    """Create a mock clustering result for testing."""
    # Distribute embeddings across clusters
    cluster_size = max(1, len(embeddings) // num_clusters)
    clusters = []

    for i in range(num_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, len(embeddings))

        if start_idx < len(embeddings):
            cluster_paths = [emb.path for emb in embeddings[start_idx:end_idx]]
            if cluster_paths:  # Only create cluster if it has paths
                cluster = create_mock_semantic_cluster(i, cluster_paths)
                clusters.append(cluster)

    size_distribution = {
        "min": min(c.size for c in clusters) if clusters else 0,
        "max": max(c.size for c in clusters) if clusters else 0,
        "mean": int(np.mean([c.size for c in clusters])) if clusters else 0,
        "std": int(np.std([c.size for c in clusters])) if clusters else 0,
    }

    return ClusteringResult(
        method=method,
        num_clusters=len(clusters),
        clusters=clusters,
        silhouette_score=0.6,
        inertia=100.0 if method == ClusteringMethod.KMEANS else None,
        cluster_size_distribution=size_distribution,
        processing_time=0.5,
    )


# Mock data generators for different test scenarios
MOCK_EMBEDDINGS = {
    "similar_group": [
        [0.8, 0.6, 0.2, 0.1] + [0.0] * 508,  # Cluster 1 - similar to next
        [0.7, 0.7, 0.3, 0.1] + [0.0] * 508,  # Cluster 1 - similar to previous
        [0.2, 0.1, 0.8, 0.6] + [0.0] * 508,  # Cluster 2 - different group
        [0.1, 0.2, 0.7, 0.7] + [0.0] * 508,  # Cluster 2 - similar to previous
    ],
    "diverse_group": [
        [1.0, 0.0, 0.0, 0.0] + [0.0] * 508,  # Completely different directions
        [0.0, 1.0, 0.0, 0.0] + [0.0] * 508,  # Completely different directions
        [0.0, 0.0, 1.0, 0.0] + [0.0] * 508,  # Completely different directions
        [0.0, 0.0, 0.0, 1.0] + [0.0] * 508,  # Completely different directions
    ],
    "uniform_group": [
        [0.6, 0.4, 0.3, 0.2] + [0.0] * 508,  # More diverse variations
        [0.3, 0.7, 0.2, 0.4] + [0.0] * 508,  # More diverse variations
        [0.4, 0.2, 0.8, 0.1] + [0.0] * 508,  # More diverse variations
        [0.2, 0.3, 0.4, 0.9] + [0.0] * 508,  # More diverse variations
    ],
}
