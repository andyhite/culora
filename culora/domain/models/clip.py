"""CLIP semantic embedding domain models."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from culora.domain.enums.clip import ClusteringMethod, SimilarityMetric


class SemanticEmbedding(BaseModel):
    """Semantic embedding vector with metadata."""

    path: Path = Field(description="Path to the source image")
    embedding: list[float] = Field(description="Normalized embedding vector")
    model_name: str = Field(description="CLIP model used for extraction")
    embedding_dimension: int = Field(description="Dimension of the embedding vector")
    extraction_time: float = Field(description="Time taken to extract embedding")
    confidence_score: float | None = Field(
        default=None, description="Model confidence in the embedding quality"
    )

    model_config = ConfigDict(json_encoders={Path: str})


class EmbeddingSimilarity(BaseModel):
    """Similarity score between two embeddings."""

    path1: Path = Field(description="Path to first image")
    path2: Path = Field(description="Path to second image")
    similarity_score: float = Field(
        description="Similarity score between 0 and 1", ge=0.0, le=1.0
    )
    distance: float = Field(description="Distance metric value")
    metric: SimilarityMetric = Field(description="Similarity calculation method")

    model_config = ConfigDict(json_encoders={Path: str})


class SemanticCluster(BaseModel):
    """Cluster of semantically similar images."""

    cluster_id: int = Field(description="Unique cluster identifier")
    image_paths: list[Path] = Field(description="Paths of images in this cluster")
    centroid: list[float] = Field(description="Cluster centroid embedding")
    intra_cluster_similarity: float = Field(
        description="Average similarity within cluster", ge=0.0, le=1.0
    )
    size: int = Field(description="Number of images in cluster", ge=1)

    model_config = ConfigDict(json_encoders={Path: str})


class SemanticAnalysisResult(BaseModel):
    """Result of semantic analysis for a single image."""

    path: Path = Field(description="Path to analyzed image")
    success: bool = Field(description="Whether analysis succeeded")
    embedding: SemanticEmbedding | None = Field(
        default=None, description="Extracted semantic embedding"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    error_code: str | None = Field(default=None, description="Error code if failed")
    analysis_duration: float = Field(description="Total analysis time in seconds")

    model_config = ConfigDict(json_encoders={Path: str})


class BatchSemanticResult(BaseModel):
    """Results of batch semantic analysis."""

    results: list[SemanticAnalysisResult] = Field(
        description="Individual analysis results"
    )
    successful_analyses: int = Field(description="Number of successful analyses")
    failed_analyses: int = Field(description="Number of failed analyses")
    total_duration: float = Field(description="Total processing time")
    embeddings_per_second: float = Field(description="Processing rate")
    mean_similarity: float = Field(
        description="Mean pairwise similarity", ge=0.0, le=1.0
    )
    embedding_statistics: dict[str, float] = Field(
        description="Statistics about embedding vectors"
    )
    clusters: list[SemanticCluster] | None = Field(
        default=None, description="Semantic clusters if clustering was performed"
    )
    cluster_statistics: dict[str, Any] | None = Field(
        default=None, description="Clustering quality metrics"
    )

    model_config = ConfigDict(json_encoders={Path: str})


class DiversityAnalysis(BaseModel):
    """Analysis of semantic diversity in a dataset."""

    total_images: int = Field(description="Total number of images analyzed")
    mean_pairwise_similarity: float = Field(
        description="Average similarity between all pairs", ge=0.0, le=1.0
    )
    diversity_score: float = Field(
        description="Overall diversity score (1 - mean_similarity)", ge=0.0, le=1.0
    )
    similarity_distribution: dict[str, float] = Field(
        description="Percentile distribution of similarity scores"
    )
    most_similar_pairs: list[EmbeddingSimilarity] = Field(
        description="Most similar image pairs"
    )
    most_diverse_pairs: list[EmbeddingSimilarity] = Field(
        description="Most diverse image pairs"
    )

    model_config = ConfigDict(json_encoders={Path: str})


class ClusteringResult(BaseModel):
    """Result of semantic clustering analysis."""

    method: ClusteringMethod = Field(description="Clustering algorithm used")
    num_clusters: int = Field(description="Number of clusters found", ge=1)
    clusters: list[SemanticCluster] = Field(description="Individual clusters")
    silhouette_score: float = Field(
        description="Clustering quality metric", ge=-1.0, le=1.0
    )
    inertia: float | None = Field(
        default=None, description="Within-cluster sum of squares (K-means only)"
    )
    cluster_size_distribution: dict[str, int] = Field(
        description="Distribution of cluster sizes"
    )
    processing_time: float = Field(description="Time taken for clustering")

    model_config = ConfigDict()


class SemanticSelectionCriteria(BaseModel):
    """Criteria for semantic diversity-based selection."""

    target_count: int = Field(description="Target number of images to select", ge=1)
    diversity_weight: float = Field(
        description="Weight for diversity optimization", ge=0.0, le=1.0, default=0.7
    )
    quality_weight: float = Field(
        description="Weight for quality preservation", ge=0.0, le=1.0, default=0.3
    )
    min_cluster_representation: int = Field(
        description="Minimum images per cluster", ge=1, default=1
    )
    max_cluster_representation: int | None = Field(
        default=None, description="Maximum images per cluster"
    )
    similarity_threshold: float = Field(
        description="Minimum similarity threshold for grouping",
        ge=0.0,
        le=1.0,
        default=0.8,
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that weights sum to 1.0."""
        if abs(self.diversity_weight + self.quality_weight - 1.0) > 1e-6:
            raise ValueError("Diversity and quality weights must sum to 1.0")


class SemanticSelectionResult(BaseModel):
    """Result of semantic diversity-based image selection."""

    selected_paths: list[Path] = Field(description="Paths of selected images")
    selection_criteria: SemanticSelectionCriteria = Field(
        description="Criteria used for selection"
    )
    diversity_score: float = Field(
        description="Diversity score of selected set", ge=0.0, le=1.0
    )
    mean_quality_score: float = Field(
        description="Mean quality score of selected images", ge=0.0, le=1.0
    )
    cluster_representation: dict[int, int] = Field(
        description="Number of images selected from each cluster"
    )
    selection_reasoning: list[str] = Field(
        description="Human-readable selection reasoning"
    )
    processing_time: float = Field(description="Time taken for selection")

    model_config = ConfigDict(json_encoders={Path: str})
