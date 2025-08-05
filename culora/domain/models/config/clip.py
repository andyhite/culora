"""CLIP semantic embedding configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from culora.domain.enums.clip import CLIPModelType, ClusteringMethod, SimilarityMetric


class CLIPConfig(BaseModel):
    """Configuration for CLIP semantic embeddings."""

    # Model settings
    model_name: CLIPModelType = Field(
        default=CLIPModelType.OPENAI_CLIP_VIT_B_32,
        description="CLIP model variant to use for embeddings",
    )
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home()
        / "Library"
        / "Application Support"
        / "culora"
        / "clip_models",
        description="Directory to cache CLIP models",
    )
    device_preference: str = Field(
        default="auto", description="Device preference: 'auto', 'cuda', 'mps', or 'cpu'"
    )

    # Embedding settings
    normalize_embeddings: bool = Field(
        default=True, description="Normalize embeddings to unit vectors"
    )
    embedding_precision: str = Field(
        default="float32", description="Embedding precision: 'float16' or 'float32'"
    )
    enable_embedding_cache: bool = Field(
        default=True, description="Cache embeddings to avoid recomputation"
    )
    cache_compression: bool = Field(
        default=True, description="Compress cached embeddings to save disk space"
    )

    # Similarity settings
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Default similarity metric for comparisons",
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Default similarity threshold for grouping",
    )

    # Clustering settings
    clustering_method: ClusteringMethod = Field(
        default=ClusteringMethod.KMEANS, description="Default clustering algorithm"
    )
    max_clusters: int = Field(
        default=20, ge=2, le=100, description="Maximum number of clusters"
    )
    min_cluster_size: int = Field(
        default=2, ge=1, description="Minimum images per cluster"
    )
    enable_auto_clustering: bool = Field(
        default=True, description="Automatically determine optimal cluster count"
    )

    # Performance settings
    batch_size: int = Field(
        default=8, ge=1, le=32, description="Batch size for embedding extraction"
    )
    max_image_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Maximum image size for CLIP processing (width, height)",
    )
    num_workers: int = Field(
        default=2, ge=1, le=8, description="Number of worker processes for batching"
    )
    memory_limit_mb: int = Field(
        default=2048, ge=512, description="Memory limit for embedding operations"
    )

    # Diversity optimization settings
    diversity_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in selection algorithms",
    )
    quality_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for quality in selection algorithms",
    )
    min_diversity_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable diversity score",
    )

    # Analysis settings
    enable_similarity_analysis: bool = Field(
        default=True, description="Enable pairwise similarity analysis"
    )
    enable_clustering_analysis: bool = Field(
        default=True, description="Enable semantic clustering analysis"
    )
    enable_diversity_analysis: bool = Field(
        default=True, description="Enable diversity scoring analysis"
    )
    max_similarity_pairs: int = Field(
        default=10, ge=1, description="Maximum similarity pairs to report"
    )

    # Export settings
    export_embeddings: bool = Field(
        default=False, description="Export raw embeddings to files"
    )
    export_similarity_matrix: bool = Field(
        default=False, description="Export full similarity matrix"
    )
    export_clusters: bool = Field(
        default=True, description="Export cluster assignments and metadata"
    )

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration consistency."""
        # Ensure weights sum to 1.0
        if abs(self.diversity_weight + self.quality_weight - 1.0) > 1e-6:
            raise ValueError("Diversity and quality weights must sum to 1.0")

        # Validate model cache directory
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
