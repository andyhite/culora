"""Pose estimation configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PoseConfig(BaseModel):
    """Configuration for MediaPipe pose estimation."""

    # MediaPipe settings
    model_complexity: int = Field(
        default=1, ge=0, le=2, description="Model complexity (0=lite, 1=full, 2=heavy)"
    )
    min_detection_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for pose detection",
    )
    min_tracking_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for pose tracking",
    )
    enable_segmentation: bool = Field(
        default=False, description="Enable pose segmentation mask"
    )
    smooth_landmarks: bool = Field(
        default=True, description="Enable landmark smoothing"
    )
    smooth_segmentation: bool = Field(
        default=True, description="Enable segmentation smoothing"
    )

    # Processing settings
    max_image_size: tuple[int, int] = Field(
        default=(1024, 1024),
        description="Maximum image size for pose processing (width, height)",
    )
    batch_size: int = Field(
        default=4, ge=1, le=16, description="Batch size for pose analysis"
    )
    enable_pose_cache: bool = Field(
        default=True, description="Cache pose analysis results"
    )
    cache_compression: bool = Field(
        default=True, description="Compress cached pose data"
    )

    # Feature extraction settings
    feature_vector_dim: int = Field(
        default=66, ge=33, le=132, description="Pose feature vector dimension"
    )
    key_landmarks_only: bool = Field(
        default=True, description="Use only key landmarks for feature vector"
    )
    normalize_coordinates: bool = Field(
        default=True, description="Normalize landmark coordinates"
    )
    include_visibility: bool = Field(
        default=True, description="Include visibility scores in feature vector"
    )

    # Quality filtering settings
    min_pose_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum pose quality score for inclusion",
    )
    min_visible_landmarks: int = Field(
        default=20, ge=10, le=33, description="Minimum visible landmarks required"
    )
    min_landmark_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum individual landmark confidence",
    )

    # Classification settings
    enable_pose_classification: bool = Field(
        default=True, description="Enable automatic pose classification"
    )
    classification_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for pose classification",
    )

    # Clustering settings
    max_clusters: int = Field(
        default=15, ge=2, le=50, description="Maximum number of pose clusters"
    )
    min_cluster_size: int = Field(
        default=2, ge=1, description="Minimum images per pose cluster"
    )
    enable_auto_clustering: bool = Field(
        default=True, description="Automatically determine optimal cluster count"
    )

    # Diversity optimization settings
    diversity_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in pose selection",
    )
    quality_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for quality in pose selection",
    )
    min_diversity_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable pose diversity score",
    )

    # Analysis settings
    enable_similarity_analysis: bool = Field(
        default=True, description="Enable pairwise pose similarity analysis"
    )
    enable_clustering_analysis: bool = Field(
        default=True, description="Enable pose clustering analysis"
    )
    enable_diversity_analysis: bool = Field(
        default=True, description="Enable pose diversity analysis"
    )
    max_similarity_pairs: int = Field(
        default=10, ge=1, description="Maximum similarity pairs to report"
    )

    # Export settings
    export_landmarks: bool = Field(
        default=False, description="Export raw pose landmarks to files"
    )
    export_pose_vectors: bool = Field(
        default=False, description="Export pose feature vectors"
    )
    export_pose_visualization: bool = Field(
        default=True, description="Export pose visualization overlays"
    )
    export_clusters: bool = Field(
        default=True, description="Export pose cluster assignments and metadata"
    )

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration consistency."""
        # Ensure weights sum to 1.0
        if abs(self.diversity_weight + self.quality_weight - 1.0) > 1e-6:
            raise ValueError("Diversity and quality weights must sum to 1.0")

        # Validate feature vector dimension
        min_dim = 33 if not self.key_landmarks_only else 21
        if self.include_visibility:
            min_dim *= 2
        if self.feature_vector_dim < min_dim:
            raise ValueError(
                f"Feature vector dimension must be at least {min_dim} "
                f"with current settings"
            )
