"""Pose estimation domain models."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PoseLandmark(BaseModel):
    """Individual pose landmark with coordinates and visibility."""

    x: float = Field(description="Normalized x coordinate (0-1)")
    y: float = Field(description="Normalized y coordinate (0-1)")
    z: float = Field(description="Depth coordinate (relative)")
    visibility: float = Field(
        description="Landmark visibility score (0-1)", ge=0.0, le=1.0
    )
    presence: float = Field(description="Landmark presence score (0-1)", ge=0.0, le=1.0)


class PoseVector(BaseModel):
    """Pose feature vector for clustering and comparison."""

    vector: list[float] = Field(description="Normalized pose feature vector")
    vector_dimension: int = Field(description="Dimension of the feature vector")
    confidence: float = Field(
        description="Overall confidence in pose detection", ge=0.0, le=1.0
    )


class PoseCategory(str, Enum):
    """Pose categories based on body position and posture."""

    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    KNEELING = "kneeling"
    CROUCHING = "crouching"
    UNKNOWN = "unknown"


class PoseOrientation(str, Enum):
    """Body orientation relative to camera."""

    FRONTAL = "frontal"
    PROFILE = "profile"
    THREE_QUARTER = "three_quarter"
    BACK = "back"
    UNKNOWN = "unknown"


class ArmPosition(str, Enum):
    """Arm positioning categories."""

    RAISED = "raised"
    EXTENDED = "extended"
    CROSSED = "crossed"
    AT_SIDES = "at_sides"
    ON_HIPS = "on_hips"
    BEHIND_BACK = "behind_back"
    UNKNOWN = "unknown"


class LegPosition(str, Enum):
    """Leg positioning categories."""

    STRAIGHT = "straight"
    BENT = "bent"
    CROSSED = "crossed"
    SPREAD = "spread"
    ONE_RAISED = "one_raised"
    UNKNOWN = "unknown"


class PoseSymmetry(str, Enum):
    """Pose symmetry classification."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    UNKNOWN = "unknown"


class PoseDynamism(str, Enum):
    """Pose dynamism level."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    ACTION = "action"
    UNKNOWN = "unknown"


class PoseClassification(BaseModel):
    """Pose classification results."""

    category: PoseCategory = Field(description="Primary pose category")
    orientation: PoseOrientation = Field(description="Body orientation")
    arm_position: ArmPosition = Field(description="Arm positioning")
    leg_position: LegPosition = Field(description="Leg positioning")
    symmetry: PoseSymmetry = Field(description="Pose symmetry")
    dynamism: PoseDynamism = Field(description="Pose dynamism level")
    confidence: float = Field(description="Classification confidence", ge=0.0, le=1.0)

    model_config = ConfigDict(use_enum_values=True)


class PoseAnalysis(BaseModel):
    """Complete pose analysis for a single image."""

    path: Path = Field(description="Path to analyzed image")
    landmarks: list[PoseLandmark] = Field(description="Detected pose landmarks")
    pose_vector: PoseVector = Field(description="Pose feature vector")
    classification: PoseClassification = Field(description="Pose classification")
    bbox: tuple[float, float, float, float] = Field(
        description="Pose bounding box (x, y, width, height)"
    )
    pose_score: float = Field(description="Overall pose quality score", ge=0.0, le=1.0)
    analysis_duration: float = Field(description="Analysis time in seconds")

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)


class PoseAnalysisResult(BaseModel):
    """Result of pose analysis for a single image."""

    path: Path = Field(description="Path to analyzed image")
    success: bool = Field(description="Whether analysis succeeded")
    pose_analysis: PoseAnalysis | None = Field(
        default=None, description="Pose analysis if successful"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    error_code: str | None = Field(default=None, description="Error code if failed")
    analysis_duration: float = Field(description="Total analysis time in seconds")

    model_config = ConfigDict(json_encoders={Path: str})


class BatchPoseResult(BaseModel):
    """Results of batch pose analysis."""

    results: list[PoseAnalysisResult] = Field(description="Individual analysis results")
    successful_analyses: int = Field(description="Number of successful analyses")
    failed_analyses: int = Field(description="Number of failed analyses")
    total_duration: float = Field(description="Total processing time")
    poses_per_second: float = Field(description="Processing rate")
    mean_pose_score: float = Field(
        description="Mean pose quality score", ge=0.0, le=1.0
    )
    pose_statistics: dict[str, float] = Field(
        description="Statistics about pose vectors"
    )

    model_config = ConfigDict(json_encoders={Path: str})


class PoseSimilarity(BaseModel):
    """Similarity score between two poses."""

    path1: Path = Field(description="Path to first image")
    path2: Path = Field(description="Path to second image")
    similarity_score: float = Field(
        description="Pose similarity score between 0 and 1", ge=0.0, le=1.0
    )
    distance: float = Field(description="Pose distance metric value")
    landmark_matches: dict[str, float] = Field(
        description="Individual landmark similarity scores"
    )

    model_config = ConfigDict(json_encoders={Path: str})


class PoseCluster(BaseModel):
    """Cluster of similar poses."""

    cluster_id: int = Field(description="Unique cluster identifier")
    image_paths: list[Path] = Field(description="Paths of images in this cluster")
    centroid_vector: list[float] = Field(description="Cluster centroid pose vector")
    intra_cluster_similarity: float = Field(
        description="Average similarity within cluster", ge=0.0, le=1.0
    )
    dominant_category: PoseCategory = Field(
        description="Most common pose category in cluster"
    )
    size: int = Field(description="Number of images in cluster", ge=1)

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)


class PoseClusteringResult(BaseModel):
    """Result of pose clustering analysis."""

    num_clusters: int = Field(description="Number of clusters found", ge=1)
    clusters: list[PoseCluster] = Field(description="Individual clusters")
    silhouette_score: float = Field(
        description="Clustering quality metric", ge=-1.0, le=1.0
    )
    cluster_size_distribution: dict[str, int] = Field(
        description="Distribution of cluster sizes"
    )
    category_distribution: dict[str, int] = Field(
        description="Distribution of pose categories across clusters"
    )
    processing_time: float = Field(description="Time taken for clustering")

    model_config = ConfigDict(use_enum_values=True)


class PoseDiversityAnalysis(BaseModel):
    """Analysis of pose diversity in a dataset."""

    total_images: int = Field(description="Total number of images analyzed")
    category_distribution: dict[str, int] = Field(
        description="Distribution of pose categories"
    )
    orientation_distribution: dict[str, int] = Field(
        description="Distribution of body orientations"
    )
    mean_pairwise_similarity: float = Field(
        description="Average similarity between all pairs", ge=0.0, le=1.0
    )
    diversity_score: float = Field(
        description="Overall diversity score (1 - mean_similarity)", ge=0.0, le=1.0
    )
    similarity_distribution: dict[str, float] = Field(
        description="Percentile distribution of similarity scores"
    )
    most_similar_pairs: list[PoseSimilarity] = Field(
        description="Most similar pose pairs"
    )
    most_diverse_pairs: list[PoseSimilarity] = Field(
        description="Most diverse pose pairs"
    )

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)


class PoseSelectionCriteria(BaseModel):
    """Criteria for pose diversity-based selection."""

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
    category_balance: bool = Field(
        default=True, description="Balance across pose categories"
    )
    orientation_balance: bool = Field(
        default=True, description="Balance across orientations"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that weights sum to 1.0."""
        if abs(self.diversity_weight + self.quality_weight - 1.0) > 1e-6:
            raise ValueError("Diversity and quality weights must sum to 1.0")


class PoseSelectionResult(BaseModel):
    """Result of pose diversity-based image selection."""

    selected_paths: list[Path] = Field(description="Paths of selected images")
    selection_criteria: PoseSelectionCriteria = Field(
        description="Criteria used for selection"
    )
    diversity_score: float = Field(
        description="Diversity score of selected set", ge=0.0, le=1.0
    )
    mean_quality_score: float = Field(
        description="Mean quality score of selected images", ge=0.0, le=1.0
    )
    category_representation: dict[str, int] = Field(
        description="Number of images selected from each pose category"
    )
    orientation_representation: dict[str, int] = Field(
        description="Number of images selected from each orientation"
    )
    cluster_representation: dict[int, int] = Field(
        description="Number of images selected from each cluster"
    )
    selection_reasoning: list[str] = Field(
        description="Human-readable selection reasoning"
    )
    processing_time: float = Field(description="Time taken for selection")

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)
