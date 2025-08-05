"""Mock objects for pose estimation testing."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from culora.domain.enums.pose import (
    ArmPosition,
    LegPosition,
    PoseCategory,
    PoseDynamism,
    PoseOrientation,
    PoseSymmetry,
)
from culora.domain.models.pose import (
    BatchPoseResult,
    PoseAnalysis,
    PoseAnalysisResult,
    PoseClassification,
    PoseCluster,
    PoseClusteringResult,
    PoseDiversityAnalysis,
    PoseLandmark,
    PoseSelectionResult,
    PoseSimilarity,
    PoseVector,
)


def create_mock_mediapipe_pose() -> MagicMock:
    """Create a mock MediaPipe pose estimator."""
    mock_pose = MagicMock()

    # Mock pose estimation results
    def mock_process(_image: np.ndarray) -> MagicMock:
        results = MagicMock()

        # Create mock landmarks (33 landmarks for full body pose)
        mock_landmarks = []
        for i in range(33):
            landmark = MagicMock()
            # Generate realistic landmark positions
            landmark.x = 0.3 + (i % 5) * 0.1  # Spread across image width
            landmark.y = 0.2 + (i // 11) * 0.25  # Three rows (head, torso, legs)
            landmark.z = np.random.normal(0, 0.1)  # Random depth
            landmark.visibility = np.random.uniform(0.7, 1.0)  # High visibility
            landmark.presence = np.random.uniform(0.8, 1.0)  # High presence
            mock_landmarks.append(landmark)

        # Mock pose landmarks container
        pose_landmarks = MagicMock()
        pose_landmarks.landmark = mock_landmarks
        results.pose_landmarks = pose_landmarks

        return results

    mock_pose.process.side_effect = mock_process
    return mock_pose


def create_mock_pose_landmark(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    visibility: float = 0.9,
    presence: float = 0.9,
) -> PoseLandmark:
    """Create a mock pose landmark."""
    return PoseLandmark(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
        presence=presence,
    )


def create_mock_pose_vector(
    dimension: int = 66,
    confidence: float = 0.8,
    seed: int = 42,
) -> PoseVector:
    """Create a mock pose vector."""
    np.random.seed(seed)
    vector = np.random.normal(0, 1, dimension).tolist()

    return PoseVector(
        vector=vector,
        vector_dimension=dimension,
        confidence=confidence,
    )


def create_mock_pose_classification(
    category: PoseCategory = PoseCategory.STANDING,
    orientation: PoseOrientation = PoseOrientation.FRONTAL,
    confidence: float = 0.8,
) -> PoseClassification:
    """Create a mock pose classification."""
    return PoseClassification(
        category=category,
        orientation=orientation,
        arm_position=ArmPosition.AT_SIDES,
        leg_position=LegPosition.STRAIGHT,
        symmetry=PoseSymmetry.SYMMETRIC,
        dynamism=PoseDynamism.STATIC,
        confidence=confidence,
    )


def create_mock_pose_analysis(
    path: Path,
    pose_score: float = 0.8,
    landmark_count: int = 33,
    category: PoseCategory = PoseCategory.STANDING,
    seed: int = 42,
) -> PoseAnalysis:
    """Create a mock pose analysis."""
    # Create landmarks
    landmarks = []
    for i in range(landmark_count):
        landmarks.append(
            create_mock_pose_landmark(
                x=0.3 + (i % 5) * 0.1,
                y=0.2 + (i // 11) * 0.25,
                visibility=np.random.RandomState(seed + i).uniform(0.7, 1.0),
            )
        )

    # Create pose vector
    pose_vector = create_mock_pose_vector(seed=seed)

    # Create classification
    classification = create_mock_pose_classification(category=category)

    return PoseAnalysis(
        path=path,
        landmarks=landmarks,
        pose_vector=pose_vector,
        classification=classification,
        bbox=(0.2, 0.1, 0.6, 0.8),
        pose_score=pose_score,
        analysis_duration=0.15,
    )


def create_mock_pose_analysis_result(
    path: Path,
    success: bool = True,
    pose_analysis: PoseAnalysis | None = None,
) -> PoseAnalysisResult:
    """Create a mock pose analysis result."""
    if success and pose_analysis is None:
        pose_analysis = create_mock_pose_analysis(path)

    return PoseAnalysisResult(
        path=path,
        success=success,
        pose_analysis=pose_analysis,
        error=None if success else "Mock error",
        error_code=None if success else "MOCK_ERROR",
        analysis_duration=0.15,
    )


def create_mock_batch_pose_result(
    paths: list[Path],
    success_rate: float = 1.0,
) -> BatchPoseResult:
    """Create a mock batch pose result."""
    results = []
    successful_count = 0

    for i, path in enumerate(paths):
        success = np.random.RandomState(i).random() < success_rate
        result = create_mock_pose_analysis_result(path, success)
        results.append(result)
        if success:
            successful_count += 1

    failed_count = len(paths) - successful_count
    poses_per_second = len(paths) / 2.0  # Mock processing rate

    return BatchPoseResult(
        results=results,
        successful_analyses=successful_count,
        failed_analyses=failed_count,
        total_duration=2.0,
        poses_per_second=poses_per_second,
        mean_pose_score=0.75,
        pose_statistics={
            "mean_confidence": 0.8,
            "vector_std": 0.5,
            "vector_mean": 0.1,
        },
    )


def create_mock_pose_similarity(
    path1: Path,
    path2: Path,
    similarity_score: float = 0.7,
) -> PoseSimilarity:
    """Create a mock pose similarity."""
    distance = 1.0 - similarity_score

    # Mock landmark matches
    landmark_matches = {f"landmark_{i}": np.random.uniform(0.5, 1.0) for i in range(10)}

    return PoseSimilarity(
        path1=path1,
        path2=path2,
        similarity_score=similarity_score,
        distance=distance,
        landmark_matches=landmark_matches,
    )


def create_mock_pose_cluster(
    cluster_id: int,
    image_paths: list[Path],
    dominant_category: PoseCategory = PoseCategory.STANDING,
) -> PoseCluster:
    """Create a mock pose cluster."""
    # Generate centroid vector
    centroid_vector = np.random.normal(0, 1, 66).tolist()

    return PoseCluster(
        cluster_id=cluster_id,
        image_paths=image_paths,
        centroid_vector=centroid_vector,
        intra_cluster_similarity=0.8,
        dominant_category=dominant_category,
        size=len(image_paths),
    )


def create_mock_pose_clustering_result(
    poses: list[PoseAnalysis],
    num_clusters: int = 3,
) -> PoseClusteringResult:
    """Create a mock pose clustering result."""
    clusters = []
    cluster_size_dist = {}
    category_dist: dict[str, int] = {}

    # Distribute poses across clusters
    poses_per_cluster = len(poses) // num_clusters
    for cluster_id in range(num_clusters):
        start_idx = cluster_id * poses_per_cluster
        end_idx = (
            start_idx + poses_per_cluster
            if cluster_id < num_clusters - 1
            else len(poses)
        )
        cluster_poses = poses[start_idx:end_idx]

        if cluster_poses:
            cluster_paths = [pose.path for pose in cluster_poses]
            dominant_category = cluster_poses[0].classification.category

            cluster = create_mock_pose_cluster(
                cluster_id, cluster_paths, dominant_category
            )
            clusters.append(cluster)

            cluster_size_dist[f"cluster_{cluster_id}"] = len(cluster_poses)
            category_dist[dominant_category.value] = (
                category_dist.get(dominant_category.value, 0) + 1
            )

    return PoseClusteringResult(
        num_clusters=len(clusters),
        clusters=clusters,
        silhouette_score=0.6,
        cluster_size_distribution=cluster_size_dist,
        category_distribution=category_dist,
        processing_time=1.5,
    )


def create_mock_pose_diversity_analysis(
    poses: list[PoseAnalysis],
) -> PoseDiversityAnalysis:
    """Create a mock pose diversity analysis."""
    # Calculate category distribution
    category_counts: dict[str, int] = {}
    orientation_counts: dict[str, int] = {}

    for pose in poses:
        cat = pose.classification.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

        orient = pose.classification.orientation.value
        orientation_counts[orient] = orientation_counts.get(orient, 0) + 1

    # Mock similarity pairs
    most_similar_pairs = []
    most_diverse_pairs = []

    if len(poses) >= 2:
        # Create a few mock similarity pairs
        for i in range(min(3, len(poses) - 1)):
            similar_pair = create_mock_pose_similarity(
                poses[i].path, poses[i + 1].path, 0.9
            )
            most_similar_pairs.append(similar_pair)

            diverse_pair = create_mock_pose_similarity(
                poses[i].path, poses[-i - 1].path, 0.2
            )
            most_diverse_pairs.append(diverse_pair)

    return PoseDiversityAnalysis(
        total_images=len(poses),
        category_distribution=category_counts,
        orientation_distribution=orientation_counts,
        mean_pairwise_similarity=0.5,
        diversity_score=0.5,
        similarity_distribution={
            "p10": 0.1,
            "p25": 0.3,
            "p50": 0.5,
            "p75": 0.7,
            "p90": 0.9,
        },
        most_similar_pairs=most_similar_pairs,
        most_diverse_pairs=most_diverse_pairs,
    )


def create_mock_pose_selection_result(
    selected_paths: list[Path],
    diversity_score: float = 0.7,
    mean_quality_score: float = 0.8,
) -> PoseSelectionResult:
    """Create a mock pose selection result."""
    from culora.domain.models.pose import PoseSelectionCriteria

    criteria = PoseSelectionCriteria(
        target_count=len(selected_paths),
        diversity_weight=0.7,
        quality_weight=0.3,
    )

    return PoseSelectionResult(
        selected_paths=selected_paths,
        selection_criteria=criteria,
        diversity_score=diversity_score,
        mean_quality_score=mean_quality_score,
        category_representation={"standing": 2, "sitting": 1},
        orientation_representation={"frontal": 2, "profile": 1},
        cluster_representation={0: 2, 1: 1},
        selection_reasoning=["Selected diverse poses from clusters"],
        processing_time=0.5,
    )


# Mock pose data for testing different scenarios
MOCK_POSE_CATEGORIES = [
    PoseCategory.STANDING,
    PoseCategory.SITTING,
    PoseCategory.LYING,
    PoseCategory.KNEELING,
    PoseCategory.CROUCHING,
]

MOCK_POSE_ORIENTATIONS = [
    PoseOrientation.FRONTAL,
    PoseOrientation.PROFILE,
    PoseOrientation.THREE_QUARTER,
    PoseOrientation.BACK,
]
