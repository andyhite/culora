"""Tests for pose estimation service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from culora.domain import CuLoRAConfig
from culora.domain.enums.pose import PoseCategory, PoseOrientation
from culora.domain.models.pose import PoseSelectionCriteria
from culora.services.pose_service import (
    PoseClusteringError,
    PoseService,
    PoseServiceError,
)
from tests.mocks.pose_mocks import (
    create_mock_mediapipe_pose,
    create_mock_pose_analysis,
    create_mock_pose_classification,
    create_mock_pose_landmark,
)


class TestPoseService:
    """Test cases for PoseService."""

    @pytest.fixture
    def config(self) -> CuLoRAConfig:
        """Create test configuration."""
        config = CuLoRAConfig()
        config.pose.model_complexity = 1
        config.pose.min_detection_confidence = 0.5
        config.pose.min_tracking_confidence = 0.5
        config.pose.batch_size = 2
        config.pose.enable_pose_cache = False  # Disable for predictable tests
        config.pose.min_pose_score = 0.3
        config.pose.min_visible_landmarks = 20
        return config

    @pytest.fixture
    def pose_service(self, config: CuLoRAConfig) -> PoseService:
        """Create PoseService instance for testing."""
        return PoseService(config)

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test PIL image."""
        return Image.new("RGB", (640, 480), color="blue")

    @pytest.fixture
    def test_images_and_paths(self, temp_dir: Path) -> list[tuple[Image.Image, Path]]:
        """Create test images and paths."""
        images_and_paths = []
        for i in range(4):
            image = Image.new(
                "RGB", (640, 480), color=["red", "green", "blue", "yellow"][i]
            )
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))
        return images_and_paths

    def test_pose_service_initialization(self, config: CuLoRAConfig) -> None:
        """Test PoseService initialization."""
        service = PoseService(config)

        assert service.config == config
        assert service.pose_config == config.pose
        assert service._pose_estimator is None
        assert service._pose_cache == {}

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_get_pose_estimator(
        self, mock_pose_class: MagicMock, pose_service: PoseService
    ) -> None:
        """Test MediaPipe pose estimator creation."""
        mock_pose_instance = create_mock_mediapipe_pose()
        mock_pose_class.return_value = mock_pose_instance

        # First call should create estimator
        estimator1 = pose_service._get_pose_estimator()
        assert estimator1 is mock_pose_instance
        mock_pose_class.assert_called_once()

        # Second call should return cached estimator
        estimator2 = pose_service._get_pose_estimator()
        assert estimator2 is mock_pose_instance
        assert mock_pose_class.call_count == 1  # Still only called once

    def test_prepare_image_for_pose(self, pose_service: PoseService) -> None:
        """Test image preparation for pose estimation."""
        # Test oversized image
        large_image = Image.new("RGB", (2000, 1500), color="red")
        prepared = pose_service._prepare_image_for_pose(large_image)

        # Should be resized to within max_image_size
        max_w, max_h = pose_service.pose_config.max_image_size
        assert prepared.shape[1] <= max_w  # width
        assert prepared.shape[0] <= max_h  # height
        assert prepared.shape[2] == 3  # RGB channels

        # Test appropriately sized image
        small_image = Image.new("RGB", (400, 300), color="blue")
        prepared = pose_service._prepare_image_for_pose(small_image)

        # Should remain the same size
        assert prepared.shape == (300, 400, 3)  # height, width, channels

    def test_extract_landmarks(self, pose_service: PoseService) -> None:
        """Test landmark extraction from MediaPipe results."""
        # Mock results with landmarks
        results = MagicMock()
        mock_landmarks = []
        for i in range(33):
            mock_landmark = MagicMock()
            mock_landmark.x = 0.3 + (i % 5) * 0.1
            mock_landmark.y = 0.2 + (i // 11) * 0.25
            mock_landmark.z = np.random.normal(0, 0.1)
            mock_landmark.visibility = 0.9
            mock_landmark.presence = 0.95
            mock_landmarks.append(mock_landmark)

        results.pose_landmarks = MagicMock()
        results.pose_landmarks.landmark = mock_landmarks

        landmarks = pose_service._extract_landmarks(results)

        assert len(landmarks) == 33
        for landmark in landmarks:
            assert isinstance(landmark.x, float)
            assert isinstance(landmark.y, float)
            assert isinstance(landmark.z, float)
            assert 0.0 <= landmark.visibility <= 1.0
            assert 0.0 <= landmark.presence <= 1.0

        # Test results without landmarks
        results.pose_landmarks = None
        landmarks = pose_service._extract_landmarks(results)
        assert landmarks == []

    def test_create_pose_vector(self, pose_service: PoseService) -> None:
        """Test pose vector creation from landmarks."""
        # Create mock landmarks
        landmarks = []
        for i in range(33):
            landmarks.append(
                create_mock_pose_landmark(
                    x=0.3 + (i % 5) * 0.1,
                    y=0.2 + (i // 11) * 0.25,
                    visibility=0.9,
                )
            )

        pose_vector = pose_service._create_pose_vector(landmarks)

        assert (
            pose_vector.vector_dimension == pose_service.pose_config.feature_vector_dim
        )
        assert len(pose_vector.vector) == pose_service.pose_config.feature_vector_dim
        assert 0.0 <= pose_vector.confidence <= 1.0

        # Test with empty landmarks
        empty_vector = pose_service._create_pose_vector([])
        assert empty_vector.confidence == 0.0
        assert len(empty_vector.vector) == pose_service.pose_config.feature_vector_dim

    def test_classify_pose(self, pose_service: PoseService) -> None:
        """Test pose classification."""
        # Create landmarks representing a standing pose
        landmarks = []
        for i in range(33):
            # Arrange landmarks to represent standing pose
            if i == 0:  # nose
                landmarks.append(create_mock_pose_landmark(x=0.5, y=0.2))
            elif i == 11:  # left shoulder
                landmarks.append(create_mock_pose_landmark(x=0.4, y=0.3))
            elif i == 12:  # right shoulder
                landmarks.append(create_mock_pose_landmark(x=0.6, y=0.3))
            elif i == 23:  # left hip
                landmarks.append(create_mock_pose_landmark(x=0.45, y=0.6))
            elif i == 24:  # right hip
                landmarks.append(create_mock_pose_landmark(x=0.55, y=0.6))
            elif i == 27:  # left ankle
                landmarks.append(create_mock_pose_landmark(x=0.45, y=0.9))
            elif i == 28:  # right ankle
                landmarks.append(create_mock_pose_landmark(x=0.55, y=0.9))
            else:
                landmarks.append(create_mock_pose_landmark())

        classification = pose_service._classify_pose(landmarks)

        assert classification.category in PoseCategory
        assert classification.orientation in PoseOrientation
        assert 0.0 <= classification.confidence <= 1.0

        # Test with insufficient landmarks
        short_landmarks = landmarks[:10]
        classification = pose_service._classify_pose(short_landmarks)
        assert classification.category == PoseCategory.UNKNOWN
        assert classification.confidence == 0.0

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_analyze_pose_success(
        self,
        mock_pose_class: MagicMock,
        pose_service: PoseService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test successful pose analysis."""
        # Setup mock
        mock_pose_instance = create_mock_mediapipe_pose()
        mock_pose_class.return_value = mock_pose_instance

        test_path = temp_dir / "test.jpg"
        result = pose_service.analyze_pose(test_image, test_path)

        # Verify result
        assert result.success is True
        assert result.pose_analysis is not None
        assert result.error is None
        assert result.analysis_duration > 0

        # Verify pose analysis
        pose = result.pose_analysis
        assert pose.path == test_path
        assert len(pose.landmarks) == 33  # Full body pose
        assert pose.pose_score > 0
        assert 0.0 <= pose.pose_score <= 1.0

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_analyze_pose_failure(
        self,
        mock_pose_class: MagicMock,
        pose_service: PoseService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test pose analysis failure handling."""
        # Setup mock to raise exception
        mock_pose_instance = MagicMock()
        mock_pose_instance.process.side_effect = Exception("Test error")
        mock_pose_class.return_value = mock_pose_instance

        test_path = temp_dir / "test.jpg"
        result = pose_service.analyze_pose(test_image, test_path)

        # Verify failure result
        assert result.success is False
        assert result.pose_analysis is None
        assert result.error is not None
        assert "Test error" in result.error
        assert result.error_code == "POSE_ANALYSIS_FAILED"

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_analyze_pose_insufficient_landmarks(
        self,
        mock_pose_class: MagicMock,
        pose_service: PoseService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test pose analysis with insufficient landmarks."""
        # Setup mock to return few landmarks
        mock_pose_instance = MagicMock()
        results = MagicMock()

        # Create only 10 landmarks (below minimum)
        mock_landmarks = []
        for _i in range(10):
            landmark = MagicMock()
            landmark.x = 0.5
            landmark.y = 0.5
            landmark.z = 0.0
            landmark.visibility = 0.9
            landmark.presence = 0.9
            mock_landmarks.append(landmark)

        results.pose_landmarks = MagicMock()
        results.pose_landmarks.landmark = mock_landmarks
        mock_pose_instance.process.return_value = results
        mock_pose_class.return_value = mock_pose_instance

        test_path = temp_dir / "test.jpg"
        result = pose_service.analyze_pose(test_image, test_path)

        # Should fail due to insufficient landmarks
        assert result.success is False
        assert result.error_code == "INSUFFICIENT_LANDMARKS"

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_analyze_batch_poses(
        self,
        mock_pose_class: MagicMock,
        pose_service: PoseService,
        test_images_and_paths: list[tuple[Image.Image, Path]],
    ) -> None:
        """Test batch pose analysis."""
        # Setup mock
        mock_pose_instance = create_mock_mediapipe_pose()
        mock_pose_class.return_value = mock_pose_instance

        result = pose_service.analyze_batch_poses(test_images_and_paths)

        # Verify batch result
        assert result.successful_analyses == len(test_images_and_paths)
        assert result.failed_analyses == 0
        assert result.poses_per_second > 0
        assert len(result.results) == len(test_images_and_paths)

        # Verify individual results
        for individual_result in result.results:
            assert individual_result.success is True
            assert individual_result.pose_analysis is not None

    def test_calculate_pose_similarity(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test pose similarity calculation."""
        pose1 = create_mock_pose_analysis(temp_dir / "image1.jpg", seed=42)
        pose2 = create_mock_pose_analysis(temp_dir / "image2.jpg", seed=43)

        similarity = pose_service.calculate_pose_similarity(pose1, pose2)

        # Verify similarity result
        assert similarity.path1 == pose1.path
        assert similarity.path2 == pose2.path
        assert 0.0 <= similarity.similarity_score <= 1.0
        assert 0.0 <= similarity.distance <= 1.0
        assert len(similarity.landmark_matches) > 0

    def test_analyze_pose_diversity(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test pose diversity analysis."""
        # Create diverse poses
        poses = []
        categories = [PoseCategory.STANDING, PoseCategory.SITTING, PoseCategory.LYING]
        for i, category in enumerate(categories):
            pose = create_mock_pose_analysis(
                temp_dir / f"image_{i}.jpg",
                category=category,
                seed=i * 10,
            )
            poses.append(pose)

        diversity = pose_service.analyze_pose_diversity(poses)

        # Verify diversity result
        assert diversity.total_images == len(poses)
        assert 0.0 <= diversity.mean_pairwise_similarity <= 1.0
        assert 0.0 <= diversity.diversity_score <= 1.0
        assert diversity.diversity_score == 1.0 - diversity.mean_pairwise_similarity

        # Verify distributions
        assert len(diversity.category_distribution) > 0
        assert len(diversity.orientation_distribution) > 0
        assert len(diversity.similarity_distribution) > 0

    def test_analyze_pose_diversity_insufficient_poses(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test diversity analysis with insufficient poses."""
        poses = [create_mock_pose_analysis(temp_dir / "image1.jpg")]

        with pytest.raises(PoseServiceError, match="Need at least 2 poses"):
            pose_service.analyze_pose_diversity(poses)

    def test_cluster_poses(self, pose_service: PoseService, temp_dir: Path) -> None:
        """Test pose clustering."""
        # Create poses for clustering
        poses = []
        for i in range(6):
            pose = create_mock_pose_analysis(
                temp_dir / f"image_{i}.jpg",
                seed=i * 5,
            )
            poses.append(pose)

        result = pose_service.cluster_poses(poses)

        # Verify clustering result
        assert result.num_clusters > 0
        assert len(result.clusters) == result.num_clusters
        assert -1.0 <= result.silhouette_score <= 1.0
        assert result.processing_time > 0

        # Verify clusters
        for cluster in result.clusters:
            assert cluster.size >= pose_service.pose_config.min_cluster_size
            assert len(cluster.image_paths) == cluster.size

    def test_cluster_poses_insufficient_poses(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test clustering with insufficient poses."""
        poses = [create_mock_pose_analysis(temp_dir / "image1.jpg")]

        with pytest.raises(PoseClusteringError, match="Need at least 2 poses"):
            pose_service.cluster_poses(poses)

    def test_select_diverse_poses(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test diverse pose selection."""
        # Create poses for selection
        poses = []
        for i in range(8):
            pose = create_mock_pose_analysis(
                temp_dir / f"image_{i}.jpg",
                pose_score=0.5 + (i * 0.05),  # Varying quality scores
                seed=i * 3,
            )
            poses.append(pose)

        criteria = PoseSelectionCriteria(
            target_count=4,
            diversity_weight=0.6,
            quality_weight=0.4,
        )

        # Mock quality scores
        quality_scores = {pose.path: 0.7 + (i * 0.03) for i, pose in enumerate(poses)}

        result = pose_service.select_diverse_poses(poses, criteria, quality_scores)

        # Verify selection result
        assert len(result.selected_paths) == criteria.target_count
        assert result.diversity_score is not None
        assert result.mean_quality_score is not None
        assert result.processing_time > 0
        assert len(result.selection_reasoning) > 0

    def test_select_diverse_poses_fewer_than_target(
        self, pose_service: PoseService, temp_dir: Path
    ) -> None:
        """Test pose selection when fewer poses available than target."""
        poses = [
            create_mock_pose_analysis(temp_dir / "image1.jpg"),
            create_mock_pose_analysis(temp_dir / "image2.jpg"),
        ]

        criteria = PoseSelectionCriteria(target_count=5)

        result = pose_service.select_diverse_poses(poses, criteria)

        # Should return all available poses
        assert len(result.selected_paths) == len(poses)
        assert all(pose.path in result.selected_paths for pose in poses)

    def test_pose_caching(
        self, config: CuLoRAConfig, test_image: Image.Image, temp_dir: Path
    ) -> None:
        """Test pose analysis caching functionality."""
        config.pose.enable_pose_cache = True

        with (
            patch(
                "culora.services.pose_service.mp.solutions.pose.Pose"
            ) as mock_pose_class,
        ):
            # Setup mock
            mock_pose_instance = create_mock_mediapipe_pose()
            mock_pose_class.return_value = mock_pose_instance

            service = PoseService(config)
            test_path = temp_dir / "test.jpg"

            # First analysis should compute pose
            result1 = service.analyze_pose(test_image, test_path)
            assert result1.success

            # Second analysis should use cache
            result2 = service.analyze_pose(test_image, test_path)
            assert result2.success
            assert result2.pose_analysis == result1.pose_analysis

    def test_pose_score_calculation(self, pose_service: PoseService) -> None:
        """Test pose quality score calculation."""
        # Create landmarks with varying visibility
        landmarks = []
        for i in range(33):
            visibility = 0.9 if i < 25 else 0.3  # Most landmarks visible
            landmarks.append(create_mock_pose_landmark(visibility=visibility))

        classification = create_mock_pose_classification(confidence=0.8)

        score = pose_service._calculate_pose_score(landmarks, classification)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for good landmarks

        # Test with poor landmarks
        poor_landmarks = []
        for _i in range(33):
            poor_landmarks.append(create_mock_pose_landmark(visibility=0.2))

        poor_classification = create_mock_pose_classification(confidence=0.3)
        poor_score = pose_service._calculate_pose_score(
            poor_landmarks, poor_classification
        )

        assert poor_score < score  # Should be lower than good pose

    def test_determine_optimal_pose_clusters(self, pose_service: PoseService) -> None:
        """Test optimal cluster count determination."""
        # Create embedding matrix for testing
        embedding_matrix = np.random.rand(10, 66)

        optimal_k = pose_service._determine_optimal_pose_clusters(embedding_matrix, 8)

        assert 2 <= optimal_k <= 8
        assert optimal_k <= len(embedding_matrix) // 2

        # Test with very few poses
        small_matrix = np.random.rand(3, 66)
        small_k = pose_service._determine_optimal_pose_clusters(small_matrix, 5)
        assert small_k == 2  # Should return minimum

    def test_pose_bbox_calculation(self, pose_service: PoseService) -> None:
        """Test pose bounding box calculation."""
        # Create landmarks spread across image
        landmarks = []
        for i in range(33):
            x = 0.2 + (i % 7) * 0.1  # Spread 0.2 to 0.8
            y = 0.1 + (i // 7) * 0.15  # Spread 0.1 to 0.85
            landmarks.append(create_mock_pose_landmark(x=x, y=y, visibility=0.9))

        bbox = pose_service._calculate_pose_bbox(landmarks)

        assert len(bbox) == 4  # x, y, width, height
        assert all(0.0 <= val <= 1.0 for val in bbox)
        assert bbox[2] > 0  # width > 0
        assert bbox[3] > 0  # height > 0

        # Test with no visible landmarks
        invisible_landmarks = [
            create_mock_pose_landmark(visibility=0.1) for _ in range(33)
        ]
        empty_bbox = pose_service._calculate_pose_bbox(invisible_landmarks)
        assert empty_bbox == (0.0, 0.0, 0.0, 0.0)


class TestPoseServiceIntegration:
    """Integration tests for PoseService."""

    @pytest.fixture
    def integration_config(self) -> CuLoRAConfig:
        """Create integration test configuration."""
        config = CuLoRAConfig()
        config.pose.batch_size = 2
        config.pose.max_clusters = 3
        config.pose.enable_pose_cache = False
        config.pose.min_pose_score = 0.2  # Lower threshold for testing
        return config

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_end_to_end_workflow(
        self,
        mock_pose_class: MagicMock,
        integration_config: CuLoRAConfig,
        temp_dir: Path,
    ) -> None:
        """Test complete end-to-end pose workflow."""
        # Setup mock
        mock_pose_instance = create_mock_mediapipe_pose()
        mock_pose_class.return_value = mock_pose_instance

        service = PoseService(integration_config)

        # Create test images
        images_and_paths = []
        for i in range(6):
            image = Image.new(
                "RGB",
                (640, 480),
                color=["red", "green", "blue", "yellow", "purple", "orange"][i],
            )
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))

        # Extract poses
        batch_result = service.analyze_batch_poses(images_and_paths)
        assert batch_result.successful_analyses == len(images_and_paths)

        # Get poses for further analysis
        poses = [
            r.pose_analysis
            for r in batch_result.results
            if r.success and r.pose_analysis
        ]

        # Perform clustering
        clustering_result = service.cluster_poses(poses)
        assert clustering_result.num_clusters > 0

        # Analyze diversity
        diversity_analysis = service.analyze_pose_diversity(poses)
        assert diversity_analysis.total_images == len(poses)

        # Select diverse subset
        criteria = PoseSelectionCriteria(
            target_count=3,
            diversity_weight=0.7,
            quality_weight=0.3,
        )
        selection_result = service.select_diverse_poses(poses, criteria)
        assert len(selection_result.selected_paths) == 3

    def test_error_handling_and_recovery(
        self, integration_config: CuLoRAConfig, temp_dir: Path
    ) -> None:
        """Test error handling and recovery scenarios."""
        service = PoseService(integration_config)

        # Test with invalid image
        invalid_image = Image.new("RGB", (0, 0))  # Invalid dimensions
        test_path = temp_dir / "invalid.jpg"

        # Should handle gracefully
        result = service.analyze_pose(invalid_image, test_path)
        # Result may succeed or fail depending on implementation, but shouldn't crash
        assert result is not None

    @patch("culora.services.pose_service.mp.solutions.pose.Pose")
    def test_memory_efficiency(
        self,
        mock_pose_class: MagicMock,
        integration_config: CuLoRAConfig,
        temp_dir: Path,
    ) -> None:
        """Test memory efficiency with large batches."""
        # Setup mock
        mock_pose_instance = create_mock_mediapipe_pose()
        mock_pose_class.return_value = mock_pose_instance

        service = PoseService(integration_config)

        # Create larger batch to test memory handling
        images_and_paths = []
        for i in range(20):
            image = Image.new("RGB", (320, 240), color="blue")
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))

        # Should complete without memory issues
        result = service.analyze_batch_poses(images_and_paths)
        assert result.successful_analyses > 0
