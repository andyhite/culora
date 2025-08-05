"""Tests for CLIP semantic embedding service."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]

from culora.domain import CuLoRAConfig
from culora.domain.enums.clip import CLIPModelType, ClusteringMethod, SimilarityMetric
from culora.domain.models.clip import SemanticSelectionCriteria
from culora.services.clip_service import (
    CLIPService,
    CLIPServiceError,
    ClusteringError,
)
from tests.mocks.clip_mocks import (
    MOCK_EMBEDDINGS,
    create_mock_clip_model,
    create_mock_clip_processor,
    create_mock_semantic_embedding,
)


class TestCLIPService:
    """Test cases for CLIPService."""

    @pytest.fixture
    def config(self) -> CuLoRAConfig:
        """Create test configuration."""
        config = CuLoRAConfig()
        config.clip.model_name = CLIPModelType.OPENAI_CLIP_VIT_B_32
        config.clip.batch_size = 2
        config.clip.normalize_embeddings = True
        config.clip.enable_embedding_cache = False  # Disable for predictable tests
        return config

    @pytest.fixture
    def clip_service(self, config: CuLoRAConfig) -> CLIPService:
        """Create CLIPService instance for testing."""
        return CLIPService(config)

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test PIL image."""
        return Image.new("RGB", (224, 224), color="red")

    @pytest.fixture
    def test_images_and_paths(self, temp_dir: Path) -> list[tuple[Image.Image, Path]]:
        """Create test images and paths."""
        images_and_paths = []
        for i in range(4):
            image = Image.new(
                "RGB", (224, 224), color=["red", "green", "blue", "yellow"][i]
            )
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))
        return images_and_paths

    @pytest.fixture
    def mock_embeddings(self, temp_dir: Path) -> list:
        """Create mock embeddings for testing."""
        embeddings = []
        for i, embedding_data in enumerate(MOCK_EMBEDDINGS["similar_group"]):
            path = temp_dir / f"test_{i}.jpg"
            # Normalize the embedding
            embedding_array = np.array(embedding_data)
            embedding_array = embedding_array / np.linalg.norm(embedding_array)

            embedding = create_mock_semantic_embedding(
                path=path,
                embedding_dim=512,
            )
            embedding.embedding = embedding_array.tolist()
            embeddings.append(embedding)
        return embeddings

    def test_clip_service_initialization(self, config: CuLoRAConfig) -> None:
        """Test CLIPService initialization."""
        service = CLIPService(config)

        assert service.config == config
        assert service.clip_config == config.clip
        assert service._model is None
        assert service._processor is None
        assert service._embedding_cache == {}

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_model_loading(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        clip_service: CLIPService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test CLIP model loading."""
        # Setup mocks
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model_class.from_pretrained.return_value = create_mock_clip_model()

        # Test model loading on first use
        test_path = temp_dir / "test.jpg"
        clip_service.extract_embedding(test_image, test_path)

        # Verify model was loaded
        assert clip_service._model is not None
        assert clip_service._processor is not None
        mock_model_class.from_pretrained.assert_called_once()
        mock_processor_class.from_pretrained.assert_called_once()

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_embedding_extraction_success(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        clip_service: CLIPService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test successful embedding extraction."""
        # Setup mocks
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model_class.from_pretrained.return_value = create_mock_clip_model()

        test_path = temp_dir / "test.jpg"
        result = clip_service.extract_embedding(test_image, test_path)

        # Verify result
        assert result.success is True
        assert result.embedding is not None
        assert result.error is None
        assert result.analysis_duration > 0

        # Verify embedding properties
        embedding = result.embedding
        assert embedding.path == test_path
        assert embedding.embedding_dimension == 512
        assert len(embedding.embedding) == 512
        assert embedding.confidence_score is not None

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_embedding_extraction_failure(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        clip_service: CLIPService,
        test_image: Image.Image,
        temp_dir: Path,
    ) -> None:
        """Test embedding extraction failure handling."""
        # Setup mocks to raise exception
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model = create_mock_clip_model()
        mock_model.get_image_features.side_effect = Exception("Test error")
        mock_model_class.from_pretrained.return_value = mock_model

        test_path = temp_dir / "test.jpg"
        result = clip_service.extract_embedding(test_image, test_path)

        # Verify failure result
        assert result.success is False
        assert result.embedding is None
        assert result.error is not None
        assert "Test error" in result.error
        assert result.error_code == "EMBEDDING_EXTRACTION_FAILED"

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_batch_embedding_extraction(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        clip_service: CLIPService,
        test_images_and_paths: list[tuple[Image.Image, Path]],
    ) -> None:
        """Test batch embedding extraction."""
        # Setup mocks
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model_class.from_pretrained.return_value = create_mock_clip_model()

        result = clip_service.extract_batch_embeddings(test_images_and_paths)

        # Verify batch result
        assert result.successful_analyses == len(test_images_and_paths)
        assert result.failed_analyses == 0
        assert result.embeddings_per_second > 0
        assert len(result.results) == len(test_images_and_paths)

        # Verify individual results
        for individual_result in result.results:
            assert individual_result.success is True
            assert individual_result.embedding is not None

    def test_similarity_calculation(
        self, clip_service: CLIPService, mock_embeddings: list
    ) -> None:
        """Test similarity calculation between embeddings."""
        embedding1, embedding2 = mock_embeddings[0], mock_embeddings[1]

        similarity = clip_service.calculate_similarity(embedding1, embedding2)

        # Verify similarity result
        assert similarity.path1 == embedding1.path
        assert similarity.path2 == embedding2.path
        assert 0.0 <= similarity.similarity_score <= 1.0
        assert similarity.metric == "cosine"  # type: ignore[comparison-overlap]

    def test_similarity_calculation_different_metrics(
        self, config: CuLoRAConfig, mock_embeddings: list
    ) -> None:
        """Test similarity calculation with different metrics."""
        embedding1, embedding2 = mock_embeddings[0], mock_embeddings[1]

        # Test different similarity metrics
        for metric in SimilarityMetric:
            config.clip.similarity_metric = metric
            service = CLIPService(config)

            similarity = service.calculate_similarity(embedding1, embedding2)
            assert similarity.metric == metric.value  # type: ignore[comparison-overlap]
            assert 0.0 <= similarity.similarity_score <= 1.0

    def test_diversity_analysis(
        self, clip_service: CLIPService, mock_embeddings: list
    ) -> None:
        """Test diversity analysis on embeddings."""
        diversity = clip_service.analyze_diversity(mock_embeddings)

        # Verify diversity result
        assert diversity.total_images == len(mock_embeddings)
        assert 0.0 <= diversity.mean_pairwise_similarity <= 1.0
        assert 0.0 <= diversity.diversity_score <= 1.0
        assert diversity.diversity_score == 1.0 - diversity.mean_pairwise_similarity

        # Verify similarity distribution
        assert len(diversity.similarity_distribution) > 0
        for _percentile, score in diversity.similarity_distribution.items():
            assert 0.0 <= score <= 1.0

    def test_diversity_analysis_insufficient_embeddings(
        self, clip_service: CLIPService, mock_embeddings: list
    ) -> None:
        """Test diversity analysis with insufficient embeddings."""
        with pytest.raises(CLIPServiceError, match="Need at least 2 embeddings"):
            clip_service.analyze_diversity(mock_embeddings[:1])

    def test_kmeans_clustering(
        self, config: CuLoRAConfig, mock_embeddings: list
    ) -> None:
        """Test K-means clustering."""
        config.clip.clustering_method = ClusteringMethod.KMEANS
        service = CLIPService(config)

        result = service.cluster_embeddings(mock_embeddings)

        # Verify clustering result
        assert result.method == "kmeans"  # type: ignore[comparison-overlap]
        assert result.num_clusters > 0  # type: ignore[unreachable]
        assert len(result.clusters) == result.num_clusters
        assert result.silhouette_score is not None
        assert result.inertia is not None  # K-means specific
        assert result.processing_time > 0

    @patch("sklearn.cluster.AgglomerativeClustering")
    def test_hierarchical_clustering(
        self, mock_clustering: MagicMock, config: CuLoRAConfig, mock_embeddings: list
    ) -> None:
        """Test hierarchical clustering."""
        # Mock sklearn clustering
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_clustering.return_value = mock_instance

        config.clip.clustering_method = ClusteringMethod.HIERARCHICAL
        service = CLIPService(config)

        result = service.cluster_embeddings(mock_embeddings)

        # Verify clustering result
        assert result.method == "hierarchical"  # type: ignore[comparison-overlap]
        # Not applicable for hierarchical clustering
        assert result.inertia is None  # type: ignore[unreachable]

    @patch("sklearn.cluster.DBSCAN")
    def test_dbscan_clustering(
        self, mock_dbscan: MagicMock, config: CuLoRAConfig, mock_embeddings: list
    ) -> None:
        """Test DBSCAN clustering."""
        # Mock sklearn DBSCAN
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 1, -1])  # -1 is noise
        mock_dbscan.return_value = mock_instance

        config.clip.clustering_method = ClusteringMethod.DBSCAN
        service = CLIPService(config)

        result = service.cluster_embeddings(mock_embeddings)

        # Verify clustering result (noise points filtered out)
        assert result.method == "dbscan"  # type: ignore[comparison-overlap]
        # Not applicable for DBSCAN clustering
        assert result.inertia is None  # type: ignore[unreachable]

    def test_clustering_insufficient_embeddings(
        self, clip_service: CLIPService, mock_embeddings: list
    ) -> None:
        """Test clustering with insufficient embeddings."""
        with pytest.raises(ClusteringError, match="Need at least 2 embeddings"):
            clip_service.cluster_embeddings(mock_embeddings[:1])

    def test_diverse_image_selection(
        self, clip_service: CLIPService, mock_embeddings: list
    ) -> None:
        """Test diverse image selection algorithm."""
        criteria = SemanticSelectionCriteria(
            target_count=2,
            diversity_weight=0.7,
            quality_weight=0.3,
            min_cluster_representation=1,
        )

        # Mock quality scores
        quality_scores = {emb.path: 0.8 for emb in mock_embeddings}

        result = clip_service.select_diverse_images(
            mock_embeddings, criteria, quality_scores
        )

        # Verify selection result
        assert len(result.selected_paths) == criteria.target_count
        assert result.diversity_score is not None
        assert result.mean_quality_score is not None
        assert result.processing_time > 0
        assert len(result.selection_reasoning) > 0

    def test_image_preparation(self, clip_service: CLIPService) -> None:
        """Test image preparation for CLIP processing."""
        # Test oversized image
        large_image = Image.new("RGB", (2000, 1500), color="red")
        prepared = clip_service._prepare_image_for_clip(large_image)

        # Should be resized to within max_image_size
        max_w, max_h = clip_service.clip_config.max_image_size
        assert prepared.size[0] <= max_w
        assert prepared.size[1] <= max_h

        # Test appropriately sized image
        small_image = Image.new("RGB", (200, 200), color="blue")
        prepared = clip_service._prepare_image_for_clip(small_image)

        # Should remain unchanged
        assert prepared.size == small_image.size

    def test_embedding_caching(
        self, config: CuLoRAConfig, test_image: Image.Image, temp_dir: Path
    ) -> None:
        """Test embedding caching functionality."""
        config.clip.enable_embedding_cache = True

        with (
            patch("culora.services.clip_service.CLIPModel") as mock_model_class,
            patch("culora.services.clip_service.CLIPProcessor") as mock_processor_class,
        ):

            # Setup mocks
            mock_processor_class.from_pretrained.return_value = (
                create_mock_clip_processor()
            )
            mock_model_class.from_pretrained.return_value = create_mock_clip_model()

            service = CLIPService(config)
            test_path = temp_dir / "test.jpg"

            # First extraction should compute embedding
            result1 = service.extract_embedding(test_image, test_path)
            assert result1.success

            # Second extraction should use cache
            result2 = service.extract_embedding(test_image, test_path)
            assert result2.success
            assert result2.embedding == result1.embedding

    def test_confidence_score_calculation(self, clip_service: CLIPService) -> None:
        """Test embedding confidence score calculation."""
        # Test with different embedding magnitudes
        high_magnitude = np.array([3.0, 4.0, 0.0, 0.0])
        low_magnitude = np.array([0.1, 0.1, 0.0, 0.0])

        high_confidence = clip_service._calculate_embedding_confidence(high_magnitude)
        low_confidence = clip_service._calculate_embedding_confidence(low_magnitude)

        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
        assert high_confidence > low_confidence

    def test_optimal_cluster_determination(self, clip_service: CLIPService) -> None:
        """Test optimal cluster count determination."""
        # Create embedding matrix for testing
        embedding_matrix = np.random.rand(10, 512)

        optimal_k = clip_service._determine_optimal_clusters(embedding_matrix)

        assert 2 <= optimal_k <= clip_service.clip_config.max_clusters
        assert optimal_k <= len(embedding_matrix) // 2


class TestCLIPServiceIntegration:
    """Integration tests for CLIPService."""

    @pytest.fixture
    def integration_config(self) -> CuLoRAConfig:
        """Create integration test configuration."""
        config = CuLoRAConfig()
        config.clip.batch_size = 2
        config.clip.max_clusters = 2  # Reduced to avoid convergence warnings
        config.clip.enable_embedding_cache = False
        return config

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_end_to_end_workflow(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        integration_config: CuLoRAConfig,
        temp_dir: Path,
    ) -> None:
        """Test complete end-to-end CLIP workflow."""
        # Setup mocks
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model_class.from_pretrained.return_value = create_mock_clip_model()

        service = CLIPService(integration_config)

        # Create test images
        images_and_paths = []
        for i in range(6):
            image = Image.new(
                "RGB",
                (224, 224),
                color=["red", "green", "blue", "yellow", "purple", "orange"][i],
            )
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))

        # Extract embeddings
        batch_result = service.extract_batch_embeddings(images_and_paths)
        assert batch_result.successful_analyses == len(images_and_paths)

        # Get embeddings for further analysis
        embeddings = [
            r.embedding for r in batch_result.results if r.success and r.embedding
        ]

        # Perform clustering with warnings suppressed for known sklearn convergence issues
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clustering_result = service.cluster_embeddings(embeddings)
        assert clustering_result.num_clusters > 0

        # Analyze diversity
        diversity_analysis = service.analyze_diversity(embeddings)
        assert diversity_analysis.total_images == len(embeddings)

        # Select diverse subset with warnings suppressed
        criteria = SemanticSelectionCriteria(
            target_count=2,  # Reduced to match max_clusters
            diversity_weight=0.8,
            quality_weight=0.2,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            selection_result = service.select_diverse_images(embeddings, criteria)
        assert len(selection_result.selected_paths) == 2

    def test_error_handling_and_recovery(
        self, integration_config: CuLoRAConfig, temp_dir: Path
    ) -> None:
        """Test error handling and recovery scenarios."""
        service = CLIPService(integration_config)

        # Test with invalid image
        invalid_image = Image.new("RGB", (0, 0))  # Invalid dimensions
        test_path = temp_dir / "invalid.jpg"

        # Should handle gracefully
        result = service.extract_embedding(invalid_image, test_path)
        # Result may succeed or fail depending on implementation, but shouldn't crash
        assert result is not None

    @patch("culora.services.clip_service.CLIPModel")
    @patch("culora.services.clip_service.CLIPProcessor")
    def test_memory_efficiency(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        integration_config: CuLoRAConfig,
        temp_dir: Path,
    ) -> None:
        """Test memory efficiency with large batches."""
        # Setup mocks
        mock_processor_class.from_pretrained.return_value = create_mock_clip_processor()
        mock_model_class.from_pretrained.return_value = create_mock_clip_model()

        service = CLIPService(integration_config)

        # Create larger batch to test memory handling
        images_and_paths = []
        for i in range(20):
            image = Image.new("RGB", (100, 100), color="red")
            path = temp_dir / f"test_{i}.jpg"
            images_and_paths.append((image, path))

        # Should complete without memory issues
        result = service.extract_batch_embeddings(images_and_paths)
        assert result.successful_analyses > 0
