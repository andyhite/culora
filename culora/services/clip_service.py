"""CLIP semantic embedding service using transformers."""

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from transformers import CLIPModel, CLIPProcessor

from culora.core import CuLoRAError
from culora.domain import CuLoRAConfig
from culora.domain.enums.clip import ClusteringMethod, SimilarityMetric
from culora.domain.models.clip import (
    BatchSemanticResult,
    ClusteringResult,
    DiversityAnalysis,
    EmbeddingSimilarity,
    SemanticAnalysisResult,
    SemanticCluster,
    SemanticEmbedding,
    SemanticSelectionCriteria,
    SemanticSelectionResult,
)
from culora.services.device_service import get_device_service


class CLIPServiceError(CuLoRAError):
    """Base exception for CLIP service errors."""

    def __init__(self, message: str, error_code: str = "CLIP_SERVICE_ERROR") -> None:
        super().__init__(message, error_code)


class EmbeddingExtractionError(CLIPServiceError):
    """Exception for embedding extraction failures."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message, "EMBEDDING_EXTRACTION_ERROR")
        self.path = path


class ClusteringError(CLIPServiceError):
    """Exception for clustering analysis failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CLUSTERING_ERROR")


class CLIPService:
    """Service for CLIP semantic embeddings and diversity analysis.

    Provides comprehensive semantic analysis including:
    - CLIP embedding extraction for images
    - Similarity calculation between embeddings
    - Semantic clustering for diversity analysis
    - Selection algorithms for optimal diversity
    """

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize CLIP service with configuration.

        Args:
            config: CuLoRA configuration containing CLIP settings
        """
        self.config = config
        self.clip_config = config.clip
        self.device_service = get_device_service()

        # Model components (loaded on first use)
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._device: torch.device | None = None

        # Embedding cache
        self._embedding_cache: dict[str, SemanticEmbedding] = {}

    def extract_embedding(
        self, image: Image.Image, path: Path
    ) -> SemanticAnalysisResult:
        """Extract semantic embedding for a single image.

        Args:
            image: PIL Image to analyze
            path: Path to the image file for error reporting

        Returns:
            Complete semantic analysis result
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = str(path)
            if (
                self.clip_config.enable_embedding_cache
                and cache_key in self._embedding_cache
            ):
                cached_embedding = self._embedding_cache[cache_key]
                duration = time.time() - start_time
                return SemanticAnalysisResult(
                    path=path,
                    success=True,
                    embedding=cached_embedding,
                    analysis_duration=duration,
                )

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Prepare image for CLIP processing
            prepared_image = self._prepare_image_for_clip(image)

            # Extract embedding
            embedding_vector = self._extract_clip_embedding(prepared_image)

            # Create embedding object
            embedding = SemanticEmbedding(
                path=path,
                embedding=embedding_vector.tolist(),
                model_name=self.clip_config.model_name.value,
                embedding_dimension=len(embedding_vector),
                extraction_time=time.time() - start_time,
                confidence_score=self._calculate_embedding_confidence(embedding_vector),
            )

            # Cache embedding
            if self.clip_config.enable_embedding_cache:
                self._embedding_cache[cache_key] = embedding

            duration = time.time() - start_time

            return SemanticAnalysisResult(
                path=path,
                success=True,
                embedding=embedding,
                analysis_duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Embedding extraction failed for {path}: {e}"

            return SemanticAnalysisResult(
                path=path,
                success=False,
                error=error_msg,
                error_code="EMBEDDING_EXTRACTION_FAILED",
                analysis_duration=duration,
            )

    def extract_batch_embeddings(
        self, images_and_paths: list[tuple[Image.Image, Path]]
    ) -> BatchSemanticResult:
        """Extract semantic embeddings for a batch of images.

        Args:
            images_and_paths: List of (image, path) tuples to analyze

        Returns:
            Batch analysis results with statistics
        """
        start_time = time.time()
        results: list[SemanticAnalysisResult] = []

        # Process images in batches for efficiency
        batch_size = self.clip_config.batch_size
        for i in range(0, len(images_and_paths), batch_size):
            batch = images_and_paths[i : i + batch_size]

            for image, path in batch:
                result = self.extract_embedding(image, path)
                results.append(result)

        # Calculate batch statistics
        total_duration = time.time() - start_time
        successful_results = [
            r for r in results if r.success and r.embedding is not None
        ]

        batch_result = self._calculate_batch_statistics(
            results, successful_results, total_duration
        )

        return batch_result

    def calculate_similarity(
        self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding
    ) -> EmbeddingSimilarity:
        """Calculate similarity between two semantic embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity analysis result
        """
        vec1 = np.array(embedding1.embedding)
        vec2 = np.array(embedding2.embedding)

        metric = self.clip_config.similarity_metric

        if metric == SimilarityMetric.COSINE:
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            similarity = dot_product / norms if norms > 0 else 0.0
            distance = 1.0 - similarity
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (convert to similarity)
            distance = float(np.linalg.norm(vec1 - vec2))
            similarity = 1.0 / (1.0 + distance)
        elif metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product similarity
            similarity = np.dot(vec1, vec2)
            distance = -similarity  # Inverse for distance
        else:
            raise CLIPServiceError(f"Unsupported similarity metric: {metric}")

        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))

        return EmbeddingSimilarity(
            path1=embedding1.path,
            path2=embedding2.path,
            similarity_score=similarity,
            distance=distance,
            metric=metric,
        )

    def analyze_diversity(
        self, embeddings: list[SemanticEmbedding]
    ) -> DiversityAnalysis:
        """Analyze semantic diversity in a set of embeddings.

        Args:
            embeddings: List of semantic embeddings to analyze

        Returns:
            Complete diversity analysis
        """
        if len(embeddings) < 2:
            raise CLIPServiceError("Need at least 2 embeddings for diversity analysis")

        # Calculate all pairwise similarities
        similarities: list[EmbeddingSimilarity] = []
        similarity_scores: list[float] = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.calculate_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
                similarity_scores.append(sim.similarity_score)

        # Calculate statistics
        mean_similarity = float(np.mean(similarity_scores))
        diversity_score = 1.0 - mean_similarity

        # Calculate percentile distribution
        sorted_scores = sorted(similarity_scores)
        percentiles = [10, 25, 50, 75, 90, 95]
        similarity_distribution = {}
        for p in percentiles:
            idx = int((p / 100.0) * (len(sorted_scores) - 1))
            similarity_distribution[f"p{p}"] = sorted_scores[idx]

        # Find most similar and diverse pairs
        similarities.sort(key=lambda x: x.similarity_score)
        most_similar = similarities[-self.clip_config.max_similarity_pairs :]
        most_diverse = similarities[: self.clip_config.max_similarity_pairs]

        return DiversityAnalysis(
            total_images=len(embeddings),
            mean_pairwise_similarity=mean_similarity,
            diversity_score=diversity_score,
            similarity_distribution=similarity_distribution,
            most_similar_pairs=most_similar,
            most_diverse_pairs=most_diverse,
        )

    def cluster_embeddings(
        self,
        embeddings: list[SemanticEmbedding],
        method: ClusteringMethod | None = None,
    ) -> ClusteringResult:
        """Perform semantic clustering on embeddings.

        Args:
            embeddings: List of embeddings to cluster
            method: Clustering method to use (defaults to config)

        Returns:
            Clustering analysis result
        """
        if len(embeddings) < 2:
            raise ClusteringError("Need at least 2 embeddings for clustering")

        method = method or self.clip_config.clustering_method
        start_time = time.time()

        # Prepare embedding matrix
        embedding_matrix = np.array([emb.embedding for emb in embeddings])

        # Normalize embeddings if requested
        if self.clip_config.normalize_embeddings:
            embedding_matrix = embedding_matrix / np.linalg.norm(
                embedding_matrix, axis=1, keepdims=True
            )

        try:
            if method == ClusteringMethod.KMEANS:
                _, labels, inertia = self._kmeans_clustering(embedding_matrix)
            elif method == ClusteringMethod.HIERARCHICAL:
                _, labels, inertia = self._hierarchical_clustering(embedding_matrix)
            elif method == ClusteringMethod.DBSCAN:
                _, labels, inertia = self._dbscan_clustering(embedding_matrix)
            else:
                raise ClusteringError(f"Unsupported clustering method: {method}")

            # Calculate silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(embedding_matrix, labels)
            else:
                silhouette = 0.0

            # Create cluster objects
            semantic_clusters = self._create_cluster_objects(
                embeddings, labels, embedding_matrix
            )

            # Calculate cluster size distribution
            cluster_sizes = [cluster.size for cluster in semantic_clusters]
            size_distribution = {
                "min": min(cluster_sizes),
                "max": max(cluster_sizes),
                "mean": int(np.mean(cluster_sizes)),
                "std": int(np.std(cluster_sizes)),
            }

            processing_time = time.time() - start_time

            return ClusteringResult(
                method=method,
                num_clusters=len(semantic_clusters),
                clusters=semantic_clusters,
                silhouette_score=float(silhouette),
                inertia=inertia,
                cluster_size_distribution=size_distribution,
                processing_time=processing_time,
            )

        except Exception as e:
            raise ClusteringError(f"Clustering failed: {e}") from e

    def select_diverse_images(
        self,
        embeddings: list[SemanticEmbedding],
        criteria: SemanticSelectionCriteria,
        quality_scores: dict[Path, float] | None = None,
    ) -> SemanticSelectionResult:
        """Select diverse images based on semantic embeddings and criteria.

        Args:
            embeddings: Available embeddings to select from
            criteria: Selection criteria and parameters
            quality_scores: Optional quality scores for quality-weighted selection

        Returns:
            Selection result with chosen images and statistics
        """
        start_time = time.time()

        # First cluster the embeddings
        clustering_result = self.cluster_embeddings(embeddings)

        # Apply selection algorithm
        selected_paths, reasoning = self._apply_diversity_selection(
            clustering_result, criteria, quality_scores
        )

        # Calculate diversity and quality metrics for selected set
        selected_embeddings = [emb for emb in embeddings if emb.path in selected_paths]
        diversity_analysis = self.analyze_diversity(selected_embeddings)

        # Calculate mean quality score
        if quality_scores:
            selected_quality_scores = [
                quality_scores.get(path, 0.0) for path in selected_paths
            ]
            mean_quality = float(np.mean(selected_quality_scores))
        else:
            mean_quality = 0.0

        # Calculate cluster representation
        cluster_representation = {}
        for cluster in clustering_result.clusters:
            count = len(
                [path for path in selected_paths if path in cluster.image_paths]
            )
            if count > 0:
                cluster_representation[cluster.cluster_id] = count

        processing_time = time.time() - start_time

        return SemanticSelectionResult(
            selected_paths=selected_paths,
            selection_criteria=criteria,
            diversity_score=diversity_analysis.diversity_score,
            mean_quality_score=mean_quality,
            cluster_representation=cluster_representation,
            selection_reasoning=reasoning,
            processing_time=processing_time,
        )

    def _ensure_model_loaded(self) -> None:
        """Ensure CLIP model is loaded and ready."""
        if self._model is not None and self._processor is not None:
            return

        try:
            # Get optimal device
            device_info = self.device_service.get_selected_device()
            device_str = device_info.device_type.value
            if device_str == "cuda":
                self._device = torch.device("cuda")
            elif device_str == "mps":
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # Load model and processor
            model_name = self.clip_config.model_name.value
            cache_dir = str(self.clip_config.model_cache_dir)

            self._processor = CLIPProcessor.from_pretrained(
                model_name, cache_dir=cache_dir, use_fast=True
            )

            # Load model with appropriate precision
            if (
                self.clip_config.embedding_precision == "float16"
                and device_str == "cuda"
            ):
                self._model = CLIPModel.from_pretrained(
                    model_name, cache_dir=cache_dir, torch_dtype=torch.float16
                )
            else:
                self._model = CLIPModel.from_pretrained(
                    model_name, cache_dir=cache_dir, torch_dtype=torch.float32
                )

            # Move model to device (guaranteed to be non-None at this point)
            assert self._model is not None
            assert self._device is not None
            self._model = self._model.to(self._device)  # type: ignore[arg-type]
            self._model.eval()

        except Exception as e:
            raise CLIPServiceError(f"Failed to load CLIP model: {e}") from e

    def _prepare_image_for_clip(self, image: Image.Image) -> Image.Image:
        """Prepare image for CLIP processing.

        Args:
            image: Original PIL Image

        Returns:
            Prepared image for model input
        """
        # Resize if needed
        max_width, max_height = self.clip_config.max_image_size
        width, height = image.size

        if width <= max_width and height <= max_height:
            return image

        # Calculate aspect-preserving resize
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _extract_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP embedding from prepared image.

        Args:
            image: Prepared PIL Image

        Returns:
            Normalized embedding vector
        """
        if self._model is None or self._processor is None:
            raise CLIPServiceError("CLIP model not loaded")

        try:
            # Process image
            inputs = self._processor(images=image, return_tensors="pt")

            # Move inputs to device
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                pixel_values = inputs["pixel_values"]
                # Ensure pixel_values is a tensor and cast to FloatTensor
                if not isinstance(pixel_values, torch.Tensor):
                    raise EmbeddingExtractionError("Expected tensor for pixel_values")
                # Cast to FloatTensor as expected by the model
                float_tensor = cast(torch.FloatTensor, pixel_values.float())
                image_features = self._model.get_image_features(
                    pixel_values=float_tensor
                )

            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy().flatten()

            if self.clip_config.normalize_embeddings:
                norm = float(np.linalg.norm(embedding))
                if norm > 0:
                    embedding = embedding / norm
                embedding = np.asarray(embedding)

            return np.asarray(embedding, dtype=np.float32)

        except Exception as e:
            raise EmbeddingExtractionError(
                f"Failed to extract CLIP embedding: {e}"
            ) from e

    def _calculate_embedding_confidence(self, embedding: np.ndarray) -> float:
        """Calculate confidence score for an embedding.

        Args:
            embedding: Embedding vector

        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: higher norm suggests more confident embedding
        norm = np.linalg.norm(embedding)
        # Normalize to [0, 1] range (assuming typical norms are < 10)
        confidence = min(1.0, float(norm) / 10.0)
        return float(confidence)

    def _calculate_batch_statistics(
        self,
        all_results: list[SemanticAnalysisResult],
        successful_results: list[SemanticAnalysisResult],
        total_duration: float,
    ) -> BatchSemanticResult:
        """Calculate statistics for batch semantic analysis.

        Args:
            all_results: All analysis results including failures
            successful_results: Only successful analysis results
            total_duration: Total processing time

        Returns:
            Batch statistics and results
        """
        successful_count = len(successful_results)
        failed_count = len(all_results) - successful_count

        # Calculate processing rate
        embeddings_per_second = (
            len(all_results) / total_duration if total_duration > 0 else 0.0
        )

        # Extract successful embeddings
        embeddings = [
            r.embedding for r in successful_results if r.embedding is not None
        ]

        # Calculate mean similarity if we have enough embeddings
        mean_similarity = 0.0
        if len(embeddings) >= 2:
            diversity_analysis = self.analyze_diversity(embeddings)
            mean_similarity = diversity_analysis.mean_pairwise_similarity

        # Calculate embedding statistics
        if embeddings:
            embedding_matrix = np.array([emb.embedding for emb in embeddings])
            embedding_stats = {
                "mean_norm": float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
                "dimension": len(embeddings[0].embedding),
                "sparsity": float(np.mean(embedding_matrix == 0.0)),
            }
        else:
            embedding_stats = {}

        return BatchSemanticResult(
            results=all_results,
            successful_analyses=successful_count,
            failed_analyses=failed_count,
            total_duration=total_duration,
            embeddings_per_second=embeddings_per_second,
            mean_similarity=mean_similarity,
            embedding_statistics=embedding_stats,
        )

    def _kmeans_clustering(
        self, embedding_matrix: np.ndarray
    ) -> tuple[Any, np.ndarray, float | None]:
        """Perform K-means clustering on embeddings."""
        # Determine optimal number of clusters
        if self.clip_config.enable_auto_clustering:
            n_clusters = self._determine_optimal_clusters(embedding_matrix)
        else:
            n_clusters = min(self.clip_config.max_clusters, len(embedding_matrix) // 2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embedding_matrix)

        return kmeans, labels, kmeans.inertia_

    def _hierarchical_clustering(
        self, embedding_matrix: np.ndarray
    ) -> tuple[Any, np.ndarray, None]:
        """Perform hierarchical clustering on embeddings."""
        from sklearn.cluster import AgglomerativeClustering

        # Determine number of clusters
        n_clusters = min(self.clip_config.max_clusters, len(embedding_matrix) // 2)

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clustering.fit_predict(embedding_matrix)

        return clustering, labels, None

    def _dbscan_clustering(
        self, embedding_matrix: np.ndarray
    ) -> tuple[Any, np.ndarray, None]:
        """Perform DBSCAN clustering on embeddings."""
        # Use similarity threshold to determine eps
        eps = 1.0 - self.clip_config.similarity_threshold
        min_samples = max(2, self.clip_config.min_cluster_size)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embedding_matrix)

        return dbscan, labels, None

    def _determine_optimal_clusters(self, embedding_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method."""
        max_k = min(self.clip_config.max_clusters, len(embedding_matrix) // 2, 10)

        if max_k < 2:
            return 2

        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(embedding_matrix)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            # Find the point where the rate of decrease slows down most
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
            return k_range[elbow_idx]

        return k_range[len(k_range) // 2]  # Default to middle value

    def _create_cluster_objects(
        self,
        embeddings: list[SemanticEmbedding],
        labels: np.ndarray,
        embedding_matrix: np.ndarray,
    ) -> list[SemanticCluster]:
        """Create SemanticCluster objects from clustering results."""
        clusters = []
        unique_labels = set(labels)

        # Filter out noise points (label -1 in DBSCAN)
        unique_labels.discard(-1)

        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            cluster_paths = [emb.path for emb in cluster_embeddings]

            # Calculate centroid
            cluster_vectors = embedding_matrix[cluster_indices]
            centroid = np.mean(cluster_vectors, axis=0)

            # Calculate intra-cluster similarity
            if len(cluster_vectors) > 1:
                pairwise_sims = []
                for i in range(len(cluster_vectors)):
                    for j in range(i + 1, len(cluster_vectors)):
                        sim = np.dot(cluster_vectors[i], cluster_vectors[j])
                        pairwise_sims.append(sim)
                intra_similarity = float(np.mean(pairwise_sims))
            else:
                intra_similarity = 1.0

            cluster = SemanticCluster(
                cluster_id=int(cluster_id),
                image_paths=cluster_paths,
                centroid=centroid.tolist(),
                intra_cluster_similarity=intra_similarity,
                size=len(cluster_paths),
            )
            clusters.append(cluster)

        return clusters

    def _apply_diversity_selection(
        self,
        clustering_result: ClusteringResult,
        criteria: SemanticSelectionCriteria,
        quality_scores: dict[Path, float] | None,
    ) -> tuple[list[Path], list[str]]:
        """Apply diversity-based selection algorithm."""
        selected_paths = []
        reasoning = []

        # Sort clusters by size (largest first for better representation)
        sorted_clusters = sorted(
            clustering_result.clusters, key=lambda c: c.size, reverse=True
        )

        # Calculate target per cluster
        total_clusters = len(sorted_clusters)
        base_per_cluster = max(1, criteria.target_count // total_clusters)
        remaining_slots = criteria.target_count - (base_per_cluster * total_clusters)

        reasoning.append(
            f"Distributing {criteria.target_count} selections across {total_clusters} clusters"
        )
        reasoning.append(
            f"Base allocation: {base_per_cluster} per cluster, {remaining_slots} extra slots"
        )

        for i, cluster in enumerate(sorted_clusters):
            # Calculate allocation for this cluster
            cluster_allocation = base_per_cluster
            if i < remaining_slots:  # Give extra slots to largest clusters
                cluster_allocation += 1

            # Apply min/max constraints
            cluster_allocation = max(
                criteria.min_cluster_representation, cluster_allocation
            )
            if criteria.max_cluster_representation:
                cluster_allocation = min(
                    criteria.max_cluster_representation, cluster_allocation
                )
            cluster_allocation = min(cluster_allocation, cluster.size)

            # Select best images from cluster
            cluster_paths = cluster.image_paths[:cluster_allocation]

            # If we have quality scores, sort by quality within cluster
            if quality_scores:
                cluster_paths = sorted(
                    cluster.image_paths,
                    key=lambda p: quality_scores.get(p, 0.0),
                    reverse=True,
                )[:cluster_allocation]

            selected_paths.extend(cluster_paths)
            reasoning.append(
                f"Cluster {cluster.cluster_id}: selected {len(cluster_paths)}/{cluster.size} images"
            )

            if len(selected_paths) >= criteria.target_count:
                break

        # Trim to exact target count if we went over
        if len(selected_paths) > criteria.target_count:
            selected_paths = selected_paths[: criteria.target_count]
            reasoning.append(
                f"Trimmed selection to exactly {criteria.target_count} images"
            )

        return selected_paths, reasoning


# Global service instance
_clip_service: CLIPService | None = None


def get_clip_service(config: CuLoRAConfig | None = None) -> CLIPService:
    """Get or create CLIP service instance.

    Args:
        config: Configuration to use for service creation

    Returns:
        CLIP service instance
    """
    global _clip_service

    if _clip_service is None:
        if config is None:
            from culora.services.config_service import get_config_service

            config = get_config_service().get_config()
        _clip_service = CLIPService(config)

    return _clip_service
