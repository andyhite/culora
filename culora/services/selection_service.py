"""Multi-criteria selection service for intelligent dataset curation."""

import time
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from culora.core.exceptions import (
    SelectionConfigurationError,
    SelectionError,
    SelectionInsufficientDataError,
)
from culora.domain.models.config.selection import (
    SelectionConfig,
    SelectionConstraints,
)
from culora.domain.models.selection import (
    DistributionAnalysis,
    SelectionCandidate,
    SelectionCriteria,
    SelectionResult,
    SelectionStageResult,
    SelectionSummary,
)
from culora.services.clip_service import get_clip_service
from culora.services.duplicate_service import get_duplicate_service
from culora.services.pose_service import get_pose_service
from culora.services.quality_service import get_quality_service


class SelectionService:
    """Service for multi-criteria image selection with quality, diversity, and distribution optimization."""

    def __init__(self) -> None:
        """Initialize the selection service."""
        self._quality_service = get_quality_service()
        self._duplicate_service = get_duplicate_service()
        self._pose_service = get_pose_service()
        self._clip_service = get_clip_service()

    def select_images(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        constraints: SelectionConstraints | None = None,
    ) -> SelectionResult:
        """Perform multi-criteria image selection.

        Args:
            candidates: List of candidate images with analysis results
            config: Selection configuration and parameters
            constraints: Runtime constraints (auto-generated if not provided)

        Returns:
            Complete selection results with analysis and statistics

        Raises:
            SelectionError: If selection process fails
            SelectionConfigurationError: If configuration is invalid
            SelectionInsufficientDataError: If insufficient data for selection
        """
        start_time = time.time()
        stage_results: list[SelectionStageResult] = []

        try:
            # Generate constraints if not provided
            if constraints is None:
                constraints = self._generate_constraints(candidates)

            # Validate configuration
            self._validate_selection_config(config, constraints)

            # Calculate effective target count
            effective_target = constraints.calculate_effective_target(config)

            # Create selection criteria
            criteria = self._create_selection_criteria(config, effective_target)

            # Stage 1: Quality filtering
            current_candidates = candidates
            if config.enable_early_filtering:
                stage_result = self._apply_quality_filtering(
                    current_candidates, config, "quality_filtering"
                )
                stage_results.append(stage_result)
                current_candidates = stage_result.candidates

                if not current_candidates:
                    raise SelectionInsufficientDataError(
                        "No candidates passed quality filtering"
                    )

            # Stage 2: Duplicate removal
            if config.enable_duplicate_removal:
                stage_result = self._apply_duplicate_filtering(
                    current_candidates, config, "duplicate_removal"
                )
                stage_results.append(stage_result)
                current_candidates = stage_result.candidates

                if not current_candidates:
                    raise SelectionInsufficientDataError(
                        "No candidates remained after duplicate removal"
                    )

            # Stage 3: Multi-criteria selection
            final_stage_result = self._apply_multi_criteria_selection(
                current_candidates, config, criteria, effective_target
            )
            stage_results.append(final_stage_result)

            selected_candidates = final_stage_result.candidates
            rejected_candidates = self._get_rejected_candidates(
                candidates, selected_candidates
            )

            # Generate analysis and statistics
            quality_distribution = self._analyze_quality_distribution(
                selected_candidates
            )
            composition_distribution = self._analyze_composition_distribution(
                selected_candidates, config
            )
            diversity_analysis = self._analyze_diversity(selected_candidates)

            # Calculate performance metrics
            total_duration = time.time() - start_time
            selection_efficiency = (
                len(candidates) / total_duration if total_duration > 0 else 0.0
            )

            # Calculate success metrics
            target_fulfillment = len(selected_candidates) / effective_target
            quality_improvement = self._calculate_quality_improvement(
                candidates, selected_candidates
            )

            # Create final result
            result = SelectionResult(
                selected_candidates=selected_candidates,
                rejected_candidates=rejected_candidates,
                total_processed=len(candidates),
                criteria=criteria,
                constraints_applied=self._serialize_constraints(constraints),
                stage_results=stage_results,
                quality_distribution=quality_distribution,
                composition_distribution=composition_distribution,
                diversity_analysis=diversity_analysis,
                total_duration=total_duration,
                selection_efficiency=selection_efficiency,
                success=len(selected_candidates) > 0,
                target_fulfillment_ratio=target_fulfillment,
                quality_improvement_ratio=quality_improvement,
            )

            return result

        except Exception as e:
            if isinstance(
                e,
                SelectionError
                | SelectionConfigurationError
                | SelectionInsufficientDataError,
            ):
                raise
            raise SelectionError(f"Selection process failed: {e}") from e

    def _generate_constraints(
        self, candidates: list[SelectionCandidate]
    ) -> SelectionConstraints:
        """Generate selection constraints from candidate analysis."""
        quality_count = sum(1 for c in candidates if c.has_quality_analysis)
        # face_count = sum(1 for c in candidates if c.has_face_analysis)  # Not used
        composition_count = sum(1 for c in candidates if c.has_composition_analysis)
        pose_count = sum(1 for c in candidates if c.has_pose_analysis)
        semantic_count = sum(1 for c in candidates if c.has_semantic_analysis)
        reference_count = sum(1 for c in candidates if c.has_reference_match)
        duplicate_count = sum(1 for c in candidates if not c.is_duplicate)

        return SelectionConstraints(
            available_images=len(candidates),
            quality_filtered_count=quality_count,
            duplicate_filtered_count=duplicate_count,
            composition_analyzed_count=composition_count,
            pose_analyzed_count=pose_count,
            semantic_analyzed_count=semantic_count,
            reference_matched_count=reference_count,
        )

    def _validate_selection_config(
        self, config: SelectionConfig, constraints: SelectionConstraints
    ) -> None:
        """Validate selection configuration against constraints."""
        if config.target_count <= 0:
            raise SelectionConfigurationError("Target count must be positive")

        if constraints.available_images == 0:
            raise SelectionInsufficientDataError("No candidate images available")

        max_possible = config.calculate_max_selection(constraints.available_images)
        if max_possible == 0:
            raise SelectionInsufficientDataError(
                "No images can be selected with current configuration"
            )

    def _create_selection_criteria(
        self, config: SelectionConfig, target_count: int
    ) -> SelectionCriteria:
        """Create selection criteria from configuration."""
        # Calculate weights based on configuration
        diversity_weight = config.diversity_settings.diversity_weight
        quality_weight = config.diversity_settings.quality_vs_diversity_balance
        distribution_weight = 0.1 if config.enable_distribution_enforcement else 0.0
        reference_weight = (
            config.reference_match_weight if config.enable_reference_matching else 0.0
        )

        # Normalize weights
        total_weight = (
            quality_weight + diversity_weight + distribution_weight + reference_weight
        )
        if total_weight > 0:
            quality_weight /= total_weight
            diversity_weight /= total_weight
            distribution_weight /= total_weight
            reference_weight /= total_weight

        return SelectionCriteria(
            target_count=target_count,
            min_quality_threshold=config.quality_thresholds.min_composite_quality,
            enable_duplicate_removal=config.enable_duplicate_removal,
            enable_diversity_optimization=(
                config.diversity_settings.enable_pose_diversity
                or config.diversity_settings.enable_semantic_diversity
            ),
            enable_distribution_balancing=config.enable_distribution_enforcement,
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
            distribution_weight=distribution_weight,
            reference_match_weight=reference_weight,
            selection_strategy=config.selection_strategy,
        )

    def _apply_quality_filtering(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        stage_name: str,
    ) -> SelectionStageResult:
        """Apply quality-based filtering."""
        start_time = time.time()
        input_count = len(candidates)

        # Filter by quality thresholds
        filtered_candidates = []
        for candidate in candidates:
            if (
                candidate.effective_quality_score
                >= config.quality_thresholds.min_composite_quality
            ):
                filtered_candidates.append(candidate)

        duration = time.time() - start_time
        filtered_count = input_count - len(filtered_candidates)

        # Generate quality statistics
        quality_stats = self._calculate_quality_stats(filtered_candidates)

        return SelectionStageResult(
            stage_name=stage_name,
            input_count=input_count,
            output_count=len(filtered_candidates),
            filtered_count=filtered_count,
            candidates=filtered_candidates,
            duration=duration,
            success=True,
            quality_stats=quality_stats,
        )

    def _apply_duplicate_filtering(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        stage_name: str,
    ) -> SelectionStageResult:
        """Apply duplicate removal filtering."""
        start_time = time.time()
        input_count = len(candidates)

        # Group candidates by duplicate group
        non_duplicates = [c for c in candidates if not c.is_duplicate]
        duplicate_representatives = [
            c for c in candidates if c.is_duplicate_representative
        ]

        # Combine non-duplicates with best representatives
        filtered_candidates = non_duplicates + duplicate_representatives

        duration = time.time() - start_time
        filtered_count = input_count - len(filtered_candidates)

        return SelectionStageResult(
            stage_name=stage_name,
            input_count=input_count,
            output_count=len(filtered_candidates),
            filtered_count=filtered_count,
            candidates=filtered_candidates,
            duration=duration,
            success=True,
        )

    def _apply_multi_criteria_selection(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        criteria: SelectionCriteria,
        target_count: int,
    ) -> SelectionStageResult:
        """Apply multi-criteria selection algorithm."""
        start_time = time.time()
        input_count = len(candidates)

        try:
            if config.selection_strategy == "quality_first":
                selected = self._select_by_quality_first(candidates, target_count)
            elif config.selection_strategy == "diversity_first":
                selected = self._select_by_diversity_first(
                    candidates, config, target_count
                )
            elif config.selection_strategy == "balanced":
                selected = self._select_balanced(candidates, config, target_count)
            else:  # multi_stage
                selected = self._select_multi_stage(candidates, config, target_count)

            duration = time.time() - start_time

            # Calculate distribution statistics
            distribution_stats = self._calculate_distribution_stats(selected)
            diversity_stats = (
                self._calculate_diversity_stats(selected)
                if config.diversity_settings.diversity_weight > 0
                else None
            )

            return SelectionStageResult(
                stage_name="multi_criteria_selection",
                input_count=input_count,
                output_count=len(selected),
                filtered_count=input_count - len(selected),
                candidates=selected,
                duration=duration,
                success=True,
                distribution_stats=distribution_stats,
                diversity_stats=diversity_stats,
            )

        except Exception as e:
            duration = time.time() - start_time
            return SelectionStageResult(
                stage_name="multi_criteria_selection",
                input_count=input_count,
                output_count=0,
                filtered_count=input_count,
                candidates=[],
                duration=duration,
                success=False,
                error_message=str(e),
            )

    def _select_by_quality_first(
        self, candidates: list[SelectionCandidate], target_count: int
    ) -> list[SelectionCandidate]:
        """Select images prioritizing quality above all else."""
        # Sort by quality score (descending)
        sorted_candidates = sorted(
            candidates, key=lambda c: c.effective_quality_score, reverse=True
        )

        return sorted_candidates[:target_count]

    def _select_by_diversity_first(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        target_count: int,
    ) -> list[SelectionCandidate]:
        """Select images prioritizing diversity above quality."""
        # Use clustering for diversity selection
        if config.diversity_settings.enable_pose_diversity:
            return self._select_by_pose_diversity(candidates, target_count)
        elif config.diversity_settings.enable_semantic_diversity:
            return self._select_by_semantic_diversity(candidates, target_count)
        else:
            # Fallback to quality-based selection
            return self._select_by_quality_first(candidates, target_count)

    def _select_balanced(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        target_count: int,
    ) -> list[SelectionCandidate]:
        """Select images with balanced quality and diversity scoring."""
        # Calculate composite scores combining quality and diversity
        scored_candidates = []

        for candidate in candidates:
            quality_score = candidate.effective_quality_score
            diversity_score = self._calculate_candidate_diversity_score(
                candidate, candidates
            )

            composite_score = (
                config.diversity_settings.quality_vs_diversity_balance * quality_score
                + (1 - config.diversity_settings.quality_vs_diversity_balance)
                * diversity_score
            )

            # Create new candidate with updated selection score
            scored_candidate = SelectionCandidate(
                path=candidate.path,
                file_size=candidate.file_size,
                quality_assessment=candidate.quality_assessment,
                composite_quality_score=candidate.composite_quality_score,
                face_analysis=candidate.face_analysis,
                reference_match=candidate.reference_match,
                composition_analysis=candidate.composition_analysis,
                pose_analysis=candidate.pose_analysis,
                semantic_embedding=candidate.semantic_embedding,
                duplicate_group_id=candidate.duplicate_group_id,
                is_duplicate_representative=candidate.is_duplicate_representative,
                duplicate_quality_rank=candidate.duplicate_quality_rank,
                selection_score=composite_score,
                quality_percentile=candidate.quality_percentile,
                diversity_score=diversity_score,
                distribution_bonus=candidate.distribution_bonus,
            )
            scored_candidates.append(scored_candidate)

        # Sort by composite score and select top candidates
        sorted_candidates = sorted(
            scored_candidates, key=lambda c: c.selection_score or 0.0, reverse=True
        )

        return sorted_candidates[:target_count]

    def _select_multi_stage(
        self,
        candidates: list[SelectionCandidate],
        config: SelectionConfig,
        target_count: int,
    ) -> list[SelectionCandidate]:
        """Multi-stage selection with quality, diversity, and distribution balancing."""
        # Stage 1: Pre-filter by quality (keep top 3x target for diversity analysis)
        quality_filtered_count = min(target_count * 3, len(candidates))
        quality_sorted = sorted(
            candidates, key=lambda c: c.effective_quality_score, reverse=True
        )
        quality_filtered = quality_sorted[:quality_filtered_count]

        # Stage 2: Diversity-based selection from quality-filtered candidates
        if (
            config.diversity_settings.enable_pose_diversity
            and len(quality_filtered) > target_count
        ):
            diversity_selected = self._select_by_pose_diversity(
                quality_filtered, target_count
            )
        elif (
            config.diversity_settings.enable_semantic_diversity
            and len(quality_filtered) > target_count
        ):
            diversity_selected = self._select_by_semantic_diversity(
                quality_filtered, target_count
            )
        else:
            diversity_selected = quality_filtered[:target_count]

        # Stage 3: Distribution balancing (if enabled and insufficient diversity)
        if (
            config.enable_distribution_enforcement
            and len(diversity_selected) < target_count
        ):
            final_selected = self._balance_distribution(
                diversity_selected, candidates, config, target_count
            )
        else:
            final_selected = diversity_selected

        return final_selected[:target_count]

    def _select_by_pose_diversity(
        self, candidates: list[SelectionCandidate], target_count: int
    ) -> list[SelectionCandidate]:
        """Select images using pose-based diversity clustering."""
        pose_candidates = [c for c in candidates if c.has_pose_analysis]

        if len(pose_candidates) < target_count:
            # Not enough pose data, fallback to quality selection
            return self._select_by_quality_first(candidates, target_count)

        # Extract pose vectors
        pose_vectors = []
        for candidate in pose_candidates:
            if candidate.pose_analysis and candidate.pose_analysis.pose_vector:
                pose_vectors.append(candidate.pose_analysis.pose_vector.vector)

        if len(pose_vectors) < 5:
            # Insufficient data for clustering
            return self._select_by_quality_first(candidates, target_count)

        # Perform K-means clustering
        n_clusters = min(target_count, len(pose_vectors) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(pose_vectors)

        # Select best candidate from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_candidates = [
                pose_candidates[i]
                for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]

            if cluster_candidates:
                # Select highest quality candidate from cluster
                best_in_cluster = max(
                    cluster_candidates, key=lambda c: c.effective_quality_score
                )
                selected.append(best_in_cluster)

        # Fill remaining slots with highest quality candidates not yet selected
        if len(selected) < target_count:
            selected_paths = {c.path for c in selected}
            remaining_candidates = [
                c for c in candidates if c.path not in selected_paths
            ]
            remaining_sorted = sorted(
                remaining_candidates,
                key=lambda c: c.effective_quality_score,
                reverse=True,
            )
            selected.extend(remaining_sorted[: target_count - len(selected)])

        return selected[:target_count]

    def _select_by_semantic_diversity(
        self, candidates: list[SelectionCandidate], target_count: int
    ) -> list[SelectionCandidate]:
        """Select images using semantic embedding diversity."""
        semantic_candidates = [c for c in candidates if c.has_semantic_analysis]

        if len(semantic_candidates) < target_count:
            return self._select_by_quality_first(candidates, target_count)

        # Extract semantic embeddings
        embeddings = []
        for candidate in semantic_candidates:
            if candidate.semantic_embedding:
                embeddings.append(candidate.semantic_embedding.embedding)

        if len(embeddings) < 5:
            return self._select_by_quality_first(candidates, target_count)

        # Perform K-means clustering on embeddings
        n_clusters = min(target_count, len(embeddings) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select best candidate from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_candidates = [
                semantic_candidates[i]
                for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]

            if cluster_candidates:
                best_in_cluster = max(
                    cluster_candidates, key=lambda c: c.effective_quality_score
                )
                selected.append(best_in_cluster)

        # Fill remaining slots
        if len(selected) < target_count:
            selected_paths = {c.path for c in selected}
            remaining_candidates = [
                c for c in candidates if c.path not in selected_paths
            ]
            remaining_sorted = sorted(
                remaining_candidates,
                key=lambda c: c.effective_quality_score,
                reverse=True,
            )
            selected.extend(remaining_sorted[: target_count - len(selected)])

        return selected[:target_count]

    def _balance_distribution(
        self,
        current_selection: list[SelectionCandidate],
        all_candidates: list[SelectionCandidate],
        config: SelectionConfig,
        target_count: int,
    ) -> list[SelectionCandidate]:
        """Balance composition distribution in selection."""
        # Analyze current distribution
        current_shot_types: Counter[Any] = Counter()
        current_scene_types: Counter[Any] = Counter()

        for candidate in current_selection:
            if candidate.has_composition_analysis and candidate.composition_analysis:
                current_shot_types[candidate.composition_analysis.shot_type] += 1
                current_scene_types[candidate.composition_analysis.scene_type] += 1

        # Identify under-represented categories
        selected_paths = {c.path for c in current_selection}
        remaining_candidates = [
            c for c in all_candidates if c.path not in selected_paths
        ]

        # Add candidates to balance distribution
        additional_selections = []
        remaining_slots = target_count - len(current_selection)

        # Simple balancing: prioritize under-represented shot types
        shot_type_targets = config.shot_type_distribution.fallback_distribution
        for shot_type, target_ratio in shot_type_targets.items():
            target_count_for_type = int(target_count * target_ratio)
            current_count = current_shot_types[shot_type]

            if current_count < target_count_for_type and remaining_slots > 0:
                # Find candidates with this shot type
                type_candidates = [
                    c
                    for c in remaining_candidates
                    if c.has_composition_analysis
                    and c.composition_analysis
                    and c.composition_analysis.shot_type == shot_type
                ]

                # Select best quality candidates of this type
                type_candidates.sort(
                    key=lambda c: c.effective_quality_score, reverse=True
                )
                needed = min(
                    target_count_for_type - current_count,
                    len(type_candidates),
                    remaining_slots,
                )

                additional_selections.extend(type_candidates[:needed])
                remaining_slots -= needed

                # Remove selected candidates from remaining pool
                selected_paths.update(c.path for c in type_candidates[:needed])
                remaining_candidates = [
                    c for c in remaining_candidates if c.path not in selected_paths
                ]

        return current_selection + additional_selections

    def _calculate_candidate_diversity_score(
        self, candidate: SelectionCandidate, all_candidates: list[SelectionCandidate]
    ) -> float:
        """Calculate diversity score for a single candidate relative to others."""
        if candidate.has_pose_analysis and candidate.pose_analysis:
            # Use pose-based diversity
            return self._calculate_pose_diversity_score(candidate, all_candidates)
        elif candidate.has_semantic_analysis and candidate.semantic_embedding:
            # Use semantic diversity
            return self._calculate_semantic_diversity_score(candidate, all_candidates)
        else:
            # No diversity data available
            return 0.5  # Neutral diversity score

    def _calculate_pose_diversity_score(
        self, candidate: SelectionCandidate, all_candidates: list[SelectionCandidate]
    ) -> float:
        """Calculate pose diversity score for a candidate."""
        if not candidate.pose_analysis or not candidate.pose_analysis.pose_vector:
            return 0.0

        candidate_vector = np.array(candidate.pose_analysis.pose_vector.vector)

        # Calculate distances to other candidates with pose data
        distances = []
        for other in all_candidates:
            if (
                other.path != candidate.path
                and other.has_pose_analysis
                and other.pose_analysis
                and other.pose_analysis.pose_vector
            ):

                other_vector = np.array(other.pose_analysis.pose_vector.vector)
                distance = np.linalg.norm(candidate_vector - other_vector)
                distances.append(distance)

        if not distances:
            return 0.5  # Neutral score if no comparisons possible

        # Higher average distance means more diverse
        avg_distance = np.mean(distances)
        # Normalize to 0-1 range (assuming max distance of ~10 for pose vectors)
        return float(min(avg_distance / 10.0, 1.0))

    def _calculate_semantic_diversity_score(
        self, candidate: SelectionCandidate, all_candidates: list[SelectionCandidate]
    ) -> float:
        """Calculate semantic diversity score for a candidate."""
        if not candidate.semantic_embedding:
            return 0.0

        candidate_embedding = np.array(candidate.semantic_embedding.embedding)

        # Calculate cosine similarities to other candidates
        similarities = []
        for other in all_candidates:
            if (
                other.path != candidate.path
                and other.has_semantic_analysis
                and other.semantic_embedding
            ):

                other_embedding = np.array(other.semantic_embedding.embedding)
                # Cosine similarity
                similarity = np.dot(candidate_embedding, other_embedding) / (
                    np.linalg.norm(candidate_embedding)
                    * np.linalg.norm(other_embedding)
                )
                similarities.append(similarity)

        if not similarities:
            return 0.5  # Neutral score if no comparisons possible

        # Lower average similarity means more diverse
        avg_similarity = np.mean(similarities)
        return float(1.0 - avg_similarity)  # Convert similarity to diversity score

    def _get_rejected_candidates(
        self,
        all_candidates: list[SelectionCandidate],
        selected_candidates: list[SelectionCandidate],
    ) -> list[SelectionCandidate]:
        """Get list of rejected candidates."""
        selected_paths = {c.path for c in selected_candidates}
        return [c for c in all_candidates if c.path not in selected_paths]

    def _analyze_quality_distribution(
        self, candidates: list[SelectionCandidate]
    ) -> dict[str, float]:
        """Analyze quality score distribution of selected candidates."""
        if not candidates:
            return {}

        scores = [c.effective_quality_score for c in candidates]
        return {
            "min_quality": min(scores),
            "max_quality": max(scores),
            "mean_quality": float(np.mean(scores)),
            "median_quality": float(np.median(scores)),
            "std_quality": float(np.std(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
        }

    def _analyze_composition_distribution(
        self, candidates: list[SelectionCandidate], config: SelectionConfig
    ) -> DistributionAnalysis | None:
        """Analyze composition distribution of selected candidates."""
        composition_candidates = [c for c in candidates if c.has_composition_analysis]

        if not composition_candidates:
            return None

        # Count actual distributions
        shot_type_counts: Counter[str] = Counter()
        scene_type_counts: Counter[str] = Counter()

        for candidate in composition_candidates:
            if candidate.composition_analysis:
                shot_type_counts[candidate.composition_analysis.shot_type.value] += 1
                scene_type_counts[candidate.composition_analysis.scene_type.value] += 1

        # Get target distributions from config
        target_shot_types = config.shot_type_distribution.fallback_distribution
        target_scene_types = config.scene_type_distribution.fallback_distribution

        # Calculate target counts
        total_selected = len(composition_candidates)
        target_shot_counts = {
            shot_type.value: int(total_selected * ratio)
            for shot_type, ratio in target_shot_types.items()
        }
        target_scene_counts = {
            scene_type.value: int(total_selected * ratio)
            for scene_type, ratio in target_scene_types.items()
        }

        # Combine for overall analysis
        all_target_counts = {**target_shot_counts, **target_scene_counts}
        all_actual_counts = {**dict(shot_type_counts), **dict(scene_type_counts)}

        # Calculate ratios and fulfillment
        target_ratios = {}
        actual_ratios = {}
        fulfillment_ratios = {}

        for category in all_target_counts:
            target_count = all_target_counts[category]
            actual_count = all_actual_counts.get(category, 0)

            target_ratios[category] = (
                target_count / total_selected if total_selected > 0 else 0
            )
            actual_ratios[category] = (
                actual_count / total_selected if total_selected > 0 else 0
            )
            fulfillment_ratios[category] = (
                actual_count / target_count if target_count > 0 else 0
            )

        # Calculate overall distribution score
        fulfillment_scores = [min(ratio, 1.0) for ratio in fulfillment_ratios.values()]
        overall_score = (
            float(np.mean(fulfillment_scores)) if fulfillment_scores else 0.0
        )

        # Identify missing and over-represented categories
        missing_categories = [
            cat for cat, ratio in fulfillment_ratios.items() if ratio < 0.5
        ]
        over_represented = [
            cat for cat, ratio in fulfillment_ratios.items() if ratio > 1.5
        ]

        return DistributionAnalysis(
            target_counts=all_target_counts,
            actual_counts=all_actual_counts,
            target_ratios=target_ratios,
            actual_ratios=actual_ratios,
            fulfillment_ratios=fulfillment_ratios,
            overall_distribution_score=overall_score,
            missing_categories=missing_categories,
            over_represented_categories=over_represented,
        )

    def _analyze_diversity(
        self, candidates: list[SelectionCandidate]
    ) -> dict[str, float] | None:
        """Analyze diversity metrics of selected candidates."""
        if len(candidates) < 2:
            return None

        diversity_metrics = {}

        # Pose diversity analysis
        pose_candidates = [c for c in candidates if c.has_pose_analysis]
        if len(pose_candidates) >= 2:
            pose_vectors = []
            for candidate in pose_candidates:
                if candidate.pose_analysis and candidate.pose_analysis.pose_vector:
                    pose_vectors.append(candidate.pose_analysis.pose_vector.vector)

            if len(pose_vectors) >= 2:
                pose_array = np.array(pose_vectors)
                distances = euclidean_distances(pose_array)
                # Average pairwise distance as diversity metric
                avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                diversity_metrics["pose_diversity"] = min(avg_distance / 10.0, 1.0)

        # Semantic diversity analysis
        semantic_candidates = [c for c in candidates if c.has_semantic_analysis]
        if len(semantic_candidates) >= 2:
            embeddings = []
            for candidate in semantic_candidates:
                if candidate.semantic_embedding:
                    embeddings.append(candidate.semantic_embedding.embedding)

            if len(embeddings) >= 2:
                # embedding_array = np.array(embeddings)  # Not used
                # Calculate pairwise cosine similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i])
                            * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(similarity)

                # Average dissimilarity as diversity metric
                avg_similarity = np.mean(similarities)
                diversity_metrics["semantic_diversity"] = 1.0 - avg_similarity

        return diversity_metrics if diversity_metrics else None

    def _calculate_quality_improvement(
        self,
        all_candidates: list[SelectionCandidate],
        selected_candidates: list[SelectionCandidate],
    ) -> float:
        """Calculate quality improvement ratio between selected and all candidates."""
        if not all_candidates or not selected_candidates:
            return 0.0

        all_scores = [c.effective_quality_score for c in all_candidates]
        selected_scores = [c.effective_quality_score for c in selected_candidates]

        avg_all_quality = np.mean(all_scores)
        avg_selected_quality = np.mean(selected_scores)

        if avg_all_quality == 0:
            return 0.0

        return float(avg_selected_quality / avg_all_quality)

    def _calculate_quality_stats(
        self, candidates: list[SelectionCandidate]
    ) -> dict[str, float]:
        """Calculate quality statistics for a set of candidates."""
        if not candidates:
            return {}

        scores = [c.effective_quality_score for c in candidates]
        return {
            "count": len(candidates),
            "mean_quality": float(np.mean(scores)),
            "median_quality": float(np.median(scores)),
            "min_quality": float(min(scores)),
            "max_quality": float(max(scores)),
            "std_quality": float(np.std(scores)),
        }

    def _calculate_distribution_stats(
        self, candidates: list[SelectionCandidate]
    ) -> dict[str, int]:
        """Calculate composition distribution statistics."""
        shot_type_counts: Counter[str] = Counter()
        scene_type_counts: Counter[str] = Counter()

        for candidate in candidates:
            if candidate.has_composition_analysis and candidate.composition_analysis:
                shot_type_counts[candidate.composition_analysis.shot_type.value] += 1
                scene_type_counts[candidate.composition_analysis.scene_type.value] += 1

        return {
            **{f"shot_type_{k}": v for k, v in shot_type_counts.items()},
            **{f"scene_type_{k}": v for k, v in scene_type_counts.items()},
        }

    def _calculate_diversity_stats(
        self, candidates: list[SelectionCandidate]
    ) -> dict[str, float]:
        """Calculate diversity statistics."""
        stats = {}

        # Count candidates with diversity data
        pose_count = sum(1 for c in candidates if c.has_pose_analysis)
        semantic_count = sum(1 for c in candidates if c.has_semantic_analysis)

        stats["pose_data_ratio"] = pose_count / len(candidates) if candidates else 0.0
        stats["semantic_data_ratio"] = (
            semantic_count / len(candidates) if candidates else 0.0
        )

        # Add diversity scores if available
        diversity_analysis = self._analyze_diversity(candidates)
        if diversity_analysis:
            stats.update(diversity_analysis)

        return stats

    def _serialize_constraints(
        self, constraints: SelectionConstraints
    ) -> dict[str, Any]:
        """Serialize constraints to dictionary for result storage."""
        return {
            "available_images": constraints.available_images,
            "quality_filtered_count": constraints.quality_filtered_count,
            "duplicate_filtered_count": constraints.duplicate_filtered_count,
            "composition_analyzed_count": constraints.composition_analyzed_count,
            "pose_analyzed_count": constraints.pose_analyzed_count,
            "semantic_analyzed_count": constraints.semantic_analyzed_count,
            "reference_matched_count": constraints.reference_matched_count,
        }

    def generate_selection_summary(self, result: SelectionResult) -> SelectionSummary:
        """Generate a high-level selection summary for reporting."""
        # Calculate quality percentile distribution
        quality_percentiles = {}
        if result.selected_candidates:
            scores = [c.effective_quality_score for c in result.selected_candidates]
            percentile_ranges = [(90, 100), (80, 90), (70, 80), (60, 70), (0, 60)]

            for low, high in percentile_ranges:
                low_threshold = (
                    np.percentile(scores, low) if len(scores) > 1 else scores[0]
                )
                high_threshold = (
                    np.percentile(scores, high) if len(scores) > 1 else scores[0]
                )
                count = sum(
                    1 for score in scores if low_threshold <= score <= high_threshold
                )
                quality_percentiles[f"{low}-{high}"] = count

        # Extract distribution data
        shot_type_dist = {}
        scene_type_dist = {}
        if result.composition_distribution:
            for key, count in result.composition_distribution.actual_counts.items():
                if key.startswith(
                    ("closeup", "portrait", "headshot", "medium", "full")
                ):
                    shot_type_dist[key] = count
                else:
                    scene_type_dist[key] = count

        # Calculate success metrics
        fulfillment_pct = result.target_fulfillment_ratio * 100
        distribution_score = (
            result.composition_distribution.overall_distribution_score
            if result.composition_distribution
            else 0.5
        )

        # Identify issues and recommendations
        issues = list(result.errors)
        recommendations = []

        if result.target_fulfillment_ratio < 0.8:
            issues.append(f"Low selection fulfillment: {fulfillment_pct:.1f}%")
            recommendations.append(
                "Consider relaxing quality thresholds or expanding input dataset"
            )

        if distribution_score < 0.6:
            issues.append("Poor composition distribution balance")
            recommendations.append(
                "Review distribution targets or increase dataset diversity"
            )

        if not result.has_diversity_analysis:
            recommendations.append(
                "Enable pose or semantic analysis for better diversity optimization"
            )

        return SelectionSummary(
            input_count=result.total_processed,
            selected_count=result.selection_count,
            target_count=result.criteria.target_count,
            fulfillment_percentage=fulfillment_pct,
            avg_quality_improvement=result.quality_improvement_ratio,
            quality_percentile_distribution=quality_percentiles,
            shot_type_distribution=shot_type_dist,
            scene_type_distribution=scene_type_dist,
            distribution_balance_score=distribution_score,
            diversity_score=(
                result.diversity_analysis.get("overall_diversity")
                if result.diversity_analysis
                else None
            ),
            pose_diversity_score=(
                result.diversity_analysis.get("pose_diversity")
                if result.diversity_analysis
                else None
            ),
            semantic_diversity_score=(
                result.diversity_analysis.get("semantic_diversity")
                if result.diversity_analysis
                else None
            ),
            total_processing_time=result.total_duration,
            processing_rate=result.selection_efficiency,
            selection_success=result.success,
            major_issues=issues,
            recommendations=recommendations,
        )


# Global service instance
_selection_service: SelectionService | None = None


def get_selection_service() -> SelectionService:
    """Get the global selection service instance."""
    global _selection_service
    if _selection_service is None:
        _selection_service = SelectionService()
    return _selection_service
