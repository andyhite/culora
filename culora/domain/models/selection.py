"""Domain models for multi-criteria selection algorithms."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from culora.domain.models.clip import SemanticEmbedding
from culora.domain.models.composition import CompositionAnalysis
from culora.domain.models.face import FaceAnalysisResult
from culora.domain.models.face_reference import ReferenceMatchResult
from culora.domain.models.pose import PoseAnalysis
from culora.domain.models.quality import QualityScore


@dataclass(frozen=True)
class SelectionCandidate:
    """A candidate image for selection with all analysis results."""

    # Core image information
    path: Path
    file_size: int

    # Quality analysis
    quality_assessment: QualityScore | None = None
    composite_quality_score: float | None = None

    # Face analysis
    face_analysis: FaceAnalysisResult | None = None
    reference_match: ReferenceMatchResult | None = None

    # Composition analysis
    composition_analysis: CompositionAnalysis | None = None

    # Diversity analysis
    pose_analysis: PoseAnalysis | None = None
    semantic_embedding: SemanticEmbedding | None = None

    # Duplicate information
    duplicate_group_id: int | None = None
    is_duplicate_representative: bool = False
    duplicate_quality_rank: int | None = None

    # Selection scoring
    selection_score: float | None = None
    quality_percentile: float | None = None
    diversity_score: float | None = None
    distribution_bonus: float | None = None

    @property
    def has_quality_analysis(self) -> bool:
        """Check if candidate has quality analysis."""
        return self.quality_assessment is not None

    @property
    def has_face_analysis(self) -> bool:
        """Check if candidate has face analysis."""
        return self.face_analysis is not None and self.face_analysis.success

    @property
    def has_composition_analysis(self) -> bool:
        """Check if candidate has composition analysis."""
        return self.composition_analysis is not None

    @property
    def has_pose_analysis(self) -> bool:
        """Check if candidate has pose analysis."""
        return self.pose_analysis is not None

    @property
    def has_semantic_analysis(self) -> bool:
        """Check if candidate has semantic analysis."""
        return self.semantic_embedding is not None

    @property
    def has_reference_match(self) -> bool:
        """Check if candidate has reference identity match."""
        return (
            self.reference_match is not None
            and self.reference_match.primary_match is not None
        )

    @property
    def effective_quality_score(self) -> float:
        """Get effective quality score with fallback."""
        if self.composite_quality_score is not None:
            return self.composite_quality_score
        if self.quality_assessment is not None:
            return self.quality_assessment.overall_score
        return 0.0

    @property
    def is_duplicate(self) -> bool:
        """Check if this candidate is part of a duplicate group."""
        return self.duplicate_group_id is not None


@dataclass(frozen=True)
class SelectionStageResult:
    """Results from a single stage of the selection pipeline."""

    stage_name: str
    input_count: int
    output_count: int
    filtered_count: int
    candidates: list[SelectionCandidate]
    duration: float
    success: bool
    error_message: str | None = None

    # Stage-specific metrics
    quality_stats: dict[str, float] | None = None
    distribution_stats: dict[str, int] | None = None
    diversity_stats: dict[str, float] | None = None

    @property
    def reduction_ratio(self) -> float:
        """Calculate reduction ratio for this stage."""
        if self.input_count == 0:
            return 0.0
        return self.filtered_count / self.input_count

    @property
    def retention_ratio(self) -> float:
        """Calculate retention ratio for this stage."""
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count


@dataclass(frozen=True)
class DistributionAnalysis:
    """Analysis of how well distribution targets were met."""

    target_counts: dict[str, int]
    actual_counts: dict[str, int]
    target_ratios: dict[str, float]
    actual_ratios: dict[str, float]
    fulfillment_ratios: dict[str, float]  # actual/target for each category
    overall_distribution_score: float  # 0.0-1.0 how well targets were met
    missing_categories: list[str]
    over_represented_categories: list[str]

    @property
    def is_well_distributed(self) -> bool:
        """Check if distribution is reasonably well balanced."""
        return self.overall_distribution_score >= 0.7

    @property
    def total_target_count(self) -> int:
        """Total targeted selection count."""
        return sum(self.target_counts.values())

    @property
    def total_actual_count(self) -> int:
        """Total actual selection count."""
        return sum(self.actual_counts.values())


@dataclass(frozen=True)
class SelectionCriteria:
    """Criteria used for selection process."""

    target_count: int
    min_quality_threshold: float
    enable_duplicate_removal: bool
    enable_diversity_optimization: bool
    enable_distribution_balancing: bool
    quality_weight: float
    diversity_weight: float
    distribution_weight: float
    reference_match_weight: float

    # Distribution targets
    shot_type_targets: dict[str, int] | None = None
    scene_type_targets: dict[str, int] | None = None

    # Selection strategy
    selection_strategy: str = "multi_stage"
    diversity_method: str = (
        "pose_clustering"  # pose_clustering, semantic_clustering, hybrid
    )

    def __post_init__(self) -> None:
        """Validate criteria after initialization."""
        total_weight = (
            self.quality_weight
            + self.diversity_weight
            + self.distribution_weight
            + self.reference_match_weight
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Selection weights must sum to 1.0, got {total_weight}")


@dataclass(frozen=True)
class SelectionResult:
    """Complete results from multi-criteria selection process."""

    # Selection outcomes
    selected_candidates: list[SelectionCandidate]
    rejected_candidates: list[SelectionCandidate]
    total_processed: int

    # Selection criteria and configuration
    criteria: SelectionCriteria
    constraints_applied: dict[str, Any]

    # Stage results
    stage_results: list[SelectionStageResult]

    # Performance metrics
    total_duration: float
    selection_efficiency: float  # selections per second

    # Success metrics
    success: bool
    target_fulfillment_ratio: float  # actual_count / target_count
    quality_improvement_ratio: float  # avg_selected_quality / avg_input_quality

    # Analysis and statistics
    quality_distribution: dict[str, float]  # percentile distributions

    # Optional fields with defaults
    composition_distribution: DistributionAnalysis | None = None
    diversity_analysis: dict[str, float] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def selection_count(self) -> int:
        """Number of selected candidates."""
        return len(self.selected_candidates)

    @property
    def rejection_count(self) -> int:
        """Number of rejected candidates."""
        return len(self.rejected_candidates)

    @property
    def selection_ratio(self) -> float:
        """Ratio of selected to total processed candidates."""
        if self.total_processed == 0:
            return 0.0
        return self.selection_count / self.total_processed

    @property
    def average_selected_quality(self) -> float:
        """Average quality score of selected candidates."""
        if not self.selected_candidates:
            return 0.0

        scores = [
            candidate.effective_quality_score for candidate in self.selected_candidates
        ]
        return sum(scores) / len(scores)

    @property
    def quality_range(self) -> tuple[float, float]:
        """Quality score range of selected candidates."""
        if not self.selected_candidates:
            return (0.0, 0.0)

        scores = [
            candidate.effective_quality_score for candidate in self.selected_candidates
        ]
        return (min(scores), max(scores))

    @property
    def has_composition_distribution(self) -> bool:
        """Check if composition distribution analysis is available."""
        return self.composition_distribution is not None

    @property
    def has_diversity_analysis(self) -> bool:
        """Check if diversity analysis is available."""
        return self.diversity_analysis is not None

    def get_stage_result(self, stage_name: str) -> SelectionStageResult | None:
        """Get results for a specific selection stage."""
        for stage_result in self.stage_results:
            if stage_result.stage_name == stage_name:
                return stage_result
        return None

    def get_candidates_by_quality_percentile(
        self, min_percentile: float, max_percentile: float
    ) -> list[SelectionCandidate]:
        """Get selected candidates within quality percentile range."""
        return [
            candidate
            for candidate in self.selected_candidates
            if candidate.quality_percentile is not None
            and min_percentile <= candidate.quality_percentile <= max_percentile
        ]


@dataclass(frozen=True)
class SelectionSummary:
    """High-level summary of selection results for reporting."""

    # Basic counts
    input_count: int
    selected_count: int
    target_count: int
    fulfillment_percentage: float

    # Quality metrics
    avg_quality_improvement: float
    quality_percentile_distribution: dict[str, int]  # "90-100": 5, "80-90": 10, etc.

    # Distribution metrics
    shot_type_distribution: dict[str, int]
    scene_type_distribution: dict[str, int]
    distribution_balance_score: float  # 0.0-1.0

    # Processing metrics
    total_processing_time: float
    processing_rate: float  # images per second

    # Success indicators
    selection_success: bool

    # Optional fields with defaults
    diversity_score: float | None = None
    pose_diversity_score: float | None = None
    semantic_diversity_score: float | None = None
    major_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_successful_selection(self) -> bool:
        """Determine if selection was generally successful."""
        return (
            self.selection_success
            and self.fulfillment_percentage >= 0.8  # Got at least 80% of target
            and self.distribution_balance_score >= 0.6  # Reasonable distribution
            and not self.major_issues  # No major issues
        )

    @property
    def needs_attention(self) -> bool:
        """Determine if selection results need user attention."""
        return (
            not self.is_successful_selection
            or self.fulfillment_percentage < 0.7
            or self.distribution_balance_score < 0.5
        )


@dataclass(frozen=True)
class ClusterSelection:
    """Results from cluster-based diversity selection."""

    cluster_id: int
    cluster_center: list[float] | None  # embedding center
    candidates_in_cluster: list[SelectionCandidate]
    selected_from_cluster: list[SelectionCandidate]
    cluster_quality_score: float
    cluster_diversity_score: float
    selection_rationale: str

    @property
    def cluster_size(self) -> int:
        """Number of candidates in this cluster."""
        return len(self.candidates_in_cluster)

    @property
    def selection_count(self) -> int:
        """Number of candidates selected from this cluster."""
        return len(self.selected_from_cluster)

    @property
    def selection_ratio(self) -> float:
        """Ratio of selected to total candidates in cluster."""
        if self.cluster_size == 0:
            return 0.0
        return self.selection_count / self.cluster_size


@dataclass(frozen=True)
class DiversitySelectionResult:
    """Results from diversity-based selection algorithms."""

    method_used: str  # "pose_clustering", "semantic_clustering", "hybrid"
    cluster_selections: list[ClusterSelection]
    total_clusters: int
    clusters_with_selections: int
    diversity_optimization_score: float  # 0.0-1.0
    quality_preservation_score: float  # 0.0-1.0

    # Diversity metrics
    average_intra_cluster_distance: float | None = None
    average_inter_cluster_distance: float | None = None
    diversity_entropy: float | None = None  # Shannon entropy of cluster distribution

    @property
    def cluster_utilization_ratio(self) -> float:
        """Ratio of clusters that contributed to selection."""
        if self.total_clusters == 0:
            return 0.0
        return self.clusters_with_selections / self.total_clusters

    @property
    def total_selected(self) -> int:
        """Total number of candidates selected across all clusters."""
        return sum(cluster.selection_count for cluster in self.cluster_selections)

    def get_cluster_selection(self, cluster_id: int) -> ClusterSelection | None:
        """Get selection results for a specific cluster."""
        for cluster_selection in self.cluster_selections:
            if cluster_selection.cluster_id == cluster_id:
                return cluster_selection
        return None
