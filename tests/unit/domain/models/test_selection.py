"""Tests for selection domain models."""

from pathlib import Path

import pytest

from culora.domain.models.quality import QualityScore
from culora.domain.models.selection import (
    ClusterSelection,
    DistributionAnalysis,
    DiversitySelectionResult,
    SelectionCandidate,
    SelectionCriteria,
    SelectionResult,
    SelectionStageResult,
    SelectionSummary,
)


class TestSelectionCandidate:
    """Test cases for SelectionCandidate."""

    @pytest.fixture
    def quality_score(self) -> QualityScore:
        """Create a sample quality assessment."""
        return QualityScore(
            technical_score=0.76,
            overall_score=0.81,
            passes_threshold=True,
            sharpness_contribution=0.16,
            brightness_contribution=0.14,
            contrast_contribution=0.18,
            color_contribution=0.12,
            noise_contribution=0.16,
            perceptual_score=0.75,
            brisque_contribution=0.0,
            face_quality_bonus=0.1,
            reference_match_bonus=0.05,
        )

    @pytest.fixture
    def basic_candidate(self, quality_score: QualityScore) -> SelectionCandidate:
        """Create a basic selection candidate."""
        return SelectionCandidate(
            path=Path("test_image.jpg"),
            file_size=1024000,
            quality_assessment=quality_score,
            composite_quality_score=0.81,
        )

    def test_candidate_creation(self, basic_candidate: SelectionCandidate) -> None:
        """Test basic candidate creation."""
        assert basic_candidate.path == Path("test_image.jpg")
        assert basic_candidate.file_size == 1024000
        assert basic_candidate.composite_quality_score == 0.81

    def test_has_quality_analysis(self, basic_candidate: SelectionCandidate) -> None:
        """Test quality analysis detection."""
        assert basic_candidate.has_quality_analysis

        candidate_no_quality = SelectionCandidate(
            path=Path("test.jpg"),
            file_size=1000,
        )
        assert not candidate_no_quality.has_quality_analysis

    def test_effective_quality_score(
        self, basic_candidate: SelectionCandidate, quality_score: QualityScore
    ) -> None:
        """Test effective quality score calculation."""
        # With composite quality score
        assert basic_candidate.effective_quality_score == 0.81

        # Without composite quality score but with assessment
        candidate_no_composite = SelectionCandidate(
            path=Path("test.jpg"),
            file_size=1000,
            quality_assessment=quality_score,
        )
        assert (
            candidate_no_composite.effective_quality_score
            == quality_score.overall_score
        )

        # Without any quality data
        candidate_no_quality = SelectionCandidate(
            path=Path("test.jpg"),
            file_size=1000,
        )
        assert candidate_no_quality.effective_quality_score == 0.0

    def test_is_duplicate(self, basic_candidate: SelectionCandidate) -> None:
        """Test duplicate detection."""
        assert not basic_candidate.is_duplicate

        duplicate_candidate = SelectionCandidate(
            path=Path("test.jpg"),
            file_size=1000,
            duplicate_group_id=1,
        )
        assert duplicate_candidate.is_duplicate

    def test_has_analysis_properties(self) -> None:
        """Test various analysis detection properties."""
        candidate = SelectionCandidate(
            path=Path("test.jpg"),
            file_size=1000,
        )

        assert not candidate.has_face_analysis
        assert not candidate.has_composition_analysis
        assert not candidate.has_pose_analysis
        assert not candidate.has_semantic_analysis
        assert not candidate.has_reference_match


class TestSelectionStageResult:
    """Test cases for SelectionStageResult."""

    @pytest.fixture
    def stage_result(self) -> SelectionStageResult:
        """Create a sample stage result."""
        return SelectionStageResult(
            stage_name="quality_filtering",
            input_count=100,
            output_count=75,
            filtered_count=25,
            candidates=[],
            duration=1.5,
            success=True,
        )

    def test_stage_result_creation(self, stage_result: SelectionStageResult) -> None:
        """Test stage result creation."""
        assert stage_result.stage_name == "quality_filtering"
        assert stage_result.input_count == 100
        assert stage_result.output_count == 75
        assert stage_result.filtered_count == 25
        assert stage_result.success

    def test_reduction_ratio(self, stage_result: SelectionStageResult) -> None:
        """Test reduction ratio calculation."""
        assert stage_result.reduction_ratio == 0.25  # 25/100

    def test_retention_ratio(self, stage_result: SelectionStageResult) -> None:
        """Test retention ratio calculation."""
        assert stage_result.retention_ratio == 0.75  # 75/100

    def test_zero_input_ratios(self) -> None:
        """Test ratio calculations with zero input."""
        stage_result = SelectionStageResult(
            stage_name="test_stage",
            input_count=0,
            output_count=0,
            filtered_count=0,
            candidates=[],
            duration=0.0,
            success=True,
        )

        assert stage_result.reduction_ratio == 0.0
        assert stage_result.retention_ratio == 0.0


class TestDistributionAnalysis:
    """Test cases for DistributionAnalysis."""

    @pytest.fixture
    def distribution_analysis(self) -> DistributionAnalysis:
        """Create a sample distribution analysis."""
        return DistributionAnalysis(
            target_counts={"closeup": 5, "portrait": 3, "full_body": 2},
            actual_counts={"closeup": 4, "portrait": 3, "full_body": 3},
            target_ratios={"closeup": 0.5, "portrait": 0.3, "full_body": 0.2},
            actual_ratios={"closeup": 0.4, "portrait": 0.3, "full_body": 0.3},
            fulfillment_ratios={"closeup": 0.8, "portrait": 1.0, "full_body": 1.5},
            overall_distribution_score=0.75,
            missing_categories=[],
            over_represented_categories=["full_body"],
        )

    def test_distribution_analysis_creation(
        self, distribution_analysis: DistributionAnalysis
    ) -> None:
        """Test distribution analysis creation."""
        assert distribution_analysis.overall_distribution_score == 0.75
        assert "full_body" in distribution_analysis.over_represented_categories

    def test_is_well_distributed(
        self, distribution_analysis: DistributionAnalysis
    ) -> None:
        """Test well-distributed check."""
        assert distribution_analysis.is_well_distributed  # Score 0.75 >= 0.7

        poor_distribution = DistributionAnalysis(
            target_counts={"closeup": 5},
            actual_counts={"closeup": 1},
            target_ratios={"closeup": 1.0},
            actual_ratios={"closeup": 1.0},
            fulfillment_ratios={"closeup": 0.2},
            overall_distribution_score=0.5,
            missing_categories=[],
            over_represented_categories=[],
        )
        assert not poor_distribution.is_well_distributed

    def test_total_counts(self, distribution_analysis: DistributionAnalysis) -> None:
        """Test total count calculations."""
        assert distribution_analysis.total_target_count == 10  # 5+3+2
        assert distribution_analysis.total_actual_count == 10  # 4+3+3


class TestSelectionCriteria:
    """Test cases for SelectionCriteria."""

    def test_criteria_creation(self) -> None:
        """Test criteria creation and validation."""
        criteria = SelectionCriteria(
            target_count=50,
            min_quality_threshold=0.5,
            enable_duplicate_removal=True,
            enable_diversity_optimization=True,
            enable_distribution_balancing=True,
            quality_weight=0.5,
            diversity_weight=0.3,
            distribution_weight=0.1,
            reference_match_weight=0.1,
        )

        assert criteria.target_count == 50
        assert criteria.quality_weight == 0.5

    def test_weight_validation(self) -> None:
        """Test weight sum validation."""
        with pytest.raises(ValueError, match="Selection weights must sum to 1.0"):
            SelectionCriteria(
                target_count=50,
                min_quality_threshold=0.5,
                enable_duplicate_removal=True,
                enable_diversity_optimization=True,
                enable_distribution_balancing=True,
                quality_weight=0.5,
                diversity_weight=0.3,
                distribution_weight=0.1,
                reference_match_weight=0.2,  # Total = 1.1
            )


class TestSelectionResult:
    """Test cases for SelectionResult."""

    @pytest.fixture
    def sample_criteria(self) -> SelectionCriteria:
        """Create sample selection criteria."""
        return SelectionCriteria(
            target_count=10,
            min_quality_threshold=0.5,
            enable_duplicate_removal=True,
            enable_diversity_optimization=True,
            enable_distribution_balancing=True,
            quality_weight=0.6,
            diversity_weight=0.2,
            distribution_weight=0.1,
            reference_match_weight=0.1,
        )

    @pytest.fixture
    def sample_candidates(self) -> list[SelectionCandidate]:
        """Create sample candidates."""
        candidates = []
        for i in range(5):
            candidate = SelectionCandidate(
                path=Path(f"image_{i}.jpg"),
                file_size=1000 * (i + 1),
                composite_quality_score=0.5 + (i * 0.1),
            )
            candidates.append(candidate)
        return candidates

    @pytest.fixture
    def selection_result(
        self,
        sample_criteria: SelectionCriteria,
        sample_candidates: list[SelectionCandidate],
    ) -> SelectionResult:
        """Create a sample selection result."""
        selected = sample_candidates[:3]
        rejected = sample_candidates[3:]

        return SelectionResult(
            selected_candidates=selected,
            rejected_candidates=rejected,
            total_processed=5,
            criteria=sample_criteria,
            constraints_applied={},
            stage_results=[],
            quality_distribution={},
            total_duration=2.5,
            selection_efficiency=2.0,
            success=True,
            target_fulfillment_ratio=0.3,
            quality_improvement_ratio=1.2,
        )

    def test_selection_result_creation(self, selection_result: SelectionResult) -> None:
        """Test selection result creation."""
        assert selection_result.success
        assert selection_result.total_processed == 5
        assert selection_result.target_fulfillment_ratio == 0.3

    def test_selection_count(self, selection_result: SelectionResult) -> None:
        """Test selection count property."""
        assert selection_result.selection_count == 3

    def test_rejection_count(self, selection_result: SelectionResult) -> None:
        """Test rejection count property."""
        assert selection_result.rejection_count == 2

    def test_selection_ratio(self, selection_result: SelectionResult) -> None:
        """Test selection ratio calculation."""
        assert selection_result.selection_ratio == 0.6  # 3/5

    def test_average_selected_quality(self, selection_result: SelectionResult) -> None:
        """Test average selected quality calculation."""
        # Expected: (0.5 + 0.6 + 0.7) / 3 = 0.6
        assert abs(selection_result.average_selected_quality - 0.6) < 0.001

    def test_quality_range(self, selection_result: SelectionResult) -> None:
        """Test quality range calculation."""
        min_quality, max_quality = selection_result.quality_range
        assert min_quality == 0.5
        assert max_quality == 0.7

    def test_get_stage_result(self, selection_result: SelectionResult) -> None:
        """Test getting stage results by name."""
        # Add a stage result
        stage_result = SelectionStageResult(
            stage_name="test_stage",
            input_count=5,
            output_count=3,
            filtered_count=2,
            candidates=[],
            duration=1.0,
            success=True,
        )
        selection_result.stage_results.append(stage_result)

        found_stage = selection_result.get_stage_result("test_stage")
        assert found_stage is not None
        assert found_stage.stage_name == "test_stage"

        not_found = selection_result.get_stage_result("nonexistent_stage")
        assert not_found is None

    def test_empty_candidates_quality_calculation(self) -> None:
        """Test quality calculations with empty candidate lists."""
        empty_result = SelectionResult(
            selected_candidates=[],
            rejected_candidates=[],
            total_processed=0,
            criteria=SelectionCriteria(
                target_count=0,
                min_quality_threshold=0.5,
                enable_duplicate_removal=False,
                enable_diversity_optimization=False,
                enable_distribution_balancing=False,
                quality_weight=1.0,
                diversity_weight=0.0,
                distribution_weight=0.0,
                reference_match_weight=0.0,
            ),
            constraints_applied={},
            stage_results=[],
            quality_distribution={},
            total_duration=0.0,
            selection_efficiency=0.0,
            success=False,
            target_fulfillment_ratio=0.0,
            quality_improvement_ratio=0.0,
        )

        assert empty_result.selection_count == 0
        assert empty_result.average_selected_quality == 0.0
        assert empty_result.quality_range == (0.0, 0.0)
        assert empty_result.selection_ratio == 0.0


class TestSelectionSummary:
    """Test cases for SelectionSummary."""

    @pytest.fixture
    def selection_summary(self) -> SelectionSummary:
        """Create a sample selection summary."""
        return SelectionSummary(
            input_count=100,
            selected_count=45,
            target_count=50,
            fulfillment_percentage=90.0,
            avg_quality_improvement=1.3,
            quality_percentile_distribution={"90-100": 5, "80-90": 15, "70-80": 25},
            shot_type_distribution={"closeup": 20, "portrait": 15, "full_body": 10},
            scene_type_distribution={"indoor": 25, "outdoor": 20},
            distribution_balance_score=0.75,
            diversity_score=0.8,
            total_processing_time=120.0,
            processing_rate=0.83,
            selection_success=True,
            major_issues=[],
            recommendations=["Consider adding more outdoor scenes"],
        )

    def test_summary_creation(self, selection_summary: SelectionSummary) -> None:
        """Test summary creation."""
        assert selection_summary.input_count == 100
        assert selection_summary.selected_count == 45
        assert selection_summary.fulfillment_percentage == 90.0

    def test_is_successful_selection(self, selection_summary: SelectionSummary) -> None:
        """Test successful selection detection."""
        assert selection_summary.is_successful_selection

        # Create unsuccessful summary
        unsuccessful_summary = SelectionSummary(
            input_count=100,
            selected_count=30,  # Low fulfillment
            target_count=50,
            fulfillment_percentage=60.0,  # < 80%
            avg_quality_improvement=1.1,
            quality_percentile_distribution={},
            shot_type_distribution={},
            scene_type_distribution={},
            distribution_balance_score=0.4,  # < 0.6
            total_processing_time=60.0,
            processing_rate=1.67,
            selection_success=True,
            major_issues=["Insufficient diversity"],
            recommendations=[],
        )
        assert not unsuccessful_summary.is_successful_selection

    def test_needs_attention(self, selection_summary: SelectionSummary) -> None:
        """Test attention needed detection."""
        assert not selection_summary.needs_attention  # Good summary

        # Create summary that needs attention
        attention_needed = SelectionSummary(
            input_count=100,
            selected_count=25,  # Very low fulfillment
            target_count=50,
            fulfillment_percentage=50.0,  # < 70%
            avg_quality_improvement=1.0,
            quality_percentile_distribution={},
            shot_type_distribution={},
            scene_type_distribution={},
            distribution_balance_score=0.3,  # < 0.5
            total_processing_time=60.0,
            processing_rate=1.67,
            selection_success=True,
            major_issues=[],
            recommendations=[],
        )
        assert attention_needed.needs_attention


class TestClusterSelection:
    """Test cases for ClusterSelection."""

    @pytest.fixture
    def cluster_selection(self) -> ClusterSelection:
        """Create a sample cluster selection."""
        candidates = [
            SelectionCandidate(path=Path(f"img_{i}.jpg"), file_size=1000)
            for i in range(5)
        ]
        selected = candidates[:2]

        return ClusterSelection(
            cluster_id=1,
            cluster_center=[0.1, 0.2, 0.3],
            candidates_in_cluster=candidates,
            selected_from_cluster=selected,
            cluster_quality_score=0.75,
            cluster_diversity_score=0.8,
            selection_rationale="Top quality candidates from cluster",
        )

    def test_cluster_selection_creation(
        self, cluster_selection: ClusterSelection
    ) -> None:
        """Test cluster selection creation."""
        assert cluster_selection.cluster_id == 1
        assert cluster_selection.cluster_quality_score == 0.75

    def test_cluster_size(self, cluster_selection: ClusterSelection) -> None:
        """Test cluster size property."""
        assert cluster_selection.cluster_size == 5

    def test_selection_count(self, cluster_selection: ClusterSelection) -> None:
        """Test selection count property."""
        assert cluster_selection.selection_count == 2

    def test_selection_ratio(self, cluster_selection: ClusterSelection) -> None:
        """Test selection ratio calculation."""
        assert cluster_selection.selection_ratio == 0.4  # 2/5

    def test_empty_cluster_selection_ratio(self) -> None:
        """Test selection ratio with empty cluster."""
        empty_cluster = ClusterSelection(
            cluster_id=1,
            cluster_center=None,
            candidates_in_cluster=[],
            selected_from_cluster=[],
            cluster_quality_score=0.0,
            cluster_diversity_score=0.0,
            selection_rationale="Empty cluster",
        )

        assert empty_cluster.selection_ratio == 0.0


class TestDiversitySelectionResult:
    """Test cases for DiversitySelectionResult."""

    @pytest.fixture
    def diversity_result(self) -> DiversitySelectionResult:
        """Create a sample diversity selection result."""
        cluster_selections = [
            ClusterSelection(
                cluster_id=i,
                cluster_center=None,
                candidates_in_cluster=[],
                selected_from_cluster=[],
                cluster_quality_score=0.7,
                cluster_diversity_score=0.8,
                selection_rationale="Test cluster",
            )
            for i in range(3)
        ]

        return DiversitySelectionResult(
            method_used="pose_clustering",
            cluster_selections=cluster_selections,
            total_clusters=5,
            clusters_with_selections=3,
            diversity_optimization_score=0.8,
            quality_preservation_score=0.75,
            average_intra_cluster_distance=0.2,
            average_inter_cluster_distance=0.8,
            diversity_entropy=1.5,
        )

    def test_diversity_result_creation(
        self, diversity_result: DiversitySelectionResult
    ) -> None:
        """Test diversity result creation."""
        assert diversity_result.method_used == "pose_clustering"
        assert diversity_result.total_clusters == 5
        assert diversity_result.clusters_with_selections == 3

    def test_cluster_utilization_ratio(
        self, diversity_result: DiversitySelectionResult
    ) -> None:
        """Test cluster utilization ratio calculation."""
        assert diversity_result.cluster_utilization_ratio == 0.6  # 3/5

    def test_total_selected(self, diversity_result: DiversitySelectionResult) -> None:
        """Test total selected count."""
        assert (
            diversity_result.total_selected == 0
        )  # All clusters have 0 selections in fixture

    def test_get_cluster_selection(
        self, diversity_result: DiversitySelectionResult
    ) -> None:
        """Test getting cluster selection by ID."""
        cluster = diversity_result.get_cluster_selection(1)
        assert cluster is not None
        assert cluster.cluster_id == 1

        non_existent = diversity_result.get_cluster_selection(99)
        assert non_existent is None

    def test_zero_clusters_utilization_ratio(self) -> None:
        """Test utilization ratio with zero clusters."""
        zero_cluster_result = DiversitySelectionResult(
            method_used="test",
            cluster_selections=[],
            total_clusters=0,
            clusters_with_selections=0,
            diversity_optimization_score=0.0,
            quality_preservation_score=0.0,
        )

        assert zero_cluster_result.cluster_utilization_ratio == 0.0
