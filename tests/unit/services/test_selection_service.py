"""Tests for the SelectionService."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from culora.core.exceptions import (
    SelectionInsufficientDataError,
)
from culora.domain.models.clip import SemanticEmbedding
from culora.domain.models.composition import (
    BackgroundComplexity,
    CameraAngle,
    CompositionAnalysis,
    FacialExpression,
    LightingQuality,
    SceneType,
    ShotType,
)
from culora.domain.models.config.selection import SelectionConfig, SelectionConstraints
from culora.domain.models.pose import (
    ArmPosition,
    LegPosition,
    PoseAnalysis,
    PoseCategory,
    PoseClassification,
    PoseLandmark,
    PoseOrientation,
    PoseSymmetry,
    PoseVector,
)
from culora.domain.models.quality import QualityScore
from culora.domain.models.selection import SelectionCandidate
from culora.services.selection_service import SelectionService


class TestSelectionService:
    """Test cases for SelectionService."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self) -> Generator[None, None, None]:
        """Set up mocks for all dependencies."""
        with (
            patch(
                "culora.services.selection_service.get_quality_service"
            ) as self.mock_quality_service,
            patch(
                "culora.services.selection_service.get_duplicate_service"
            ) as self.mock_duplicate_service,
            patch(
                "culora.services.selection_service.get_pose_service"
            ) as self.mock_pose_service,
            patch(
                "culora.services.selection_service.get_clip_service"
            ) as self.mock_clip_service,
        ):
            yield

    @pytest.fixture
    def selection_service(self) -> SelectionService:
        """Create a SelectionService instance for testing."""
        return SelectionService()

    @pytest.fixture
    def sample_config(self) -> SelectionConfig:
        """Create a sample selection configuration."""
        return SelectionConfig(
            target_count=10,
            selection_strategy="multi_stage",
        )

    @pytest.fixture
    def quality_score(self) -> QualityScore:
        """Create a sample quality score."""
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
    def composition_analysis(self) -> CompositionAnalysis:
        """Create a sample composition analysis."""
        return CompositionAnalysis(
            shot_type=ShotType.CLOSEUP,
            scene_type=SceneType.INDOOR,
            lighting_quality=LightingQuality.GOOD,
            background_complexity=BackgroundComplexity.SIMPLE,
            facial_expression=FacialExpression.NEUTRAL,
            camera_angle=CameraAngle.EYE_LEVEL,
            confidence_score=0.85,
        )

    @pytest.fixture
    def pose_analysis(self) -> PoseAnalysis:
        """Create a sample pose analysis."""
        from culora.domain.models.pose import (
            PoseDynamism,
        )

        # Create mock landmarks
        landmarks = [
            {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9, "presence": 0.9}
            for _ in range(33)
        ]

        return PoseAnalysis(
            path=Path("/test/image.jpg"),
            landmarks=[PoseLandmark(**landmark) for landmark in landmarks],
            pose_vector=PoseVector(
                vector=list(np.random.rand(66)), vector_dimension=66, confidence=0.9
            ),
            classification=PoseClassification(
                category=PoseCategory.STANDING,
                orientation=PoseOrientation.FRONTAL,
                arm_position=ArmPosition.AT_SIDES,
                leg_position=LegPosition.STRAIGHT,
                symmetry=PoseSymmetry.SYMMETRIC,
                dynamism=PoseDynamism.STATIC,
                confidence=0.85,
            ),
            bbox=(0.1, 0.1, 0.8, 0.8),
            pose_score=0.9,
            analysis_duration=0.1,
        )

    @pytest.fixture
    def semantic_embedding(self) -> SemanticEmbedding:
        """Create a sample semantic embedding."""
        return SemanticEmbedding(
            path=Path("/test/image.jpg"),
            embedding=list(np.random.rand(512)),
            model_name="clip-vit-base-patch32",
            embedding_dimension=512,
            extraction_time=0.1,
        )

    @pytest.fixture
    def sample_candidates(
        self,
        quality_score: QualityScore,
        composition_analysis: CompositionAnalysis,
        pose_analysis: PoseAnalysis,
        semantic_embedding: SemanticEmbedding,
    ) -> list[SelectionCandidate]:
        """Create sample selection candidates."""
        candidates = []

        for i in range(20):
            # Vary quality scores for diversity
            varied_quality = QualityScore(
                technical_score=0.5 + (i * 0.025),
                overall_score=0.5 + (i * 0.025),
                passes_threshold=True,
                sharpness_contribution=0.1 + (i * 0.005),
                brightness_contribution=0.1 + (i * 0.004),
                contrast_contribution=0.1 + (i * 0.003),
                color_contribution=0.1 + (i * 0.006),
                noise_contribution=0.1 + (i * 0.002),
                perceptual_score=0.5 + (i * 0.02),
                brisque_contribution=0.0,
                face_quality_bonus=0.0,
                reference_match_bonus=0.0,
            )

            # Vary composition types
            shot_types = list(ShotType)
            scene_types = list(SceneType)

            varied_composition = CompositionAnalysis(
                shot_type=shot_types[i % len(shot_types)],
                scene_type=scene_types[i % len(scene_types)],
                lighting_quality=LightingQuality.GOOD,
                background_complexity=BackgroundComplexity.SIMPLE,
                facial_expression=FacialExpression.NEUTRAL,
                camera_angle=CameraAngle.EYE_LEVEL,
                confidence_score=0.8 + (i * 0.01),
            )

            # Vary pose vectors for diversity
            from culora.domain.models.pose import (
                PoseDynamism,
            )

            pose_categories = [
                PoseCategory.STANDING,
                PoseCategory.SITTING,
                PoseCategory.LYING,
            ]
            orientations = [
                PoseOrientation.FRONTAL,
                PoseOrientation.PROFILE,
                PoseOrientation.BACK,
            ]

            varied_pose = PoseAnalysis(
                path=Path(f"test_image_{i:03d}.jpg"),
                landmarks=[
                    PoseLandmark(x=0.5, y=0.3, z=0.0, visibility=0.9, presence=0.9)
                    for _ in range(33)
                ],
                pose_vector=PoseVector(
                    vector=list(np.random.rand(66)),
                    vector_dimension=66,
                    confidence=0.8 + (i * 0.01),
                ),
                classification=PoseClassification(
                    category=pose_categories[i % len(pose_categories)],
                    orientation=orientations[i % len(orientations)],
                    arm_position=ArmPosition.AT_SIDES,
                    leg_position=LegPosition.STRAIGHT,
                    symmetry=PoseSymmetry.SYMMETRIC,
                    dynamism=PoseDynamism.STATIC,
                    confidence=0.85,
                ),
                bbox=(0.1, 0.1, 0.8, 0.8),
                pose_score=0.8 + (i * 0.01),
                analysis_duration=0.1,
            )

            # Vary semantic embeddings
            varied_embedding = SemanticEmbedding(
                path=Path(f"test_image_{i:03d}.jpg"),
                embedding=list(np.random.rand(512)),
                model_name="clip-vit-base-patch32",
                embedding_dimension=512,
                extraction_time=0.1,
            )

            candidate = SelectionCandidate(
                path=Path(f"test_image_{i:03d}.jpg"),
                file_size=1024 * (100 + i * 10),
                quality_assessment=varied_quality,
                composite_quality_score=varied_quality.overall_score,
                composition_analysis=varied_composition,
                pose_analysis=varied_pose,
                semantic_embedding=varied_embedding,
                duplicate_group_id=None,
                is_duplicate_representative=False,
            )

            candidates.append(candidate)

        return candidates

    def test_select_images_basic(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test basic image selection functionality."""
        result = selection_service.select_images(sample_candidates, sample_config)

        assert result.success
        assert len(result.selected_candidates) <= sample_config.target_count
        assert len(result.selected_candidates) > 0
        assert result.total_processed == len(sample_candidates)
        assert len(result.stage_results) > 0
        assert result.total_duration > 0

    def test_select_images_quality_first_strategy(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test quality-first selection strategy."""
        config = SelectionConfig(
            target_count=5,
            selection_strategy="quality_first",
        )

        result = selection_service.select_images(sample_candidates, config)

        assert result.success
        assert len(result.selected_candidates) == 5

        # Verify quality-based ordering (highest quality first)
        selected_qualities = [
            c.effective_quality_score for c in result.selected_candidates
        ]
        assert selected_qualities == sorted(selected_qualities, reverse=True)

    def test_select_images_diversity_first_strategy(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test diversity-first selection strategy."""
        config = SelectionConfig(
            target_count=8,
            selection_strategy="diversity_first",
        )
        config.diversity_settings.enable_pose_diversity = True

        result = selection_service.select_images(sample_candidates, config)

        assert result.success
        assert len(result.selected_candidates) <= 8
        # Diversity selection may select fewer if clustering results in fewer groups

    def test_select_images_balanced_strategy(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test balanced selection strategy."""
        config = SelectionConfig(
            target_count=7,
            selection_strategy="balanced",
        )

        result = selection_service.select_images(sample_candidates, config)

        assert result.success
        assert len(result.selected_candidates) == 7

        # Check that selection scores were calculated
        for candidate in result.selected_candidates:
            assert candidate.selection_score is not None

    def test_select_images_multi_stage_strategy(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test multi-stage selection strategy."""
        config = SelectionConfig(
            target_count=6,
            selection_strategy="multi_stage",
        )

        result = selection_service.select_images(sample_candidates, config)

        assert result.success
        assert len(result.selected_candidates) <= 6
        assert (
            len(result.stage_results) >= 1
        )  # At least the multi-criteria selection stage

    def test_quality_filtering_stage(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test quality filtering stage."""
        config = SelectionConfig(
            target_count=10,
            enable_early_filtering=True,
        )
        config.quality_thresholds.min_composite_quality = 0.7  # High threshold

        result = selection_service.select_images(sample_candidates, config)

        # Find quality filtering stage
        quality_stage = None
        for stage in result.stage_results:
            if stage.stage_name == "quality_filtering":
                quality_stage = stage
                break

        assert quality_stage is not None
        assert quality_stage.success
        assert (
            quality_stage.output_count < quality_stage.input_count
        )  # Some should be filtered

    def test_duplicate_removal_stage(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test duplicate removal stage."""
        # Create candidates with duplicate information
        from dataclasses import replace

        sample_candidates[0] = replace(sample_candidates[0], duplicate_group_id=1)
        sample_candidates[1] = replace(
            sample_candidates[1], duplicate_group_id=1, is_duplicate_representative=True
        )

        config = SelectionConfig(
            target_count=10,
            enable_duplicate_removal=True,
        )

        result = selection_service.select_images(sample_candidates, config)

        # Find duplicate removal stage
        duplicate_stage = None
        for stage in result.stage_results:
            if stage.stage_name == "duplicate_removal":
                duplicate_stage = stage
                break

        assert duplicate_stage is not None
        assert duplicate_stage.success
        assert (
            duplicate_stage.output_count == duplicate_stage.input_count - 1
        )  # One duplicate removed

    def test_insufficient_candidates_error(
        self, selection_service: SelectionService
    ) -> None:
        """Test error handling when no candidates are available."""
        with pytest.raises(SelectionInsufficientDataError):
            selection_service.select_images([], SelectionConfig(target_count=10))

    def test_configuration_adaptation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test configuration adaptation for large target counts."""
        config = SelectionConfig(target_count=1000)  # Target exceeds available images

        result = selection_service.select_images(sample_candidates, config)

        # Service should adapt and select all available candidates
        assert result.success
        assert len(result.selected_candidates) <= len(sample_candidates)

    def test_pose_diversity_selection(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test pose-based diversity selection."""
        selected = selection_service._select_by_pose_diversity(sample_candidates, 5)

        assert len(selected) <= 5
        assert all(candidate.has_pose_analysis for candidate in selected)

    def test_semantic_diversity_selection(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test semantic-based diversity selection."""
        selected = selection_service._select_by_semantic_diversity(sample_candidates, 5)

        assert len(selected) <= 5
        assert all(candidate.has_semantic_analysis for candidate in selected)

    def test_quality_distribution_analysis(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test quality distribution analysis."""
        distribution = selection_service._analyze_quality_distribution(
            sample_candidates[:10]
        )

        assert "min_quality" in distribution
        assert "max_quality" in distribution
        assert "mean_quality" in distribution
        assert "median_quality" in distribution
        assert distribution["min_quality"] <= distribution["max_quality"]

    def test_composition_distribution_analysis(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test composition distribution analysis."""
        distribution = selection_service._analyze_composition_distribution(
            sample_candidates[:10], sample_config
        )

        assert distribution is not None
        assert distribution.total_actual_count > 0
        assert distribution.overall_distribution_score >= 0.0
        assert distribution.overall_distribution_score <= 1.0

    def test_diversity_analysis(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test diversity analysis."""
        diversity = selection_service._analyze_diversity(sample_candidates[:10])

        assert diversity is not None
        assert "pose_diversity" in diversity or "semantic_diversity" in diversity

    def test_constraints_generation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test constraint generation from candidates."""
        constraints = selection_service._generate_constraints(sample_candidates)

        assert constraints.available_images == len(sample_candidates)
        assert constraints.quality_filtered_count > 0
        assert constraints.composition_analyzed_count > 0
        assert constraints.pose_analyzed_count > 0
        assert constraints.semantic_analyzed_count > 0

    def test_selection_criteria_creation(
        self, selection_service: SelectionService, sample_config: SelectionConfig
    ) -> None:
        """Test selection criteria creation."""
        criteria = selection_service._create_selection_criteria(sample_config, 10)

        assert criteria.target_count == 10
        assert criteria.quality_weight >= 0.0
        assert criteria.diversity_weight >= 0.0
        assert criteria.distribution_weight >= 0.0
        assert criteria.reference_match_weight >= 0.0

    def test_candidate_diversity_score_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test diversity score calculation for individual candidates."""
        candidate = sample_candidates[0]
        diversity_score = selection_service._calculate_candidate_diversity_score(
            candidate, sample_candidates
        )

        assert 0.0 <= diversity_score <= 1.0

    def test_pose_diversity_score_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test pose diversity score calculation."""
        candidate = sample_candidates[0]
        score = selection_service._calculate_pose_diversity_score(
            candidate, sample_candidates
        )

        assert 0.0 <= score <= 1.0

    def test_semantic_diversity_score_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test semantic diversity score calculation."""
        candidate = sample_candidates[0]
        score = selection_service._calculate_semantic_diversity_score(
            candidate, sample_candidates
        )

        assert 0.0 <= score <= 1.0

    def test_quality_improvement_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test quality improvement ratio calculation."""
        # Select top 5 candidates by quality
        selected = sorted(
            sample_candidates, key=lambda c: c.effective_quality_score, reverse=True
        )[:5]

        improvement = selection_service._calculate_quality_improvement(
            sample_candidates, selected
        )

        assert improvement >= 1.0  # Selected should have higher average quality

    def test_distribution_balancing(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test distribution balancing functionality."""
        # Start with a small selection
        current_selection = sample_candidates[:3]

        balanced = selection_service._balance_distribution(
            current_selection, sample_candidates, sample_config, 8
        )

        assert len(balanced) >= len(current_selection)
        assert len(balanced) <= 8

    def test_quality_stats_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test quality statistics calculation."""
        stats = selection_service._calculate_quality_stats(sample_candidates[:5])

        assert "count" in stats
        assert "mean_quality" in stats
        assert "min_quality" in stats
        assert "max_quality" in stats
        assert stats["count"] == 5

    def test_distribution_stats_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test distribution statistics calculation."""
        stats = selection_service._calculate_distribution_stats(sample_candidates[:5])

        assert len(stats) > 0
        # Should have shot_type and scene_type counts
        shot_type_keys = [k for k in stats if k.startswith("shot_type_")]
        scene_type_keys = [k for k in stats if k.startswith("scene_type_")]

        assert len(shot_type_keys) > 0
        assert len(scene_type_keys) > 0

    def test_diversity_stats_calculation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test diversity statistics calculation."""
        stats = selection_service._calculate_diversity_stats(sample_candidates[:5])

        assert "pose_data_ratio" in stats
        assert "semantic_data_ratio" in stats
        assert 0.0 <= stats["pose_data_ratio"] <= 1.0
        assert 0.0 <= stats["semantic_data_ratio"] <= 1.0

    def test_selection_summary_generation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test selection summary generation."""
        result = selection_service.select_images(sample_candidates, sample_config)
        summary = selection_service.generate_selection_summary(result)

        assert summary.input_count == len(sample_candidates)
        assert summary.selected_count == len(result.selected_candidates)
        assert summary.target_count == sample_config.target_count
        assert summary.fulfillment_percentage >= 0.0
        assert summary.total_processing_time > 0.0
        assert summary.processing_rate > 0.0

    def test_empty_candidate_handling(
        self, selection_service: SelectionService
    ) -> None:
        """Test handling of empty candidate lists."""
        candidates: list[SelectionCandidate] = []
        config = SelectionConfig(target_count=5)

        with pytest.raises(SelectionInsufficientDataError):
            selection_service.select_images(candidates, config)

    def test_single_candidate_selection(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test selection with only one candidate."""
        single_candidate = sample_candidates[:1]
        config = SelectionConfig(target_count=1, max_selection_ratio=1.0)

        result = selection_service.select_images(single_candidate, config)

        assert result.success
        assert len(result.selected_candidates) == 1
        assert result.target_fulfillment_ratio == 1.0  # 1/1

    def test_target_count_exceeds_candidates(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test selection when target count exceeds available candidates."""
        few_candidates = sample_candidates[:5]
        config = SelectionConfig(target_count=10)

        result = selection_service.select_images(few_candidates, config)

        assert result.success
        assert len(result.selected_candidates) <= 5  # Can't select more than available

    def test_configuration_validation(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test configuration validation."""
        constraints = selection_service._generate_constraints(sample_candidates)

        # Valid configuration should not raise
        valid_config = SelectionConfig(target_count=10)
        selection_service._validate_selection_config(valid_config, constraints)

        # Test empty candidates constraint validation
        empty_constraints = SelectionConstraints(
            available_images=0,
            quality_filtered_count=0,
            duplicate_filtered_count=0,
            composition_analyzed_count=0,
            pose_analyzed_count=0,
            semantic_analyzed_count=0,
            reference_matched_count=0,
        )
        with pytest.raises(SelectionInsufficientDataError):
            selection_service._validate_selection_config(
                valid_config, empty_constraints
            )

    def test_stage_result_properties(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test stage result properties and calculations."""
        result = selection_service.select_images(sample_candidates, sample_config)

        for stage in result.stage_results:
            assert stage.reduction_ratio >= 0.0
            assert stage.retention_ratio >= 0.0
            assert stage.retention_ratio <= 1.0
            assert stage.duration >= 0.0

    def test_selection_result_properties(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test selection result properties and calculations."""
        result = selection_service.select_images(sample_candidates, sample_config)

        assert result.selection_count == len(result.selected_candidates)
        assert result.rejection_count == len(result.rejected_candidates)
        assert result.selection_ratio >= 0.0
        assert result.selection_ratio <= 1.0
        assert result.average_selected_quality > 0.0

        min_quality, max_quality = result.quality_range
        assert min_quality <= max_quality

    def test_get_candidates_by_quality_percentile(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
        sample_config: SelectionConfig,
    ) -> None:
        """Test filtering candidates by quality percentile."""
        result = selection_service.select_images(sample_candidates, sample_config)

        # This would work if quality percentiles were calculated
        # For now, just test that the method exists and doesn't crash
        filtered = result.get_candidates_by_quality_percentile(0.5, 1.0)
        assert isinstance(filtered, list)

    def test_pose_diversity_clustering(
        self,
        selection_service: SelectionService,
        sample_candidates: list[SelectionCandidate],
    ) -> None:
        """Test pose diversity clustering with valid data."""
        # Should work with sufficient pose data (20 candidates with pose analysis)
        selected = selection_service._select_by_pose_diversity(sample_candidates, 5)

        assert len(selected) <= 5
        # All selected candidates should have pose analysis
        assert all(candidate.has_pose_analysis for candidate in selected)
