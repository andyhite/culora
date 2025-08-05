"""Tests for selection configuration models."""

import pytest

from culora.domain.enums.composition import SceneType, ShotType
from culora.domain.models.config.selection import (
    DistributionTarget,
    DiversitySettings,
    QualityThresholds,
    SceneTypeDistribution,
    SelectionConfig,
    SelectionConstraints,
    ShotTypeDistribution,
)


class TestDistributionTarget:
    """Test cases for DistributionTarget."""

    def test_target_creation_with_ratio(self) -> None:
        """Test creating target with ratio specification."""
        target = DistributionTarget(
            category_name="closeup",
            target_ratio=0.3,
            min_count=2,
        )

        assert target.category_name == "closeup"
        assert target.target_ratio == 0.3
        assert target.target_count is None
        assert target.min_count == 2

    def test_target_creation_with_count(self) -> None:
        """Test creating target with count specification."""
        target = DistributionTarget(
            category_name="portrait",
            target_count=10,
            max_count=15,
        )

        assert target.category_name == "portrait"
        assert target.target_count == 10
        assert target.target_ratio is None
        assert target.max_count == 15

    def test_target_creation_with_both(self) -> None:
        """Test creating target with both ratio and count (count should take precedence)."""
        target = DistributionTarget(
            category_name="full_body",
            target_ratio=0.2,
            target_count=5,
        )

        assert target.target_ratio == 0.2
        assert target.target_count == 5

    def test_target_creation_with_neither(self) -> None:
        """Test that creating target with neither ratio nor count raises error."""
        with pytest.raises(
            ValueError, match="Either target_ratio or target_count must be specified"
        ):
            DistributionTarget(category_name="test")

    def test_max_count_validation(self) -> None:
        """Test max_count validation against min_count."""
        with pytest.raises(
            ValueError, match="max_count must be greater than or equal to min_count"
        ):
            DistributionTarget(
                category_name="test",
                target_ratio=0.5,
                min_count=10,
                max_count=5,
            )

    def test_valid_max_count(self) -> None:
        """Test valid max_count configuration."""
        target = DistributionTarget(
            category_name="test",
            target_ratio=0.5,
            min_count=5,
            max_count=10,
        )

        assert target.min_count == 5
        assert target.max_count == 10


class TestQualityThresholds:
    """Test cases for QualityThresholds."""

    def test_default_thresholds(self) -> None:
        """Test default quality thresholds."""
        thresholds = QualityThresholds()

        assert thresholds.min_composite_quality == 0.4
        assert thresholds.min_technical_quality == 0.3
        assert thresholds.enable_quality_distribution is True
        assert len(thresholds.quality_distribution_percentiles) == 3

    def test_custom_thresholds(self) -> None:
        """Test custom quality thresholds."""
        thresholds = QualityThresholds(
            min_composite_quality=0.6,
            min_technical_quality=0.5,
            min_brisque_quality=0.7,
        )

        assert thresholds.min_composite_quality == 0.6
        assert thresholds.min_technical_quality == 0.5
        assert thresholds.min_brisque_quality == 0.7

    def test_percentiles_validation_empty(self) -> None:
        """Test validation of empty percentiles list."""
        with pytest.raises(
            ValueError, match="Quality distribution percentiles cannot be empty"
        ):
            QualityThresholds(quality_distribution_percentiles=[])

    def test_percentiles_validation_invalid_range(self) -> None:
        """Test validation of percentiles outside valid range."""
        with pytest.raises(
            ValueError, match="Percentile .* must be between 0.0 and 1.0"
        ):
            QualityThresholds(quality_distribution_percentiles=[0.5, 1.2])

    def test_percentiles_validation_not_ascending(self) -> None:
        """Test validation of percentiles not in ascending order."""
        with pytest.raises(
            ValueError,
            match="Quality distribution percentiles must be in ascending order",
        ):
            QualityThresholds(quality_distribution_percentiles=[0.9, 0.7, 0.8])

    def test_valid_percentiles(self) -> None:
        """Test valid percentiles configuration."""
        thresholds = QualityThresholds(
            quality_distribution_percentiles=[0.6, 0.8, 0.9, 0.95]
        )

        assert thresholds.quality_distribution_percentiles == [0.6, 0.8, 0.9, 0.95]


class TestDiversitySettings:
    """Test cases for DiversitySettings."""

    def test_default_settings(self) -> None:
        """Test default diversity settings."""
        settings = DiversitySettings()

        assert settings.enable_pose_diversity is True
        assert settings.enable_semantic_diversity is True
        assert settings.diversity_weight == 0.3
        assert settings.quality_vs_diversity_balance == 0.7

    def test_custom_settings(self) -> None:
        """Test custom diversity settings."""
        settings = DiversitySettings(
            enable_pose_diversity=False,
            diversity_weight=0.5,
            min_cluster_separation=0.3,
            max_selections_per_cluster=2,
        )

        assert settings.enable_pose_diversity is False
        assert settings.diversity_weight == 0.5
        assert settings.min_cluster_separation == 0.3
        assert settings.max_selections_per_cluster == 2


class TestShotTypeDistribution:
    """Test cases for ShotTypeDistribution."""

    def test_default_distribution(self) -> None:
        """Test default shot type distribution."""
        distribution = ShotTypeDistribution()

        assert distribution.enable_balancing is True
        assert len(distribution.targets) == 0
        assert ShotType.CLOSEUP in distribution.fallback_distribution
        assert distribution.fallback_distribution[ShotType.CLOSEUP] == 0.3

    def test_custom_targets(self) -> None:
        """Test custom shot type targets."""
        target = DistributionTarget(
            category_name="closeup",
            target_ratio=0.5,
        )

        distribution = ShotTypeDistribution(
            targets={ShotType.CLOSEUP: target},
            enable_balancing=False,
        )

        assert distribution.enable_balancing is False
        assert ShotType.CLOSEUP in distribution.targets
        assert distribution.targets[ShotType.CLOSEUP].target_ratio == 0.5


class TestSceneTypeDistribution:
    """Test cases for SceneTypeDistribution."""

    def test_default_distribution(self) -> None:
        """Test default scene type distribution."""
        distribution = SceneTypeDistribution()

        assert distribution.enable_balancing is True
        assert len(distribution.targets) == 0
        assert SceneType.INDOOR in distribution.fallback_distribution
        assert distribution.fallback_distribution[SceneType.INDOOR] == 0.4

    def test_custom_targets(self) -> None:
        """Test custom scene type targets."""
        target = DistributionTarget(
            category_name="outdoor",
            target_count=15,
        )

        distribution = SceneTypeDistribution(targets={SceneType.OUTDOOR: target})

        assert SceneType.OUTDOOR in distribution.targets
        assert distribution.targets[SceneType.OUTDOOR].target_count == 15


class TestSelectionConfig:
    """Test cases for SelectionConfig."""

    def test_default_config(self) -> None:
        """Test default selection configuration."""
        config = SelectionConfig()

        assert config.target_count == 50
        assert config.max_selection_ratio == 0.8
        assert config.selection_strategy == "multi_stage"
        assert config.enable_duplicate_removal is True
        assert config.enable_reference_matching is True

    def test_custom_config(self) -> None:
        """Test custom selection configuration."""
        config = SelectionConfig(
            target_count=100,
            selection_strategy="quality_first",
            enable_distribution_enforcement=False,
        )

        assert config.target_count == 100
        assert config.selection_strategy == "quality_first"
        assert config.enable_distribution_enforcement is False

    def test_target_count_validation(self) -> None:
        """Test target count validation."""
        with pytest.raises(ValueError, match="Target count too large"):
            SelectionConfig(target_count=15000)

    def test_calculate_max_selection(self) -> None:
        """Test max selection calculation."""
        config = SelectionConfig(target_count=50, max_selection_ratio=0.8)

        # With plenty of input images
        max_selection = config.calculate_max_selection(100)
        assert max_selection == 50  # Limited by target_count

        # With limited input images
        max_selection = config.calculate_max_selection(50)
        assert max_selection == 40  # Limited by ratio: 50 * 0.8 = 40

        # With very few input images
        max_selection = config.calculate_max_selection(10)
        assert max_selection == 8  # Limited by ratio: 10 * 0.8 = 8

    def test_validate_distribution_targets_shot_types(self) -> None:
        """Test validation of shot type distribution targets."""
        target1 = DistributionTarget(
            category_name="closeup", target_ratio=0.5, min_count=30
        )
        target2 = DistributionTarget(
            category_name="portrait", target_ratio=0.3, min_count=25
        )

        # Should raise error because min_counts (30 + 25 = 55) > target_count (50)
        with pytest.raises(
            ValueError, match="Shot type minimum counts .* exceed target count"
        ):
            SelectionConfig(
                target_count=50,
                shot_type_distribution=ShotTypeDistribution(
                    targets={
                        ShotType.CLOSEUP: target1,
                        ShotType.PORTRAIT: target2,
                    }
                ),
            )

    def test_validate_distribution_targets_scene_types(self) -> None:
        """Test validation of scene type distribution targets."""
        target1 = DistributionTarget(
            category_name="indoor", target_ratio=0.6, min_count=35
        )
        target2 = DistributionTarget(
            category_name="outdoor", target_ratio=0.4, min_count=20
        )

        # Should raise error because min_counts (35 + 20 = 55) > target_count (50)
        with pytest.raises(
            ValueError, match="Scene type minimum counts .* exceed target count"
        ):
            SelectionConfig(
                target_count=50,
                scene_type_distribution=SceneTypeDistribution(
                    targets={
                        SceneType.INDOOR: target1,
                        SceneType.OUTDOOR: target2,
                    }
                ),
            )

    def test_valid_distribution_targets(self) -> None:
        """Test valid distribution targets configuration."""
        target1 = DistributionTarget(
            category_name="closeup", target_ratio=0.5, min_count=20
        )
        target2 = DistributionTarget(
            category_name="portrait", target_ratio=0.3, min_count=15
        )

        config = SelectionConfig(
            target_count=50,
            shot_type_distribution=ShotTypeDistribution(
                targets={
                    ShotType.CLOSEUP: target1,
                    ShotType.PORTRAIT: target2,
                }
            ),
        )

        # Should not raise error because min_counts (20 + 15 = 35) <= target_count (50)
        config.validate_distribution_targets(50)

    def test_model_post_init_validation(self) -> None:
        """Test post-initialization validation."""
        # This should trigger validation during construction
        target = DistributionTarget(
            category_name="test", target_ratio=0.5, min_count=60
        )

        with pytest.raises(ValueError):
            SelectionConfig(
                target_count=50,
                shot_type_distribution=ShotTypeDistribution(
                    targets={ShotType.CLOSEUP: target}
                ),
            )


class TestSelectionConstraints:
    """Test cases for SelectionConstraints."""

    @pytest.fixture
    def sample_constraints(self) -> SelectionConstraints:
        """Create sample selection constraints."""
        return SelectionConstraints(
            available_images=100,
            quality_filtered_count=85,
            duplicate_filtered_count=80,
            composition_analyzed_count=75,
            pose_analyzed_count=60,
            semantic_analyzed_count=70,
            reference_matched_count=40,
        )

    def test_constraints_creation(
        self, sample_constraints: SelectionConstraints
    ) -> None:
        """Test constraints creation."""
        assert sample_constraints.available_images == 100
        assert sample_constraints.quality_filtered_count == 85
        assert sample_constraints.duplicate_filtered_count == 80

    def test_calculate_effective_target(
        self, sample_constraints: SelectionConstraints
    ) -> None:
        """Test effective target calculation."""
        config = SelectionConfig(target_count=50)

        effective_target = sample_constraints.calculate_effective_target(config)

        # Should be min of target_count (50), duplicate_filtered_count (80), quality_filtered_count (85)
        assert effective_target == 50

    def test_calculate_effective_target_limited_by_duplicates(self) -> None:
        """Test effective target limited by duplicate filtering."""
        constraints = SelectionConstraints(
            available_images=100,
            quality_filtered_count=85,
            duplicate_filtered_count=30,  # Lower limit
            composition_analyzed_count=75,
            pose_analyzed_count=60,
            semantic_analyzed_count=70,
            reference_matched_count=40,
        )

        config = SelectionConfig(target_count=50)
        effective_target = constraints.calculate_effective_target(config)

        assert effective_target == 30  # Limited by duplicate_filtered_count

    def test_calculate_effective_target_limited_by_quality(self) -> None:
        """Test effective target limited by quality filtering."""
        constraints = SelectionConstraints(
            available_images=100,
            quality_filtered_count=25,  # Lower limit
            duplicate_filtered_count=80,
            composition_analyzed_count=75,
            pose_analyzed_count=60,
            semantic_analyzed_count=70,
            reference_matched_count=40,
        )

        config = SelectionConfig(target_count=50)
        effective_target = constraints.calculate_effective_target(config)

        assert effective_target == 25  # Limited by quality_filtered_count

    def test_has_sufficient_diversity_data(
        self, sample_constraints: SelectionConstraints
    ) -> None:
        """Test diversity data sufficiency check."""
        assert sample_constraints.has_sufficient_diversity_data  # Both > 10

        insufficient_constraints = SelectionConstraints(
            available_images=100,
            quality_filtered_count=85,
            duplicate_filtered_count=80,
            composition_analyzed_count=75,
            pose_analyzed_count=5,  # Too few
            semantic_analyzed_count=8,  # Too few
            reference_matched_count=40,
        )

        assert not insufficient_constraints.has_sufficient_diversity_data

    def test_composition_coverage_ratio(
        self, sample_constraints: SelectionConstraints
    ) -> None:
        """Test composition coverage ratio calculation."""
        ratio = sample_constraints.composition_coverage_ratio
        assert ratio == 0.75  # 75/100

    def test_diversity_coverage_ratio(
        self, sample_constraints: SelectionConstraints
    ) -> None:
        """Test diversity coverage ratio calculation."""
        ratio = sample_constraints.diversity_coverage_ratio
        # Should be max(pose_analyzed_count, semantic_analyzed_count) / available_images
        # max(60, 70) / 100 = 0.7
        assert ratio == 0.7

    def test_coverage_ratios_zero_available(self) -> None:
        """Test coverage ratios with zero available images."""
        zero_constraints = SelectionConstraints(
            available_images=0,
            quality_filtered_count=0,
            duplicate_filtered_count=0,
            composition_analyzed_count=0,
            pose_analyzed_count=0,
            semantic_analyzed_count=0,
            reference_matched_count=0,
        )

        assert zero_constraints.composition_coverage_ratio == 0.0
        assert zero_constraints.diversity_coverage_ratio == 0.0
