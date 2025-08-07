"""Configuration models for multi-criteria selection algorithms."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from culora.domain.models.composition import (
    SceneType,
    ShotType,
)


class DistributionTarget(BaseModel):
    """Target distribution specification for a composition category.

    Can specify either a ratio (0.0-1.0) or absolute count.
    If both are provided, count takes precedence.
    """

    category_name: str = Field(description="Name of the composition category")
    target_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Target ratio of total selection (0.0-1.0)",
    )
    target_count: int | None = Field(
        default=None, ge=0, description="Target absolute count"
    )
    min_count: int = Field(
        default=0, ge=0, description="Minimum required count for this category"
    )
    max_count: int | None = Field(
        default=None, ge=0, description="Maximum allowed count for this category"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Priority level (1=lowest, 10=highest) for distribution conflicts",
    )

    @field_validator("max_count")
    @classmethod
    def validate_max_count(cls, v: int | None, info: Any) -> int | None:
        """Validate max_count is greater than min_count."""
        if v is not None and "min_count" in info.data and v < info.data["min_count"]:
            raise ValueError("max_count must be greater than or equal to min_count")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        if self.target_ratio is None and self.target_count is None:
            raise ValueError("Either target_ratio or target_count must be specified")


class ShotTypeDistribution(BaseModel):
    """Shot type distribution targets."""

    targets: dict[ShotType, DistributionTarget] = Field(
        default_factory=dict, description="Distribution targets by shot type"
    )
    enable_balancing: bool = Field(
        default=True, description="Enable automatic balancing across shot types"
    )
    fallback_distribution: dict[ShotType, float] = Field(
        default_factory=lambda: {
            ShotType.CLOSEUP: 0.3,
            ShotType.MEDIUM_SHOT: 0.25,
            ShotType.PORTRAIT: 0.2,
            ShotType.HEADSHOT: 0.15,
            ShotType.FULL_BODY: 0.1,
        },
        description="Fallback distribution ratios when specific targets not met",
    )


class SceneTypeDistribution(BaseModel):
    """Scene type distribution targets."""

    targets: dict[SceneType, DistributionTarget] = Field(
        default_factory=dict, description="Distribution targets by scene type"
    )
    enable_balancing: bool = Field(
        default=True, description="Enable automatic balancing across scene types"
    )
    fallback_distribution: dict[SceneType, float] = Field(
        default_factory=lambda: {
            SceneType.INDOOR: 0.4,
            SceneType.OUTDOOR: 0.3,
            SceneType.STUDIO: 0.2,
            SceneType.NATURAL: 0.1,
        },
        description="Fallback distribution ratios when specific targets not met",
    )


class QualityThresholds(BaseModel):
    """Quality-based filtering thresholds."""

    min_composite_quality: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum composite quality score for selection",
    )
    min_technical_quality: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum technical quality score for selection",
    )
    min_brisque_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum BRISQUE quality score (if enabled)",
    )
    enable_quality_distribution: bool = Field(
        default=True,
        description="Ensure quality distribution across selection (not just top quality)",
    )
    quality_distribution_percentiles: list[float] = Field(
        default=[0.7, 0.85, 0.95],
        description="Target percentiles for quality distribution",
    )

    @field_validator("quality_distribution_percentiles")
    @classmethod
    def validate_percentiles(cls, v: list[float]) -> list[float]:
        """Validate percentiles are in ascending order and valid range."""
        if not v:
            raise ValueError("Quality distribution percentiles cannot be empty")

        for percentile in v:
            if not (0.0 <= percentile <= 1.0):
                raise ValueError(f"Percentile {percentile} must be between 0.0 and 1.0")

        if sorted(v) != v:
            raise ValueError(
                "Quality distribution percentiles must be in ascending order"
            )

        return v


class DiversitySettings(BaseModel):
    """Diversity optimization settings."""

    enable_pose_diversity: bool = Field(
        default=True, description="Enable pose-based diversity optimization"
    )
    enable_semantic_diversity: bool = Field(
        default=True, description="Enable CLIP semantic diversity optimization"
    )
    diversity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in selection scoring (0=ignore, 1=maximize)",
    )
    quality_vs_diversity_balance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Balance between quality (1.0) and diversity (0.0)",
    )
    min_cluster_separation: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum separation between selected items in embedding space",
    )
    max_selections_per_cluster: int = Field(
        default=3,
        ge=1,
        description="Maximum selections from a single diversity cluster",
    )


class SelectionConfig(BaseModel):
    """Configuration for multi-criteria selection algorithms."""

    # Target selection size
    target_count: int = Field(
        default=50, ge=1, description="Target number of images to select"
    )
    max_selection_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Maximum ratio of input images to select (prevents over-selection)",
    )

    # Quality filtering
    quality_thresholds: QualityThresholds = Field(
        default_factory=QualityThresholds,
        description="Quality-based filtering settings",
    )

    # Distribution targets
    shot_type_distribution: ShotTypeDistribution = Field(
        default_factory=ShotTypeDistribution,
        description="Shot type distribution targets",
    )
    scene_type_distribution: SceneTypeDistribution = Field(
        default_factory=SceneTypeDistribution,
        description="Scene type distribution targets",
    )
    enable_distribution_enforcement: bool = Field(
        default=True,
        description="Enforce distribution targets even if it means lower quality selections",
    )

    # Diversity optimization
    diversity_settings: DiversitySettings = Field(
        default_factory=DiversitySettings, description="Diversity optimization settings"
    )

    # Duplicate handling
    enable_duplicate_removal: bool = Field(
        default=True, description="Remove duplicates before selection"
    )
    duplicate_quality_preference: str = Field(
        default="highest",
        pattern="^(highest|lowest|first|last)$",
        description="Preference for duplicate selection: highest/lowest quality, first/last found",
    )

    # Reference matching
    enable_reference_matching: bool = Field(
        default=True, description="Enable reference identity matching bonuses"
    )
    reference_match_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for reference matching in selection scoring",
    )

    # Selection strategy
    selection_strategy: str = Field(
        default="multi_stage",
        pattern="^(multi_stage|quality_first|diversity_first|balanced)$",
        description="Selection strategy: multi_stage, quality_first, diversity_first, balanced",
    )
    enable_fallback_selection: bool = Field(
        default=True,
        description="Enable fallback selection when targets cannot be met",
    )

    # Performance optimization
    enable_early_filtering: bool = Field(
        default=True,
        description="Apply quality filters early to reduce processing load",
    )
    batch_size: int = Field(
        default=100, ge=1, description="Batch size for processing large datasets"
    )

    @field_validator("target_count")
    @classmethod
    def validate_target_count(cls, v: int) -> int:
        """Validate target count is reasonable."""
        if v > 10000:
            raise ValueError("Target count too large (maximum 10,000)")
        return v

    def calculate_max_selection(self, input_count: int) -> int:
        """Calculate maximum selection size based on input count and ratio."""
        max_by_ratio = int(input_count * self.max_selection_ratio)
        return min(self.target_count, max_by_ratio)

    def validate_distribution_targets(self, target_count: int) -> None:
        """Validate that distribution targets are achievable with target count."""
        # Check shot type distribution
        if self.shot_type_distribution.targets:
            total_min_count = sum(
                target.min_count
                for target in self.shot_type_distribution.targets.values()
            )
            if total_min_count > target_count:
                raise ValueError(
                    f"Shot type minimum counts ({total_min_count}) exceed target count ({target_count})"
                )

        # Check scene type distribution
        if self.scene_type_distribution.targets:
            total_min_count = sum(
                target.min_count
                for target in self.scene_type_distribution.targets.values()
            )
            if total_min_count > target_count:
                raise ValueError(
                    f"Scene type minimum counts ({total_min_count}) exceed target count ({target_count})"
                )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        self.validate_distribution_targets(self.target_count)


class SelectionConstraints(BaseModel):
    """Runtime constraints for selection algorithms."""

    available_images: int = Field(
        description="Number of available images for selection"
    )
    quality_filtered_count: int = Field(
        description="Number of images passing quality filters"
    )
    duplicate_filtered_count: int = Field(
        description="Number of images after duplicate removal"
    )
    composition_analyzed_count: int = Field(
        description="Number of images with composition analysis"
    )
    pose_analyzed_count: int = Field(description="Number of images with pose analysis")
    semantic_analyzed_count: int = Field(
        description="Number of images with semantic analysis"
    )
    reference_matched_count: int = Field(
        description="Number of images with reference matches"
    )

    def calculate_effective_target(self, config: SelectionConfig) -> int:
        """Calculate effective target selection count based on constraints."""
        max_possible = min(
            self.duplicate_filtered_count,
            self.quality_filtered_count,
        )
        return min(config.target_count, max_possible)

    @property
    def has_sufficient_diversity_data(self) -> bool:
        """Check if we have sufficient data for diversity analysis."""
        return (
            self.pose_analyzed_count > 10 or self.semantic_analyzed_count > 10
        )  # Arbitrary threshold

    @property
    def composition_coverage_ratio(self) -> float:
        """Calculate ratio of images with composition analysis."""
        if self.available_images == 0:
            return 0.0
        return self.composition_analyzed_count / self.available_images

    @property
    def diversity_coverage_ratio(self) -> float:
        """Calculate ratio of images with diversity analysis."""
        if self.available_images == 0:
            return 0.0
        max_diversity = max(self.pose_analyzed_count, self.semantic_analyzed_count)
        return max_diversity / self.available_images
