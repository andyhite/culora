"""Configuration models for duplicate detection."""

from pydantic import BaseModel, Field

from culora.domain.models.duplicate import DuplicateConfig as DomainDuplicateConfig


class DuplicateConfig(BaseModel):
    """Configuration for duplicate detection and analysis."""

    # Hash algorithm settings
    hash_algorithm: str = Field(
        default="perceptual",
        description="Perceptual hash algorithm: average, perceptual, difference, wavelet",
    )

    # Threshold settings
    similarity_threshold: int = Field(
        default=10,
        ge=0,
        le=64,
        description="Hamming distance threshold for duplicate detection (0-64)",
    )

    group_threshold: int = Field(
        default=5,
        ge=0,
        le=64,
        description="Hamming distance threshold for grouping similar images (0-64)",
    )

    # Selection strategy
    removal_strategy: str = Field(
        default="highest_quality",
        description="Strategy for selecting representatives: highest_quality, first, smallest_file, largest_file",
    )

    # Feature flags
    enable_exact_matching: bool = Field(
        default=True, description="Enable exact duplicate detection (distance=0)"
    )

    enable_near_matching: bool = Field(
        default=True, description="Enable near duplicate detection (distance<=5)"
    )

    # Performance settings
    max_group_size: int = Field(
        default=50, ge=2, description="Maximum number of images per duplicate group"
    )

    progress_reporting: bool = Field(
        default=True, description="Enable progress reporting during analysis"
    )

    def to_domain_config(self) -> DomainDuplicateConfig:
        """Convert to domain duplicate configuration."""
        from culora.domain.models.duplicate import (
            DuplicateRemovalStrategy,
            DuplicateThreshold,
            HashAlgorithm,
        )

        # Map string values to enums
        algorithm_map = {
            "average": HashAlgorithm.AVERAGE,
            "perceptual": HashAlgorithm.PERCEPTUAL,
            "difference": HashAlgorithm.DIFFERENCE,
            "wavelet": HashAlgorithm.WAVELET,
        }

        strategy_map = {
            "highest_quality": DuplicateRemovalStrategy.KEEP_HIGHEST_QUALITY,
            "first": DuplicateRemovalStrategy.KEEP_FIRST,
            "smallest_file": DuplicateRemovalStrategy.KEEP_SMALLEST_FILE,
            "largest_file": DuplicateRemovalStrategy.KEEP_LARGEST_FILE,
        }

        threshold = DuplicateThreshold(
            hash_algorithm=algorithm_map[self.hash_algorithm],
            similarity_threshold=self.similarity_threshold,
            group_threshold=self.group_threshold,
        )

        return DomainDuplicateConfig(
            threshold=threshold,
            removal_strategy=strategy_map[self.removal_strategy],
            enable_exact_matching=self.enable_exact_matching,
            enable_near_matching=self.enable_near_matching,
            max_group_size=self.max_group_size,
            progress_reporting=self.progress_reporting,
        )
