"""Domain models for duplicate detection and analysis."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class HashAlgorithm(str, Enum):
    """Supported perceptual hash algorithms."""

    AVERAGE = "average"
    PERCEPTUAL = "perceptual"
    DIFFERENCE = "difference"
    WAVELET = "wavelet"


class DuplicateThreshold(BaseModel):
    """Threshold configuration for duplicate detection."""

    hash_algorithm: HashAlgorithm = Field(
        default=HashAlgorithm.PERCEPTUAL, description="Perceptual hash algorithm to use"
    )
    similarity_threshold: int = Field(
        default=10,
        ge=0,
        le=64,
        description="Hamming distance threshold for duplicate detection (0=identical, 64=completely different)",
    )
    group_threshold: int = Field(
        default=5,
        ge=0,
        le=64,
        description="Hamming distance threshold for grouping similar images",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate threshold relationships."""
        if self.group_threshold > self.similarity_threshold:
            raise ValueError(
                f"Group threshold ({self.group_threshold}) must be <= similarity threshold ({self.similarity_threshold})"
            )


class ImageHash(BaseModel):
    """Perceptual hash for an image."""

    image_path: Path = Field(description="Path to the image file")
    hash_value: str = Field(description="Hexadecimal hash string")
    hash_algorithm: HashAlgorithm = Field(description="Algorithm used for hashing")
    hash_size: int = Field(description="Hash size in bits")

    @property
    def hash_bits(self) -> int:
        """Calculate number of bits from hash string length."""
        return len(self.hash_value) * 4  # Each hex character = 4 bits


class DuplicateMatch(BaseModel):
    """A duplicate match between two images."""

    image1_path: Path = Field(description="Path to first image")
    image2_path: Path = Field(description="Path to second image")
    hamming_distance: int = Field(
        ge=0, le=64, description="Hamming distance between hashes"
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score (1.0 = identical, 0.0 = completely different)",
    )
    hash_algorithm: HashAlgorithm = Field(description="Algorithm used for comparison")

    @property
    def is_exact_duplicate(self) -> bool:
        """Check if this is an exact duplicate (distance = 0)."""
        return self.hamming_distance == 0

    @property
    def is_near_duplicate(self) -> bool:
        """Check if this is a near duplicate (low distance)."""
        return self.hamming_distance <= 5


class DuplicateGroup(BaseModel):
    """A group of duplicate or near-duplicate images."""

    group_id: str = Field(description="Unique identifier for the group")
    image_paths: list[Path] = Field(description="Paths to all images in the group")
    representative_path: Path | None = Field(
        default=None, description="Path to the selected representative image"
    )
    quality_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores for each image (path_str -> score)",
    )
    max_distance: int = Field(
        ge=0, description="Maximum hamming distance within the group"
    )
    hash_algorithm: HashAlgorithm = Field(description="Algorithm used for grouping")

    @property
    def image_count(self) -> int:
        """Number of images in the group."""
        return len(self.image_paths)

    @property
    def has_representative(self) -> bool:
        """Check if a representative has been selected."""
        return self.representative_path is not None

    def select_representative(
        self, quality_scores: dict[str, float] | None = None
    ) -> Path:
        """Select the best representative image based on quality scores."""
        if not self.image_paths:
            raise ValueError("Cannot select representative from empty group")

        # Use provided quality scores or fall back to stored ones
        scores = quality_scores or self.quality_scores

        if scores:
            # Select image with highest quality score
            best_path = max(self.image_paths, key=lambda p: scores.get(str(p), 0.0))
        else:
            # Fall back to first image if no quality scores available
            best_path = self.image_paths[0]

        self.representative_path = best_path
        return best_path


class DuplicateAnalysis(BaseModel):
    """Complete duplicate detection analysis results."""

    total_images: int = Field(ge=0, description="Total number of images analyzed")
    total_hashes: int = Field(ge=0, description="Total number of hashes generated")
    total_matches: int = Field(
        ge=0, description="Total number of duplicate matches found"
    )
    total_groups: int = Field(ge=0, description="Total number of duplicate groups")
    exact_duplicates: int = Field(
        ge=0, description="Number of exact duplicates (distance=0)"
    )
    near_duplicates: int = Field(
        ge=0, description="Number of near duplicates (distance<=5)"
    )
    hash_algorithm: HashAlgorithm = Field(description="Algorithm used for analysis")
    threshold_config: DuplicateThreshold = Field(
        description="Threshold configuration used"
    )

    duplicate_groups: list[DuplicateGroup] = Field(
        default_factory=list, description="All detected duplicate groups"
    )
    matches: list[DuplicateMatch] = Field(
        default_factory=list, description="All duplicate matches found"
    )
    unique_images: list[Path] = Field(
        default_factory=list, description="Images with no duplicates found"
    )

    @property
    def duplicate_rate(self) -> float:
        """Percentage of images that are duplicates."""
        if self.total_images == 0:
            return 0.0
        duplicate_images = self.total_images - len(self.unique_images)
        return (duplicate_images / self.total_images) * 100.0

    @property
    def reduction_rate(self) -> float:
        """Percentage reduction in dataset size after duplicate removal."""
        if self.total_images == 0:
            return 0.0
        images_after_dedup = len(self.unique_images) + self.total_groups
        return ((self.total_images - images_after_dedup) / self.total_images) * 100.0

    @property
    def images_after_deduplication(self) -> int:
        """Number of images remaining after deduplication."""
        return len(self.unique_images) + self.total_groups

    def get_representative_images(self) -> list[Path]:
        """Get list of all representative images (unique + group representatives)."""
        representatives = list(self.unique_images)

        for group in self.duplicate_groups:
            if group.has_representative and group.representative_path is not None:
                representatives.append(group.representative_path)
            elif group.image_paths:
                # Fall back to first image if no representative selected
                representatives.append(group.image_paths[0])

        return representatives


class DuplicateRemovalStrategy(str, Enum):
    """Strategy for handling duplicate removal."""

    KEEP_HIGHEST_QUALITY = "highest_quality"
    KEEP_FIRST = "first"
    KEEP_SMALLEST_FILE = "smallest_file"
    KEEP_LARGEST_FILE = "largest_file"


class DuplicateConfig(BaseModel):
    """Configuration for duplicate detection and removal."""

    threshold: DuplicateThreshold = Field(
        default_factory=DuplicateThreshold,
        description="Threshold configuration for duplicate detection",
    )
    removal_strategy: DuplicateRemovalStrategy = Field(
        default=DuplicateRemovalStrategy.KEEP_HIGHEST_QUALITY,
        description="Strategy for selecting representatives from duplicate groups",
    )
    enable_exact_matching: bool = Field(
        default=True, description="Enable exact duplicate detection (distance=0)"
    )
    enable_near_matching: bool = Field(
        default=True, description="Enable near duplicate detection (low distance)"
    )
    max_group_size: int = Field(
        default=50, ge=2, description="Maximum number of images per duplicate group"
    )
    progress_reporting: bool = Field(
        default=True, description="Enable progress reporting during analysis"
    )
