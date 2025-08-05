"""Service for duplicate image detection and analysis."""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from culora.core.exceptions import CuLoRAImageError, CuLoRAServiceError
from culora.domain.models.duplicate import (
    DuplicateAnalysis,
    DuplicateConfig,
    DuplicateGroup,
    DuplicateMatch,
    HashAlgorithm,
    ImageHash,
)
from culora.utils import get_logger

logger = get_logger(__name__)


class DuplicateService:
    """Service for perceptual duplicate detection and analysis."""

    def __init__(self, config: DuplicateConfig) -> None:
        """Initialize the duplicate detection service.

        Args:
            config: Configuration for duplicate detection
        """
        self.config = config
        self._suppress_pil_warnings()

    def _suppress_pil_warnings(self) -> None:
        """Suppress verbose PIL warnings."""
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)

    def calculate_hash(self, image_path: Path) -> ImageHash:
        """Calculate perceptual hash for an image.

        Args:
            image_path: Path to the image file

        Returns:
            ImageHash object with hash information

        Raises:
            CuLoRAImageError: If image cannot be loaded or hashed
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Calculate hash based on algorithm
                hash_func = self._get_hash_function()
                hash_obj = hash_func(img)

                return ImageHash(
                    image_path=image_path,
                    hash_value=str(hash_obj),
                    hash_algorithm=self.config.threshold.hash_algorithm,
                    hash_size=len(str(hash_obj)) * 4,  # hex chars to bits
                )

        except Exception as e:
            raise CuLoRAImageError(
                f"Failed to calculate hash for {image_path}: {e}"
            ) from e

    def _get_hash_function(self) -> Any:
        """Get the appropriate hash function based on configuration."""
        algorithm = self.config.threshold.hash_algorithm

        if algorithm == HashAlgorithm.AVERAGE:
            return imagehash.average_hash
        elif algorithm == HashAlgorithm.PERCEPTUAL:
            return imagehash.phash
        elif algorithm == HashAlgorithm.DIFFERENCE:
            return imagehash.dhash
        elif algorithm == HashAlgorithm.WAVELET:
            return imagehash.whash
        else:
            raise CuLoRAServiceError(f"Unsupported hash algorithm: {algorithm}")

    def calculate_batch_hashes(self, image_paths: list[Path]) -> list[ImageHash]:
        """Calculate perceptual hashes for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of ImageHash objects (successful hashes only)
        """
        hashes = []
        failed_count = 0

        for image_path in image_paths:
            try:
                image_hash = self.calculate_hash(image_path)
                hashes.append(image_hash)
            except CuLoRAImageError as e:
                logger.warning(
                    "Failed to hash image", image_path=str(image_path), error=str(e)
                )
                failed_count += 1

        if failed_count > 0:
            logger.info(
                "Hash calculation completed with failures",
                successful=len(hashes),
                failed=failed_count,
                total=len(image_paths),
            )

        return hashes

    def find_duplicates(self, hashes: list[ImageHash]) -> list[DuplicateMatch]:
        """Find duplicate matches between image hashes.

        Args:
            hashes: List of image hashes to compare

        Returns:
            List of duplicate matches found
        """
        matches = []
        threshold = self.config.threshold.similarity_threshold

        # Compare each hash with every other hash
        for i, hash1 in enumerate(hashes):
            for _j, hash2 in enumerate(hashes[i + 1 :], i + 1):
                try:
                    # Calculate Hamming distance
                    hash1_obj = imagehash.hex_to_hash(hash1.hash_value)
                    hash2_obj = imagehash.hex_to_hash(hash2.hash_value)
                    distance = hash1_obj - hash2_obj

                    # Check if within threshold
                    if distance <= threshold:
                        similarity_score = 1.0 - (distance / 64.0)  # Normalize to 0-1

                        match = DuplicateMatch(
                            image1_path=hash1.image_path,
                            image2_path=hash2.image_path,
                            hamming_distance=distance,
                            similarity_score=similarity_score,
                            hash_algorithm=hash1.hash_algorithm,
                        )
                        matches.append(match)

                except Exception as e:
                    logger.warning(
                        "Failed to compare hashes",
                        hash1=hash1.hash_value,
                        hash2=hash2.hash_value,
                        error=str(e),
                    )

        return matches

    def group_duplicates(
        self,
        matches: list[DuplicateMatch],
        quality_scores: dict[str, float] | None = None,
    ) -> list[DuplicateGroup]:
        """Group duplicate matches into clusters.

        Args:
            matches: List of duplicate matches
            quality_scores: Optional quality scores for representative selection

        Returns:
            List of duplicate groups
        """
        if not matches:
            return []

        # Build adjacency graph
        graph: dict[Path, set[Path]] = defaultdict(set)
        for match in matches:
            if match.hamming_distance <= self.config.threshold.group_threshold:
                graph[match.image1_path].add(match.image2_path)
                graph[match.image2_path].add(match.image1_path)

        # Find connected components using DFS
        visited: set[Path] = set()
        groups = []
        group_id = 0

        for image_path in graph:
            if image_path not in visited:
                # Find all connected images
                component = []
                stack = [image_path]
                max_distance = 0

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        # Add neighbors to stack
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                                # Track maximum distance in group
                                for match in matches:
                                    if (
                                        match.image1_path == current
                                        and match.image2_path == neighbor
                                    ) or (
                                        match.image2_path == current
                                        and match.image1_path == neighbor
                                    ):
                                        max_distance = max(
                                            max_distance, match.hamming_distance
                                        )

                # Create group if it has multiple images
                if len(component) > 1:
                    group = DuplicateGroup(
                        group_id=f"group_{group_id:04d}",
                        image_paths=component,
                        quality_scores=quality_scores or {},
                        max_distance=max_distance,
                        hash_algorithm=matches[0].hash_algorithm,
                    )

                    # Select representative if quality scores available
                    if quality_scores:
                        group.select_representative(quality_scores)

                    groups.append(group)
                    group_id += 1

        return groups

    def analyze_duplicates(
        self, image_paths: list[Path], quality_scores: dict[str, float] | None = None
    ) -> DuplicateAnalysis:
        """Perform complete duplicate detection analysis.

        Args:
            image_paths: List of image paths to analyze
            quality_scores: Optional quality scores for representative selection

        Returns:
            Complete duplicate analysis results
        """
        logger.info(
            "Starting duplicate analysis",
            total_images=len(image_paths),
            algorithm=self.config.threshold.hash_algorithm.value,
            threshold=self.config.threshold.similarity_threshold,
        )

        # Calculate hashes
        hashes = self.calculate_batch_hashes(image_paths)

        if not hashes:
            logger.warning("No valid hashes generated")
            return DuplicateAnalysis(
                total_images=len(image_paths),
                total_hashes=0,
                total_matches=0,
                total_groups=0,
                exact_duplicates=0,
                near_duplicates=0,
                hash_algorithm=self.config.threshold.hash_algorithm,
                threshold_config=self.config.threshold,
                unique_images=image_paths,
            )

        # Find duplicate matches
        matches = self.find_duplicates(hashes)

        # Group duplicates
        groups = self.group_duplicates(matches, quality_scores)

        # Find unique images (not in any group)
        grouped_images = set()
        for group in groups:
            grouped_images.update(group.image_paths)

        unique_images = [
            hash_obj.image_path
            for hash_obj in hashes
            if hash_obj.image_path not in grouped_images
        ]

        # Count exact and near duplicates
        exact_duplicates = sum(1 for match in matches if match.is_exact_duplicate)
        near_duplicates = sum(1 for match in matches if match.is_near_duplicate)

        analysis = DuplicateAnalysis(
            total_images=len(image_paths),
            total_hashes=len(hashes),
            total_matches=len(matches),
            total_groups=len(groups),
            exact_duplicates=exact_duplicates,
            near_duplicates=near_duplicates,
            hash_algorithm=self.config.threshold.hash_algorithm,
            threshold_config=self.config.threshold,
            duplicate_groups=groups,
            matches=matches,
            unique_images=unique_images,
        )

        logger.info(
            "Duplicate analysis completed",
            total_hashes=analysis.total_hashes,
            total_matches=analysis.total_matches,
            total_groups=analysis.total_groups,
            duplicate_rate=f"{analysis.duplicate_rate:.1f}%",
            reduction_rate=f"{analysis.reduction_rate:.1f}%",
        )

        return analysis

    def get_file_sizes(self, image_paths: list[Path]) -> dict[str, int]:
        """Get file sizes for images.

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping path strings to file sizes in bytes
        """
        sizes = {}
        for path in image_paths:
            try:
                sizes[str(path)] = os.path.getsize(path)
            except OSError as e:
                logger.warning(
                    "Failed to get file size", image_path=str(path), error=str(e)
                )
                sizes[str(path)] = 0
        return sizes


# Global service instance
_duplicate_service: DuplicateService | None = None


def get_duplicate_service(config: DuplicateConfig | None = None) -> DuplicateService:
    """Get or create the global duplicate detection service.

    Args:
        config: Optional configuration. If not provided, uses default config.

    Returns:
        DuplicateService instance
    """
    global _duplicate_service

    if _duplicate_service is None or config is not None:
        service_config = config or DuplicateConfig()
        _duplicate_service = DuplicateService(service_config)

    return _duplicate_service
