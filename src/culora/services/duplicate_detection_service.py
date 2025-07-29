"""Deduplication service for CuLoRA."""

import imagehash
from PIL import Image

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.models.duplicate_detection_result import DuplicateDetectionResult


class DuplicateDetectionService:
    """Service for detecting and handling duplicate images using perceptual hashing."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        """Initialize the deduplication service.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()

    def analyze_image(self, image: Image.Image) -> DuplicateDetectionResult:
        """Generate perceptual hash for duplicate detection.

        Generates a dHash for the image that will be used for duplicate detection
        by the selection service.

        Args:
            image: PIL Image object.

        Returns:
            DuplicateDetectionResult with hash data.
        """
        try:
            config = self._config_manager.get_config(AnalysisStage.DEDUPLICATION)

            # Use dHash for speed and effectiveness with photo duplicates
            dhash = imagehash.dhash(image, hash_size=config.hash_size)
            hash_str = str(dhash)

            return DuplicateDetectionResult(
                hash_value=hash_str,
            )

        except Exception:
            # Return empty result on failure - SelectionService will handle this
            return DuplicateDetectionResult(
                hash_value="",
            )
