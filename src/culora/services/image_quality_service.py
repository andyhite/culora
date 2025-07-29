"""Image quality service for CuLoRA."""

from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from culora.managers.config_manager import ConfigManager
from culora.models.image_quality_result import (
    ImageQualityResult,
)

# QualityConfig accessed via ConfigManager


class ImageQualityService:
    """Service for assessing image quality using OpenCV metrics."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        """Initialize the quality quality service.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()

    def analyze_image(self, image: Image.Image) -> ImageQualityResult:
        """Analyze image quality using OpenCV metrics.

        Evaluates sharpness, brightness, and contrast of the image and calculates
        a composite quality score. Based on research documented in docs/analysis-libraries.md.

        Args:
            image: PIL Image object.

        Returns:
            QualityResult with quality metrics.
        """
        try:
            # Config not needed for current implementation but keep for future use
            # config = self._config_manager.get_stage_config(AnalysisStage.QUALITY)

            # Convert PIL image to OpenCV format
            # PIL uses RGB, OpenCV uses BGR, so convert
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Calculate brightness (mean pixel intensity)
            brightness = float(np.mean(cast(np.ndarray[Any, np.dtype[Any]], gray)))

            # Calculate contrast (standard deviation)
            contrast = float(np.std(cast(np.ndarray[Any, np.dtype[Any]], gray)))

            # Calculate composite quality score from image quality metrics
            composite_score = 0.0
            # Sharpness component (40 points max) - most important for LoRA training
            composite_score += min(laplacian_var / 1000.0, 1.0) * 40
            # Brightness component (30 points max) - optimal at 128, penalize extremes
            brightness_score_factor = 1.0 - abs(brightness - 128) / 128.0
            composite_score += brightness_score_factor * 30
            # Contrast component (30 points max) - important for detail
            composite_score += min(contrast / 100.0, 1.0) * 30

            return ImageQualityResult(
                sharpness_score=laplacian_var,
                brightness_score=brightness,
                contrast_score=contrast,
                composite_score=composite_score,
            )

        except Exception:
            # Return empty/zero result on failure - SelectionService will handle this
            return ImageQualityResult(
                sharpness_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                composite_score=0.0,
            )
