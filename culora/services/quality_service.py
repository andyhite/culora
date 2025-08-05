"""Image quality assessment service."""

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from culora.core import CuLoRAError
from culora.domain import CuLoRAConfig
from culora.domain.models.quality import (
    BatchQualityResult,
    ImageQualityResult,
    QualityScore,
    TechnicalQualityMetrics,
)
from culora.utils import get_logger

logger = get_logger(__name__)


class QualityServiceError(CuLoRAError):
    """Base exception for quality service errors."""

    def __init__(self, message: str, error_code: str = "QUALITY_SERVICE_ERROR") -> None:
        super().__init__(message, error_code)


class QualityAnalysisError(QualityServiceError):
    """Exception for quality analysis failures."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message, "QUALITY_ANALYSIS_ERROR")
        self.path = path


class QualityService:
    """Service for image quality assessment and scoring.

    Provides comprehensive technical quality analysis including:
    - Sharpness assessment using Laplacian variance
    - Brightness and contrast evaluation
    - Color quality and saturation analysis
    - Noise detection and scoring
    - Composite quality score calculation
    """

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize quality service with configuration.

        Args:
            config: CuLoRA configuration containing quality settings
        """
        self.config = config
        self.quality_config = config.quality
        logger.info("Quality service initialized")

    def analyze_image(self, image: Image.Image, path: Path) -> ImageQualityResult:
        """Analyze quality metrics for a single image.

        Args:
            image: PIL Image to analyze
            path: Path to the image file for error reporting

        Returns:
            Complete quality analysis result
        """
        start_time = time.time()

        try:
            # Prepare image for analysis
            analysis_image, was_resized = self._prepare_image_for_analysis(image)

            # Calculate technical quality metrics
            metrics = self._calculate_technical_metrics(analysis_image, was_resized)

            # Calculate composite quality score
            score = self._calculate_quality_score(metrics)

            duration = time.time() - start_time

            return ImageQualityResult(
                path=path,
                success=True,
                metrics=metrics,
                score=score,
                analysis_duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Quality analysis failed for {path}: {e}"
            logger.warning(error_msg)

            return ImageQualityResult(
                path=path,
                success=False,
                error=error_msg,
                error_code="QUALITY_ANALYSIS_FAILED",
                analysis_duration=duration,
            )

    def analyze_batch(
        self, images_and_paths: list[tuple[Image.Image, Path]]
    ) -> BatchQualityResult:
        """Analyze quality for a batch of images.

        Args:
            images_and_paths: List of (image, path) tuples to analyze

        Returns:
            Batch analysis results with statistics
        """
        start_time = time.time()
        results: list[ImageQualityResult] = []

        logger.info(f"Starting quality analysis for {len(images_and_paths)} images")

        # Analyze individual images
        for image, path in images_and_paths:
            result = self.analyze_image(image, path)
            results.append(result)

        # Calculate batch statistics
        total_duration = time.time() - start_time
        successful_results = [r for r in results if r.success and r.score is not None]

        batch_result = self._calculate_batch_statistics(
            results, successful_results, total_duration
        )

        # Add percentile information to individual results
        self._add_percentile_rankings(successful_results, batch_result)

        logger.info(
            f"Quality analysis completed: {batch_result.successful_analyses}/"
            f"{len(results)} successful, "
            f"mean score: {batch_result.mean_quality_score:.3f}"
        )

        return batch_result

    def _prepare_image_for_analysis(
        self, image: Image.Image
    ) -> tuple[Image.Image, bool]:
        """Prepare image for quality analysis.

        Args:
            image: Original PIL Image

        Returns:
            Tuple of (prepared_image, was_resized)
        """
        if not self.quality_config.resize_for_analysis:
            return image, False

        max_width, max_height = self.quality_config.max_analysis_size
        width, height = image.size

        if width <= max_width and height <= max_height:
            return image, False

        # Calculate aspect-preserving resize
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, True

    def _calculate_technical_metrics(
        self, image: Image.Image, was_resized: bool
    ) -> TechnicalQualityMetrics:
        """Calculate technical quality metrics for an image.

        Args:
            image: PIL Image to analyze
            was_resized: Whether the image was resized for analysis

        Returns:
            Technical quality metrics
        """
        # Convert to grayscale for some calculations
        gray_array = np.array(image.convert("L"))

        # Convert to RGB array for color analysis
        rgb_array = np.array(image.convert("RGB"))

        width, height = image.size

        # Calculate sharpness using Laplacian variance
        laplacian_var = self._calculate_sharpness(gray_array)
        sharpness_score = self._normalize_sharpness(laplacian_var)

        # Calculate brightness metrics
        mean_brightness = float(np.mean(gray_array) / 255.0)
        brightness_score = self._score_brightness(mean_brightness)

        # Calculate contrast using standard deviation
        contrast_value = float(np.std(gray_array) / 255.0)
        contrast_score = self._score_contrast(contrast_value)

        # Calculate color quality (saturation)
        mean_saturation = self._calculate_color_quality(rgb_array)
        color_score = self._score_color_quality(mean_saturation)

        # Calculate noise level
        noise_level = self._calculate_noise_level(gray_array)
        noise_score = self._score_noise_level(noise_level)

        return TechnicalQualityMetrics(
            sharpness=sharpness_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            color_quality=color_score,
            noise_score=noise_score,
            laplacian_variance=laplacian_var,
            mean_brightness=mean_brightness,
            contrast_value=contrast_value,
            mean_saturation=mean_saturation,
            noise_level=noise_level,
            analysis_width=width,
            analysis_height=height,
            was_resized=was_resized,
        )

    def _calculate_sharpness(self, gray_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance.

        Args:
            gray_array: Grayscale image as numpy array

        Returns:
            Laplacian variance (higher = sharper)
        """
        kernel_size = self.quality_config.sharpness_kernel_size
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F, ksize=kernel_size)
        return float(laplacian.var())

    def _normalize_sharpness(self, laplacian_var: float) -> float:
        """Convert Laplacian variance to 0-1 score.

        Args:
            laplacian_var: Raw Laplacian variance

        Returns:
            Normalized sharpness score (0-1)
        """
        # Empirically determined ranges for different image types
        # These can be adjusted based on training data characteristics
        min_sharp = 50.0  # Below this is very blurry
        max_sharp = 2000.0  # Above this is very sharp

        return min(1.0, max(0.0, (laplacian_var - min_sharp) / (max_sharp - min_sharp)))

    def _score_brightness(self, mean_brightness: float) -> float:
        """Score brightness based on optimal range.

        Args:
            mean_brightness: Mean brightness (0-1)

        Returns:
            Brightness score (0-1)
        """
        min_optimal, max_optimal = self.quality_config.optimal_brightness_range

        if min_optimal <= mean_brightness <= max_optimal:
            return 1.0
        elif mean_brightness < min_optimal:
            # Too dark - score decreases as it gets darker
            return mean_brightness / min_optimal
        else:
            # Too bright - score decreases as it gets brighter
            return (1.0 - mean_brightness) / (1.0 - max_optimal)

    def _score_contrast(self, contrast_value: float) -> float:
        """Score contrast based on standard deviation.

        Args:
            contrast_value: Contrast as normalized standard deviation

        Returns:
            Contrast score (0-1)
        """
        threshold = self.quality_config.high_contrast_threshold
        return min(1.0, contrast_value / threshold)

    def _calculate_color_quality(self, rgb_array: np.ndarray) -> float:
        """Calculate color quality based on saturation.

        Args:
            rgb_array: RGB image as numpy array

        Returns:
            Mean saturation (0-1)
        """
        # Convert RGB to HSV to get saturation
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        return float(np.mean(saturation))

    def _score_color_quality(self, mean_saturation: float) -> float:
        """Score color quality based on saturation levels.

        Args:
            mean_saturation: Mean saturation (0-1)

        Returns:
            Color quality score (0-1)
        """
        min_sat = self.quality_config.min_saturation
        max_sat = self.quality_config.max_saturation

        if mean_saturation < min_sat:
            # Too desaturated
            return mean_saturation / min_sat
        elif mean_saturation > max_sat:
            # Oversaturated
            return 1.0 - ((mean_saturation - max_sat) / (1.0 - max_sat))
        else:
            # Good saturation range
            return 1.0

    def _calculate_noise_level(self, gray_array: np.ndarray) -> float:
        """Calculate noise level using local standard deviation.

        Args:
            gray_array: Grayscale image as numpy array

        Returns:
            Estimated noise level
        """
        # Use Laplacian to highlight edges, then measure variation
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F, ksize=1)
        return float(np.std(laplacian))

    def _score_noise_level(self, noise_level: float) -> float:
        """Score noise level (higher noise = lower score).

        Args:
            noise_level: Calculated noise level

        Returns:
            Noise score (0-1, higher is better)
        """
        threshold = self.quality_config.noise_threshold
        # Invert score - less noise is better
        return max(0.0, 1.0 - (noise_level / threshold))

    def _calculate_quality_score(
        self, metrics: TechnicalQualityMetrics
    ) -> QualityScore:
        """Calculate composite quality score from technical metrics.

        Args:
            metrics: Technical quality metrics

        Returns:
            Composite quality score
        """
        # Calculate weighted contributions
        sharpness_contrib = metrics.sharpness * self.quality_config.sharpness_weight
        brightness_contrib = (
            metrics.brightness_score * self.quality_config.brightness_weight
        )
        contrast_contrib = metrics.contrast_score * self.quality_config.contrast_weight
        color_contrib = metrics.color_quality * self.quality_config.color_weight
        noise_contrib = metrics.noise_score * self.quality_config.noise_weight

        # Calculate composite technical score
        technical_score = (
            sharpness_contrib
            + brightness_contrib
            + contrast_contrib
            + color_contrib
            + noise_contrib
        )

        # For now, overall score is same as technical score
        # Future versions may include perceptual quality (BRISQUE)
        overall_score = technical_score

        # Check if passes minimum threshold
        passes_threshold = overall_score >= self.quality_config.min_quality_score

        return QualityScore(
            technical_score=technical_score,
            overall_score=overall_score,
            sharpness_contribution=sharpness_contrib,
            brightness_contribution=brightness_contrib,
            contrast_contribution=contrast_contrib,
            color_contribution=color_contrib,
            noise_contribution=noise_contrib,
            passes_threshold=passes_threshold,
        )

    def _calculate_batch_statistics(
        self,
        all_results: list[ImageQualityResult],
        successful_results: list[ImageQualityResult],
        total_duration: float,
    ) -> BatchQualityResult:
        """Calculate statistics for batch quality analysis.

        Args:
            all_results: All analysis results including failures
            successful_results: Only successful analysis results
            total_duration: Total processing time

        Returns:
            Batch statistics and results
        """
        successful_count = len(successful_results)
        failed_count = len(all_results) - successful_count

        if successful_count == 0:
            # No successful analyses
            return BatchQualityResult(
                results=all_results,
                successful_analyses=0,
                failed_analyses=failed_count,
                mean_quality_score=0.0,
                median_quality_score=0.0,
                quality_score_std=0.0,
                quality_score_range=(0.0, 0.0),
                total_duration=total_duration,
                images_per_second=0.0,
                scores_by_percentile={},
                passing_threshold_count=0,
            )

        # Extract quality scores
        scores = [r.score.overall_score for r in successful_results if r.score]

        # Calculate statistics
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        score_range = (min(scores), max(scores))

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        scores_by_percentile = {}
        for p in percentiles:
            if scores:
                idx = int((p / 100.0) * (len(scores) - 1))
                sorted_scores = sorted(scores)
                scores_by_percentile[p] = sorted_scores[idx]

        # Count images passing threshold
        passing_count = sum(
            1 for r in successful_results if r.score and r.score.passes_threshold
        )

        # Calculate processing rate
        images_per_second = (
            len(all_results) / total_duration if total_duration > 0 else 0.0
        )

        return BatchQualityResult(
            results=all_results,
            successful_analyses=successful_count,
            failed_analyses=failed_count,
            mean_quality_score=mean_score,
            median_quality_score=median_score,
            quality_score_std=score_std,
            quality_score_range=score_range,
            total_duration=total_duration,
            images_per_second=images_per_second,
            scores_by_percentile=scores_by_percentile,
            passing_threshold_count=passing_count,
        )

    def _add_percentile_rankings(
        self,
        successful_results: list[ImageQualityResult],
        batch_result: BatchQualityResult,
    ) -> None:
        """Add percentile rankings to individual results.

        Args:
            successful_results: Successful analysis results to update
            batch_result: Batch result containing percentile data
        """
        if not successful_results:
            return

        # Sort results by quality score
        sorted_results = sorted(
            successful_results, key=lambda r: r.score.overall_score if r.score else 0.0
        )

        # Calculate percentile for each result
        total_count = len(sorted_results)
        for i, result in enumerate(sorted_results):
            if result.score:
                percentile = (i / (total_count - 1)) * 100 if total_count > 1 else 100.0
                # Update the score with percentile (requires creating new immutable object)
                result.score.__dict__["quality_percentile"] = percentile


# Global service instance
_quality_service: QualityService | None = None


def get_quality_service(config: CuLoRAConfig | None = None) -> QualityService:
    """Get or create quality service instance.

    Args:
        config: Configuration to use for service creation

    Returns:
        Quality service instance
    """
    global _quality_service

    if _quality_service is None:
        if config is None:
            from culora.services.config_service import get_config_service

            config = get_config_service().get_config()
        _quality_service = QualityService(config)

    return _quality_service
