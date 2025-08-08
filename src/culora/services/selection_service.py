"""Selection service for CuLoRA."""

import shutil
from pathlib import Path

import cv2

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.models.directory_analysis import DirectoryAnalysis
from culora.models.face_detection_result import FaceDetectionResult
from culora.models.image_analysis import ImageAnalysis
from culora.models.image_quality_result import ImageQualityResult
from culora.utils.console import get_console

console = get_console()


class SelectionService:
    """Service for selecting and copying curated images."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        """Initialize the selection service.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()

    def select_images(
        self,
        analysis: DirectoryAnalysis,
        output_dir: str,
        draw_boxes: bool = False,
        dry_run: bool = False,
        max_images: int | None = None,
    ) -> tuple[int, int]:
        """Select and copy curated images based on analysis results.

        Uses a two-tier approach:
        1. First tier: Apply individual heuristics as minimum thresholds
        2. Second tier: Rank remaining images by composite score and select top N

        Args:
            analysis: Directory analysis results.
            output_dir: Directory to copy selected images to.
            draw_boxes: Whether to draw bounding boxes on faces.
            dry_run: Whether to perform a dry run (no actual copying).
            max_images: Maximum number of images to select (top N by score).

        Returns:
            Tuple of (selected_count, total_count).

        Raises:
            RuntimeError: If output directory cannot be created or other errors.
        """
        output_path = Path(output_dir).resolve()

        # Two-tier selection approach:
        # Tier 1: Apply individual heuristics as minimum thresholds (culling)
        # Tier 2: Rank by composite score and select top N

        enabled_stages = self._config_manager.analysis_config.enabled_stages

        # Tier 1: Cull images that don't meet minimum thresholds
        qualified_images: list[ImageAnalysis] = []
        for img in analysis.images:
            if self._evaluate_image_results(img, enabled_stages, analysis.images):
                qualified_images.append(img)

        if not qualified_images:
            return 0, len(analysis.images)

        # Tier 2: Sort by composite score (best first) and select top N
        qualified_images.sort(key=lambda img: img.score, reverse=True)

        # Apply max_images limit if specified
        if max_images is not None:
            selected_images = qualified_images[:max_images]
        else:
            selected_images = qualified_images

        if not selected_images:
            return 0, len(analysis.images)

        # Show selected filenames in dry-run mode
        if dry_run:
            console.info("Two-tier selection results:")
            console.print(
                f"  Tier 1 (Qualified): {len(qualified_images)} images passed minimum thresholds"
            )
            if max_images is not None:
                console.print(
                    f"  Tier 2 (Selected): Top {max_images} images by composite score"
                )
            else:
                console.print("  Tier 2 (Selected): All qualified images (no limit)")
            console.print()
            console.info("Selected images for curation:")
            for idx, image in enumerate(selected_images, 1):
                source_path = Path(image.file_path)
                extension = source_path.suffix.lower()
                target_filename = f"{idx:04d}{extension}"
                score_str = f"(score: {image.score:.3f})"
                console.print(f"  {source_path.name} → {target_filename} {score_str}")
            return len(selected_images), len(analysis.images)

        # Create output directory if it doesn't exist (unless dry run)
        if not dry_run:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Could not create output directory: {e}") from e

        selected_count = 0

        for idx, image in enumerate(selected_images, 1):
            source_path = Path(image.file_path)

            if not source_path.exists():
                continue  # Skip missing files

            # Generate target filename with sequential numbering
            extension = source_path.suffix.lower()
            target_filename = f"{idx:04d}{extension}"
            target_path = output_path / target_filename

            if not dry_run:
                if draw_boxes:
                    # Check if this image has face detection results
                    face_result: FaceDetectionResult | None = image.results.get_face()
                    if face_result and face_result.face_count > 0:
                        # Draw bounding boxes and save annotated image
                        self._draw_face_bounding_boxes(
                            source_path, target_path, face_result
                        )
                    else:
                        # No face detection data, just copy original
                        shutil.copy2(source_path, target_path)
                else:
                    # Just copy the original file
                    shutil.copy2(source_path, target_path)

            selected_count += 1

        return selected_count, len(analysis.images)

    def _draw_face_bounding_boxes(
        self, image_path: Path, output_path: Path, face_result: FaceDetectionResult
    ) -> None:
        """Draw bounding boxes on image for detected faces with confidence scores.

        Args:
            image_path: Path to the source image.
            output_path: Path where the annotated image will be saved.
            face_result: Face detection result containing bounding boxes and confidences.
        """
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:  # type: ignore[unreachable]
            # Fallback to copying original if image can't be loaded
            shutil.copy2(image_path, output_path)
            return

        # Draw each bounding box from face result
        for face in face_result.faces:
            try:
                # Extract coordinates from face object
                x1, y1, x2, y2 = map(int, face.bounding_box)

                # Get confidence from face object
                confidence = face.confidence

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw confidence score
                confidence_text = f"Face: {confidence:.2f}"
                text_size = cv2.getTextSize(
                    confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0]

                # Position text above the bounding box
                text_x = x1
                text_y = max(y1 - 10, text_size[1] + 5)

                # Draw text background
                cv2.rectangle(
                    image,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0], text_y + 5),
                    (0, 255, 0),
                    -1,
                )

                # Draw text
                cv2.putText(
                    image,
                    confidence_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            except (ValueError, IndexError):
                # Skip malformed bounding box data
                continue

        # Save the annotated image
        cv2.imwrite(str(output_path), image)

    def _evaluate_image_results(
        self,
        image: ImageAnalysis,
        enabled_stages: list[AnalysisStage],
        all_images: list[ImageAnalysis],
    ) -> bool:
        """Check if an image passes all enabled analysis stages based on analysis data.

        Args:
            image: Image analysis to check
            enabled_stages: List of enabled analysis stages
            all_images: All images in the dataset (needed for deduplication checking)

        Returns:
            True if the image passes all enabled stages, False otherwise
        """
        for stage in enabled_stages:
            if stage == AnalysisStage.QUALITY:
                quality_result = image.results.get_quality()
                if not quality_result or not self._evaluate_image_quality_results(
                    quality_result
                ):
                    return False
            elif stage == AnalysisStage.FACE:
                face_result = image.results.get_face()
                if not face_result or not self._evaluate_face_detection_results(
                    face_result
                ):
                    return False
            elif stage == AnalysisStage.DEDUPLICATION:
                dedup_result = image.results.get_deduplication()
                if not dedup_result or not self._evaluate_duplicate_detection_results(
                    image, all_images
                ):
                    return False
        return True

    def _evaluate_image_quality_results(
        self, quality_result: ImageQualityResult
    ) -> bool:
        """Check if quality result meets quality thresholds.

        Args:
            quality_result: Quality analysis result

        Returns:
            True if quality passes thresholds, False otherwise
        """
        config = self._config_manager.get_stage_config(AnalysisStage.QUALITY)

        # Check individual quality metrics against thresholds
        if quality_result.sharpness_score < config.sharpness_threshold:
            return False
        if not (
            config.brightness_min
            <= quality_result.brightness_score
            <= config.brightness_max
        ):
            return False
        if quality_result.contrast_score < config.contrast_threshold:
            return False

        return True

    def _evaluate_face_detection_results(
        self, face_result: FaceDetectionResult
    ) -> bool:
        """Check if face result meets face detection requirements.

        Args:
            face_result: Face detection result

        Returns:
            True if face detection passes requirements, False otherwise
        """
        config = self._config_manager.get_stage_config(AnalysisStage.FACE)

        # Check if we have faces and they meet confidence threshold
        if face_result.face_count == 0:
            return False

        # Check if highest confidence face meets threshold
        if face_result.highest_confidence < config.confidence_threshold:
            return False

        return True

    def _calculate_face_size(
        self, face_bounding_box: tuple[float, float, float, float]
    ) -> tuple[int, int]:
        """Calculate face size from bounding box.

        Args:
            face_bounding_box: Bounding box coordinates (x1, y1, x2, y2)

        Returns:
            Tuple of (width, height) in pixels
        """
        x1, y1, x2, y2 = face_bounding_box
        width = int(x2 - x1)
        height = int(y2 - y1)
        return width, height

    def _calculate_face_percentage(
        self,
        face_bounding_box: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> float:
        """Calculate what percentage of the image the face occupies.

        Args:
            face_bounding_box: Bounding box coordinates (x1, y1, x2, y2)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Percentage of image area occupied by the face (0.0 to 1.0)
        """
        face_width, face_height = self._calculate_face_size(face_bounding_box)
        face_area = face_width * face_height
        image_area = image_width * image_height
        return face_area / image_area if image_area > 0 else 0.0

    def _evaluate_duplicate_detection_results(
        self, image: ImageAnalysis, all_images: list[ImageAnalysis]
    ) -> bool:
        """Check if an image should be kept based on deduplication logic.

        For duplicates, keeps only the image with the highest composite score that
        considers quality, face confidence, face size, and face-to-image ratio.

        Args:
            image: Image to evaluate
            all_images: All images in the dataset

        Returns:
            True if the image should be kept, False if it's a lower-quality duplicate
        """
        config = self._config_manager.get_stage_config(AnalysisStage.DEDUPLICATION)
        if not config or not config.enabled:
            return True  # Pass if deduplication is disabled

        dedup_result = image.results.get_deduplication()
        if not dedup_result or not dedup_result.hash_value:
            return False  # No hash data, can't evaluate

        current_hash = dedup_result.hash_value
        current_score = image.score

        # Find all images similar to this one
        for other_image in all_images:
            if other_image is image:  # Skip self
                continue

            other_dedup_result = other_image.results.get_deduplication()
            if not other_dedup_result or not other_dedup_result.hash_value:
                continue

            other_hash = other_dedup_result.hash_value

            # Calculate Hamming distance between hashes
            try:
                hash1_int = int(current_hash, 16)
                hash2_int = int(other_hash, 16)
                hamming_distance = bin(hash1_int ^ hash2_int).count("1")

                if hamming_distance <= config.threshold:
                    # Images are duplicates - check composite score
                    other_score = other_image.score

                    if other_score > current_score:
                        return False  # Other image has better overall score
                    elif other_score == current_score:
                        # Same score - use stable ordering (e.g., by filename)
                        if (
                            Path(other_image.file_path).name
                            < Path(image.file_path).name
                        ):
                            return False  # Other image comes first alphabetically

            except (ValueError, TypeError):
                # Invalid hash format, skip comparison
                continue

        return True  # Keep this image (no better duplicates found)
