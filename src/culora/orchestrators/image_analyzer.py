"""Image analyzer orchestrator for CuLoRA."""

from datetime import datetime
from pathlib import Path

from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.managers.image_manager import ImageManager
from culora.models.directory_analysis import DirectoryAnalysis
from culora.models.image_analysis import ImageAnalysis
from culora.services.duplicate_detection_service import DuplicateDetectionService
from culora.services.face_detection_service import FaceDetectionService
from culora.services.image_quality_service import ImageQualityService
from culora.utils.console import get_console

console = get_console()


class ImageAnalyzer:
    """Orchestrator for analyzing images using analysis services."""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        image_manager: ImageManager | None = None,
        quality_service: ImageQualityService | None = None,
        face_service: FaceDetectionService | None = None,
        deduplication_service: DuplicateDetectionService | None = None,
    ) -> None:
        """Initialize the image analyzer.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
            image_manager: Image manager instance. If None, uses singleton.
            image_quality_service: Image quality service. If None, creates new instance.
            face_service: Face detection service. If None, creates new instance.
            deduplication_service: Deduplication service. If None, creates new instance.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()
        self._image_manager = image_manager or ImageManager.get_instance()
        self._image_quality_service = quality_service or ImageQualityService(
            self._config_manager
        )
        self._face_detection_service = face_service or FaceDetectionService(
            self._config_manager
        )
        self._deduplication_service = (
            deduplication_service or DuplicateDetectionService(self._config_manager)
        )

    def analyze_directory(self, input_directory: Path) -> DirectoryAnalysis:
        """Analyze all images in a directory.

        Args:
            input_directory: Directory containing images to analyze.

        Returns:
            Analysis results for the directory.

        Raises:
            FileNotFoundError: If input directory doesn't exist.
            NotADirectoryError: If input path is not a directory.
        """
        # Validate directory
        self._image_manager.validate_directory(input_directory)

        # Find images
        image_paths = list(
            self._image_manager.find_images_in_directory(input_directory)
        )

        # Handle empty directory case
        if not image_paths:
            return DirectoryAnalysis(
                input_directory=str(input_directory.resolve()),
                analysis_time=datetime.now(),
                analysis_config=self._config_manager.config,
                images=[],
            )

        # Analyze images using enabled analysis services
        analyzed_images: list[ImageAnalysis] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console.rich_console,
            transient=True,
        ) as progress:
            # Initialize all images with metadata
            for image_path in image_paths:
                metadata = self._image_manager.get_image_metadata(image_path)
                analyzed_images.append(
                    ImageAnalysis(
                        file_path=metadata["file_path"],
                        file_size=metadata["file_size"],
                        modified_time=metadata["modified_time"],
                    )
                )

            # Analyze all images with enabled services
            task = progress.add_task(
                f"Analyzing images in: {input_directory.name}",
                total=len(image_paths),
            )
            self._analyze_images(analyzed_images, progress, task)

        return DirectoryAnalysis(
            input_directory=str(input_directory.resolve()),
            analysis_time=datetime.now(),
            analysis_config=self._config_manager.config,
            images=analyzed_images,
        )

    def _analyze_images(
        self,
        analyzed_images: list[ImageAnalysis],
        progress: Progress,
        task: TaskID,
    ) -> None:
        """Analyze all images using enabled services.

        Args:
            analyzed_images: List of image analysis objects to update
            progress: Rich progress bar instance
            task: Progress task to update
        """
        enabled_stages = [
            stage
            for stage in AnalysisStage
            if getattr(self._config_manager.config, stage.value).enabled
        ]

        for idx, image_analysis in enumerate(analyzed_images):
            image_path = Path(image_analysis.file_path)
            progress.update(
                task,
                description=f"Analyzing: {image_path.name} ({idx + 1}/{len(analyzed_images)})",
                completed=idx,
            )

            # Load image once and pass to all services
            try:
                with self._image_manager.load_image(image_path) as loaded_image:
                    for stage in enabled_stages:
                        service = self._get_service_for_stage(stage)
                        if service is None:
                            continue

                        # Run analysis for this stage with loaded image
                        result = service.analyze_image(loaded_image)
                        image_analysis.results.set(stage, result)

                    # Calculate and set composite score after all analysis is complete
                    image_analysis.score = self._calculate_image_score(
                        image_analysis, loaded_image
                    )

            except (FileNotFoundError, ValueError):
                console.warning(
                    f"Failed to load image {image_path.name}, skipping analysis."
                )

            progress.update(task, completed=idx + 1)

    def _get_service_for_stage(
        self, stage: AnalysisStage
    ) -> ImageQualityService | FaceDetectionService | DuplicateDetectionService | None:
        """Get the appropriate service for an analysis stage.

        Args:
            stage: The analysis stage

        Returns:
            The service instance for the stage, or None if unknown
        """
        if stage == AnalysisStage.QUALITY:
            return self._image_quality_service
        elif stage == AnalysisStage.FACE:
            return self._face_detection_service
        elif stage == AnalysisStage.DEDUPLICATION:
            return self._deduplication_service
        return None

    def _calculate_image_score(
        self, image: ImageAnalysis, loaded_image: Image.Image
    ) -> float:
        """Calculate a composite score for an image to rank selection candidates.

        Improved scoring with relative face sizing, confidence gating, face count
        penalties, and better weight distribution.

        Args:
            image: Image analysis to score
            loaded_image: PIL Image object for dimensions

        Returns:
            Composite score (0.0 to 1.0, higher is better)
        """
        score = 0.0
        image_width, image_height = loaded_image.size
        image_area = image_width * image_height

        # Get scoring configuration
        scoring_config = self._config_manager.config.scoring

        # Quality component
        quality_result = image.results.get_quality()
        if quality_result:
            # Normalize quality score to 0-1 range (assuming max of 100)
            quality_component = min(quality_result.composite_score / 100.0, 1.0)
            score += quality_component * scoring_config.quality_weight

        # Face component
        face_result = image.results.get_face()
        if face_result and face_result.face_count > 0:
            # Get best face by confidence
            best_face = max(face_result.faces, key=lambda f: f.confidence)
            face_confidence = best_face.confidence

            # Calculate face area as percentage of image
            x1, y1, x2, y2 = best_face.bounding_box
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            face_area_ratio = face_area / image_area if image_area > 0 else 0.0

            # Face area scoring with configurable sigmoid-like curve
            face_min = scoring_config.face_area_min
            face_peak = scoring_config.face_area_peak
            face_max = scoring_config.face_area_max

            if face_area_ratio < face_min:
                # Too small - linear up to minimum
                area_score = face_area_ratio / face_min
            elif face_area_ratio <= face_peak:
                # Sweet spot - linear growth to peak
                area_score = (
                    0.5 + (face_area_ratio - face_min) / (face_peak - face_min) * 0.5
                )
            elif face_area_ratio <= face_max:
                # Good but declining
                decline_factor = (face_area_ratio - face_peak) / (face_max - face_peak)
                area_score = 1.0 - decline_factor * 0.2  # Declines to 0.8 at max
            else:
                # Too large - continues declining
                excess_factor = (face_area_ratio - face_max) / face_max
                area_score = max(0.2, 0.8 - excess_factor * 2.0)

            area_score = max(0.0, min(1.0, area_score))

            # Face confidence gates the area score (confidence as multiplier)
            face_component = face_confidence * area_score

            # Face count penalty using configurable values
            if face_result.face_count > 1:
                penalty_factor = max(
                    1.0 - scoring_config.max_face_penalty,
                    1.0
                    - (face_result.face_count - 1) * scoring_config.multi_face_penalty,
                )
                face_component *= penalty_factor

            score += face_component * scoring_config.face_weight

        # Clamp final score to [0.0, 1.0]
        return max(0.0, min(1.0, score))
