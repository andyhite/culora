"""Composition analysis service using vision-language models."""

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from culora.core import CuLoRAError
from culora.domain import CuLoRAConfig
from culora.domain.enums.composition import (
    BackgroundComplexity,
    CameraAngle,
    FacialExpression,
    LightingQuality,
    SceneType,
    ShotType,
)
from culora.domain.models.composition import (
    BatchCompositionResult,
    CompositionAnalysis,
    CompositionResult,
)
from culora.domain.models.config.composition import COMPOSITION_ANALYSIS_PROMPT
from culora.services.device_service import get_device_service
from culora.utils import get_logger

logger = get_logger(__name__)


class CompositionServiceError(CuLoRAError):
    """Base exception for composition service errors."""

    def __init__(
        self, message: str, error_code: str = "COMPOSITION_SERVICE_ERROR"
    ) -> None:
        super().__init__(message, error_code)


class CompositionAnalysisError(CompositionServiceError):
    """Exception for composition analysis failures."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message, "COMPOSITION_ANALYSIS_ERROR")
        self.path = path


class CompositionService:
    """Service for composition analysis using vision-language models.

    Provides comprehensive composition analysis including:
    - Shot type classification (closeup, medium shot, full body, etc.)
    - Scene type analysis (indoor, outdoor, studio, etc.)
    - Lighting quality assessment
    - Background complexity analysis
    - Facial expression recognition
    - Camera angle detection
    """

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize composition service with configuration.

        Args:
            config: CuLoRA configuration containing composition settings
        """
        self.config = config
        self.composition_config = config.composition
        self.device_service = get_device_service()

        # Model components (loaded on first use)
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: torch.device | None = None

        logger.info("Composition service initialized")

    def analyze_image(self, image: Image.Image, path: Path) -> CompositionResult:
        """Analyze composition for a single image.

        Args:
            image: PIL Image to analyze
            path: Path to the image file for error reporting

        Returns:
            Complete composition analysis result
        """
        start_time = time.time()

        try:
            # Ensure model is loaded
            self._ensure_model_loaded()

            # Prepare image for analysis
            prepared_image = self._prepare_image_for_analysis(image)

            # Generate composition analysis
            response = self._generate_analysis(prepared_image)

            # Parse response into structured analysis
            analysis = self._parse_response(response)

            duration = time.time() - start_time

            return CompositionResult(
                path=path,
                success=True,
                analysis=analysis,
                analysis_duration=duration,
                model_response=response,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Composition analysis failed for {path}: {e}"
            logger.warning(error_msg)

            return CompositionResult(
                path=path,
                success=False,
                error=error_msg,
                error_code="COMPOSITION_ANALYSIS_FAILED",
                analysis_duration=duration,
            )

    def analyze_batch(
        self, images_and_paths: list[tuple[Image.Image, Path]]
    ) -> BatchCompositionResult:
        """Analyze composition for a batch of images.

        Args:
            images_and_paths: List of (image, path) tuples to analyze

        Returns:
            Batch analysis results with statistics
        """
        start_time = time.time()
        results: list[CompositionResult] = []

        logger.info(f"Starting composition analysis for {len(images_and_paths)} images")

        # Analyze individual images
        for image, path in images_and_paths:
            result = self.analyze_image(image, path)
            results.append(result)

        # Calculate batch statistics
        total_duration = time.time() - start_time
        successful_results = [
            r for r in results if r.success and r.analysis is not None
        ]

        batch_result = self._calculate_batch_statistics(
            results, successful_results, total_duration
        )

        logger.info(
            f"Composition analysis completed: {batch_result.successful_analyses}/"
            f"{len(results)} successful"
        )

        return batch_result

    def _ensure_model_loaded(self) -> None:
        """Ensure vision-language model is loaded and ready."""
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(
            f"Loading vision-language model: {self.composition_config.model_name}"
        )

        try:
            # Get optimal device
            device_info = self.device_service.get_selected_device()
            device_str = device_info.device_type.value
            if device_str == "cuda":
                self._device = torch.device("cuda")
            elif device_str == "mps":
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.composition_config.model_name,
                cache_dir=str(self.composition_config.model_cache_dir),
                trust_remote_code=True,
            )

            # Load model with device-specific settings
            # Use float16 for CUDA, float32 for others
            use_fp16 = device_str == "cuda"
            model_kwargs = {
                "cache_dir": str(self.composition_config.model_cache_dir),
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            }

            self._model = AutoModelForCausalLM.from_pretrained(
                self.composition_config.model_name, **model_kwargs
            )

            # Move model to device and set to evaluation mode
            if self._model is not None and self._device is not None:
                self._model = self._model.to(self._device)  # type: ignore[union-attr]
                self._model.eval()  # type: ignore[union-attr]

            logger.info(f"Model loaded successfully on {self._device}")

        except Exception as e:
            raise CompositionServiceError(
                f"Failed to load vision-language model: {e}"
            ) from e

    def _prepare_image_for_analysis(self, image: Image.Image) -> Image.Image:
        """Prepare image for composition analysis.

        Args:
            image: Original PIL Image

        Returns:
            Prepared image for model input
        """
        # Resize if needed
        max_width, max_height = self.composition_config.max_image_size
        width, height = image.size

        if width <= max_width and height <= max_height:
            return image

        # Calculate aspect-preserving resize
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _generate_analysis(self, image: Image.Image) -> str:
        """Generate composition analysis using vision-language model.

        Args:
            image: Prepared PIL Image

        Returns:
            Raw model response
        """
        if self._model is None or self._tokenizer is None:
            raise CompositionServiceError("Model not loaded")

        try:
            # Create prompt
            prompt = COMPOSITION_ANALYSIS_PROMPT

            # For Moondream, we need to use the specific API
            if hasattr(self._model, "encode_image") and hasattr(
                self._model, "answer_question"
            ):
                # Encode the image first
                encoded = self._model.encode_image(image)
                # Get the analysis response
                response = self._model.answer_question(encoded, prompt, self._tokenizer)
                return str(response)
            else:
                # Fallback for other vision-language models
                raise CompositionServiceError(
                    f"Unsupported model type: {type(self._model)}. "
                    "Model must support encode_image and answer_question methods."
                )

        except Exception as e:
            raise CompositionAnalysisError(f"Failed to generate analysis: {e}") from e

    def _parse_response(self, response: str) -> CompositionAnalysis:
        """Parse model response into structured composition analysis.

        Args:
            response: Raw model response

        Returns:
            Parsed composition analysis

        Raises:
            CompositionAnalysisError: If response cannot be parsed
        """
        try:
            # Try to extract JSON from response using proper brace matching
            data = self._extract_json_from_response(response)
            if data is None:
                # Fallback parsing if no JSON found
                data = self._fallback_parse_response(response)

            # Extract and validate fields
            shot_type = self._parse_enum_field(
                data.get("shot_type"), ShotType, ShotType.UNKNOWN
            )
            scene_type = self._parse_enum_field(
                data.get("scene_type"), SceneType, SceneType.UNKNOWN
            )
            lighting_quality = self._parse_enum_field(
                data.get("lighting_quality"), LightingQuality, LightingQuality.UNKNOWN
            )
            background_complexity = self._parse_enum_field(
                data.get("background_complexity"),
                BackgroundComplexity,
                BackgroundComplexity.UNKNOWN,
            )

            # Optional fields
            facial_expression = self._parse_enum_field(
                data.get("facial_expression"), FacialExpression, None
            )
            camera_angle = self._parse_enum_field(
                data.get("camera_angle"), CameraAngle, None
            )

            # Extract confidence and description
            confidence_score = data.get("confidence")
            if confidence_score is not None:
                confidence_score = float(confidence_score)
                confidence_score = max(0.0, min(1.0, confidence_score))

            raw_description = data.get("description", response[:200])

            return CompositionAnalysis(
                shot_type=shot_type,
                scene_type=scene_type,
                lighting_quality=lighting_quality,
                background_complexity=background_complexity,
                facial_expression=facial_expression,
                camera_angle=camera_angle,
                confidence_score=confidence_score,
                raw_description=raw_description,
            )

        except Exception as e:
            raise CompositionAnalysisError(f"Failed to parse response: {e}") from e

    def _extract_json_from_response(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from model response using proper brace matching.

        Args:
            response: Raw model response

        Returns:
            Parsed JSON data or None if no valid JSON found
        """
        # Find the first opening brace
        start_idx = response.find("{")
        if start_idx == -1:
            return None

        # Count braces to find the matching closing brace
        brace_count = 0
        for i, char in enumerate(response[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    json_str = response[start_idx : i + 1]
                    try:
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                        return None
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}")
                        logger.debug(f"Attempted to parse: {json_str}")
                        return None

        # No matching closing brace found
        logger.debug("No matching closing brace found in response")
        return None

    def _parse_enum_field(self, value: Any, enum_class: type[Any], default: Any) -> Any:
        """Parse and validate enum field from response.

        Args:
            value: Raw value from response
            enum_class: Enum class to validate against
            default: Default value if parsing fails

        Returns:
            Valid enum value or default
        """
        if value is None:
            return default

        if isinstance(value, str):
            # Try to find matching enum value
            for enum_value in enum_class:
                if enum_value.value.lower() == value.lower():
                    return enum_value
                if enum_value.name.lower() == value.lower():
                    return enum_value

        return default

    def _fallback_parse_response(self, response: str) -> dict[str, Any]:
        """Fallback parsing when JSON extraction fails.

        Args:
            response: Raw model response

        Returns:
            Parsed data dictionary
        """
        # Simple keyword-based extraction
        data: dict[str, Any] = {}

        # Look for common patterns
        response_lower = response.lower()
        if "closeup" in response_lower:
            data["shot_type"] = "closeup"
        elif "medium" in response_lower:
            data["shot_type"] = "medium_shot"
        elif "full body" in response_lower:
            data["shot_type"] = "full_body"

        if "outdoor" in response_lower:
            data["scene_type"] = "outdoor"
        elif "indoor" in response_lower:
            data["scene_type"] = "indoor"
        elif "studio" in response_lower:
            data["scene_type"] = "studio"

        # Set default confidence for fallback parsing
        data["confidence"] = 0.5
        data["description"] = response[:100]

        return data

    def _calculate_batch_statistics(
        self,
        all_results: list[CompositionResult],
        successful_results: list[CompositionResult],
        total_duration: float,
    ) -> BatchCompositionResult:
        """Calculate statistics for batch composition analysis.

        Args:
            all_results: All analysis results including failures
            successful_results: Only successful analysis results
            total_duration: Total processing time

        Returns:
            Batch statistics and results
        """
        successful_count = len(successful_results)
        failed_count = len(all_results) - successful_count

        # Initialize distribution counters
        shot_type_dist: dict[ShotType, int] = {}
        scene_type_dist: dict[SceneType, int] = {}
        lighting_dist: dict[LightingQuality, int] = {}
        background_dist: dict[BackgroundComplexity, int] = {}
        expression_dist: dict[FacialExpression, int] = {}
        angle_dist: dict[CameraAngle, int] = {}

        confidence_scores = []

        # Calculate distributions
        for result in successful_results:
            if result.analysis is None:
                continue

            analysis = result.analysis

            # Count distributions
            shot_type_dist[analysis.shot_type] = (
                shot_type_dist.get(analysis.shot_type, 0) + 1
            )
            scene_type_dist[analysis.scene_type] = (
                scene_type_dist.get(analysis.scene_type, 0) + 1
            )
            lighting_dist[analysis.lighting_quality] = (
                lighting_dist.get(analysis.lighting_quality, 0) + 1
            )
            background_dist[analysis.background_complexity] = (
                background_dist.get(analysis.background_complexity, 0) + 1
            )

            if analysis.facial_expression:
                expression_dist[analysis.facial_expression] = (
                    expression_dist.get(analysis.facial_expression, 0) + 1
                )

            if analysis.camera_angle:
                angle_dist[analysis.camera_angle] = (
                    angle_dist.get(analysis.camera_angle, 0) + 1
                )

            if analysis.confidence_score is not None:
                confidence_scores.append(analysis.confidence_score)

        # Calculate confidence statistics
        mean_confidence = (
            statistics.mean(confidence_scores) if confidence_scores else 0.0
        )
        confidence_distribution = {}
        if confidence_scores:
            sorted_scores = sorted(confidence_scores)
            for percentile in [25, 50, 75, 90, 95]:
                idx = int((percentile / 100.0) * (len(sorted_scores) - 1))
                confidence_distribution[f"p{percentile}"] = sorted_scores[idx]

        # Calculate processing rate
        images_per_second = (
            len(all_results) / total_duration if total_duration > 0 else 0.0
        )

        return BatchCompositionResult(
            results=all_results,
            successful_analyses=successful_count,
            failed_analyses=failed_count,
            total_duration=total_duration,
            images_per_second=images_per_second,
            shot_type_distribution=shot_type_dist,
            scene_type_distribution=scene_type_dist,
            lighting_distribution=lighting_dist,
            background_distribution=background_dist,
            expression_distribution=expression_dist,
            angle_distribution=angle_dist,
            mean_confidence=mean_confidence,
            confidence_distribution=confidence_distribution,
        )


# Global service instance
_composition_service: CompositionService | None = None


def get_composition_service(config: CuLoRAConfig | None = None) -> CompositionService:
    """Get or create composition service instance.

    Args:
        config: Configuration to use for service creation

    Returns:
        Composition service instance
    """
    global _composition_service

    if _composition_service is None:
        if config is None:
            from culora.services.config_service import get_config_service

            config = get_config_service().get_config()
        _composition_service = CompositionService(config)

    return _composition_service
