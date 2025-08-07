"""Face analysis service using InsightFace."""

import contextlib
import io
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from culora.core.exceptions import CuLoRAError
from culora.domain import (
    BatchFaceAnalysisResult,
    CuLoRAConfig,
    FaceAnalysisResult,
    FaceDetection,
    ImageLoadResult,
)
from culora.services.device_service import get_device_service


class FaceAnalysisServiceError(CuLoRAError):
    """Face analysis service specific errors."""

    pass


class FaceAnalysisService:
    """Service for face detection and analysis using InsightFace.

    Provides comprehensive face analysis capabilities including detection,
    embedding extraction, and batch processing with device optimization.
    """

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize face analysis service.

        Args:
            config: Application configuration
        """
        self.config = config
        self.face_config = config.faces
        self._model: Any | None = None
        self._device_context: dict[str, Any] | None = None

    def _initialize_model(self) -> None:
        """Initialize InsightFace model with device optimization.

        Raises:
            FaceAnalysisServiceError: If model initialization fails
        """
        if self._model is not None:
            return

        try:
            # Import InsightFace here to allow for optional dependency
            import insightface

            # Get device information from DeviceService
            device_service = get_device_service()
            device_info = device_service.get_selected_device()

            # Determine execution providers based on device
            if self.face_config.device_preference == "auto":
                if device_info.device_type.value == "cuda":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                elif device_info.device_type.value == "mps":
                    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
            else:
                providers = self.face_config.get_device_context_providers()

            # Initialize and prepare the model with suppressed output
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                # Initialize the model
                self._model = insightface.app.FaceAnalysis(
                    name=self.face_config.model_name,
                    root=str(self.face_config.model_cache_dir),
                    providers=providers,
                )

                # Prepare the model with context size
                # Use a reasonable context size for face detection
                ctx_id = 0 if "CUDA" in providers[0] else -1
                if self._model is not None:
                    self._model.prepare(ctx_id=ctx_id, det_size=(640, 640))

            self._device_context = {
                "providers": providers,
                "device_type": device_info.device_type.value,
                "ctx_id": ctx_id,
            }

        except ImportError as e:
            raise FaceAnalysisServiceError(
                "InsightFace not available. Install with: pip install insightface"
            ) from e
        except Exception as e:
            raise FaceAnalysisServiceError(
                f"Failed to initialize InsightFace model: {e}"
            ) from e

    def analyze_image(self, image_result: ImageLoadResult) -> FaceAnalysisResult:
        """Analyze faces in a single image.

        Args:
            image_result: Result from ImageService containing loaded image

        Returns:
            Face analysis result with detected faces and metadata
        """
        start_time = time.time()

        # Check if image loading was successful
        if not image_result.success or image_result.image is None:
            return FaceAnalysisResult(
                image_path=image_result.metadata.path,
                success=False,
                faces=[],
                processing_duration=time.time() - start_time,
                processed_at=datetime.now(),
                image_width=image_result.metadata.width,
                image_height=image_result.metadata.height,
                error=f"Image loading failed: {image_result.error}",
                error_code="IMAGE_LOAD_FAILED",
            )

        try:
            # Initialize model if needed
            self._initialize_model()

            # Convert PIL image to numpy array (RGB format)
            img_array = np.array(image_result.image)

            # InsightFace expects BGR format
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1]  # RGB to BGR

            # Perform face detection and analysis
            if self._model is None:
                raise FaceAnalysisServiceError("Model not initialized")
            faces_data = self._model.get(img_array)

            # Process detected faces
            detected_faces = []
            image_area = image_result.metadata.width * image_result.metadata.height

            for face_data in faces_data:
                # Extract face information
                bbox = face_data.bbox.astype(float)  # [x1, y1, x2, y2]
                confidence = float(face_data.det_score)

                # Skip faces below confidence threshold
                if confidence < self.face_config.confidence_threshold:
                    continue

                # Calculate face area ratio
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height
                face_area_ratio = face_area / image_area if image_area > 0 else 0.0

                # Extract features based on configuration
                landmarks = None
                if self.face_config.extract_landmarks and hasattr(face_data, "kps"):
                    landmarks = face_data.kps.astype(float)

                embedding = None
                if self.face_config.extract_embeddings and hasattr(
                    face_data, "embedding"
                ):
                    embedding = face_data.embedding.astype(float)
                    if self.face_config.normalize_embeddings:
                        # Normalize to unit length
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                # Extract attributes if available and requested
                age = None
                gender = None
                if self.face_config.extract_attributes:
                    if hasattr(face_data, "age"):
                        age = int(face_data.age)
                    if hasattr(face_data, "gender"):
                        gender = "male" if face_data.gender == 1 else "female"

                # Create face detection object
                face_detection = FaceDetection(
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    confidence=confidence,
                    landmarks=landmarks,
                    embedding=embedding,
                    face_area_ratio=face_area_ratio,
                    age=age,
                    gender=gender,
                )

                detected_faces.append(face_detection)

                # Limit number of faces if configured
                if len(detected_faces) >= self.face_config.max_faces_per_image:
                    break

            processing_duration = time.time() - start_time

            return FaceAnalysisResult(
                image_path=image_result.metadata.path,
                success=True,
                faces=detected_faces,
                processing_duration=processing_duration,
                processed_at=datetime.now(),
                image_width=image_result.metadata.width,
                image_height=image_result.metadata.height,
            )

        except Exception as e:
            processing_duration = time.time() - start_time

            return FaceAnalysisResult(
                image_path=image_result.metadata.path,
                success=False,
                faces=[],
                processing_duration=processing_duration,
                processed_at=datetime.now(),
                image_width=image_result.metadata.width,
                image_height=image_result.metadata.height,
                error=str(e),
                error_code="ANALYSIS_FAILED",
            )

    def analyze_batch(
        self, image_results: list[ImageLoadResult]
    ) -> BatchFaceAnalysisResult:
        """Analyze faces in a batch of images.

        Args:
            image_results: List of image load results from ImageService

        Returns:
            Batch face analysis result with aggregated statistics
        """
        start_time = time.time()
        results = []

        # Process each image
        for image_result in image_results:
            face_result = self.analyze_image(image_result)
            results.append(face_result)

        # Calculate aggregated statistics
        successful_analyses = sum(1 for r in results if r.success)
        failed_analyses = len(results) - successful_analyses

        total_faces_detected = sum(len(r.faces) for r in results if r.success)
        images_with_faces = sum(1 for r in results if r.success and r.has_faces)
        images_without_faces = successful_analyses - images_with_faces

        processing_duration = time.time() - start_time

        return BatchFaceAnalysisResult(
            results=results,
            processing_duration=processing_duration,
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            total_faces_detected=total_faces_detected,
            images_with_faces=images_with_faces,
            images_without_faces=images_without_faces,
        )

    def analyze_directory_batch(
        self, directory: Path, batch_size: int | None = None
    ) -> Generator[BatchFaceAnalysisResult, None, None]:
        """Analyze faces in all images in a directory using batch processing.

        Args:
            directory: Directory containing images to analyze
            batch_size: Optional batch size override

        Yields:
            Batch face analysis results for each batch of images
        """
        # Import ImageService here to avoid circular imports
        from culora.services.image_service import get_image_service

        # Use configured batch size if not provided
        if batch_size is None:
            # Use configured batch size
            batch_size = self.face_config.batch_size

        image_service = get_image_service()

        # Process directory in batches
        for batch_result in image_service.load_directory_batch(directory):
            # Analyze faces in this batch
            face_batch_result = self.analyze_batch(batch_result.results)
            yield face_batch_result

    def analyze_with_reference(
        self,
        image_result: ImageLoadResult,
        reference_service: Any | None = None,
        reference_set: Any | None = None,
    ) -> FaceAnalysisResult:
        """Analyze faces in image with optional reference matching for primary face selection.

        Args:
            image_result: Result from ImageService containing loaded image
            reference_service: Optional FaceReferenceService instance for dependency injection
            reference_set: Optional ReferenceSet for primary face selection

        Returns:
            Face analysis result with primary face selected based on reference matching
        """
        # Perform standard face analysis first
        face_result = self.analyze_image(image_result)

        # If analysis failed or no reference set provided, return as-is
        if not face_result.success or reference_set is None or not face_result.faces:
            return face_result

        # Use injected reference service for primary face selection
        if reference_service is not None:
            primary_face = reference_service.select_primary_face(
                face_result.faces, reference_set
            )

            # If we found a primary face, reorder the faces list to put it first
            if primary_face is not None:
                faces_reordered = [primary_face]
                faces_reordered.extend(
                    [f for f in face_result.faces if f != primary_face]
                )

                # Create new result with reordered faces
                return FaceAnalysisResult(
                    image_path=face_result.image_path,
                    success=face_result.success,
                    faces=faces_reordered,
                    processing_duration=face_result.processing_duration,
                    processed_at=face_result.processed_at,
                    image_width=face_result.image_width,
                    image_height=face_result.image_height,
                    error=face_result.error,
                    error_code=face_result.error_code,
                )

        return face_result

    def get_model_info(self) -> dict[str, str]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self._model is None:
            return {
                "status": "not_initialized",
                "model_name": self.face_config.model_name,
                "cache_dir": str(self.face_config.model_cache_dir),
            }

        device_context_str = (
            str(self._device_context) if self._device_context else "unknown"
        )
        return {
            "status": "initialized",
            "model_name": self.face_config.model_name,
            "cache_dir": str(self.face_config.model_cache_dir),
            "device_context": device_context_str,
            "confidence_threshold": str(self.face_config.confidence_threshold),
            "max_faces_per_image": str(self.face_config.max_faces_per_image),
        }


# Global service instance
_face_analysis_service: FaceAnalysisService | None = None


def initialize_face_analysis_service(config: CuLoRAConfig) -> FaceAnalysisService:
    """Initialize global FaceAnalysisService instance.

    Args:
        config: Application configuration

    Returns:
        Initialized FaceAnalysisService instance
    """
    global _face_analysis_service
    _face_analysis_service = FaceAnalysisService(config)
    return _face_analysis_service


def get_face_analysis_service() -> FaceAnalysisService:
    """Get global FaceAnalysisService instance.

    Returns:
        Global FaceAnalysisService instance
    """
    global _face_analysis_service
    if _face_analysis_service is None:
        from culora.services import get_config_service

        config_service = get_config_service()

        # Load default config if not already loaded
        try:
            config = config_service.get_config()
        except Exception:
            config = config_service.load_config()

        _face_analysis_service = FaceAnalysisService(config)

    return _face_analysis_service
