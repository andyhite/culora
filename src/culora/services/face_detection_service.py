"""Face detection service for CuLoRA."""

from typing import Any

from PIL import Image

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.managers.model_manager import ModelManager
from culora.models.face_detection_result import Face, FaceDetectionResult

# FaceConfig accessed via ConfigManager


class FaceDetectionService:
    """Service for detecting faces using specialized YOLO11 face model."""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        """Initialize the face detection service.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
            model_manager: Model manager instance. If None, uses singleton.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()
        self._model_manager = model_manager or ModelManager.get_instance()

    def analyze_image(self, image: Image.Image) -> FaceDetectionResult:
        """Analyze image for face detection using specialized YOLO11 face model.

        Uses AdamCodd/YOLOv11n-face-detection model for direct face detection.
        Based on research documented in docs/analysis-libraries.md.

        Args:
            image: PIL Image object.

        Returns:
            FaceResult with detection data.
        """
        try:
            config = self._config_manager.get_stage_config(AnalysisStage.FACE)

            # Extract configuration parameters
            confidence_threshold = config.confidence_threshold
            max_detections = config.max_detections
            iou_threshold = config.iou_threshold
            use_half_precision = config.use_half_precision
            device_setting = config.device

            # Determine device to use
            if device_setting == "auto":
                device = self._model_manager.detect_optimal_device()
            else:
                device = device_setting

            # Get face detection model identifier and load cached model
            model_identifier = f"{config.model_repo}:{config.model_filename}"
            model = self._model_manager.get_cached_model(
                "face_detection", model_identifier
            )

            # Run inference on the PIL image with optimized parameters
            results: Any = model(  # pyright: ignore[reportUnknownVariableType]
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                device=device,
                half=use_half_precision,
                verbose=False,
            )

            if not results:
                return FaceDetectionResult(
                    faces=[],
                    model_used=model_identifier,
                    device_used=device,
                )

            # Extract detections from the first (and only) result
            result = results[0]  # type: ignore[misc]

            # Extract all face detections (specialized face model detects faces directly)
            faces: list[Face] = []
            if hasattr(result, "boxes") and result.boxes is not None:  # type: ignore[misc]
                for box in result.boxes:  # type: ignore[misc]
                    # Face detection model returns faces directly - no class filtering needed
                    if hasattr(box, "conf") and hasattr(box, "xyxy"):  # type: ignore[misc]
                        conf_tensor = box.conf  # type: ignore[misc]
                        xyxy_tensor = box.xyxy  # type: ignore[misc]
                        if len(conf_tensor) > 0 and len(xyxy_tensor) > 0:  # type: ignore[misc]
                            confidence = float(conf_tensor[0])  # type: ignore[misc]
                            # Extract bounding box coordinates (x1, y1, x2, y2)
                            bbox = xyxy_tensor[0].tolist()  # type: ignore[misc]
                            faces.append(
                                Face(
                                    confidence=confidence,
                                    bounding_box=tuple(bbox),
                                )
                            )

            # Sort faces by confidence (highest first)
            faces.sort(key=lambda face: face.confidence, reverse=True)

            return FaceDetectionResult(
                faces=faces,
                model_used=model_identifier,
                device_used=device,
            )

        except Exception:
            # Return empty result on failure - SelectionService will handle this
            return FaceDetectionResult(
                faces=[],
                model_used="",
                device_used="",
            )
