"""Tests for new service architecture."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image

from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import FaceDetectionResult
from culora.models.image_quality_result import ImageQualityResult
from culora.services.duplicate_detection_service import DuplicateDetectionService
from culora.services.face_detection_service import FaceDetectionService
from culora.services.image_quality_service import ImageQualityService


class TestDuplicateDetectionService:
    """Tests for DuplicateDetectionService."""

    @patch("culora.services.duplicate_detection_service.imagehash.dhash")
    def test_analyze_image_success(self, mock_dhash: Any) -> None:
        """Test successful deduplication analysis."""
        # Mock imagehash
        mock_hash = MagicMock()
        mock_hash.__str__ = Mock(return_value="1234567890abcdef")
        mock_dhash.return_value = mock_hash

        # Create mock PIL image
        mock_pil_image = MagicMock(spec=Image.Image)

        # Test
        service = DuplicateDetectionService()
        result = service.analyze_image(mock_pil_image)

        # Assertions
        assert isinstance(result, DuplicateDetectionResult)
        assert result.hash_value == "1234567890abcdef"


class TestImageQualityService:
    """Tests for ImageQualityService."""

    @patch("culora.services.image_quality_service.cv2.cvtColor")
    @patch("culora.services.image_quality_service.cv2.Laplacian")
    @patch("culora.services.image_quality_service.np.mean")
    @patch("culora.services.image_quality_service.np.std")
    @patch("culora.services.image_quality_service.np.array")
    def test_analyze_image_pass(
        self,
        mock_array: Any,
        mock_std: Any,
        mock_mean: Any,
        mock_laplacian: Any,
        mock_cvt_color: Any,
    ) -> None:
        """Test quality analysis with passing metrics."""
        # Create mock PIL image
        mock_pil_image = MagicMock(spec=Image.Image)

        # Mock OpenCV operations
        mock_rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)

        mock_array.return_value = mock_rgb_array  # PIL to numpy array
        mock_cvt_color.side_effect = [
            mock_bgr_image,
            mock_gray,
        ]  # RGB->BGR then BGR->Gray

        # Mock Laplacian to return high sharpness
        mock_laplacian_result = MagicMock()
        mock_laplacian_result.var.return_value = 200.0  # Above threshold
        mock_laplacian.return_value = mock_laplacian_result

        # Mock mean for good brightness
        mock_mean.return_value = 120.0  # Within range

        # Mock std for good contrast
        mock_std.return_value = 50.0  # Above threshold

        # Test
        service = ImageQualityService()
        result = service.analyze_image(mock_pil_image)

        # Assertions
        assert isinstance(result, ImageQualityResult)
        assert result.sharpness_score == 200.0
        assert result.brightness_score == 120.0
        assert result.contrast_score == 50.0
        # Should have a positive composite score
        assert result.composite_score > 0


class TestFaceDetectionService:
    """Tests for FaceDetectionService."""

    @patch("culora.services.face_detection_service.ModelManager.get_instance")
    def test_analyze_image_faces_detected(self, mock_model_manager: Any) -> None:
        """Test face detection with faces found."""
        # Mock model manager and model
        mock_manager_instance = MagicMock()
        mock_model_manager.return_value = mock_manager_instance
        mock_manager_instance.detect_optimal_device.return_value = "cpu"

        mock_model = MagicMock()
        mock_manager_instance.get_cached_model.return_value = mock_model

        # Mock YOLO results with faces
        mock_box = MagicMock()
        # Mock tensor-like conf attribute
        mock_conf_tensor = MagicMock()
        mock_conf_tensor.__len__ = Mock(return_value=1)
        mock_conf_tensor.__getitem__ = Mock(return_value=0.8)
        mock_box.conf = mock_conf_tensor

        # Mock tensor-like xyxy attribute
        mock_xyxy_tensor = MagicMock()
        mock_xyxy_tensor.__len__ = Mock(return_value=1)
        mock_xyxy_item = MagicMock()
        mock_xyxy_item.tolist = Mock(return_value=[100, 100, 200, 200])
        mock_xyxy_tensor.__getitem__ = Mock(return_value=mock_xyxy_item)
        mock_box.xyxy = mock_xyxy_tensor

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model.return_value = [mock_result]

        # Create mock PIL image
        mock_pil_image = MagicMock(spec=Image.Image)

        # Test
        service = FaceDetectionService()
        result = service.analyze_image(mock_pil_image)

        # Assertions
        assert isinstance(result, FaceDetectionResult)
        assert result.face_count == 1
        assert result.highest_confidence == 0.8
        assert len(result.confidence_scores) == 1
        assert result.confidence_scores[0] == 0.8
        assert len(result.bounding_boxes) == 1
        # Check new faces structure
        assert len(result.faces) == 1
        assert result.faces[0].confidence == 0.8
        assert result.faces[0].bounding_box == (100, 100, 200, 200)
