"""Tests for analyzer module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from culora.analysis.analyzer import (
    analyze_deduplication,
    analyze_face,
    analyze_image,
    analyze_quality,
)
from culora.models.analysis import AnalysisResult, AnalysisStage, StageResult


class TestAnalyzeDeduplication:
    """Tests for analyze_deduplication function."""

    @patch("culora.analysis.analyzer.Image.open")
    @patch("culora.analysis.analyzer.imagehash.dhash")
    def test_analyze_deduplication_success(
        self, mock_dhash: Any, mock_image_open: Any
    ) -> None:
        """Test successful deduplication analysis."""
        # Mock PIL Image and imagehash
        mock_image = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_hash = MagicMock()
        mock_hash.__str__ = Mock(return_value="1234567890abcdef")
        mock_dhash.return_value = mock_hash

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_deduplication(image_path)

        # Assertions
        assert result.stage == AnalysisStage.DEDUPLICATION
        assert result.result == AnalysisResult.PASS
        assert result.reason is not None
        assert "Generated hash: 1234567890abcdef" in result.reason
        assert result.metadata == {
            "hash": "1234567890abcdef",
            "hash_type": "dhash",
            "hash_size": "8",
        }

        # Verify calls
        mock_image_open.assert_called_once_with(image_path)
        mock_dhash.assert_called_once_with(mock_image, hash_size=8)

    @patch("culora.analysis.analyzer.Image.open")
    def test_analyze_deduplication_file_error(self, mock_image_open: Any) -> None:
        """Test deduplication analysis with file access error."""
        # Mock file error
        mock_image_open.side_effect = OSError("Cannot open image")

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_deduplication(image_path)

        # Assertions
        assert result.stage == AnalysisStage.DEDUPLICATION
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "Failed to generate image hash: Cannot open image" in result.reason
        assert result.metadata == {}

    @patch("culora.analysis.analyzer.Image.open")
    @patch("culora.analysis.analyzer.imagehash.dhash")
    def test_analyze_deduplication_hash_error(
        self, mock_dhash: Any, mock_image_open: Any
    ) -> None:
        """Test deduplication analysis with hashing error."""
        # Mock PIL Image but imagehash error
        mock_image = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_dhash.side_effect = Exception("Hash calculation failed")

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_deduplication(image_path)

        # Assertions
        assert result.stage == AnalysisStage.DEDUPLICATION
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "Failed to generate image hash: Hash calculation failed" in result.reason


class TestAnalyzeQuality:
    """Tests for analyze_quality function."""

    @patch("culora.analysis.analyzer.cv2.imread")
    @patch("culora.analysis.analyzer.cv2.cvtColor")
    @patch("culora.analysis.analyzer.cv2.Laplacian")
    @patch("culora.analysis.analyzer.np.mean")
    @patch("culora.analysis.analyzer.np.std")
    def test_analyze_quality_pass_all_metrics(
        self,
        mock_std: Any,
        mock_mean: Any,
        mock_laplacian: Any,
        mock_cvtcolor: Any,
        mock_imread: Any,
    ) -> None:
        """Test quality analysis where all metrics pass."""
        # Setup mocks
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_laplacian_result = MagicMock()
        mock_laplacian_result.var.return_value = 200.0  # Above threshold

        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_gray
        mock_laplacian.return_value = mock_laplacian_result
        mock_mean.return_value = 120.0  # Within range
        mock_std.return_value = 50.0  # Above threshold

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_quality(image_path)

        # Assertions
        assert result.stage == AnalysisStage.QUALITY
        assert result.result == AnalysisResult.PASS
        assert result.reason is not None
        assert "Quality metrics passed" in result.reason
        assert "sharpness: 200.0" in result.reason
        assert "brightness: 120.0" in result.reason
        assert "contrast: 50.0" in result.reason

        # Check metadata
        assert result.metadata["sharpness_laplacian"] == "200.00"
        assert result.metadata["brightness_mean"] == "120.00"
        assert result.metadata["contrast_std"] == "50.00"
        assert result.metadata["sharpness_pass"] == "True"
        assert result.metadata["brightness_pass"] == "True"
        assert result.metadata["contrast_pass"] == "True"

    @patch("culora.analysis.analyzer.cv2.imread")
    @patch("culora.analysis.analyzer.cv2.cvtColor")
    @patch("culora.analysis.analyzer.cv2.Laplacian")
    @patch("culora.analysis.analyzer.np.mean")
    @patch("culora.analysis.analyzer.np.std")
    def test_analyze_quality_fail_sharpness(
        self,
        mock_std: Any,
        mock_mean: Any,
        mock_laplacian: Any,
        mock_cvtcolor: Any,
        mock_imread: Any,
    ) -> None:
        """Test quality analysis where sharpness fails."""
        # Setup mocks with low sharpness
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_laplacian_result = MagicMock()
        mock_laplacian_result.var.return_value = 100.0  # Below threshold

        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_gray
        mock_laplacian.return_value = mock_laplacian_result
        mock_mean.return_value = 120.0  # Within range
        mock_std.return_value = 50.0  # Above threshold

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_quality(image_path)

        # Assertions
        assert result.stage == AnalysisStage.QUALITY
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "Quality metrics failed" in result.reason
        assert "sharpness: 100.0 < 150" in result.reason
        assert result.metadata["sharpness_pass"] == "False"
        assert result.metadata["brightness_pass"] == "True"
        assert result.metadata["contrast_pass"] == "True"

    @patch("culora.analysis.analyzer.cv2.imread")
    @patch("culora.analysis.analyzer.cv2.cvtColor")
    @patch("culora.analysis.analyzer.cv2.Laplacian")
    @patch("culora.analysis.analyzer.np.mean")
    @patch("culora.analysis.analyzer.np.std")
    def test_analyze_quality_fail_brightness(
        self,
        mock_std: Any,
        mock_mean: Any,
        mock_laplacian: Any,
        mock_cvtcolor: Any,
        mock_imread: Any,
    ) -> None:
        """Test quality analysis where brightness fails."""
        # Setup mocks with poor brightness
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_laplacian_result = MagicMock()
        mock_laplacian_result.var.return_value = 200.0  # Above threshold

        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_gray
        mock_laplacian.return_value = mock_laplacian_result
        mock_mean.return_value = 30.0  # Below range
        mock_std.return_value = 50.0  # Above threshold

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_quality(image_path)

        # Assertions
        assert result.stage == AnalysisStage.QUALITY
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "Quality metrics failed" in result.reason
        assert "brightness: 30.0 not in range [60.0-200.0]" in result.reason
        assert result.metadata["sharpness_pass"] == "True"
        assert result.metadata["brightness_pass"] == "False"
        assert result.metadata["contrast_pass"] == "True"

    @patch("culora.analysis.analyzer.cv2.imread")
    @patch("culora.analysis.analyzer.cv2.cvtColor")
    @patch("culora.analysis.analyzer.cv2.Laplacian")
    @patch("culora.analysis.analyzer.np.mean")
    @patch("culora.analysis.analyzer.np.std")
    def test_analyze_quality_fail_contrast(
        self,
        mock_std: Any,
        mock_mean: Any,
        mock_laplacian: Any,
        mock_cvtcolor: Any,
        mock_imread: Any,
    ) -> None:
        """Test quality analysis where contrast fails."""
        # Setup mocks with low contrast
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_laplacian_result = MagicMock()
        mock_laplacian_result.var.return_value = 200.0  # Above threshold

        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_gray
        mock_laplacian.return_value = mock_laplacian_result
        mock_mean.return_value = 120.0  # Within range
        mock_std.return_value = 20.0  # Below threshold

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_quality(image_path)

        # Assertions
        assert result.stage == AnalysisStage.QUALITY
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "Quality metrics failed" in result.reason
        assert "contrast: 20.0 < 40" in result.reason
        assert result.metadata["sharpness_pass"] == "True"
        assert result.metadata["brightness_pass"] == "True"
        assert result.metadata["contrast_pass"] == "False"

    @patch("culora.analysis.analyzer.cv2.imread")
    def test_analyze_quality_opencv_error(self, mock_imread: Any) -> None:
        """Test quality analysis with OpenCV error."""
        # Mock OpenCV error
        mock_imread.side_effect = Exception("OpenCV failed to read image")

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_quality(image_path)

        # Assertions
        assert result.stage == AnalysisStage.QUALITY
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert (
            "Failed to analyze image quality: OpenCV failed to read image"
            in result.reason
        )


class TestAnalyzeFace:
    """Tests for analyze_face function."""

    @patch("culora.analysis.analyzer.YOLO")
    def test_analyze_face_people_detected(self, mock_yolo: Any) -> None:
        """Test face analysis with people detected using YOLO11."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock YOLOv8 results
        mock_box1 = MagicMock()
        mock_box1.cls = [0]  # Person class
        mock_box1.conf = [0.85]

        mock_box2 = MagicMock()
        mock_box2.cls = [0]  # Person class
        mock_box2.conf = [0.92]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]

        mock_model.return_value = [mock_result]

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_face(image_path)

        # Assertions
        assert result.stage == AnalysisStage.FACE
        assert result.result == AnalysisResult.PASS
        assert result.reason is not None
        assert "Detected 2 person(s)" in result.reason
        assert "average confidence 0.885" in result.reason

        # Check metadata
        assert result.metadata["face_count"] == "2"
        assert result.metadata["confidence_scores"] == "0.850,0.920"
        assert result.metadata["average_confidence"] == "0.885"
        assert result.metadata["model"] == "yolo11n.pt"
        assert result.metadata["detection_type"] == "person"

    @patch("culora.analysis.analyzer.YOLO")
    def test_analyze_face_no_people(self, mock_yolo: Any) -> None:
        """Test face analysis with no people detected."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock YOLOv8 results with no person detections
        mock_result = MagicMock()
        mock_result.boxes = None  # No detections

        mock_model.return_value = [mock_result]

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_face(image_path)

        # Assertions
        assert result.stage == AnalysisStage.FACE
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert "No people detected in image" in result.reason

        # Check metadata
        assert result.metadata["face_count"] == "0"
        assert result.metadata["confidence_scores"] == ""
        assert result.metadata["average_confidence"] == "0.000"

    @patch("culora.analysis.analyzer.YOLO")
    def test_analyze_face_yolo_error(self, mock_yolo: Any) -> None:
        """Test face analysis with YOLO error."""
        # Mock YOLO error
        mock_yolo.side_effect = Exception("YOLO model failed to load")

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_face(image_path)

        # Assertions
        assert result.stage == AnalysisStage.FACE
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert (
            "Failed to analyze face detection: YOLO model failed to load"
            in result.reason
        )

    @patch("culora.analysis.analyzer.YOLO")
    def test_analyze_face_yolo_inference_error(self, mock_yolo: Any) -> None:
        """Test face analysis with YOLO inference error."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock YOLO inference error
        mock_model.side_effect = Exception("YOLO inference failed")

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_face(image_path)

        # Assertions
        assert result.stage == AnalysisStage.FACE
        assert result.result == AnalysisResult.FAIL
        assert result.reason is not None
        assert (
            "Failed to analyze face detection: YOLO inference failed" in result.reason
        )

    @patch("culora.analysis.analyzer.detect_optimal_device")
    @patch("culora.analysis.analyzer.YOLO")
    def test_analyze_face_with_custom_config(
        self, mock_yolo: Any, mock_device_detect: Any
    ) -> None:
        """Test face analysis with custom configuration parameters."""
        from culora.models.analysis import StageConfig

        # Setup mocks
        mock_device_detect.return_value = "cuda"
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock YOLO results
        mock_box = MagicMock()
        mock_box.cls = [0]  # Person class
        mock_box.conf = [0.75]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # Create custom config
        custom_config = StageConfig(
            stage=AnalysisStage.FACE,
            config={
                "confidence_threshold": "0.7",
                "model_name": "yolo11s.pt",
                "max_detections": "5",
                "iou_threshold": "0.4",
                "use_half_precision": "false",
                "device": "auto",
            },
            version="2.0",
        )

        # Test
        image_path = Path("/fake/path/image.jpg")
        result = analyze_face(image_path, custom_config)

        # Verify YOLO was called with correct parameters
        mock_model.assert_called_once()
        call_args = mock_model.call_args
        assert call_args[0][0] == str(image_path)  # First positional arg
        assert call_args[1]["conf"] == 0.7
        assert call_args[1]["iou"] == 0.4
        assert call_args[1]["max_det"] == 5
        assert call_args[1]["device"] == "cuda"
        assert call_args[1]["half"] is False
        assert call_args[1]["verbose"] is False

        # Check result metadata includes config info
        assert result.metadata["model"] == "yolo11s.pt"
        assert result.metadata["device_used"] == "cuda"
        assert result.metadata["confidence_threshold"] == "0.7"
        assert result.metadata["max_detections"] == "5"
        assert result.metadata["iou_threshold"] == "0.4"
        assert result.metadata["half_precision"] == "False"


class TestAnalyzeImage:
    """Tests for analyze_image function."""

    @patch("culora.analysis.analyzer.analyze_deduplication")
    @patch("culora.analysis.analyzer.analyze_quality")
    @patch("culora.analysis.analyzer.analyze_face")
    def test_analyze_image_all_stages(
        self,
        mock_analyze_face: Any,
        mock_analyze_quality: Any,
        mock_analyze_deduplication: Any,
        tmp_path: Path,
    ) -> None:
        """Test analyze_image with all stages enabled."""
        # Create a temporary image file
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        # Mock analysis functions
        mock_dedup_result = StageResult(
            stage=AnalysisStage.DEDUPLICATION,
            result=AnalysisResult.PASS,
            reason="Generated hash: abc123",
        )
        mock_quality_result = StageResult(
            stage=AnalysisStage.QUALITY,
            result=AnalysisResult.PASS,
            reason="Quality metrics passed",
        )
        mock_face_result = StageResult(
            stage=AnalysisStage.FACE,
            result=AnalysisResult.FAIL,
            reason="No faces detected",
        )

        mock_analyze_deduplication.return_value = mock_dedup_result
        mock_analyze_quality.return_value = mock_quality_result
        mock_analyze_face.return_value = mock_face_result

        # Test
        enabled_stages = [
            AnalysisStage.DEDUPLICATION,
            AnalysisStage.QUALITY,
            AnalysisStage.FACE,
        ]
        result = analyze_image(image_file, enabled_stages)

        # Assertions
        assert result.file_path == str(image_file.resolve())
        assert result.file_size == 0  # Empty file
        assert len(result.stage_results) == 3

        # Check stage results
        assert result.stage_results[0] == mock_dedup_result
        assert result.stage_results[1] == mock_quality_result
        assert result.stage_results[2] == mock_face_result

        # Verify analysis functions were called with stage config
        mock_analyze_deduplication.assert_called_once()
        mock_analyze_quality.assert_called_once()
        mock_analyze_face.assert_called_once()

        # Verify first arguments are the image paths
        assert mock_analyze_deduplication.call_args[0][0] == image_file
        assert mock_analyze_quality.call_args[0][0] == image_file
        assert mock_analyze_face.call_args[0][0] == image_file

    @patch("culora.analysis.analyzer.analyze_deduplication")
    def test_analyze_image_single_stage(
        self, mock_analyze_deduplication: Any, tmp_path: Path
    ) -> None:
        """Test analyze_image with only deduplication stage."""
        # Create a temporary image file
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        # Mock analysis function
        mock_result = StageResult(
            stage=AnalysisStage.DEDUPLICATION,
            result=AnalysisResult.PASS,
            reason="Generated hash: abc123",
        )
        mock_analyze_deduplication.return_value = mock_result

        # Test
        enabled_stages = [AnalysisStage.DEDUPLICATION]
        result = analyze_image(image_file, enabled_stages)

        # Assertions
        assert len(result.stage_results) == 1
        assert result.stage_results[0] == mock_result
        # Verify analysis function was called with stage config
        mock_analyze_deduplication.assert_called_once()
        assert mock_analyze_deduplication.call_args[0][0] == image_file

    def test_analyze_image_empty_stages(self, tmp_path: Path) -> None:
        """Test analyze_image with no stages enabled."""
        # Create a temporary image file
        image_file = tmp_path / "test.jpg"
        image_file.touch()

        # Test
        enabled_stages: list[AnalysisStage] = []
        result = analyze_image(image_file, enabled_stages)

        # Assertions
        assert len(result.stage_results) == 0
        assert result.file_path == str(image_file.resolve())
