"""Comprehensive unit tests for SelectionService."""

import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest

from culora.config import AnalysisStage, CuLoRAConfig
from culora.models.analysis_result import AnalysisResult
from culora.models.directory_analysis import DirectoryAnalysis
from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import Face, FaceDetectionResult
from culora.models.image_analysis import ImageAnalysis
from culora.models.image_quality_result import ImageQualityResult
from culora.services.selection_service import SelectionService


class TestSelectionServiceComprehensive:
    """Comprehensive tests for SelectionService actual functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a real config manager with default config for testing
        config = CuLoRAConfig()
        mock_config_manager = Mock()
        mock_config_manager.config = config

        # Mock the get_config method to return stage-specific configs
        def mock_get_config(stage: AnalysisStage) -> Any:
            if stage == AnalysisStage.QUALITY:
                return config.quality
            elif stage == AnalysisStage.FACE:
                return config.face
            elif stage == AnalysisStage.DEDUPLICATION:
                return config.deduplication
            return None

        mock_config_manager.get_config = mock_get_config
        self.selection_service = SelectionService(mock_config_manager)

    def _create_mock_image_analysis(
        self,
        file_path: str,
        score: float = 0.5,
        quality_passed: bool = True,
        face_passed: bool = True,
        dedup_passed: bool = True,
        hash_value: str = "unique_hash",
    ) -> ImageAnalysis:
        """Helper to create mock ImageAnalysis objects."""
        # Create analysis results

        analysis_result = AnalysisResult()

        if quality_passed:
            quality_result = ImageQualityResult(
                sharpness_score=200.0,
                brightness_score=120.0,
                contrast_score=60.0,
                composite_score=0.8,
            )
            analysis_result.set_quality(quality_result)

        if face_passed:
            faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]
            face_result = FaceDetectionResult(
                faces=faces,
                model_used="test_model",
                device_used="cpu",
            )
            analysis_result.set_face(face_result)

        if dedup_passed:
            dedup_result = DuplicateDetectionResult(hash_value=hash_value)
            analysis_result.set_deduplication(dedup_result)

        return ImageAnalysis(
            file_path=file_path,
            file_size=1024,
            modified_time=datetime.fromisoformat("2023-01-01T00:00:00"),
            results=analysis_result,
            score=score,
        )

    def _create_mock_directory_analysis(
        self, images: list[ImageAnalysis]
    ) -> DirectoryAnalysis:
        """Helper to create mock DirectoryAnalysis."""
        config = CuLoRAConfig()
        return DirectoryAnalysis(
            input_directory="/test/input",
            analysis_time="2023-01-01T00:00:00",  # type: ignore
            analysis_config=config,
            images=images,
        )

    def test_select_images_with_qualifying_images(self):
        """Test selection with images that pass all stages."""
        # Create test images with different scores and unique hashes
        images = [
            self._create_mock_image_analysis(
                "/test/image1.jpg", score=0.9, hash_value="hash1"
            ),
            self._create_mock_image_analysis(
                "/test/image2.jpg", score=0.7, hash_value="hash2"
            ),
            self._create_mock_image_analysis(
                "/test/image3.jpg", score=0.5, hash_value="hash3"
            ),
        ]

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("shutil.copy2") as mock_copy,
            ):

                selected, total = self.selection_service.select_images(
                    analysis, temp_dir
                )

                assert selected == 3
                assert total == 3
                assert mock_copy.call_count == 3

    def test_select_images_with_max_images_limit(self):
        """Test selection respects max_images parameter."""
        # Create test images with different scores and unique hashes
        images = [
            self._create_mock_image_analysis(
                "/test/image1.jpg", score=0.9, hash_value="hash1"
            ),
            self._create_mock_image_analysis(
                "/test/image2.jpg", score=0.7, hash_value="hash2"
            ),
            self._create_mock_image_analysis(
                "/test/image3.jpg", score=0.5, hash_value="hash3"
            ),
        ]

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("shutil.copy2") as mock_copy,
            ):

                selected, total = self.selection_service.select_images(
                    analysis, temp_dir, max_images=2
                )

                assert selected == 2
                assert total == 3
                assert mock_copy.call_count == 2

    def test_select_images_dry_run(self):
        """Test dry run mode doesn't copy files."""
        images = [
            self._create_mock_image_analysis(
                "/test/image1.jpg", score=0.8, hash_value="hash1"
            ),
        ]

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("shutil.copy2") as mock_copy:

                selected, total = self.selection_service.select_images(
                    analysis, temp_dir, dry_run=True
                )

                assert selected == 1
                assert total == 1
                # Should not copy files in dry run
                mock_copy.assert_not_called()

    def test_select_images_no_qualifying_images(self):
        """Test selection when no images pass qualifications."""
        # Create images that fail quality checks
        images = [
            self._create_mock_image_analysis("/test/image1.jpg", quality_passed=False),
            self._create_mock_image_analysis("/test/image2.jpg", face_passed=False),
        ]

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            selected, total = self.selection_service.select_images(analysis, temp_dir)

            assert selected == 0
            assert total == 2

    def test_select_images_missing_source_files(self):
        """Test selection handles missing source files gracefully."""
        images = [
            self._create_mock_image_analysis(
                "/test/missing.jpg", score=0.8, hash_value="hash1"
            ),
        ]

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock file as not existing
            with patch("pathlib.Path.exists", return_value=False):

                selected, total = self.selection_service.select_images(
                    analysis, temp_dir
                )

                # Should skip missing files
                assert selected == 0
                assert total == 1

    def test_draw_face_bounding_boxes(self):
        """Test drawing face bounding boxes on images."""
        face = Face(bounding_box=(10, 10, 50, 50), confidence=0.8)
        face_result = FaceDetectionResult(
            faces=[face], model_used="test_model", device_used="cpu"
        )

        images = [self._create_mock_image_analysis("/test/face_image.jpg", score=0.8)]
        # Add face result to the image
        images[0].results.set_face(face_result)

        analysis = self._create_mock_directory_analysis(images)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("cv2.imread") as mock_imread,
                patch("cv2.imwrite") as mock_imwrite,
                patch("cv2.rectangle") as mock_rectangle,
                patch("cv2.putText") as mock_puttext,
                patch("cv2.getTextSize", return_value=((100, 20), 5)),
            ):

                # Mock opencv image loading
                mock_imread.return_value = "mock_image_array"

                selected, total = self.selection_service.select_images(
                    analysis, temp_dir, draw_boxes=True
                )

                assert selected == 1
                assert total == 1

                # Verify opencv functions were called
                mock_imread.assert_called_once()
                mock_rectangle.assert_called()  # For bounding box and text background
                mock_puttext.assert_called_once()
                mock_imwrite.assert_called_once()

    def test_evaluate_image_quality_results(self):
        """Test quality evaluation logic."""
        # Create quality result that passes
        quality_result = ImageQualityResult(
            sharpness_score=200.0,  # > 150 threshold
            brightness_score=120.0,  # within 60-200 range
            contrast_score=60.0,  # > 40 threshold
            composite_score=0.8,
        )

        result = self.selection_service._evaluate_image_quality_results(quality_result)
        assert result is True

        # Create quality result that fails sharpness
        quality_result_fail = ImageQualityResult(
            sharpness_score=50.0,  # < 150 threshold
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=0.3,
        )

        result = self.selection_service._evaluate_image_quality_results(
            quality_result_fail
        )
        assert result is False

    def test_evaluate_face_detection_results(self):
        """Test face detection evaluation logic."""
        faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]

        # Create face result that passes
        face_result = FaceDetectionResult(
            faces=faces,
            model_used="test_model",
            device_used="cpu",
        )

        result = self.selection_service._evaluate_face_detection_results(face_result)
        assert result is True

        # Create face result with no faces
        face_result_no_faces = FaceDetectionResult(
            faces=[], model_used="test_model", device_used="cpu"
        )

        result = self.selection_service._evaluate_face_detection_results(
            face_result_no_faces
        )
        assert result is False

    def test_evaluate_duplicate_detection_results(self):
        """Test deduplication evaluation logic."""
        # Create test images with same hash (duplicates)
        image1 = self._create_mock_image_analysis("/test/image1.jpg", score=0.8)
        image2 = self._create_mock_image_analysis(
            "/test/image2.jpg", score=0.6
        )  # Lower score

        # Set same hash for both (making them duplicates)
        dedup_result1 = DuplicateDetectionResult(hash_value="abc123")
        dedup_result2 = DuplicateDetectionResult(hash_value="abc123")

        image1.results.set_deduplication(dedup_result1)
        image2.results.set_deduplication(dedup_result2)

        all_images = [image1, image2]

        # Higher score image should be kept
        result1 = self.selection_service._evaluate_duplicate_detection_results(
            image1, all_images
        )
        assert result1 is True

        # Lower score image should be rejected
        result2 = self.selection_service._evaluate_duplicate_detection_results(
            image2, all_images
        )
        assert result2 is False

    def test_calculate_face_size_and_percentage(self):
        """Test face size and percentage calculations."""
        bounding_box = (10.0, 10.0, 60.0, 40.0)  # 50x30 face

        width, height = self.selection_service._calculate_face_size(bounding_box)
        assert width == 50
        assert height == 30

        percentage = self.selection_service._calculate_face_percentage(
            bounding_box, image_width=200, image_height=100
        )
        # Face area: 50*30 = 1500, Image area: 200*100 = 20000
        # Percentage: 1500/20000 = 0.075 (7.5%)
        assert percentage == 0.075

    def test_output_directory_creation_error(self):
        """Test handling of output directory creation errors."""
        images = [self._create_mock_image_analysis("/test/image1.jpg", score=0.8)]
        analysis = self._create_mock_directory_analysis(images)

        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(RuntimeError, match="Could not create output directory"):
                self.selection_service.select_images(analysis, "/invalid/path")
