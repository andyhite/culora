"""Unit tests for ImageAnalyzer orchestrator."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.managers.image_manager import ImageManager
from culora.models.directory_analysis import DirectoryAnalysis
from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import Face, FaceDetectionResult
from culora.models.image_analysis import ImageAnalysis
from culora.models.image_quality_result import ImageQualityResult
from culora.orchestrators.image_analyzer import ImageAnalyzer


class TestImageAnalyzer:
    """Tests for ImageAnalyzer orchestrator functionality."""

    def test_init_with_default_services(self) -> None:
        """Test initialization with default service instances."""
        with (
            patch.object(ConfigManager, "get_instance") as mock_config_manager,
            patch.object(ImageManager, "get_instance") as mock_image_manager,
        ):

            analyzer = ImageAnalyzer()

            mock_config_manager.assert_called_once()
            mock_image_manager.assert_called_once()
            assert analyzer._config_manager is mock_config_manager.return_value
            assert analyzer._image_manager is mock_image_manager.return_value

    def test_init_with_provided_services(self) -> None:
        """Test initialization with explicitly provided services."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        mock_quality_service = MagicMock()
        mock_face_service = MagicMock()
        mock_dedup_service = MagicMock()

        analyzer = ImageAnalyzer(
            config_manager=mock_config,
            image_manager=mock_image_manager,
            quality_service=mock_quality_service,
            face_service=mock_face_service,
            deduplication_service=mock_dedup_service,
        )

        assert analyzer._config_manager is mock_config
        assert analyzer._image_manager is mock_image_manager
        assert analyzer._image_quality_service is mock_quality_service
        assert analyzer._face_detection_service is mock_face_service
        assert analyzer._deduplication_service is mock_dedup_service

    def test_analyze_directory_validates_directory(self) -> None:
        """Test that analyze_directory validates the input directory."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        mock_image_manager.validate_directory.side_effect = FileNotFoundError(
            "Directory not found"
        )

        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_directory(Path("/nonexistent"))

        mock_image_manager.validate_directory.assert_called_once_with(
            Path("/nonexistent")
        )

    def test_analyze_directory_empty_directory(self) -> None:
        """Test analyzing an empty directory."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        mock_image_manager.find_images_in_directory.return_value = iter([])

        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )
        input_dir = Path("/empty")

        with patch("culora.orchestrators.image_analyzer.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

            result = analyzer.analyze_directory(input_dir)

            assert isinstance(result, DirectoryAnalysis)
            assert result.input_directory == str(input_dir.resolve())
            assert result.analysis_time == datetime(2023, 1, 1, 12, 0, 0)
            assert result.analysis_config is mock_config.config
            assert result.images == []

    @patch("culora.orchestrators.image_analyzer.Progress")
    def test_analyze_directory_with_images(
        self, mock_progress_class: MagicMock
    ) -> None:
        """Test analyzing a directory with images."""
        # Setup mocks
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        # Mock image paths
        image1 = Path("/test/image1.jpg")
        image2 = Path("/test/image2.jpg")
        mock_image_manager.find_images_in_directory.return_value = iter(
            [image1, image2]
        )

        # Mock metadata
        metadata1 = {
            "file_path": str(image1),
            "file_size": 1000,
            "modified_time": datetime(2023, 1, 1),
        }
        metadata2 = {
            "file_path": str(image2),
            "file_size": 2000,
            "modified_time": datetime(2023, 1, 2),
        }
        mock_image_manager.get_image_metadata.side_effect = [metadata1, metadata2]

        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )
        input_dir = Path("/test")

        with (
            patch("culora.orchestrators.image_analyzer.datetime") as mock_datetime,
            patch.object(analyzer, "_analyze_images") as mock_analyze_images,
        ):
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

            result = analyzer.analyze_directory(input_dir)

            # Verify directory analysis result
            assert isinstance(result, DirectoryAnalysis)
            assert result.input_directory == str(input_dir.resolve())
            assert len(result.images) == 2

            # Verify _analyze_images was called
            mock_analyze_images.assert_called_once()

            # Verify progress was set up correctly
            mock_progress.add_task.assert_called_once_with(
                f"Analyzing images in: {input_dir.name}",
                total=2,
            )

    def test_get_service_for_stage_quality(self) -> None:
        """Test _get_service_for_stage returns quality service."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        result = analyzer._get_service_for_stage(AnalysisStage.QUALITY)
        assert result is analyzer._image_quality_service

    def test_get_service_for_stage_face(self) -> None:
        """Test _get_service_for_stage returns face service."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        result = analyzer._get_service_for_stage(AnalysisStage.FACE)
        assert result is analyzer._face_detection_service

    def test_get_service_for_stage_deduplication(self) -> None:
        """Test _get_service_for_stage returns deduplication service."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        result = analyzer._get_service_for_stage(AnalysisStage.DEDUPLICATION)
        assert result is analyzer._deduplication_service

    def test_get_service_for_stage_unknown(self) -> None:
        """Test _get_service_for_stage returns None for unknown stage."""
        mock_config = MagicMock()
        mock_image_manager = MagicMock()
        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        # Create a mock stage that doesn't match any known stages
        mock_stage = MagicMock()
        mock_stage.__eq__ = MagicMock(return_value=False)

        result = analyzer._get_service_for_stage(mock_stage)
        assert result is None

    def test_analyze_images_with_all_stages_enabled(self) -> None:
        """Test _analyze_images with all analysis stages enabled."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.config.deduplication.enabled = True
        mock_config.config.quality.enabled = True
        mock_config.config.face.enabled = True

        mock_image_manager = MagicMock()
        mock_quality_service = MagicMock()
        mock_face_service = MagicMock()
        mock_dedup_service = MagicMock()

        # Mock service results
        quality_result = ImageQualityResult(
            sharpness_score=100.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=80.0,
        )
        face_result = FaceDetectionResult(
            faces=[], model_used="test_model", device_used="cpu"
        )
        dedup_result = DuplicateDetectionResult(hash_value="abc123")

        mock_quality_service.analyze_image.return_value = quality_result
        mock_face_service.analyze_image.return_value = face_result
        mock_dedup_service.analyze_image.return_value = dedup_result

        analyzer = ImageAnalyzer(
            config_manager=mock_config,
            image_manager=mock_image_manager,
            quality_service=mock_quality_service,
            face_service=mock_face_service,
            deduplication_service=mock_dedup_service,
        )

        # Create test image analysis objects
        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )
        analyzed_images = [image_analysis]

        # Mock image loading
        mock_image = MagicMock(spec=Image.Image)
        mock_image_manager.load_image.return_value.__enter__.return_value = mock_image

        # Mock progress
        mock_progress = MagicMock()
        mock_task = MagicMock()

        with patch.object(
            analyzer, "_calculate_image_score", return_value=0.85
        ) as mock_calc_score:
            analyzer._analyze_images(analyzed_images, mock_progress, mock_task)

            # Verify all services were called
            mock_quality_service.analyze_image.assert_called_once_with(mock_image)
            mock_face_service.analyze_image.assert_called_once_with(mock_image)
            mock_dedup_service.analyze_image.assert_called_once_with(mock_image)

            # Verify results were set
            assert image_analysis.results.get_quality() is quality_result
            assert image_analysis.results.get_face() is face_result
            assert image_analysis.results.get_deduplication() is dedup_result

            # Verify score was calculated and set
            mock_calc_score.assert_called_once_with(image_analysis, mock_image)
            assert image_analysis.score == 0.85

    def test_analyze_images_handles_image_load_failure(self) -> None:
        """Test _analyze_images handles image loading failures gracefully."""
        mock_config = MagicMock()
        mock_config.config.quality.enabled = True
        mock_image_manager = MagicMock()
        mock_image_manager.load_image.side_effect = FileNotFoundError("Image not found")

        analyzer = ImageAnalyzer(
            config_manager=mock_config, image_manager=mock_image_manager
        )

        image_analysis = ImageAnalysis(
            file_path="/test/missing.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )
        analyzed_images = [image_analysis]

        mock_progress = MagicMock()
        mock_task = MagicMock()

        with patch("culora.orchestrators.image_analyzer.console") as mock_console:
            analyzer._analyze_images(analyzed_images, mock_progress, mock_task)

            # Verify warning was displayed
            mock_console.warning.assert_called_once_with(
                "Failed to load image missing.jpg, skipping analysis."
            )

            # Verify progress was still updated
            mock_progress.update.assert_called_with(mock_task, completed=1)

    def test_calculate_image_score_quality_only(self) -> None:
        """Test _calculate_image_score with quality result only."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 0.6
        mock_config.config.scoring.face_weight = 0.4

        analyzer = ImageAnalyzer(config_manager=mock_config)

        # Create image with quality result
        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )
        quality_result = ImageQualityResult(
            sharpness_score=100.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=80.0,  # This should become 0.8 after normalization
        )
        image_analysis.results.set_quality(quality_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        # Score should be quality_component (0.8) * quality_weight (0.6) = 0.48
        expected_score = 0.48
        assert abs(score - expected_score) < 0.01

    def test_calculate_image_score_with_face_optimal_size(self) -> None:
        """Test _calculate_image_score with face in optimal size range."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 0.5
        mock_config.config.scoring.face_weight = 0.5
        mock_config.config.scoring.face_area_min = 0.05
        mock_config.config.scoring.face_area_peak = 0.15
        mock_config.config.scoring.face_area_max = 0.25
        mock_config.config.scoring.multi_face_penalty = 0.1
        mock_config.config.scoring.max_face_penalty = 0.5

        analyzer = ImageAnalyzer(config_manager=mock_config)

        # Create image with face result at optimal size (15% of image)
        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        # Face covers 150x150 = 22,500 pixels out of 1000x1000 = 1,000,000 (2.25%)
        # This is between min (5%) and peak (15%), so should get area_score around 0.6
        face = Face(bounding_box=(100, 100, 250, 250), confidence=0.9)
        face_result = FaceDetectionResult(
            faces=[face], model_used="test_model", device_used="cpu"
        )
        image_analysis.results.set_face(face_result)

        quality_result = ImageQualityResult(
            sharpness_score=100.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=80.0,
        )
        image_analysis.results.set_quality(quality_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        # Should be a combination of quality (0.8 * 0.5) and face components
        assert score > 0.4  # Should be greater than quality alone
        assert score <= 1.0  # Should be clamped to max 1.0

    def test_calculate_image_score_multiple_faces_penalty(self) -> None:
        """Test _calculate_image_score applies penalty for multiple faces."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 0.5
        mock_config.config.scoring.face_weight = 0.5
        mock_config.config.scoring.face_area_min = 0.05
        mock_config.config.scoring.face_area_peak = 0.15
        mock_config.config.scoring.face_area_max = 0.25
        mock_config.config.scoring.multi_face_penalty = 0.1
        mock_config.config.scoring.max_face_penalty = 0.5

        analyzer = ImageAnalyzer(config_manager=mock_config)

        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        # Create multiple faces (3 faces should get 20% penalty)
        faces = [
            Face(bounding_box=(100, 100, 250, 250), confidence=0.9),
            Face(bounding_box=(300, 100, 450, 250), confidence=0.8),
            Face(bounding_box=(500, 100, 650, 250), confidence=0.7),
        ]
        face_result = FaceDetectionResult(
            faces=faces, model_used="test_model", device_used="cpu"
        )
        image_analysis.results.set_face(face_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        # Calculate score - should be lower due to multi-face penalty
        score = analyzer._calculate_image_score(image_analysis, mock_image)

        assert 0.0 <= score <= 1.0

    def test_calculate_image_score_face_too_small(self) -> None:
        """Test _calculate_image_score with face smaller than minimum."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 0.5
        mock_config.config.scoring.face_weight = 0.5
        mock_config.config.scoring.face_area_min = 0.05  # 5%
        mock_config.config.scoring.face_area_peak = 0.15
        mock_config.config.scoring.face_area_max = 0.25

        analyzer = ImageAnalyzer(config_manager=mock_config)

        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        # Very small face (1% of image area)
        face = Face(
            bounding_box=(100, 100, 110, 110), confidence=0.9
        )  # 10x10 = 100 pixels = 0.01%
        face_result = FaceDetectionResult(
            faces=[face], model_used="test_model", device_used="cpu"
        )
        image_analysis.results.set_face(face_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        # Should still get some score but lower due to small face
        assert 0.0 <= score <= 1.0

    def test_calculate_image_score_face_too_large(self) -> None:
        """Test _calculate_image_score with face larger than maximum."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 0.5
        mock_config.config.scoring.face_weight = 0.5
        mock_config.config.scoring.face_area_min = 0.05
        mock_config.config.scoring.face_area_peak = 0.15
        mock_config.config.scoring.face_area_max = 0.25  # 25%

        analyzer = ImageAnalyzer(config_manager=mock_config)

        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        # Very large face (40% of image area)
        face = Face(
            bounding_box=(100, 100, 732, 732), confidence=0.9
        )  # ~632x632 = ~40% area
        face_result = FaceDetectionResult(
            faces=[face], model_used="test_model", device_used="cpu"
        )
        image_analysis.results.set_face(face_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        # Should get reduced score due to oversized face
        assert 0.0 <= score <= 1.0

    def test_calculate_image_score_no_results(self) -> None:
        """Test _calculate_image_score with no analysis results."""
        mock_config = MagicMock()
        analyzer = ImageAnalyzer(config_manager=mock_config)

        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        assert score == 0.0

    def test_calculate_image_score_clamping(self) -> None:
        """Test _calculate_image_score clamps results to [0.0, 1.0]."""
        mock_config = MagicMock()
        mock_config.config.scoring.quality_weight = 2.0  # Excessive weight
        mock_config.config.scoring.face_weight = 2.0  # Excessive weight

        analyzer = ImageAnalyzer(config_manager=mock_config)

        image_analysis = ImageAnalysis(
            file_path="/test/image1.jpg",
            file_size=1000,
            modified_time=datetime(2023, 1, 1),
        )

        # High scoring results
        quality_result = ImageQualityResult(
            sharpness_score=100.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=100.0,  # Max quality
        )
        image_analysis.results.set_quality(quality_result)

        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (1000, 1000)

        score = analyzer._calculate_image_score(image_analysis, mock_image)

        # Should be clamped to 1.0 despite excessive weights
        assert score == 1.0
