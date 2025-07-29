"""Shared test fixtures and configuration for CuLoRA tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a CLI runner for testing commands."""
    return CliRunner()


@pytest.fixture
def temp_image_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with mock image files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some mock image files
        image_files = [
            "image1.jpg",
            "image2.png",
            "image3.jpeg",
            "duplicate.jpg",
            "low_quality.jpg",
        ]

        for filename in image_files:
            # Create a simple test image
            img = Image.new("RGB", (512, 512), color=(255, 0, 0))
            img.save(temp_path / filename)

        # Create a non-image file (should be ignored)
        (temp_path / "readme.txt").write_text("Not an image")

        yield temp_path


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_analysis_services():
    """Mock all analysis services to return predictable results."""
    with (
        patch(
            "culora.services.duplicate_detection_service.DuplicateDetectionService"
        ) as mock_dedup,
        patch(
            "culora.services.image_quality_service.ImageQualityService"
        ) as mock_quality,
        patch(
            "culora.services.face_detection_service.FaceDetectionService"
        ) as mock_face,
    ):

        # Configure deduplication service mock
        mock_dedup_instance = MagicMock()
        mock_dedup_instance.analyze.return_value = Mock(
            hash_value="test_hash", is_duplicate=False
        )
        mock_dedup.return_value = mock_dedup_instance

        # Configure quality service mock
        mock_quality_instance = MagicMock()
        mock_quality_instance.analyze.return_value = Mock(
            sharpness=180.0, brightness=120.0, contrast=55.0, quality_score=0.75
        )
        mock_quality.return_value = mock_quality_instance

        # Configure face service mock
        mock_face_instance = MagicMock()
        mock_face_instance.analyze.return_value = Mock(
            faces_detected=1,
            face_confidences=[0.92],
            face_bounding_boxes=[(100, 100, 200, 200)],
            face_score=0.68,
        )
        mock_face.return_value = mock_face_instance

        yield {
            "dedup": mock_dedup_instance,
            "quality": mock_quality_instance,
            "face": mock_face_instance,
        }


@pytest.fixture
def mock_file_operations():
    """Mock file I/O operations while preserving directory structure."""
    with (
        patch("culora.utils.app_data.get_app_dir") as mock_app_dir,
        patch("shutil.copy2") as mock_copy,
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.read_text") as mock_read_text,
    ):

        # Mock app data directory
        mock_app_dir.return_value = Path("/tmp/culora_test")

        # Mock image copying
        mock_copy.return_value = None

        # Mock file existence and reading
        mock_exists.return_value = False  # No cache by default
        mock_read_text.return_value = "{}"

        yield {
            "app_dir": mock_app_dir,
            "copy": mock_copy,
            "exists": mock_exists,
            "read_text": mock_read_text,
        }


@pytest.fixture
def mock_model_loading():
    """Mock model loading and device detection."""
    with patch("culora.managers.model_manager.ModelManager") as mock_manager:
        mock_instance = MagicMock()
        mock_instance.get_device.return_value = "cpu"
        mock_instance.load_face_model.return_value = MagicMock()
        mock_manager.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_image_analyzer():
    """Mock the ImageAnalyzer orchestrator."""
    with patch("culora.orchestrators.image_analyzer.ImageAnalyzer") as mock_analyzer:
        mock_instance = MagicMock()

        # Mock analyze_directory to return a simple DirectoryAnalysis-like object
        mock_result = MagicMock()
        mock_result.total_images = 3
        mock_result.images = []  # List of ImageAnalysis objects

        mock_instance.analyze_directory.return_value = mock_result
        mock_analyzer.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_image_curator():
    """Mock the ImageCurator orchestrator."""
    with patch("culora.orchestrators.image_curator.ImageCurator") as mock_curator:
        mock_instance = MagicMock()
        mock_instance.select_images.return_value = (2, 3)  # selected, total
        mock_curator.return_value = mock_instance

        yield mock_instance
