"""Tests for image domain models."""

from datetime import datetime
from pathlib import Path

from PIL import Image

from culora.domain.models.image import (
    BatchLoadResult,
    DirectoryScanResult,
    ImageLoadResult,
    ImageMetadata,
)


class TestImageMetadata:
    """Test ImageMetadata model."""

    def test_image_metadata_creation(self) -> None:
        """Test creating ImageMetadata."""
        path = Path("/test/image.jpg")
        created_at = datetime.now()
        modified_at = datetime.now()

        metadata = ImageMetadata(
            path=path,
            format="JPEG",
            width=1920,
            height=1080,
            file_size=1024000,
            created_at=created_at,
            modified_at=modified_at,
            is_valid=True,
        )

        assert metadata.path == path
        assert metadata.format == "JPEG"
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.file_size == 1024000
        assert metadata.created_at == created_at
        assert metadata.modified_at == modified_at
        assert metadata.is_valid is True
        assert metadata.error_message is None

    def test_image_metadata_with_error(self) -> None:
        """Test creating ImageMetadata with error."""
        metadata = ImageMetadata(
            path=Path("/test/bad.jpg"),
            format="unknown",
            width=0,
            height=0,
            file_size=0,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_valid=False,
            error_message="Corrupted file",
        )

        assert metadata.is_valid is False
        assert metadata.error_message == "Corrupted file"


class TestImageLoadResult:
    """Test ImageLoadResult model."""

    def test_successful_load_result(self) -> None:
        """Test successful ImageLoadResult."""
        metadata = ImageMetadata(
            path=Path("/test/image.jpg"),
            format="JPEG",
            width=800,
            height=600,
            file_size=50000,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_valid=True,
        )

        # Create a test image
        image = Image.new("RGB", (800, 600), color="red")

        result = ImageLoadResult(
            success=True,
            metadata=metadata,
            image=image,
        )

        assert result.success is True
        assert result.metadata == metadata
        assert result.image is not None
        assert result.image.size == (800, 600)
        assert result.error is None
        assert result.error_code is None

    def test_failed_load_result(self) -> None:
        """Test failed ImageLoadResult."""
        metadata = ImageMetadata(
            path=Path("/test/bad.jpg"),
            format="unknown",
            width=0,
            height=0,
            file_size=0,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_valid=False,
            error_message="Corrupted file",
        )

        result = ImageLoadResult(
            success=False,
            metadata=metadata,
            error="Failed to load image",
            error_code="LOAD_FAILED",
        )

        assert result.success is False
        assert result.metadata == metadata
        assert result.image is None
        assert result.error == "Failed to load image"
        assert result.error_code == "LOAD_FAILED"


class TestDirectoryScanResult:
    """Test DirectoryScanResult model."""

    def test_directory_scan_result(self) -> None:
        """Test DirectoryScanResult creation."""
        image_paths = [
            Path("/test/image1.jpg"),
            Path("/test/image2.png"),
            Path("/test/image3.webp"),
        ]

        supported_formats = {
            ".jpg": 1,
            ".png": 1,
            ".webp": 1,
        }

        errors = ["Permission denied: /test/hidden.jpg"]

        result = DirectoryScanResult(
            total_files=10,
            valid_images=3,
            invalid_images=7,
            supported_formats=supported_formats,
            total_size=1500000,
            scan_duration=2.5,
            errors=errors,
            image_paths=image_paths,
        )

        assert result.total_files == 10
        assert result.valid_images == 3
        assert result.invalid_images == 7
        assert result.supported_formats == supported_formats
        assert result.total_size == 1500000
        assert result.scan_duration == 2.5
        assert result.errors == errors
        assert result.image_paths == image_paths


class TestBatchLoadResult:
    """Test BatchLoadResult model."""

    def test_batch_load_result(self) -> None:
        """Test BatchLoadResult creation."""
        # Create mock load results
        metadata1 = ImageMetadata(
            path=Path("/test/image1.jpg"),
            format="JPEG",
            width=800,
            height=600,
            file_size=50000,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_valid=True,
        )

        metadata2 = ImageMetadata(
            path=Path("/test/image2.jpg"),
            format="JPEG",
            width=1920,
            height=1080,
            file_size=100000,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_valid=True,
        )

        image1 = Image.new("RGB", (800, 600), color="red")
        image2 = Image.new("RGB", (1920, 1080), color="blue")

        results = [
            ImageLoadResult(success=True, metadata=metadata1, image=image1),
            ImageLoadResult(success=True, metadata=metadata2, image=image2),
        ]

        batch_result = BatchLoadResult(
            results=results,
            successful_loads=2,
            failed_loads=0,
            total_size=150000,
            processing_duration=1.2,
        )

        assert len(batch_result.results) == 2
        assert batch_result.successful_loads == 2
        assert batch_result.failed_loads == 0
        assert batch_result.total_size == 150000
        assert batch_result.processing_duration == 1.2
