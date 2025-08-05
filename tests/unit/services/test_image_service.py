"""Tests for ImageService."""

from pathlib import Path

import pytest
from PIL import Image

from culora.domain import ImageConfig
from culora.services.image_service import (
    ImageService,
    ImageServiceError,
    get_image_service,
    initialize_image_service,
)
from tests.helpers import ConfigBuilder, ImageFixtures, TempFileHelper


class TestImageService:
    """Test ImageService functionality."""

    def test_image_service_initialization(self) -> None:
        """Test ImageService initialization."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        assert service.config == config
        assert service.image_config == config.images
        assert isinstance(service.image_config, ImageConfig)

    def test_get_supported_formats(self) -> None:
        """Test getting supported image formats."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        formats = service.get_supported_formats()
        expected = [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"]
        assert formats == expected

        # Ensure returned list is a copy
        formats.append(".bmp")
        assert service.get_supported_formats() == expected

    def test_validate_image_path_valid(self) -> None:
        """Test validating a valid image path."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            image_path = ImageFixtures.create_test_image_file(
                temp_dir / "test.jpg", format="JPEG"
            )

            is_valid, error = service.validate_image_path(image_path)
            assert is_valid is True
            assert error is None

    def test_validate_image_path_nonexistent(self) -> None:
        """Test validating nonexistent file path."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        is_valid, error = service.validate_image_path(Path("/nonexistent/file.jpg"))
        assert is_valid is False
        assert error is not None and "File does not exist" in error

    def test_validate_image_path_unsupported_format(self) -> None:
        """Test validating unsupported image format."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            unsupported_file = temp_dir / "test.bmp"
            unsupported_file.write_bytes(b"fake content")

            is_valid, error = service.validate_image_path(unsupported_file)
            assert is_valid is False
            assert error is not None and "Unsupported image format" in error

    def test_validate_image_path_too_large(self) -> None:
        """Test validating file that exceeds size limit."""
        # Create config with small file size limit
        image_config = ImageConfig(max_file_size=1024)  # 1KB limit
        config = ConfigBuilder().with_image_config(image_config).build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            # Create a large image file (larger than 1KB)
            large_image_path = ImageFixtures.create_test_image_file(
                temp_dir / "large.jpg", width=1920, height=1080, format="JPEG"
            )

            is_valid, error = service.validate_image_path(large_image_path)
            assert is_valid is False
            assert error is not None and "File too large" in error

    def test_load_image_success(self) -> None:
        """Test successfully loading an image."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            image_path = ImageFixtures.create_test_image_file(
                temp_dir / "test.jpg", width=800, height=600, color="red", format="JPEG"
            )

            result = service.load_image(image_path)

            assert result.success is True
            assert result.image is not None
            assert result.image.size == (800, 600)
            assert result.image.mode == "RGB"
            assert result.metadata.path == image_path
            assert result.metadata.format == "JPEG"
            assert result.metadata.width == 800
            assert result.metadata.height == 600
            assert result.metadata.is_valid is True
            assert result.error is None
            assert result.error_code is None

    def test_load_image_convert_to_rgb(self) -> None:
        """Test loading image that gets converted to RGB."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            # Create a grayscale image
            image = Image.new("L", (400, 300), color=128)  # Grayscale
            image_path = temp_dir / "grayscale.png"
            image.save(image_path, "PNG")

            result = service.load_image(image_path)

            assert result.success is True
            assert result.image is not None
            assert result.image.mode == "RGB"  # Should be converted

    def test_load_image_too_large_dimensions(self) -> None:
        """Test loading image with dimensions too large."""
        # Create config with small dimension limits
        image_config = ImageConfig(max_image_size=(100, 100))
        config = ConfigBuilder().with_image_config(image_config).build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            # Create image larger than limits
            image_path = ImageFixtures.create_test_image_file(
                temp_dir / "large.jpg", width=200, height=200, format="JPEG"
            )

            result = service.load_image(image_path)

            assert result.success is False
            assert result.image is None
            assert result.error_code == "IMAGE_TOO_LARGE"
            assert (
                result.error is not None
                and "Image dimensions too large" in result.error
            )

    def test_load_image_corrupted_file(self) -> None:
        """Test loading corrupted image file."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            corrupted_path = ImageFixtures.create_corrupted_image_file(
                temp_dir / "corrupted.jpg"
            )

            result = service.load_image(corrupted_path)

            assert result.success is False
            assert result.image is None
            assert result.error_code == "INVALID_METADATA"
            assert result.error is not None
            assert result.metadata.is_valid is False

    def test_scan_directory_basic(self) -> None:
        """Test basic directory scanning."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            ImageFixtures.create_test_directory_structure(temp_dir)

            result = service.scan_directory(temp_dir, show_progress=False)

            assert result.total_files > 0
            assert result.valid_images >= 3  # At least the main images
            assert result.scan_duration > 0
            assert len(result.image_paths) == result.valid_images

            # Check format counts
            assert ".jpg" in result.supported_formats
            assert ".png" in result.supported_formats
            assert ".webp" in result.supported_formats

    def test_scan_directory_nonexistent(self) -> None:
        """Test scanning nonexistent directory."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with pytest.raises(ImageServiceError) as exc_info:
            service.scan_directory(Path("/nonexistent/directory"))

        assert "Directory does not exist" in str(exc_info.value)

    def test_scan_directory_not_directory(self) -> None:
        """Test scanning a file instead of directory."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            file_path = temp_dir / "not_a_dir.txt"
            file_path.write_text("This is a file")

            with pytest.raises(ImageServiceError) as exc_info:
                service.scan_directory(file_path)

            assert "Path is not a directory" in str(exc_info.value)

    def test_load_batch(self) -> None:
        """Test loading a batch of images."""
        config = ConfigBuilder().build()
        service = ImageService(config)

        with TempFileHelper.create_temp_dir() as temp_dir:
            # Create test images
            paths = [
                ImageFixtures.create_test_image_file(
                    temp_dir / "img1.jpg", width=400, height=300, format="JPEG"
                ),
                ImageFixtures.create_test_image_file(
                    temp_dir / "img2.png", width=800, height=600, format="PNG"
                ),
                ImageFixtures.create_corrupted_image_file(temp_dir / "bad.jpg"),
            ]

            result = service.load_batch(paths)

            assert len(result.results) == 3
            assert result.successful_loads == 2  # Two valid images
            assert result.failed_loads == 1  # One corrupted
            assert result.processing_duration > 0
            assert result.total_size > 0  # Size of successfully loaded images


class TestImageServiceGlobalInstance:
    """Test global ImageService instance management."""

    def test_initialize_image_service(self) -> None:
        """Test initializing global image service."""
        config = ConfigBuilder().build()

        service = initialize_image_service(config)

        assert isinstance(service, ImageService)
        assert service.config == config

    def test_get_image_service_after_initialization(self) -> None:
        """Test getting image service after initialization."""
        config = ConfigBuilder().build()

        # Initialize first
        original_service = initialize_image_service(config)

        # Get the service
        retrieved_service = get_image_service()

        assert retrieved_service is original_service

    def test_get_image_service_not_initialized(self) -> None:
        """Test getting image service before initialization auto-initializes."""
        # Reset global service
        import culora.services.image_service

        culora.services.image_service._image_service = None

        # Should auto-initialize and return a service instance
        service = get_image_service()

        assert service is not None
        assert isinstance(service, ImageService)
