"""Tests for image configuration model."""

import pytest
from pydantic import ValidationError

from culora.domain.models.config.image import ImageConfig


class TestImageConfig:
    """Test ImageConfig model."""

    def test_image_config_defaults(self) -> None:
        """Test ImageConfig with default values."""
        config = ImageConfig()

        assert config.supported_formats == [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tiff",
            ".tif",
        ]
        assert config.max_batch_size == 32
        assert config.max_image_size == (4096, 4096)
        assert config.max_file_size == 50 * 1024 * 1024  # 50MB
        assert config.recursive_scan is True
        assert config.max_scan_depth == 10
        assert config.skip_hidden_files is True
        assert config.progress_update_interval == 10

    def test_image_config_custom_values(self) -> None:
        """Test ImageConfig with custom values."""
        config = ImageConfig(
            supported_formats=[".jpg", ".png"],
            max_batch_size=16,
            max_image_size=(2048, 2048),
            max_file_size=10 * 1024 * 1024,  # 10MB
            recursive_scan=False,
            max_scan_depth=5,
            skip_hidden_files=False,
            progress_update_interval=50,
        )

        assert config.supported_formats == [".jpg", ".png"]
        assert config.max_batch_size == 16
        assert config.max_image_size == (2048, 2048)
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.recursive_scan is False
        assert config.max_scan_depth == 5
        assert config.skip_hidden_files is False
        assert config.progress_update_interval == 50

    def test_supported_formats_validation(self) -> None:
        """Test supported formats validation."""
        # Test formats without dots get normalized
        config = ImageConfig(supported_formats=["jpg", "png", "webp"])
        assert config.supported_formats == [".jpg", ".png", ".webp"]

        # Test case normalization
        config = ImageConfig(supported_formats=[".JPG", ".PNG", ".WebP"])
        assert config.supported_formats == [".jpg", ".png", ".webp"]

    def test_supported_formats_empty_validation(self) -> None:
        """Test validation error for empty supported formats."""
        with pytest.raises(ValidationError) as exc_info:
            ImageConfig(supported_formats=[])

        assert "At least one supported format must be specified" in str(exc_info.value)

    def test_max_batch_size_validation(self) -> None:
        """Test max_batch_size validation."""
        # Valid values
        config = ImageConfig(max_batch_size=1)
        assert config.max_batch_size == 1

        config = ImageConfig(max_batch_size=1000)
        assert config.max_batch_size == 1000

        # Invalid values
        with pytest.raises(ValidationError):
            ImageConfig(max_batch_size=0)

        with pytest.raises(ValidationError):
            ImageConfig(max_batch_size=1001)

    def test_max_image_size_validation(self) -> None:
        """Test max_image_size validation."""
        # Valid values
        config = ImageConfig(max_image_size=(100, 100))
        assert config.max_image_size == (100, 100)

        config = ImageConfig(max_image_size=(65535, 65535))
        assert config.max_image_size == (65535, 65535)

        # Invalid values - zero or negative
        with pytest.raises(ValidationError) as exc_info:
            ImageConfig(max_image_size=(0, 100))
        assert "Image dimensions must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ImageConfig(max_image_size=(100, -1))
        assert "Image dimensions must be positive" in str(exc_info.value)

        # Invalid values - too large
        with pytest.raises(ValidationError) as exc_info:
            ImageConfig(max_image_size=(65536, 100))
        assert "Image dimensions too large" in str(exc_info.value)

    def test_max_file_size_validation(self) -> None:
        """Test max_file_size validation."""
        # Valid values
        config = ImageConfig(max_file_size=1024)  # 1KB minimum
        assert config.max_file_size == 1024

        # Invalid values
        with pytest.raises(ValidationError):
            ImageConfig(max_file_size=1023)  # Below minimum

    def test_max_scan_depth_validation(self) -> None:
        """Test max_scan_depth validation."""
        # Valid values
        config = ImageConfig(max_scan_depth=1)
        assert config.max_scan_depth == 1

        config = ImageConfig(max_scan_depth=50)
        assert config.max_scan_depth == 50

        # Invalid values
        with pytest.raises(ValidationError):
            ImageConfig(max_scan_depth=0)

        with pytest.raises(ValidationError):
            ImageConfig(max_scan_depth=51)

    def test_progress_update_interval_validation(self) -> None:
        """Test progress_update_interval validation."""
        # Valid values
        config = ImageConfig(progress_update_interval=1)
        assert config.progress_update_interval == 1

        config = ImageConfig(progress_update_interval=1000)
        assert config.progress_update_interval == 1000

        # Invalid values
        with pytest.raises(ValidationError):
            ImageConfig(progress_update_interval=0)

        with pytest.raises(ValidationError):
            ImageConfig(progress_update_interval=1001)

    def test_from_dict(self) -> None:
        """Test creating ImageConfig from dictionary."""
        data = {
            "supported_formats": [".jpg", ".png"],
            "max_batch_size": 16,
            "recursive_scan": False,
        }

        config = ImageConfig.from_dict(data)

        assert config.supported_formats == [".jpg", ".png"]
        assert config.max_batch_size == 16
        assert config.recursive_scan is False
        # Other values should be defaults
        assert config.max_image_size == (4096, 4096)
        assert config.skip_hidden_files is True

    def test_model_dump(self) -> None:
        """Test ImageConfig serialization."""
        config = ImageConfig(
            supported_formats=[".jpg", ".png"],
            max_batch_size=16,
            recursive_scan=False,
        )

        dumped = config.model_dump()

        # Check some key values
        assert dumped["supported_formats"] == [".jpg", ".png"]
        assert dumped["max_batch_size"] == 16
        assert dumped["recursive_scan"] is False
        # Default values should be included
        assert dumped["max_image_size"] == (4096, 4096)
        assert dumped["skip_hidden_files"] is True
