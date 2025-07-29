"""Tests for custom exception hierarchy."""

import pytest

from culora.core.exceptions import (
    ConfigurationError,
    CuLoRAError,
    DeviceNotFoundError,
    FaceDetectionError,
    ImageProcessingError,
    InsufficientMemoryError,
    InvalidConfigError,
    MissingConfigError,
)


class TestCuLoRAError:
    """Test base CuLoRA exception."""

    def test_basic_error_creation(self) -> None:
        """Test creating basic error with message."""
        error = CuLoRAError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "CuLoRAError"
        assert error.user_message == "Test error message"
        assert error.context == {}

    def test_error_with_full_context(self) -> None:
        """Test creating error with all parameters."""
        context = {"key": "value", "number": 42}
        error = CuLoRAError(
            message="Detailed error message",
            error_code="CUSTOM_ERROR",
            context=context,
            user_message="User-friendly message",
        )

        assert error.message == "Detailed error message"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.context == context
        assert error.user_message == "User-friendly message"

    def test_error_to_dict(self) -> None:
        """Test converting error to dictionary."""
        context = {"operation": "test", "value": 123}
        error = CuLoRAError(
            message="Test message",
            error_code="TEST_ERROR",
            context=context,
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "CuLoRAError"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test message"
        assert error_dict["context"] == context


class TestConfigurationErrors:
    """Test configuration-specific errors."""

    def test_invalid_config_error(self) -> None:
        """Test InvalidConfigError with field information."""
        error = InvalidConfigError(
            field_name="batch_size",
            field_value=-1,
            expected="positive integer",
        )

        assert "batch_size" in error.message
        assert "-1" in error.message
        assert "positive integer" in error.message
        assert error.context["field_name"] == "batch_size"
        assert error.context["field_value"] == -1

    def test_missing_config_error(self) -> None:
        """Test MissingConfigError with helpful message."""
        error = MissingConfigError(field_name="api_key")

        assert "api_key" in error.message
        assert "api_key" in error.user_message
        assert error.context["field_name"] == "api_key"


class TestProcessingErrors:
    """Test processing-specific errors."""

    def test_image_processing_error(self) -> None:
        """Test ImageProcessingError with image context."""
        original_error = ValueError("Invalid format")
        error = ImageProcessingError(
            image_path="/path/to/image.jpg",
            operation="resize",
            original_error=original_error,
        )

        assert "/path/to/image.jpg" in error.message
        assert "resize" in error.message
        assert error.context["image_path"] == "/path/to/image.jpg"
        assert error.context["operation"] == "resize"
        assert "Invalid format" in error.context["original_error"]

    def test_face_detection_error(self) -> None:
        """Test FaceDetectionError with specific reason."""
        error = FaceDetectionError(
            image_path="/test/image.jpg",
            reason="No faces found",
        )

        assert "/test/image.jpg" in error.message
        assert "No faces found" in error.message
        assert error.context["reason"] == "No faces found"


class TestDeviceErrors:
    """Test device-specific errors."""

    def test_device_not_found_error(self) -> None:
        """Test DeviceNotFoundError with device type."""
        error = DeviceNotFoundError(device_type="cuda")

        assert "cuda" in error.message
        assert "cuda" in error.user_message
        assert error.context["device_type"] == "cuda"

    def test_insufficient_memory_error(self) -> None:
        """Test InsufficientMemoryError with memory details."""
        error = InsufficientMemoryError(
            device_type="cuda",
            required_mb=2048,
            available_mb=1024,
        )

        assert "cuda" in error.message
        assert "2048" in error.message
        assert "1024" in error.message
        assert error.context["required_mb"] == 2048
        assert error.context["available_mb"] == 1024


class TestErrorInheritance:
    """Test error inheritance hierarchy."""

    def test_configuration_error_inheritance(self) -> None:
        """Test that configuration errors inherit from base."""
        error = InvalidConfigError("field", "value", "expected")

        assert isinstance(error, ConfigurationError)
        assert isinstance(error, CuLoRAError)
        assert isinstance(error, Exception)

    def test_processing_error_inheritance(self) -> None:
        """Test that processing errors inherit correctly."""
        error = ImageProcessingError("/path", "operation")

        # Should be able to catch as any parent type
        with pytest.raises(CuLoRAError):
            raise error

        with pytest.raises(CuLoRAError):
            raise error
