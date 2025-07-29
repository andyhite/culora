"""Custom exception hierarchy for CuLoRA.

This module defines structured error handling with clear categorization
and contextual information for debugging and user feedback.
"""

from typing import Any


class CuLoRAError(Exception):
    """Base exception for all CuLoRA errors.

    Provides structured error handling with error codes, context information,
    and user-friendly messages.

    Args:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        context: Additional context information for debugging
        user_message: User-friendly message for CLI display
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
        }


class ConfigurationError(CuLoRAError):
    """Base exception for configuration-related errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(
        self,
        field_name: str,
        field_value: Any,
        expected: str,
        **kwargs: Any,
    ) -> None:
        message = f"Invalid configuration for '{field_name}': {field_value}. Expected: {expected}"
        context = {
            "field_name": field_name,
            "field_value": field_value,
            "expected": expected,
        }
        super().__init__(message, context=context, **kwargs)


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, field_name: str, **kwargs: Any) -> None:
        message = f"Missing required configuration: '{field_name}'"
        context = {"field_name": field_name}
        user_message = f"Configuration error: Please provide a value for '{field_name}'"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class ProcessingError(CuLoRAError):
    """Base exception for processing-related errors."""

    pass


class ImageProcessingError(ProcessingError):
    """Raised when image processing fails."""

    def __init__(
        self,
        image_path: str,
        operation: str,
        original_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to process image '{image_path}' during {operation}"
        context = {
            "image_path": image_path,
            "operation": operation,
            "original_error": str(original_error) if original_error else None,
        }
        user_message = f"Unable to process image: {image_path}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class FaceDetectionError(ProcessingError):
    """Raised when face detection fails."""

    def __init__(
        self,
        image_path: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        message = f"Face detection failed for '{image_path}': {reason}"
        context = {"image_path": image_path, "reason": reason}
        user_message = f"Could not detect faces in: {image_path}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class QualityAssessmentError(ProcessingError):
    """Raised when quality assessment fails."""

    def __init__(
        self,
        image_path: str,
        metric: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        message = f"Quality assessment failed for '{image_path}' ({metric}): {reason}"
        context = {"image_path": image_path, "metric": metric, "reason": reason}
        user_message = f"Quality assessment failed for: {image_path}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class ModelLoadError(ProcessingError):
    """Raised when AI model loading fails."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to load model '{model_name}': {reason}"
        context = {"model_name": model_name, "reason": reason}
        user_message = f"Unable to load AI model: {model_name}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class DeviceError(CuLoRAError):
    """Base exception for device-related errors."""

    pass


class DeviceNotFoundError(DeviceError):
    """Raised when requested device is not available."""

    def __init__(self, device_type: str, **kwargs: Any) -> None:
        message = f"Requested device type '{device_type}' is not available"
        context = {"device_type": device_type}
        user_message = f"Device '{device_type}' not found. Falling back to CPU."
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class InsufficientMemoryError(DeviceError):
    """Raised when device has insufficient memory."""

    def __init__(
        self,
        device_type: str,
        required_mb: int,
        available_mb: int,
        **kwargs: Any,
    ) -> None:
        message = f"Insufficient memory on {device_type}: required {required_mb}MB, available {available_mb}MB"
        context = {
            "device_type": device_type,
            "required_mb": required_mb,
            "available_mb": available_mb,
        }
        user_message = f"Not enough memory on {device_type}. Consider using CPU or reducing batch size."
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class ExportError(CuLoRAError):
    """Base exception for export-related errors."""

    pass


class OutputDirectoryError(ExportError):
    """Raised when output directory operations fail."""

    def __init__(self, directory_path: str, reason: str, **kwargs: Any) -> None:
        message = f"Output directory error '{directory_path}': {reason}"
        context = {"directory_path": directory_path, "reason": reason}
        user_message = f"Cannot access output directory: {directory_path}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)


class FileWriteError(ExportError):
    """Raised when file writing fails."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to write file '{file_path}': {reason}"
        context = {"file_path": file_path, "reason": reason}
        user_message = f"Cannot write file: {file_path}"
        super().__init__(message, context=context, user_message=user_message, **kwargs)
