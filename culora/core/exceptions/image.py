"""Image processing exceptions."""

from typing import Any

from .culora import CuLoRAError


class CuLoRAImageError(CuLoRAError):
    """Exception for image processing errors.

    Raised when there are issues with image loading, processing, or validation.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "IMAGE_ERROR",
            context=context,
            user_message=user_message,
        )


class ImageLoadError(CuLoRAImageError):
    """Exception for image loading errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "IMAGE_LOAD_ERROR",
            context=context,
            user_message=user_message,
        )


class ImageValidationError(CuLoRAImageError):
    """Exception for image validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "IMAGE_VALIDATION_ERROR",
            context=context,
            user_message=user_message,
        )
