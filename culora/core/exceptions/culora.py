"""Base exception class for CuLoRA."""

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
