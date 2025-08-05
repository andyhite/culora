"""Service layer exceptions."""

from typing import Any

from .culora import CuLoRAError


class CuLoRAServiceError(CuLoRAError):
    """Exception for service layer errors.

    Raised when there are issues with service initialization, configuration,
    or operation.
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
            error_code=error_code or "SERVICE_ERROR",
            context=context,
            user_message=user_message,
        )


class ServiceInitializationError(CuLoRAServiceError):
    """Exception for service initialization errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "SERVICE_INIT_ERROR",
            context=context,
            user_message=user_message,
        )


class ServiceConfigurationError(CuLoRAServiceError):
    """Exception for service configuration errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or "SERVICE_CONFIG_ERROR",
            context=context,
            user_message=user_message,
        )
