"""Config-related exceptions."""

from typing import Any

from .culora import CuLoRAError


class ConfigError(CuLoRAError):
    """Base exception for configuration-related errors."""

    pass


class InvalidConfigError(ConfigError):
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


class MissingConfigError(ConfigError):
    """Raised when required configuration is missing."""

    def __init__(self, field_name: str, **kwargs: Any) -> None:
        message = f"Missing required configuration: '{field_name}'"
        context = {"field_name": field_name}
        user_message = f"Configuration error: Please provide a value for '{field_name}'"
        super().__init__(message, context=context, user_message=user_message, **kwargs)
