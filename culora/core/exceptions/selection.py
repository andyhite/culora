"""Selection-related exceptions."""

from .culora import CuLoRAError


class SelectionError(CuLoRAError):
    """Base exception for selection-related errors."""

    pass


class SelectionConfigurationError(SelectionError):
    """Exception raised for selection configuration errors."""

    pass


class SelectionInsufficientDataError(SelectionError):
    """Exception raised when insufficient data is available for selection."""

    pass


class SelectionExecutionError(SelectionError):
    """Exception raised during selection execution."""

    pass
