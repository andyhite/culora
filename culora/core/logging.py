"""Structured logging system for CuLoRA.

This module provides structured logging that writes JSON logs to files
while keeping user-facing output through Rich console separate.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import structlog
from rich.console import Console

from .exceptions import CuLoRAError
from .types import LogLevel


class CuLoRALogger:
    """Structured logger for CuLoRA operations.

    Provides context-aware logging with structured JSON output to files
    and separation from user-facing Rich console output.
    """

    def __init__(self, name: str, context: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.context = context or {}
        self._logger = structlog.get_logger(name)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._logger.debug(message, **self._merge_context(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._logger.info(message, **self._merge_context(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._logger.warning(message, **self._merge_context(kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._logger.error(message, **self._merge_context(kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self._logger.critical(message, **self._merge_context(kwargs))

    def exception(
        self, message: str, exc_info: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Log exception with full context."""
        context = self._merge_context(kwargs)

        if exc_info and isinstance(exc_info, CuLoRAError):
            # Add structured exception information
            context.update(exc_info.to_dict())

        self._logger.error(message, exc_info=exc_info, **context)

    def bind(self, **kwargs: Any) -> "CuLoRALogger":
        """Create a new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return CuLoRALogger(self.name, new_context)

    def _merge_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge instance context with kwargs."""
        return {**self.context, **kwargs}


def setup_logging(
    log_level: LogLevel = LogLevel.INFO,
    log_dir: Optional[Path] = None,
    console_output: bool = False,
) -> None:
    """Configure structured logging for CuLoRA.

    Args:
        log_level: Minimum log level to capture
        log_dir: Directory for log files (defaults to ./logs)
        console_output: Whether to also output logs to console (for development)
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.value.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure Python logging to write to file
    logging.basicConfig(
        level=getattr(logging, log_level.value.upper()),
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_dir / "culora.log"),
            *([logging.StreamHandler(sys.stdout)] if console_output else []),
        ],
    )

    # Suppress third-party library logs unless they're warnings or errors
    for logger_name in ["PIL", "torch", "transformers", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> CuLoRALogger:
    """Get a CuLoRA logger instance.

    Args:
        name: Logger name (typically module name)
        **context: Additional context to include in all log messages

    Returns:
        Configured CuLoRA logger instance
    """
    return CuLoRALogger(name, context)


# Performance timing decorator
def log_performance(logger: CuLoRALogger, operation: str) -> Callable[..., Any]:
    """Decorator to log performance timing for operations.

    Args:
        logger: Logger instance to use
        operation: Name of the operation being timed
    """
    import time
    from functools import wraps

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger.info(f"Starting {operation}", operation=operation)

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"Completed {operation}",
                    operation=operation,
                    execution_time_seconds=execution_time,
                    status="success",
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {operation}",
                    operation=operation,
                    execution_time_seconds=execution_time,
                    status="failed",
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


# Rich console for user-facing output (separate from logging)
console = Console()


def get_console() -> Console:
    """Get the Rich console instance for user-facing output.

    This is separate from the logging system to ensure structured logs
    don't interfere with user interface.
    """
    return console
