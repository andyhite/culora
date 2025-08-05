"""CLI utility functions."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import structlog


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@contextmanager
def handle_cli_error() -> Generator[None, None, None]:
    """Context manager for handling CLI errors gracefully."""
    try:
        yield
    except Exception:
        # Re-raise exceptions to be handled by the calling code
        raise
