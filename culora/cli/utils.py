"""CLI utility functions."""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def handle_cli_error() -> Generator[None, None, None]:
    """Context manager for handling CLI errors gracefully."""
    try:
        yield
    except Exception:
        # Re-raise exceptions to be handled by the calling code
        raise
