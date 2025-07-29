"""Logging fixtures for testing."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog

from culora.core.logging import CuLoRALogger, setup_logging
from culora.core.types import LogLevel


class LogCapture:
    """Capture structured logs for testing."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def capture(
        self, logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Capture log record."""
        self.records.append(event_dict)
        return event_dict

    def clear(self) -> None:
        """Clear captured records."""
        self.records.clear()

    def get_records(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get captured records, optionally filtered by level."""
        if level is None:
            return self.records.copy()
        return [r for r in self.records if r.get("level") == level]


@pytest.fixture
def log_capture() -> LogCapture:
    """Provide a log capture instance."""
    return LogCapture()


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provide a mock logger for testing."""
    return MagicMock(spec=CuLoRALogger)


@pytest.fixture
def test_logger(temp_log_dir: Path) -> Generator[CuLoRALogger, None, None]:
    """Provide a test logger with temporary log directory."""
    # Setup logging for testing
    setup_logging(
        log_level=LogLevel.DEBUG,
        log_dir=temp_log_dir,
        console_output=False,
    )

    # Create test logger
    logger = CuLoRALogger("test_logger", {"test_context": "fixture"})

    yield logger

    # Cleanup
    logging.shutdown()


@pytest.fixture
def captured_logs(
    log_capture: LogCapture,
) -> Generator[LogCapture, None, None]:
    """Capture logs during test execution."""
    # Configure structlog to use our capture processor
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            log_capture.capture,  # type: ignore[list-item]  # Custom capture processor
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    yield log_capture

    # Cleanup
    log_capture.clear()


@pytest.fixture
def sample_log_context() -> dict[str, Any]:
    """Provide sample log context data."""
    return {
        "operation": "test_operation",
        "image_path": "/test/image.jpg",
        "processing_time": 1.23,
        "batch_size": 16,
    }
