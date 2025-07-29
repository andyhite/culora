"""Tests for structured logging system."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from culora.core.exceptions import CuLoRAError
from culora.core.logging import CuLoRALogger, get_console, get_logger, setup_logging
from culora.core.types import LogLevel


class TestCuLoRALogger:
    """Test CuLoRA logger functionality."""

    def test_logger_creation(self) -> None:
        """Test creating logger with context."""
        context = {"module": "test", "version": "1.0"}
        logger = CuLoRALogger("test_logger", context)

        assert logger.name == "test_logger"
        assert logger.context == context

    def test_logger_without_context(self) -> None:
        """Test creating logger without initial context."""
        logger = CuLoRALogger("test_logger")

        assert logger.name == "test_logger"
        assert logger.context == {}

    def test_logger_bind(self) -> None:
        """Test binding additional context to logger."""
        logger = CuLoRALogger("test", {"initial": "value"})
        bound_logger = logger.bind(operation="test_op", count=5)

        assert bound_logger.context == {
            "initial": "value",
            "operation": "test_op",
            "count": 5,
        }

        # Original logger should be unchanged
        assert logger.context == {"initial": "value"}

    def test_context_merging(self) -> None:
        """Test that context is properly merged in log calls."""
        logger = CuLoRALogger("test", {"base": "value"})

        # Test that _merge_context works correctly
        merged = logger._merge_context({"additional": "data", "base": "overridden"})

        assert merged["base"] == "overridden"  # kwargs override instance context
        assert merged["additional"] == "data"

    @patch("structlog.get_logger")
    def test_log_methods(self, mock_get_logger: Any) -> None:
        """Test that log methods call structlog correctly."""
        mock_structlog = mock_get_logger.return_value
        logger = CuLoRALogger("test", {"context": "value"})

        # Test each log level
        logger.debug("Debug message", extra="data")
        mock_structlog.debug.assert_called_with(
            "Debug message", context="value", extra="data"
        )

        logger.info("Info message", count=42)
        mock_structlog.info.assert_called_with(
            "Info message", context="value", count=42
        )

        logger.warning("Warning message")
        mock_structlog.warning.assert_called_with("Warning message", context="value")

        logger.error("Error message", error_code="TEST")
        mock_structlog.error.assert_called_with(
            "Error message", context="value", error_code="TEST"
        )

    @patch("structlog.get_logger")
    def test_exception_logging(self, mock_get_logger: Any) -> None:
        """Test logging exceptions with CuLoRA error context."""
        mock_structlog = mock_get_logger.return_value
        logger = CuLoRALogger("test")

        # Test with CuLoRAError
        culora_error = CuLoRAError(
            "Test error", error_code="TEST_ERROR", context={"key": "value"}
        )

        logger.exception("Exception occurred", exc_info=culora_error)

        # Should merge exception context
        call_kwargs = mock_structlog.error.call_args[1]
        assert call_kwargs["error_type"] == "CuLoRAError"
        assert call_kwargs["error_code"] == "TEST_ERROR"
        assert call_kwargs["exc_info"] == culora_error


class TestLoggingSetup:
    """Test logging setup and configuration."""

    def test_setup_logging_default(self, temp_log_dir: Path) -> None:
        """Test setting up logging with default parameters."""
        setup_logging(log_dir=temp_log_dir)

        # Check that log directory and file are created
        assert temp_log_dir.exists()

        # Test that a logger can be created and used
        logger = get_logger("test_module")
        assert isinstance(logger, CuLoRALogger)

    def test_setup_logging_with_level(self, temp_log_dir: Path) -> None:
        """Test setting up logging with specific log level."""
        setup_logging(
            log_level=LogLevel.DEBUG,
            log_dir=temp_log_dir,
            console_output=True,
        )

        # Verify that we can create loggers (basic functionality test)
        logger = get_logger("test_debug")
        assert isinstance(logger, CuLoRALogger)

        # Check that log file exists
        log_file = temp_log_dir / "culora.log"
        logger.debug("Test debug message")
        assert log_file.exists()

    def test_log_file_creation(self, temp_log_dir: Path) -> None:
        """Test that log files are created correctly."""
        setup_logging(log_dir=temp_log_dir)

        # Create a logger and log a message
        logger = get_logger("test_file")
        logger.info("Test message")

        # Check log file exists
        log_file = temp_log_dir / "culora.log"
        assert log_file.exists()

    def test_third_party_logger_suppression(self, temp_log_dir: Path) -> None:
        """Test that third-party loggers are suppressed."""
        setup_logging(log_dir=temp_log_dir)

        # Test that third-party loggers have warning level
        pil_logger = logging.getLogger("PIL")
        torch_logger = logging.getLogger("torch")

        assert pil_logger.level == logging.WARNING
        assert torch_logger.level == logging.WARNING


class TestGetLogger:
    """Test logger factory function."""

    def test_get_logger_basic(self) -> None:
        """Test getting logger with name only."""
        logger = get_logger("test_module")

        assert isinstance(logger, CuLoRALogger)
        assert logger.name == "test_module"
        assert logger.context == {}

    def test_get_logger_with_context(self) -> None:
        """Test getting logger with initial context."""
        context = {"component": "face_detection", "version": "1.0"}
        logger = get_logger("face_module", **context)

        assert logger.context == context


class TestConsole:
    """Test Rich console functionality."""

    def test_get_console(self) -> None:
        """Test getting Rich console instance."""
        console = get_console()

        # Should return Rich Console instance
        assert hasattr(console, "print")
        assert hasattr(console, "log")

    def test_console_singleton(self) -> None:
        """Test that console is a singleton."""
        console1 = get_console()
        console2 = get_console()

        # Should be the same instance
        assert console1 is console2


class TestLogPerformanceDecorator:
    """Test performance logging decorator."""

    @patch("time.time")
    @patch("structlog.get_logger")
    def test_performance_decorator_success(
        self, mock_get_logger: Any, mock_time: Any
    ) -> None:
        """Test performance decorator on successful function."""
        from culora.core.logging import log_performance

        # Mock time progression
        mock_time.side_effect = [100.0, 101.5]  # 1.5 second execution

        mock_structlog = mock_get_logger.return_value
        logger = CuLoRALogger("test")

        @log_performance(logger, "test_operation")
        def test_function() -> str:
            return "success"

        result = test_function()

        assert result == "success"

        # Should log start and completion
        assert mock_structlog.info.call_count == 2

        # Check completion log
        completion_call = mock_structlog.info.call_args_list[1]
        assert "Completed test_operation" in completion_call[0][0]
        assert completion_call[1]["execution_time_seconds"] == 1.5
        assert completion_call[1]["status"] == "success"

    @patch("time.time")
    @patch("structlog.get_logger")
    def test_performance_decorator_failure(
        self, mock_get_logger: Any, mock_time: Any
    ) -> None:
        """Test performance decorator on failing function."""
        from culora.core.logging import log_performance

        mock_time.side_effect = [100.0, 100.8]  # 0.8 second execution
        mock_structlog = mock_get_logger.return_value
        logger = CuLoRALogger("test")

        @log_performance(logger, "failing_operation")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Should log start and error
        mock_structlog.info.assert_called_once()  # Start log
        mock_structlog.error.assert_called_once()  # Error log

        # Check error log
        error_call = mock_structlog.error.call_args
        assert "Failed failing_operation" in error_call[0][0]
        assert (
            abs(error_call[1]["execution_time_seconds"] - 0.8) < 0.001
        )  # Allow for floating point precision
        assert error_call[1]["status"] == "failed"
