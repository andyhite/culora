"""Tests for logging utilities."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from culora.core.exceptions import CuLoRAError
from culora.domain.enums import LogLevel
from culora.utils.logging import (
    LoggingService,
    get_logger,
    log_performance,
    setup_logging,
)


class TestLoggingService:
    """Test cases for LoggingService."""

    def test_logging_service_initialization(self, mock_structlog_logger: Mock) -> None:
        """Test LoggingService initialization."""
        with patch(
            "structlog.get_logger", return_value=mock_structlog_logger
        ) as mock_get:
            service = LoggingService("test_logger", {"key": "value"})

            assert service.name == "test_logger"
            assert service.context == {"key": "value"}
            assert service._logger == mock_structlog_logger
            mock_get.assert_called_once_with("test_logger")

    def test_logging_service_initialization_no_context(
        self, mock_structlog_logger: Mock
    ) -> None:
        """Test LoggingService initialization without context."""
        with patch("structlog.get_logger", return_value=mock_structlog_logger):
            service = LoggingService("test_logger")

            assert service.name == "test_logger"
            assert service.context == {}

    def test_debug_logging(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test debug level logging."""
        logging_service.debug("Debug message", extra_key="extra_value")

        mock_structlog_logger.debug.assert_called_once_with(
            "Debug message", extra_key="extra_value"
        )

    def test_info_logging(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test info level logging."""
        logging_service.info("Info message", info_key="info_value")

        mock_structlog_logger.info.assert_called_once_with(
            "Info message", info_key="info_value"
        )

    def test_warning_logging(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test warning level logging."""
        logging_service.warning("Warning message", warning_key="warning_value")

        mock_structlog_logger.warning.assert_called_once_with(
            "Warning message", warning_key="warning_value"
        )

    def test_error_logging(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test error level logging."""
        logging_service.error("Error message", error_key="error_value")

        mock_structlog_logger.error.assert_called_once_with(
            "Error message", error_key="error_value"
        )

    def test_critical_logging(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test critical level logging."""
        logging_service.critical("Critical message", critical_key="critical_value")

        mock_structlog_logger.critical.assert_called_once_with(
            "Critical message", critical_key="critical_value"
        )

    def test_exception_logging_with_culora_error(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test exception logging with CuLoRAError."""
        error = CuLoRAError(
            "Test error", error_code="TEST_ERROR", context={"error_context": "test"}
        )

        logging_service.exception(
            "Exception occurred", exc_info=error, extra_key="extra"
        )

        expected_context = {
            "extra_key": "extra",
            "error_type": "CuLoRAError",
            "error_code": "TEST_ERROR",
            "message": "Test error",
            "user_message": "Test error",
            "context": {"error_context": "test"},
        }

        mock_structlog_logger.error.assert_called_once_with(
            "Exception occurred", exc_info=error, **expected_context
        )

    def test_exception_logging_with_regular_exception(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test exception logging with regular exception."""
        error = ValueError("Regular error")

        logging_service.exception(
            "Exception occurred", exc_info=error, extra_key="extra"
        )

        mock_structlog_logger.error.assert_called_once_with(
            "Exception occurred", exc_info=error, extra_key="extra"
        )

    def test_exception_logging_no_exc_info(
        self, logging_service: LoggingService, mock_structlog_logger: Mock
    ) -> None:
        """Test exception logging without exc_info."""
        logging_service.exception("Exception occurred", extra_key="extra")

        mock_structlog_logger.error.assert_called_once_with(
            "Exception occurred", exc_info=None, extra_key="extra"
        )

    def test_bind_creates_new_logger(self, mock_structlog_logger: Mock) -> None:
        """Test that bind creates a new logger instance."""
        with patch("structlog.get_logger", return_value=mock_structlog_logger):
            original_service = LoggingService("test_logger", {"original": "context"})

            bound_service = original_service.bind(
                new_key="new_value", original="overridden"
            )

            assert bound_service is not original_service
            assert bound_service.name == "test_logger"
            assert bound_service.context == {
                "original": "overridden",
                "new_key": "new_value",
            }

    def test_merge_context_with_instance_context(
        self, logging_service: LoggingService
    ) -> None:
        """Test context merging with instance context."""
        logging_service.context = {"instance_key": "instance_value"}

        merged = logging_service._merge_context({"kwarg_key": "kwarg_value"})

        expected = {"instance_key": "instance_value", "kwarg_key": "kwarg_value"}
        assert merged == expected

    def test_merge_context_kwargs_override(
        self, logging_service: LoggingService
    ) -> None:
        """Test that kwargs override instance context."""
        logging_service.context = {"shared_key": "instance_value"}

        merged = logging_service._merge_context({"shared_key": "kwarg_value"})

        assert merged == {"shared_key": "kwarg_value"}


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_default_values(self) -> None:
        """Test setup_logging with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch("structlog.configure"),
                patch("logging.basicConfig") as mock_basic_config,
            ):

                setup_logging(log_dir=log_dir)

                mock_basic_config.assert_called_once()

    def test_setup_logging_creates_log_directory(self) -> None:
        """Test that setup_logging creates log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "nonexistent_logs"

            assert not log_dir.exists()

            with patch("structlog.configure"), patch("logging.basicConfig"):
                setup_logging(log_dir=log_dir)

            assert log_dir.exists()

    def test_setup_logging_with_console_output(self) -> None:
        """Test setup_logging with console output enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch("structlog.configure"),
                patch("logging.basicConfig") as mock_basic_config,
            ):

                setup_logging(log_dir=log_dir, console_output=True)

                # Check that basicConfig was called with both file and console handlers
                call_args = mock_basic_config.call_args
                handlers = call_args[1]["handlers"]
                assert len(handlers) == 2  # FileHandler + StreamHandler

    def test_setup_logging_without_console_output(self) -> None:
        """Test setup_logging without console output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch("structlog.configure"),
                patch("logging.basicConfig") as mock_basic_config,
            ):

                setup_logging(log_dir=log_dir, console_output=False)

                # Check that basicConfig was called with only file handler
                call_args = mock_basic_config.call_args
                handlers = call_args[1]["handlers"]
                assert len(handlers) == 1  # Only FileHandler

    @pytest.mark.parametrize(
        "log_level",
        [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ],
    )
    def test_setup_logging_log_levels(self, log_level: LogLevel) -> None:
        """Parametrized test for different log levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch("structlog.configure"),
                patch("logging.basicConfig") as mock_basic_config,
            ):

                setup_logging(log_level=log_level, log_dir=log_dir)

                # Check that the correct log level was used
                expected_level = getattr(logging, log_level.value.upper())
                call_args = mock_basic_config.call_args
                assert call_args[1]["level"] == expected_level

    def test_setup_logging_suppresses_third_party_logs(self) -> None:
        """Test that setup_logging suppresses third-party library logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch("structlog.configure"),
                patch("logging.basicConfig"),
                patch("logging.getLogger") as mock_get_logger,
            ):

                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger

                setup_logging(log_dir=log_dir)

                # Check that third-party loggers were configured
                expected_calls: list[tuple[tuple[str, ...], dict[str, str]]] = [
                    (("PIL",), {}),
                    (("torch",), {}),
                    (("transformers",), {}),
                    (("urllib3",), {}),
                ]

                assert mock_get_logger.call_count == 4
                for call in expected_calls:
                    assert call in mock_get_logger.call_args_list

                # Check that setLevel was called with WARNING
                assert mock_logger.setLevel.call_count == 4
                for call in mock_logger.setLevel.call_args_list:
                    assert call[0][0] == logging.WARNING


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_basic(self) -> None:
        """Test get_logger basic functionality."""
        logger = get_logger("test_module")

        assert isinstance(logger, LoggingService)
        assert logger.name == "test_module"
        assert logger.context == {}

    def test_get_logger_with_context(self) -> None:
        """Test get_logger with context."""
        context = {"key1": "value1", "key2": "value2"}
        logger = get_logger("test_module", **context)

        assert isinstance(logger, LoggingService)
        assert logger.name == "test_module"
        assert logger.context == context


class TestLogPerformance:
    """Test cases for log_performance decorator."""

    def test_log_performance_success(self, mock_logger: Mock) -> None:
        """Test log_performance decorator on successful function."""

        @log_performance(mock_logger, "test_operation")
        def test_function(x: int, y: int) -> int:
            return x + y

        with patch("time.time", side_effect=[1000.0, 1001.5]):  # 1.5 second execution
            result = test_function(2, 3)

        assert result == 5

        # Check logging calls
        assert mock_logger.info.call_count == 2

        # Check start log
        start_call = mock_logger.info.call_args_list[0]
        assert start_call[0][0] == "Starting test_operation"
        assert start_call[1]["operation"] == "test_operation"

        # Check completion log
        completion_call = mock_logger.info.call_args_list[1]
        assert completion_call[0][0] == "Completed test_operation"
        assert completion_call[1]["operation"] == "test_operation"
        assert completion_call[1]["execution_time_seconds"] == 1.5
        assert completion_call[1]["status"] == "success"

    def test_log_performance_exception(self, mock_logger: Mock) -> None:
        """Test log_performance decorator on function that raises exception."""

        @log_performance(mock_logger, "test_operation")
        def test_function() -> None:
            raise ValueError("Test error")

        with (
            patch("time.time", side_effect=[1000.0, 1001.0]),  # 1 second execution
            pytest.raises(ValueError),
        ):
            test_function()

        # Check logging calls
        assert mock_logger.info.call_count == 1  # Only start log
        assert mock_logger.error.call_count == 1  # Error log

        # Check start log
        start_call = mock_logger.info.call_args_list[0]
        assert start_call[0][0] == "Starting test_operation"

        # Check error log
        error_call = mock_logger.error.call_args_list[0]
        assert error_call[0][0] == "Failed test_operation"
        assert error_call[1]["operation"] == "test_operation"
        assert error_call[1]["execution_time_seconds"] == 1.0
        assert error_call[1]["status"] == "failed"
        assert error_call[1]["error"] == "Test error"

    def test_log_performance_preserves_function_metadata(
        self, mock_logger: Mock
    ) -> None:
        """Test that log_performance preserves function metadata."""

        @log_performance(mock_logger, "test_operation")
        def test_function(x: int, y: int) -> int:
            """Test function docstring."""
            return x + y

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test function docstring."

    def test_log_performance_with_args_and_kwargs(self, mock_logger: Mock) -> None:
        """Test log_performance decorator with various arguments."""

        @log_performance(mock_logger, "test_operation")
        def test_function(
            *args: int, **kwargs: str
        ) -> dict[str, tuple[int, ...] | dict[str, str]]:
            return {"args": args, "kwargs": kwargs}

        with patch("time.time", side_effect=[1000.0, 1000.1]):
            result = test_function(1, 2, key1="value1", key2="value2")

        expected = {"args": (1, 2), "kwargs": {"key1": "value1", "key2": "value2"}}
        assert result == expected

    @pytest.mark.parametrize(
        "operation_name",
        [
            "simple_operation",
            "complex_data_processing",
            "model_inference",
            "file_operation",
        ],
    )
    def test_log_performance_different_operations(
        self, mock_logger: Mock, operation_name: str
    ) -> None:
        """Parametrized test for different operation names."""

        @log_performance(mock_logger, operation_name)
        def test_function() -> str:
            return "success"

        with patch("time.time", side_effect=[1000.0, 1000.5]):
            result = test_function()

        assert result == "success"

        # Check that the operation name is used correctly
        start_call = mock_logger.info.call_args_list[0]
        completion_call = mock_logger.info.call_args_list[1]

        assert start_call[1]["operation"] == operation_name
        assert completion_call[1]["operation"] == operation_name
