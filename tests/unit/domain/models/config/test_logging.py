"""Tests for LoggingConfig model."""

import pytest
from pydantic import ValidationError

from culora.domain.enums import LogLevel
from culora.domain.models.config.logging import LoggingConfig


class TestLoggingConfig:
    """Test cases for LoggingConfig model."""

    def test_logging_config_default_values(self) -> None:
        """Test LoggingConfig default initialization."""
        config = LoggingConfig()
        assert config.log_level == LogLevel.INFO

    def test_logging_config_with_debug(self) -> None:
        """Test LoggingConfig with DEBUG level."""
        config = LoggingConfig(log_level=LogLevel.DEBUG)
        assert config.log_level == LogLevel.DEBUG

    def test_logging_config_with_warning(self) -> None:
        """Test LoggingConfig with WARNING level."""
        config = LoggingConfig(log_level=LogLevel.WARNING)
        assert config.log_level == LogLevel.WARNING

    def test_logging_config_with_error(self) -> None:
        """Test LoggingConfig with ERROR level."""
        config = LoggingConfig(log_level=LogLevel.ERROR)
        assert config.log_level == LogLevel.ERROR

    def test_logging_config_with_critical(self) -> None:
        """Test LoggingConfig with CRITICAL level."""
        config = LoggingConfig(log_level=LogLevel.CRITICAL)
        assert config.log_level == LogLevel.CRITICAL

    def test_logging_config_from_dict(self) -> None:
        """Test LoggingConfig creation from dictionary."""
        config_dict = {"log_level": "debug"}
        config = LoggingConfig(**config_dict)  # type: ignore[arg-type]
        assert config.log_level == LogLevel.DEBUG

    def test_logging_config_model_dump(self) -> None:
        """Test LoggingConfig serialization."""
        config = LoggingConfig(log_level=LogLevel.ERROR)
        dumped = config.model_dump()
        assert dumped == {"log_level": "error"}

    def test_logging_config_model_dump_json(self) -> None:
        """Test LoggingConfig JSON serialization."""
        config = LoggingConfig(log_level=LogLevel.WARNING)
        json_str = config.model_dump_json()
        assert '"log_level":"warning"' in json_str.replace(" ", "")

    def test_logging_config_invalid_log_level(self) -> None:
        """Test LoggingConfig with invalid log level."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(log_level="invalid_level")  # type: ignore[arg-type]

        error = exc_info.value.errors()[0]
        assert error["type"] == "enum"
        assert "log_level" in str(error["loc"])

    def test_logging_config_field_validation(self) -> None:
        """Test LoggingConfig field validation."""
        # Test that None is not accepted
        with pytest.raises(ValidationError):
            LoggingConfig(log_level=None)  # type: ignore[arg-type]

    def test_logging_config_equality(self) -> None:
        """Test LoggingConfig equality comparison."""
        config1 = LoggingConfig(log_level=LogLevel.DEBUG)
        config2 = LoggingConfig(log_level=LogLevel.DEBUG)
        config3 = LoggingConfig(log_level=LogLevel.INFO)

        assert config1 == config2
        assert config1 != config3

    def test_logging_config_repr(self) -> None:
        """Test LoggingConfig string representation."""
        config = LoggingConfig(log_level=LogLevel.ERROR)
        repr_str = repr(config)
        assert "LoggingConfig" in repr_str
        assert "log_level" in repr_str

    def test_logging_config_schema(self) -> None:
        """Test LoggingConfig JSON schema."""
        schema = LoggingConfig.model_json_schema()
        assert "properties" in schema
        assert "log_level" in schema["properties"]

        log_level_schema = schema["properties"]["log_level"]
        # In Pydantic v2, enum schemas use $ref or direct enum definition
        assert (
            "$ref" in log_level_schema
            or "enum" in log_level_schema
            or "anyOf" in log_level_schema
        )

        # Check that default value is present
        assert log_level_schema.get("default") == "info"

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
    def test_logging_config_all_log_levels(self, log_level: LogLevel) -> None:
        """Parametrized test for all log levels."""
        config = LoggingConfig(log_level=log_level)
        assert config.log_level == log_level

    @pytest.mark.parametrize(
        "level_string,expected_enum",
        [
            ("debug", LogLevel.DEBUG),
            ("info", LogLevel.INFO),
            ("warning", LogLevel.WARNING),
            ("error", LogLevel.ERROR),
            ("critical", LogLevel.CRITICAL),
        ],
    )
    def test_logging_config_string_to_enum(
        self, level_string: str, expected_enum: LogLevel
    ) -> None:
        """Test LoggingConfig with string values that get converted to enums."""
        config = LoggingConfig(log_level=level_string)  # type: ignore[arg-type]
        assert config.log_level == expected_enum
