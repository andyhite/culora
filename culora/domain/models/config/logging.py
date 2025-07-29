"""Logging configuration model."""

from pydantic import BaseModel, Field

from culora.domain.enums import LogLevel


class LoggingConfig(BaseModel):
    """Configuration for logging system behavior."""

    log_level: LogLevel = Field(default=LogLevel.INFO)
