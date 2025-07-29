"""Logging level enumeration."""

from enum import Enum


class LogLevel(str, Enum):
    """Logging levels for structured logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
