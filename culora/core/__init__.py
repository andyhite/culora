"""Core module for CuLoRA.

Configuration, logging, and device management functionality.

This module provides the foundational components for the CuLoRA application:
- Structured logging with file-based JSON output
- Type-safe Pydantic configuration models
- Custom exception hierarchy with contextual information
- Configuration management with multiple source support
"""

from .config import CuLoRAConfig
from .config_manager import ConfigManager, get_config, get_config_manager
from .exceptions import (
    ConfigurationError,
    CuLoRAError,
    DeviceError,
    ExportError,
    ProcessingError,
)
from .logging import CuLoRALogger, get_console, get_logger, setup_logging
from .types import DeviceType, LogLevel, ProcessingStatus

__all__ = [
    "ConfigManager",
    "ConfigurationError",
    "CuLoRAConfig",
    "CuLoRAError",
    "CuLoRALogger",
    "DeviceError",
    "DeviceType",
    "ExportError",
    "LogLevel",
    "ProcessingError",
    "ProcessingStatus",
    "get_config",
    "get_config_manager",
    "get_console",
    "get_logger",
    "setup_logging",
]
