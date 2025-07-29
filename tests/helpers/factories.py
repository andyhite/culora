"""Test data factories for creating test objects."""

from typing import Any

from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.domain.models import CuLoRAConfig, DeviceConfig, LoggingConfig


class ConfigBuilder:
    """Builder pattern for creating test configurations."""

    def __init__(self) -> None:
        self._device_type = DeviceType.CPU
        self._log_level = LogLevel.INFO

    def with_device(self, device_type: DeviceType) -> "ConfigBuilder":
        """Set the device type."""
        self._device_type = device_type
        return self

    def with_log_level(self, log_level: LogLevel) -> "ConfigBuilder":
        """Set the log level."""
        self._log_level = log_level
        return self

    def build(self) -> CuLoRAConfig:
        """Build the configuration."""
        return CuLoRAConfig(
            device=DeviceConfig(preferred_device=self._device_type),
            logging=LoggingConfig(log_level=self._log_level),
        )


def create_test_config(**kwargs: Any) -> CuLoRAConfig:
    """Create a test configuration with optional overrides."""
    defaults: dict[str, Any] = {
        "device_type": DeviceType.CPU,
        "log_level": LogLevel.INFO,
    }
    defaults.update(kwargs)

    device_type = defaults["device_type"]
    log_level = defaults["log_level"]

    # Ensure proper types
    if not isinstance(device_type, DeviceType):
        device_type = DeviceType.CPU
    if not isinstance(log_level, LogLevel):
        log_level = LogLevel.INFO

    return CuLoRAConfig(
        device=DeviceConfig(preferred_device=device_type),
        logging=LoggingConfig(log_level=log_level),
    )
