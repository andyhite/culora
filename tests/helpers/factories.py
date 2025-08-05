"""Test data factories for creating test objects."""

from typing import Any

from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.domain.models import (
    CuLoRAConfig,
    DeviceConfig,
    FaceAnalysisConfig,
    ImageConfig,
    LoggingConfig,
    QualityConfig,
)


class ConfigBuilder:
    """Builder pattern for creating test configurations."""

    def __init__(self) -> None:
        self._device_type = DeviceType.CPU
        self._log_level = LogLevel.INFO
        self._image_config: ImageConfig | None = None
        self._face_config: FaceAnalysisConfig | None = None
        self._quality_config: QualityConfig | None = None

    def with_device(self, device_type: DeviceType) -> "ConfigBuilder":
        """Set the device type."""
        self._device_type = device_type
        return self

    def with_log_level(self, log_level: LogLevel) -> "ConfigBuilder":
        """Set the log level."""
        self._log_level = log_level
        return self

    def with_image_config(self, image_config: ImageConfig) -> "ConfigBuilder":
        """Set the image configuration."""
        self._image_config = image_config
        return self

    def with_face_config(self, face_config: FaceAnalysisConfig) -> "ConfigBuilder":
        """Set the face analysis configuration."""
        self._face_config = face_config
        return self

    def with_quality_config(self, quality_config: QualityConfig) -> "ConfigBuilder":
        """Set the quality configuration."""
        self._quality_config = quality_config
        return self

    def build(self) -> CuLoRAConfig:
        """Build the configuration."""
        device_config = DeviceConfig(preferred_device=self._device_type)
        logging_config = LoggingConfig(log_level=self._log_level)

        # Use provided configs or defaults
        image_config = self._image_config or ImageConfig()
        face_config = self._face_config or FaceAnalysisConfig()
        quality_config = self._quality_config or QualityConfig()

        return CuLoRAConfig(
            device=device_config,
            logging=logging_config,
            images=image_config,
            faces=face_config,
            quality=quality_config,
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
