"""Device configuration model."""

from pydantic import BaseModel, Field

from culora.domain.enums.device_types import DeviceType


class DeviceConfig(BaseModel):
    """Hardware device configuration."""

    preferred_device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Preferred device type for AI model execution",
    )
