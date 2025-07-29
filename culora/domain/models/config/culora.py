"""Main CuLoRA configuration model."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .device import DeviceConfig
from .logging import LoggingConfig


class CuLoRAConfig(BaseModel):
    """Main CuLoRA configuration.

    Combines all configuration sections into a single validated model.
    """

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            Path: str,
        },
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Export configuration as JSON string."""
        return super().model_dump_json(indent=2, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CuLoRAConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def get_section(self, section_name: str) -> Any:
        """Get a specific configuration section."""
        return getattr(self, section_name)
