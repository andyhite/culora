"""Main CuLoRA configuration model."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .composition import CompositionConfig
from .device import DeviceConfig
from .face import FaceAnalysisConfig
from .image import ImageConfig
from .logging import LoggingConfig
from .quality import QualityConfig


class CuLoRAConfig(BaseModel):
    """Main CuLoRA configuration.

    Combines all configuration sections into a single validated model.
    """

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    images: ImageConfig = Field(default_factory=ImageConfig)
    faces: FaceAnalysisConfig = Field(default_factory=FaceAnalysisConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    composition: CompositionConfig = Field(default_factory=CompositionConfig)

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
