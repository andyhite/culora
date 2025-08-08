"""Configuration manager for CuLoRA."""

from typing import Any

from culora.config import AnalysisConfig, AnalysisStage


class ConfigManager:
    """Singleton configuration manager for CuLoRA."""

    _instance: "ConfigManager | None" = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._analysis_config = AnalysisConfig()
        self._initialized = True

    @property
    def analysis_config(self) -> AnalysisConfig:
        """Get the current analysis configuration."""
        return self._analysis_config

    def update_analysis_config(self, **kwargs: Any) -> None:
        """Update the analysis configuration with new values."""
        if "deduplication" in kwargs:
            self._analysis_config.deduplication = kwargs["deduplication"]
        if "quality" in kwargs:
            self._analysis_config.quality = kwargs["quality"]
        if "face" in kwargs:
            self._analysis_config.face = kwargs["face"]

    def set_stage_enabled(self, stage: AnalysisStage, enabled: bool) -> None:
        """Enable or disable a specific analysis stage."""
        if stage == AnalysisStage.DEDUPLICATION:
            self._analysis_config.deduplication.enabled = enabled
        elif stage == AnalysisStage.QUALITY:
            self._analysis_config.quality.enabled = enabled
        elif stage == AnalysisStage.FACE:
            self._analysis_config.face.enabled = enabled

    def get_stage_config(self, stage: AnalysisStage) -> Any:
        """Get configuration for a specific stage."""
        return self._analysis_config.get_stage_config(stage)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._analysis_config = AnalysisConfig()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get the singleton instance of ConfigManager."""
        return cls()
