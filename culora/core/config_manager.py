"""Configuration management system for CuLoRA.

This module provides unified configuration loading, validation, and management
with support for multiple sources and override precedence.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config import CuLoRAConfig
from .exceptions import ConfigurationError, InvalidConfigError, MissingConfigError
from .logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Unified configuration manager for CuLoRA.

    Handles configuration loading from multiple sources with proper precedence:
    CLI args > environment variables > config files > defaults
    """

    def __init__(self) -> None:
        self._config: CuLoRAConfig | None = None
        self._config_sources: dict[str, str] = {}

    def load_config(
        self,
        config_file: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
        env_prefix: str = "CULORA",
    ) -> CuLoRAConfig:
        """Load and validate configuration from multiple sources.

        Args:
            config_file: Path to configuration file (YAML or JSON)
            cli_overrides: CLI argument overrides
            env_prefix: Environment variable prefix

        Returns:
            Validated CuLoRA configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        logger.info(
            "Loading configuration",
            config_file=str(config_file) if config_file else None,
        )

        try:
            # Start with default configuration
            config_dict: dict[str, Any] = {}
            self._config_sources["defaults"] = "Built-in defaults"

            # Load from config file if provided
            if config_file and config_file.exists():
                file_config = self._load_config_file(config_file)
                config_dict = self._deep_merge(config_dict, file_config)
                self._config_sources["file"] = str(config_file)
                logger.info(
                    "Loaded configuration from file", config_file=str(config_file)
                )

            # Load from environment variables
            env_config = self._load_env_config(env_prefix)
            if env_config:
                config_dict = self._deep_merge(config_dict, env_config)
                self._config_sources["environment"] = f"{env_prefix}_* variables"
                logger.info("Loaded configuration from environment variables")

            # Apply CLI overrides
            if cli_overrides:
                config_dict = self._deep_merge(config_dict, cli_overrides)
                self._config_sources["cli"] = "Command line arguments"
                logger.info("Applied CLI configuration overrides")

            # Validate and create configuration
            self._config = CuLoRAConfig.from_dict(config_dict)

            logger.info(
                "Configuration loaded successfully",
                sources=list(self._config_sources.keys()),
                device_type=self._config.device.preferred_device,
                log_level=self._config.logging.log_level,
            )

            return self._config

        except ValidationError as e:
            logger.error(
                "Configuration validation failed", validation_errors=e.errors()
            )
            raise InvalidConfigError(
                field_name="configuration",
                field_value="<validation errors>",
                expected="valid configuration",
                error_code="INVALID_CONFIG",
            ) from e

        except (InvalidConfigError, MissingConfigError):
            raise  # Re-raise config-specific errors

        except Exception as e:
            logger.exception("Failed to load configuration", exc_info=e)
            raise ConfigurationError(
                f"Configuration loading failed: {e}",
                error_code="CONFIG_LOAD_FAILED",
            ) from e

    def get_config(self) -> CuLoRAConfig:
        """Get the current configuration.

        Returns:
            Current CuLoRA configuration

        Raises:
            MissingConfigError: If configuration has not been loaded
        """
        if self._config is None:
            raise MissingConfigError(
                field_name="configuration",
                error_code="CONFIG_NOT_LOADED",
            )
        return self._config

    def export_config(self, output_path: Path, include_defaults: bool = True) -> None:
        """Export current configuration to file.

        Args:
            output_path: Path to export configuration
            include_defaults: Whether to include default values

        Raises:
            MissingConfigError: If configuration has not been loaded
        """
        config = self.get_config()

        logger.info("Exporting configuration", output_path=str(output_path))

        try:
            # Use model_dump for Pydantic v2 compatibility and ensure string serialization
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump(
                    exclude_defaults=not include_defaults,
                    mode="json",  # This ensures enums are serialized as their values
                )
            else:
                config_dict = config.dict(exclude_defaults=not include_defaults)

            if output_path.suffix.lower() in [".yaml", ".yml"]:
                with open(output_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == ".json":
                with open(output_path, "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported file format: {output_path.suffix}")

            logger.info(
                "Configuration exported successfully", output_path=str(output_path)
            )

        except Exception as e:
            logger.exception("Failed to export configuration", exc_info=e)
            raise ConfigurationError(
                f"Configuration export failed: {e}",
                error_code="CONFIG_EXPORT_FAILED",
            ) from e

    def validate_config(self, config_dict: dict[str, Any]) -> dict[str, str]:
        """Validate configuration dictionary and return validation errors.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        try:
            CuLoRAConfig.from_dict(config_dict)
            return {}
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                errors[field_path] = error["msg"]
            return errors

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for display.

        Returns:
            Configuration summary with sources and key settings
        """
        config = self.get_config()

        return {
            "sources": self._config_sources,
            "device": {
                "preferred": config.device.preferred_device,
                "fallback": config.device.fallback_device,
                "batch_size": config.device.batch_size,
            },
            "quality": {
                "brisque_enabled": config.quality_assessment.enable_brisque,
                "technical_metrics": config.quality_assessment.enable_technical_metrics,
                "min_quality": config.selection.min_quality_score,
            },
            "selection": {
                "target_count": config.selection.target_count,
                "duplicate_removal": config.selection.enable_duplicate_removal,
                "diversity_weight": config.selection.diversity_weight,
            },
            "export": {
                "output_dir": str(config.export.output_directory),
                "format": config.export.image_format,
                "sequential_naming": config.export.sequential_naming,
            },
        }

    def _load_config_file(self, config_file: Path) -> dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file) as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == ".json":
                    return json.load(f) or {}
                else:
                    raise ValueError(
                        f"Unsupported config file format: {config_file.suffix}"
                    )
        except FileNotFoundError as e:
            raise MissingConfigError(
                field_name=f"config_file:{config_file}",
                error_code="CONFIG_FILE_NOT_FOUND",
            ) from e
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise InvalidConfigError(
                field_name=f"config_file:{config_file}",
                field_value="<parse error>",
                expected="valid YAML or JSON",
                error_code="CONFIG_FILE_PARSE_ERROR",
            ) from e

    def _load_env_config(self, prefix: str) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config_dict: dict[str, Any] = {}

        # Mapping of known field paths to handle underscores in field names
        field_mappings = {
            "device_batch_size": ("device", "batch_size"),
            "device_preferred_device": ("device", "preferred_device"),
            "device_fallback_device": ("device", "fallback_device"),
            "device_memory_limit_mb": ("device", "memory_limit_mb"),
            "device_auto_detect": ("device", "auto_detect"),
            "face_analysis_detection_threshold": (
                "face_analysis",
                "detection_threshold",
            ),
            "face_analysis_similarity_threshold": (
                "face_analysis",
                "similarity_threshold",
            ),
            "face_analysis_max_faces_per_image": (
                "face_analysis",
                "max_faces_per_image",
            ),
            "face_analysis_min_face_size": ("face_analysis", "min_face_size"),
            "logging_log_level": ("logging", "log_level"),
            "logging_log_directory": ("logging", "log_directory"),
            "logging_enable_console_output": ("logging", "enable_console_output"),
        }

        for key, value in os.environ.items():
            if key.startswith(f"{prefix}_"):
                # Convert CULORA_DEVICE_BATCH_SIZE to device.batch_size
                config_key = key[len(prefix) + 1 :].lower()

                # Use field mapping if available, otherwise split on underscores
                if config_key in field_mappings:
                    section, field = field_mappings[config_key]
                    if section not in config_dict:
                        config_dict[section] = {}
                    config_dict[section][field] = self._convert_env_value(value)
                else:
                    # Fallback to simple section_field parsing
                    parts = config_key.split("_", 1)  # Split on first underscore only
                    if len(parts) == 2:
                        section, field = parts
                        if section not in config_dict:
                            config_dict[section] = {}
                        config_dict[section][field] = self._convert_env_value(value)

        return config_dict

    def _convert_env_value(self, value: str) -> str | int | float | bool:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _deep_merge(
        self, base: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return _config_manager


def get_config() -> CuLoRAConfig:
    """Get the current configuration (convenience function)."""
    return _config_manager.get_config()
