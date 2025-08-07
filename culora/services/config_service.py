"""Configuration service for CuLoRA.

This module provides unified configuration loading, validation, and management
with support for multiple sources and override precedence.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from culora.core import (
    ConfigError,
    InvalidConfigError,
    MissingConfigError,
)
from culora.domain import CuLoRAConfig
from culora.utils.app_dir import ensure_app_directories, get_default_config_file

# Default config file location (app directory only)
DEFAULT_CONFIG_PATH = get_default_config_file()


class ConfigService:
    """Unified configuration service for CuLoRA.

    Handles configuration loading from multiple sources with proper precedence:
    CLI args > environment variables > config files > defaults
    """

    def __init__(self) -> None:
        self._config: CuLoRAConfig | None = None
        self._config_file: Path | None = None
        self._env_loaded: bool = False
        self._cli_loaded: bool = False

        # Ensure app directories exist
        ensure_app_directories()

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
        try:
            # Start with default configuration
            config_dict: dict[str, Any] = {}

            # Find and load config file
            config_file = self._find_config_file(config_file)
            if config_file and config_file.exists():
                file_config = self._load_from_file(config_file)
                config_dict = self._deep_merge(config_dict, file_config)
                self._config_file = config_file

            # Load from environment variables
            env_config = self._load_from_env(env_prefix)
            if env_config:
                config_dict = self._deep_merge(config_dict, env_config)
                self._env_loaded = True

            # Apply CLI overrides
            if cli_overrides:
                config_dict = self._deep_merge(config_dict, cli_overrides)
                self._cli_loaded = True

            # Validate and create configuration
            self._config = self._create_config(config_dict)

            return self._config

        except ValidationError as e:
            raise InvalidConfigError(
                field_name="configuration",
                field_value="<validation errors>",
                expected="valid configuration",
                error_code="INVALID_CONFIG",
            ) from e

        except (InvalidConfigError, MissingConfigError):
            raise  # Re-raise config-specific errors

        except Exception as e:
            raise ConfigError(
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

    @property
    def config_sources(self) -> dict[str, str]:
        """Get configuration sources for display.

        Returns:
            Dictionary mapping source names to descriptions
        """
        sources = {"defaults": "Built-in defaults"}

        if self._config_file:
            sources["file"] = str(self._config_file)

        if self._env_loaded:
            sources["environment"] = "CULORA_* variables"

        if self._cli_loaded:
            sources["cli"] = "Command line arguments"

        return sources

    def export_config(self, output_path: Path, include_defaults: bool = True) -> None:
        """Export current configuration to file.

        Args:
            output_path: Path to export configuration
            include_defaults: Whether to include default values

        Raises:
            MissingConfigError: If configuration has not been loaded
        """
        config = self.get_config()

        try:
            # Use model_dump for Pydantic v2 compatibility and ensure string serialization
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump(
                    exclude_defaults=not include_defaults,
                    mode="json",  # This ensures enums are serialized as their values
                )
            else:
                config_dict = config.dict(exclude_defaults=not include_defaults)

            self._save_to_file(config_dict, output_path)

        except Exception as e:
            raise ConfigError(
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
            "sources": self.config_sources,
            "device": {
                "preferred": config.device.preferred_device,
            },
        }

    def get_config_value(self, key_path: str) -> Any:
        """Get a specific configuration value by key path.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'device.preferred_device')

        Returns:
            Configuration value at the specified path

        Raises:
            MissingConfigError: If configuration has not been loaded
            KeyError: If key path does not exist
        """
        config = self.get_config()

        try:
            current = config
            for key in key_path.split("."):
                if hasattr(current, key):
                    current = getattr(current, key)
                else:
                    raise KeyError(
                        f"Configuration key '{key}' not found in path '{key_path}'"
                    )

            return current

        except Exception:
            raise

    def set_config_value(
        self, key_path: str, value: Any, config_file: Path | None = None
    ) -> None:
        """Set a configuration value and save to file.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'device.preferred_device')
            value: Value to set
            config_file: Config file to save to (uses current or default if None)

        Raises:
            MissingConfigError: If configuration has not been loaded
            InvalidConfigError: If the new configuration is invalid
            ConfigError: If setting or saving fails
        """
        try:
            # Get current config as dict
            current_config = self.get_config()
            config_dict = current_config.model_dump(mode="json")

            # Set the new value in the dictionary
            self._set_nested_value(config_dict, key_path, value)

            # Validate the new configuration
            validation_errors = self.validate_config(config_dict)
            if validation_errors:
                error_details = "; ".join(
                    f"{k}: {v}" for k, v in validation_errors.items()
                )
                raise InvalidConfigError(
                    field_name=key_path,
                    field_value=str(value),
                    expected="valid configuration value",
                    error_code="INVALID_CONFIG_VALUE",
                    details=error_details,
                )

            # Update the current configuration
            self._config = CuLoRAConfig.from_dict(config_dict)

            # Save to file
            target_file = self._resolve_config_file(config_file)
            self._save_to_file(config_dict, target_file)

        except (InvalidConfigError, MissingConfigError):
            raise  # Re-raise config-specific errors

        except Exception as e:
            raise ConfigError(
                f"Failed to set configuration value '{key_path}': {e}",
                error_code="CONFIG_SET_FAILED",
            ) from e

    def get_config_file(self) -> Path | None:
        """Get the currently loaded config file path.

        Returns:
            Path to the currently loaded config file, or None if no file was loaded
        """
        return self._config_file

    def _load_from_file(self, config_file: Path) -> dict[str, Any]:
        """Load configuration from YAML or JSON file.

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            MissingConfigError: If file not found
            InvalidConfigError: If file cannot be parsed
        """
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

    def _save_to_file(self, config_dict: dict[str, Any], output_path: Path) -> None:
        """Save configuration dictionary to file.

        Args:
            config_dict: Configuration to save
            output_path: Path to save configuration

        Raises:
            ValueError: If unsupported file format
        """
        if output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

    def _load_from_env(self, prefix: str) -> dict[str, Any]:
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix (e.g., 'CULORA')

        Returns:
            Configuration dictionary from environment variables
        """
        config: dict[str, Any] = {}

        # Mapping of environment variable suffixes to config paths
        env_mappings = {
            "DEVICE_PREFERRED": ["device", "preferred_device"],
            "DEVICE_FALLBACK": ["device", "fallback_device"],
            "DEVICE_BATCH_SIZE": ["device", "batch_size"],
            "IMAGES_MAX_BATCH_SIZE": ["images", "max_batch_size"],
            "IMAGES_MAX_FILE_SIZE": ["images", "max_file_size"],
            "IMAGES_RECURSIVE_SCAN": ["images", "recursive_scan"],
            "IMAGES_MAX_SCAN_DEPTH": ["images", "max_scan_depth"],
            "FACES_MODEL_NAME": ["faces", "model_name"],
            "FACES_CONFIDENCE_THRESHOLD": ["faces", "confidence_threshold"],
            "FACES_MAX_FACES_PER_IMAGE": ["faces", "max_faces_per_image"],
            "FACES_DEVICE_PREFERENCE": ["faces", "device_preference"],
            "FACES_BATCH_SIZE": ["faces", "batch_size"],
            "FACES_EXTRACT_EMBEDDINGS": ["faces", "extract_embeddings"],
            "FACES_EXTRACT_LANDMARKS": ["faces", "extract_landmarks"],
            "FACES_EXTRACT_ATTRIBUTES": ["faces", "extract_attributes"],
            "QUALITY_BRISQUE": ["quality_assessment", "enable_brisque"],
            "QUALITY_TECHNICAL": ["quality_assessment", "enable_technical_metrics"],
            "SELECTION_TARGET_COUNT": ["selection", "target_count"],
            "SELECTION_MIN_QUALITY": ["selection", "min_quality_score"],
            "EXPORT_OUTPUT_DIR": ["export", "output_directory"],
            "EXPORT_FORMAT": ["export", "image_format"],
        }

        for env_suffix, config_path in env_mappings.items():
            env_var = f"{prefix}_{env_suffix}"
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Convert environment variable value to appropriate type
                converted_value = self._convert_env_value(env_value)

                # Set nested dictionary value
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = converted_value

        return config

    def _convert_env_value(self, value: str) -> str | int | float | bool:
        """Convert environment variable string to appropriate type.

        Args:
            value: Environment variable string value

        Returns:
            Converted value (string, int, float, or bool)
        """
        # Handle boolean values
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False

        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _deep_merge(
        self, base: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base configuration dictionary
            update: Updates to apply

        Returns:
            Merged configuration dictionary
        """
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

    def _create_config(self, config_dict: dict[str, Any]) -> CuLoRAConfig:
        """Create and validate CuLoRA configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated CuLoRA configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        return CuLoRAConfig.from_dict(config_dict)

    def _find_config_file(self, provided_file: Path | None) -> Path | None:
        """Find configuration file using provided path or default app directory.

        Args:
            provided_file: Explicitly provided config file path

        Returns:
            Path to config file if found, None otherwise
        """
        if provided_file:
            return provided_file

        # Check default app directory location
        if DEFAULT_CONFIG_PATH.exists():
            return DEFAULT_CONFIG_PATH

        return None

    def _resolve_config_file(self, provided_file: Path | None) -> Path:
        """Resolve config file for saving, creating default if needed.

        Args:
            provided_file: Explicitly provided config file path

        Returns:
            Path to use for saving configuration

        Raises:
            ConfigError: If no suitable config file can be determined
        """
        if provided_file:
            # Ensure parent directory exists
            provided_file.parent.mkdir(parents=True, exist_ok=True)
            return provided_file

        if self._config_file:
            return self._config_file

        # Use default app directory location
        default_file = DEFAULT_CONFIG_PATH
        default_file.parent.mkdir(parents=True, exist_ok=True)
        return default_file

    def _set_nested_value(
        self, config_dict: dict[str, Any], key_path: str, value: Any
    ) -> None:
        """Set a nested value in a configuration dictionary.

        Args:
            config_dict: Configuration dictionary to modify
            key_path: Dot-separated path to the value
            value: Value to set

        Raises:
            KeyError: If intermediate keys don't exist
        """
        keys = key_path.split(".")
        current = config_dict

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise KeyError(
                    f"Cannot set nested value: '{key}' is not a dictionary in path '{key_path}'"
                )
            current = current[key]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value


# Global configuration service instance
_config_service = ConfigService()


def get_config_service() -> ConfigService:
    """Get the global configuration service instance."""
    return _config_service


def get_config() -> CuLoRAConfig:
    """Get the current configuration (convenience function)."""
    return _config_service.get_config()
