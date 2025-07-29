"""Configuration manager for CuLoRA."""

import tomllib
from pathlib import Path
from typing import Any

from culora.config import CuLoRAConfig
from culora.utils.app_data import get_app_dir


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
        self._config = CuLoRAConfig()
        self._config_file_path: Path | None = None
        self._initialized = True

    @property
    def config(self) -> CuLoRAConfig:
        """Get the current configuration."""
        return self._config

    def get_config(self, key: str) -> Any:
        """Get configuration for a specific stage."""
        if not hasattr(self._config, key):
            raise ValueError(f"Configuration for key '{key}' not found")
        return getattr(self._config, key)

    def get_config_file_path(self, config_path: Path | None = None) -> Path:
        """Get the configuration file path.

        Args:
            config_path: Optional path to config file. If None, uses app dir.

        Returns:
            Path to config file.
        """
        if config_path is not None:
            return config_path
        return get_app_dir() / "config.toml"

    def load_from_file(self, config_path: Path | None = None) -> None:
        """Load configuration from TOML file.

        Args:
            config_path: Optional path to TOML config file. If None, looks in app dir.
        """
        file_path = self.get_config_file_path(config_path)
        self._config_file_path = file_path

        # If config file doesn't exist, keep defaults
        if not file_path.exists():
            return

        try:
            with open(file_path, "rb") as f:
                config_data = tomllib.load(f)
            self._config = CuLoRAConfig.model_validate(config_data)
        except Exception as e:
            # If config is invalid, warn and keep defaults
            from rich.console import Console

            console = Console()
            console.print(
                f"[yellow]Warning: Invalid config file {file_path}: {e}[/yellow]"
            )
            console.print("[yellow]Using default configuration[/yellow]")

    def save_to_file(self, config_path: Path | None = None) -> Path:
        """Save current configuration to TOML file.

        Args:
            config_path: Optional path where to save config. If None, saves to app dir.

        Returns:
            Path where config was saved.
        """
        file_path = self.get_config_file_path(config_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and create TOML content
        config_dict = self._config.model_dump()

        toml_content = """# CuLoRA Configuration File
# Values can be overridden via CLI flags.

"""

        # Add each section
        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict):
                toml_content += f"[{section_name}]\n"
                for key, value in section_data.items():  # type: ignore[misc]
                    key_str = str(key)  # type: ignore[arg-type]
                    if isinstance(value, str):
                        toml_content += f'{key_str} = "{value}"\n'
                    elif isinstance(value, bool):
                        toml_content += f"{key_str} = {str(value).lower()}\n"
                    else:
                        toml_content += f"{key_str} = {value!s}\n"
                toml_content += "\n"

        with open(file_path, "w") as f:
            f.write(toml_content)

        self._config_file_path = file_path
        return file_path

    def save_defaults_to_file(self, config_path: Path | None = None) -> Path:
        """Save default configuration to TOML file.

        Args:
            config_path: Optional path where to save config. If None, saves to app dir.

        Returns:
            Path where config was saved.
        """
        # Temporarily store current config
        current_config = self._config

        # Reset to defaults and save
        self._config = CuLoRAConfig()
        file_path = self.save_to_file(config_path)

        # Restore current config
        self._config = current_config

        return file_path

    def delete_config_file(self, config_path: Path | None = None) -> bool:
        """Delete the configuration file.

        Args:
            config_path: Optional path to config file. If None, uses app dir.

        Returns:
            True if file was deleted, False if it didn't exist.
        """
        file_path = self.get_config_file_path(config_path)

        if file_path.exists():
            file_path.unlink()
            # Reset to defaults after deletion
            self._config = CuLoRAConfig()
            self._config_file_path = None
            return True
        return False

    def config_file_exists(self, config_path: Path | None = None) -> bool:
        """Check if configuration file exists.

        Args:
            config_path: Optional path to config file. If None, checks app dir.

        Returns:
            True if config file exists.
        """
        file_path = self.get_config_file_path(config_path)
        return file_path.exists()

    def get_config_value(self, key: str) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key in format 'section.field'

        Returns:
            Configuration value.

        Raises:
            ValueError: If key format is invalid or key not found.
        """
        key_parts = key.split(".")
        if len(key_parts) != 2:
            raise ValueError("Key must be in format 'section.field'")

        section, field = key_parts

        if not hasattr(self._config, section):
            raise ValueError(f"Section '{section}' not found")

        section_config = getattr(self._config, section)
        if not hasattr(section_config, field):
            raise ValueError(f"Field '{field}' not found in section '{section}'")

        return getattr(section_config, field)

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key: Configuration key in format 'section.field'
            value: Value to set (will be type-converted)

        Raises:
            ValueError: If key format is invalid or key not found.
        """
        key_parts = key.split(".")
        if len(key_parts) != 2:
            raise ValueError("Key must be in format 'section.field'")

        section, field = key_parts

        if not hasattr(self._config, section):
            raise ValueError(f"Section '{section}' not found")

        section_config = getattr(self._config, section)
        if not hasattr(section_config, field):
            raise ValueError(f"Field '{field}' not found in section '{section}'")

        # Get current value for type conversion
        current_value = getattr(section_config, field)

        # Convert string value to appropriate type
        if isinstance(current_value, bool):
            converted_value = str(value).lower() in ("true", "yes", "1", "on")
        elif isinstance(current_value, int):
            converted_value = int(value)
        elif isinstance(current_value, float):
            converted_value = float(value)
        elif isinstance(current_value, str):
            converted_value = str(value)
        else:
            converted_value = value

        # Set the value
        setattr(section_config, field, converted_value)

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get the singleton instance of ConfigManager."""
        return cls()
