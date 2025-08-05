"""Application directory utilities using Typer's cross-platform support."""

from pathlib import Path

import typer

# Application name for directory creation
APP_NAME = "culora"


def get_app_dir() -> Path:
    """Get the cross-platform application directory.

    Returns:
        Path to the application directory
    """
    return Path(typer.get_app_dir(APP_NAME))


def get_config_dir() -> Path:
    """Get the configuration directory within the app directory.

    Returns:
        Path to the configuration directory
    """
    return get_app_dir() / "config"


def get_cache_dir() -> Path:
    """Get the cache directory within the app directory.

    Returns:
        Path to the cache directory
    """
    return get_app_dir() / "cache"


def get_models_dir() -> Path:
    """Get the models directory within the app directory.

    Returns:
        Path to the models directory
    """
    return get_app_dir() / "models"


def get_logs_dir() -> Path:
    """Get the logs directory within the app directory.

    Returns:
        Path to the logs directory
    """
    return get_app_dir() / "logs"


def get_default_config_file() -> Path:
    """Get the default configuration file path.

    Returns:
        Path to the default config file
    """
    return get_config_dir() / "config.yaml"


def ensure_app_directories() -> None:
    """Ensure all application directories exist."""
    get_app_dir().mkdir(parents=True, exist_ok=True)
    get_config_dir().mkdir(parents=True, exist_ok=True)
    get_cache_dir().mkdir(parents=True, exist_ok=True)
    get_models_dir().mkdir(parents=True, exist_ok=True)
    get_logs_dir().mkdir(parents=True, exist_ok=True)
