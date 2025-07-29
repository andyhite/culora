"""App data directory utilities for CuLoRA."""

from pathlib import Path

import typer

APP_NAME = "culora"


def get_app_dir() -> Path:
    """Get the cross-platform app data directory for CuLoRA.

    Returns:
        Path to the app data directory where cache files are stored.
    """
    app_dir = typer.get_app_dir(APP_NAME, force_posix=True)
    return Path(app_dir)


def get_models_dir() -> Path:
    """Get the models directory for ML model storage.

    Returns:
        Path to the models directory, creating it if it doesn't exist.
    """
    models_dir = get_app_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
