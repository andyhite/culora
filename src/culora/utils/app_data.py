"""App data directory utilities for CuLoRA."""

import hashlib
from pathlib import Path

import typer

APP_NAME = "culora"


def get_app_data_dir() -> Path:
    """Get the cross-platform app data directory for CuLoRA.

    Returns:
        Path to the app data directory where cache files are stored.
    """
    app_dir = typer.get_app_dir(APP_NAME)
    return Path(app_dir)


def get_cache_dir() -> Path:
    """Get the cache directory for analysis results.

    Returns:
        Path to the cache directory, creating it if it doesn't exist.
    """
    cache_dir = get_app_data_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_file_path(input_directory: Path) -> Path:
    """Get the cache file path for a given input directory.

    Creates a unique cache file name based on the absolute path of the
    input directory to avoid collisions between different directories.

    Args:
        input_directory: The directory being analyzed.

    Returns:
        Path to the JSON cache file for this directory.
    """
    # Create a hash of the absolute path to ensure unique cache files
    abs_path = str(input_directory.resolve())
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    # Use directory name + hash for readability
    dir_name = input_directory.name or "root"
    cache_filename = f"{dir_name}_{path_hash}.json"

    return get_cache_dir() / cache_filename
