"""Cache management utilities for CuLoRA."""

import json
from datetime import datetime
from pathlib import Path

from culora.models.analysis import DirectoryAnalysis
from culora.utils.app_data import get_cache_file_path


def save_analysis_cache(analysis: DirectoryAnalysis) -> None:
    """Save analysis results to cache file.

    Args:
        analysis: Analysis results to save.
    """
    input_dir = Path(analysis.input_directory)
    cache_file = get_cache_file_path(input_dir)

    # Ensure cache directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(analysis.model_dump(), f, indent=2, default=str)


def load_analysis_cache(input_directory: Path) -> DirectoryAnalysis | None:
    """Load analysis results from cache file.

    Args:
        input_directory: Directory that was analyzed.

    Returns:
        Cached analysis results if available and valid, None otherwise.
    """
    cache_file = get_cache_file_path(input_directory)

    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse datetime strings back to datetime objects
        if "analysis_time" in data:
            data["analysis_time"] = datetime.fromisoformat(data["analysis_time"])

        for image in data.get("images", []):
            if "modified_time" in image:
                image["modified_time"] = datetime.fromisoformat(image["modified_time"])

        return DirectoryAnalysis.model_validate(data)

    except (json.JSONDecodeError, ValueError, KeyError):
        # Cache file is corrupted or invalid, ignore it
        return None


def is_cache_valid(analysis: DirectoryAnalysis, input_directory: Path) -> bool:
    """Check if cached analysis is still valid.

    Args:
        analysis: Cached analysis results.
        input_directory: Directory being analyzed.

    Returns:
        True if cache is valid and can be reused.
    """
    # Check if directory path matches
    if analysis.input_directory != str(input_directory.resolve()):
        return False

    # Check if any image files have been modified since analysis
    for image_analysis in analysis.images:
        image_path = Path(image_analysis.file_path)

        if not image_path.exists():
            # Image was deleted
            return False

        try:
            stat = image_path.stat()
            current_size = stat.st_size
            current_mtime = datetime.fromtimestamp(stat.st_mtime)

            # Check if file size or modification time changed
            if (
                current_size != image_analysis.file_size
                or current_mtime != image_analysis.modified_time
            ):
                return False

        except OSError:
            # Can't access file
            return False

    return True


def clear_cache_file(input_directory: Path) -> bool:
    """Clear the cache file for a directory.

    Args:
        input_directory: Directory whose cache should be cleared.

    Returns:
        True if cache file was deleted, False if it didn't exist.
    """
    cache_file = get_cache_file_path(input_directory)

    if cache_file.exists():
        cache_file.unlink()
        return True

    return False
