"""Cache management utilities for CuLoRA."""

import json
from datetime import datetime
from pathlib import Path

from culora.models.analysis import (
    AnalysisStage,
    DirectoryAnalysis,
    StageConfig,
    StageResult,
)
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


def get_stages_needing_analysis(
    cached_analysis: DirectoryAnalysis | None,
    requested_stages: list[AnalysisStage],
    requested_configs: list[StageConfig],
    input_directory: Path,
) -> list[AnalysisStage]:
    """Determine which stages need to be analyzed based on cache state.

    Args:
        cached_analysis: Previously cached analysis results (if any).
        requested_stages: List of stages requested for analysis.
        requested_configs: Current configurations for requested stages.
        input_directory: Directory being analyzed.

    Returns:
        List of stages that need to be (re-)analyzed.
    """
    if not cached_analysis:
        # No cache - need to analyze all requested stages
        return requested_stages

    if not is_cache_valid(cached_analysis, input_directory):
        # Cache is invalid due to file changes - need to analyze all requested stages
        return requested_stages

    stages_to_analyze: list[AnalysisStage] = []

    for stage in requested_stages:
        needs_analysis = False

        # Find the requested configuration for this stage
        requested_config = None
        for config in requested_configs:
            if config.stage == stage:
                requested_config = config
                break

        if not requested_config:
            # No configuration found - shouldn't happen but be safe
            needs_analysis = True
        else:
            # Check if this stage was previously analyzed
            cached_config = cached_analysis.get_stage_config(stage)

            if not cached_config:
                # Stage was not previously analyzed
                needs_analysis = True
            elif cached_config != requested_config:
                # Configuration changed - need to re-analyze
                needs_analysis = True
            elif not cached_analysis.has_stage_results(stage):
                # Config exists but no actual results - need to analyze
                needs_analysis = True

        if needs_analysis:
            stages_to_analyze.append(stage)

    return stages_to_analyze


def merge_analysis_results(
    cached_analysis: DirectoryAnalysis | None, new_analysis: DirectoryAnalysis
) -> DirectoryAnalysis:
    """Merge new analysis results with cached results.

    Args:
        cached_analysis: Previously cached analysis results (if any).
        new_analysis: New analysis results to merge.

    Returns:
        Merged analysis results.
    """
    if not cached_analysis:
        return new_analysis

    # Start with the new analysis as base
    merged = DirectoryAnalysis(
        input_directory=new_analysis.input_directory,
        analysis_time=new_analysis.analysis_time,
        enabled_stages=new_analysis.enabled_stages,
        stage_configs=new_analysis.stage_configs.copy(),
        images=[],
    )

    # Create a mapping of new analysis results by file path
    new_results_by_path = {img.file_path: img for img in new_analysis.images}

    # Process each image from cached analysis
    for cached_image in cached_analysis.images:
        if cached_image.file_path in new_results_by_path:
            # This image was re-analyzed - use new results but preserve old stages not re-run
            new_image = new_results_by_path[cached_image.file_path]

            # Combine stage results from both analyses
            combined_stage_results: list[StageResult] = []

            # Add results from new analysis
            new_stages = {result.stage for result in new_image.stage_results}
            combined_stage_results.extend(new_image.stage_results)

            # Add results from cached analysis for stages not in new analysis
            for cached_result in cached_image.stage_results:
                if cached_result.stage not in new_stages:
                    combined_stage_results.append(cached_result)

            # Create merged image analysis
            from culora.models.analysis import ImageAnalysis

            merged_image = ImageAnalysis(
                file_path=new_image.file_path,
                file_size=new_image.file_size,
                modified_time=new_image.modified_time,
                stage_results=combined_stage_results,
            )
            merged.images.append(merged_image)
        else:
            # This image wasn't in new analysis - keep cached version as-is
            merged.images.append(cached_image)

    # Add any completely new images from new analysis
    cached_paths = {img.file_path for img in cached_analysis.images}
    for new_image in new_analysis.images:
        if new_image.file_path not in cached_paths:
            merged.images.append(new_image)

    # Update stage configs - prefer new configs, keep old ones for stages not re-run
    config_by_stage = {}

    # Add cached configs first
    for config in cached_analysis.stage_configs:
        config_by_stage[config.stage] = config

    # Override with new configs
    for config in new_analysis.stage_configs:
        config_by_stage[config.stage] = config

    merged.stage_configs = list(config_by_stage.values())  # type: ignore

    return merged
