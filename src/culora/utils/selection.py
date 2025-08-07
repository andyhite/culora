"""Shared selection utilities for CuLoRA CLI commands."""

import shutil
from pathlib import Path

from rich.console import Console

from culora.cli.select import draw_face_bounding_boxes
from culora.models.analysis import AnalysisResult, AnalysisStage, DirectoryAnalysis
from culora.utils.cache import load_analysis_cache


def perform_selection(
    input_dir: str,
    output_dir: str,
    draw_boxes: bool = False,
    dry_run: bool = False,
    console: Console | None = None,
    analysis_results: DirectoryAnalysis | None = None,
) -> tuple[int, int]:
    """Perform image selection logic shared between analyze and select commands.

    Args:
        input_dir: Directory containing analyzed images
        output_dir: Directory to copy selected images to
        draw_boxes: Whether to draw bounding boxes on faces
        dry_run: Whether to perform a dry run (no actual copying)
        console: Rich console for output (will create if None)
        analysis_results: Optional pre-loaded analysis results (avoids cache lookup)

    Returns:
        Tuple of (selected_count, total_count)

    Raises:
        FileNotFoundError: If input directory doesn't exist
        RuntimeError: If no analysis results found or other errors
    """
    if console is None:
        console = Console()

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    # Use provided analysis results or load from cache
    if analysis_results:
        analysis = analysis_results
    else:
        analysis = load_analysis_cache(input_path)
        if not analysis:
            raise RuntimeError("No analysis results found. Run 'analyze' first.")

    # Get images that passed all stages
    selected_images = analysis.passed_images

    if not selected_images:
        return 0, len(analysis.images)

    # Create output directory if it doesn't exist (unless dry run)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Copy and rename selected images
    copied_count = 0
    for i, image in enumerate(selected_images, 1):
        source_path = Path(image.file_path)
        # Keep original extension, use sequential numbering
        extension = source_path.suffix
        target_filename = f"{i:03d}{extension}"  # 001.jpg, 002.jpg, etc.
        target_path = output_path / target_filename

        if not dry_run:
            if draw_boxes:
                # Check if this image has face detection results
                face_metadata = None
                for stage_result in image.stage_results:
                    if (
                        stage_result.stage == AnalysisStage.FACE
                        and stage_result.result == AnalysisResult.PASS
                        and stage_result.metadata
                    ):
                        face_metadata = stage_result.metadata
                        break

                if face_metadata:
                    # Draw bounding boxes and save annotated image
                    draw_face_bounding_boxes(source_path, target_path, face_metadata)
                else:
                    # No face detection data, just copy original
                    shutil.copy2(source_path, target_path)
            else:
                # Normal copy without annotations
                shutil.copy2(source_path, target_path)
            copied_count += 1

    return len(selected_images), len(analysis.images)
