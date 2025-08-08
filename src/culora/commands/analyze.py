"""Analyze command for CuLoRA CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from culora.config import AnalysisConfig, AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.models.directory_analysis import (
    DirectoryAnalysis,
)
from culora.models.duplicate_detection_result import (
    DuplicateDetectionResult,
)
from culora.models.face_detection_result import FaceDetectionResult
from culora.models.image_quality_result import ImageQualityResult
from culora.orchestrators.image_analyzer import ImageAnalyzer
from culora.orchestrators.image_curator import ImageCurator
from culora.utils.console import get_console

console = get_console()


def analyze_command(
    input_dir: Annotated[
        str, typer.Argument(help="Directory containing images to analyze")
    ],
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Automatically select and copy curated images to this directory after analysis",
        ),
    ] = None,
    no_dedupe: Annotated[
        bool, typer.Option(help="Disable image deduplication")
    ] = False,
    no_quality: Annotated[
        bool, typer.Option(help="Disable image quality assessment")
    ] = False,
    no_face: Annotated[bool, typer.Option(help="Disable face detection")] = False,
    draw_boxes: Annotated[
        bool,
        typer.Option(
            "--draw-boxes",
            help="Draw bounding boxes on detected faces with confidence scores (only used with --output)",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview selection results without copying files (only used with --output)",
        ),
    ] = False,
    max_images: Annotated[
        int | None,
        typer.Option(
            "--max-images",
            help="Maximum number of images to select, ranked by composite score (only used with --output)",
        ),
    ] = None,
) -> None:
    """Analyze images in a directory for curation.

    This command analyzes all images in the specified directory using multiple
    stages: deduplication, quality assessment, and face detection.

    Face detection uses a specialized YOLO11 face model. On first run, it will
    download the model weights (~12MB) which are cached for future use.

    If --output is specified, curated images will be automatically copied to
    the output directory after analysis is complete. Use --draw-boxes with
    --output to annotate faces with bounding boxes and confidence scores.
    """
    input_path = Path(input_dir)

    # Create analysis configuration
    config = AnalysisConfig()
    config.deduplication.enabled = not no_dedupe
    config.quality.enabled = not no_quality
    config.face.enabled = not no_face

    # Set up configuration manager
    config_manager = ConfigManager.get_instance()
    config_manager._analysis_config = config  # type: ignore[attr-defined]

    # Create orchestrators
    analyzer = ImageAnalyzer(config_manager)
    curator = ImageCurator(config_manager)

    # Validate options
    if draw_boxes and not output_dir:
        console.error("--draw-boxes can only be used with --output")
        raise typer.Exit(1)

    if dry_run and not output_dir:
        console.error("--dry-run can only be used with --output")
        raise typer.Exit(1)

    if max_images is not None and not output_dir:
        console.error("--max-images can only be used with --output")
        raise typer.Exit(1)

    if max_images is not None and max_images <= 0:
        console.error("--max-images must be a positive integer")
        raise typer.Exit(1)

    try:
        # Run the analysis (includes all enabled stages: quality, face, and deduplication)
        analysis = analyzer.analyze_directory(input_path)

        # Display results summary
        _display_analysis_summary(analysis)

        # If output directory is specified, automatically run selection
        if output_dir:
            console.header("\nSelecting curated images...")

            # Perform selection using the curator
            try:
                selected_count, total_count = curator.select_images(
                    analysis=analysis,
                    output_dir=output_dir,
                    draw_boxes=draw_boxes,
                    dry_run=dry_run,
                    max_images=max_images,
                )
                if dry_run:
                    console.success(
                        f"Would select {selected_count} of {total_count} images"
                    )
                else:
                    console.success(
                        f"Selected {selected_count} of {total_count} images"
                    )
            except RuntimeError as e:
                console.error(f"Selection Error: {e}")
                raise typer.Exit(1) from e

    except FileNotFoundError:
        console.error(f"Directory not found: {input_path}")
        raise typer.Exit(1) from None
    except NotADirectoryError:
        console.error(f"Path is not a directory: {input_path}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.error(f"Error during analysis: {e}")
        raise typer.Exit(1) from e


def _display_analysis_summary(analysis: DirectoryAnalysis) -> None:
    """Display detailed analysis results for all images.

    Args:
        analysis: DirectoryAnalysis results to display.
    """
    table = Table(title="Image Analysis Results")
    table.add_column("Image", style="cyan", width=25)

    enabled_stages = analysis.analysis_config.enabled_stages

    # Add a column for each enabled stage
    for stage in enabled_stages:
        if stage == AnalysisStage.DEDUPLICATION:
            table.add_column("Hash", justify="center", width=12)
        elif stage == AnalysisStage.QUALITY:
            # Add separate columns for each quality metric
            table.add_column("Sharpness", justify="center", width=9, style="bold")
            table.add_column("Brightness", justify="center", width=9, style="bold")
            table.add_column("Contrast", justify="center", width=9, style="bold")
            table.add_column("Quality", justify="center", width=8, style="bold")
        elif stage == AnalysisStage.FACE:
            table.add_column("Faces", justify="left", width=20)

    # Add score column (always shown for transparency)
    table.add_column("Score", justify="center", width=8, style="bold")

    # Sort images by filename for consistent output
    sorted_images = sorted(
        analysis.images, key=lambda img: Path(img.file_path).name.lower()
    )

    # Add results for each image
    for image in sorted_images:
        row_data = [Path(image.file_path).name]  # Just the filename

        # Add data for each enabled stage
        for stage in enabled_stages:
            if stage == AnalysisStage.QUALITY:
                quality_result = image.results.get_quality()
                row_data.extend(_format_quality_data(quality_result))
            elif stage == AnalysisStage.FACE:
                face_result = image.results.get_face()
                row_data.append(_format_face_data(face_result))
            elif stage == AnalysisStage.DEDUPLICATION:
                dedup_result = image.results.get_deduplication()
                row_data.append(_format_dedup_data(dedup_result))

        # Add score (always shown)
        score_str = f"{image.score:.2f}"
        if image.score >= 0.7:
            score_str = f"[green]{score_str}[/green]"
        elif image.score >= 0.4:
            score_str = f"[yellow]{score_str}[/yellow]"
        else:
            score_str = f"[red]{score_str}[/red]"
        row_data.append(score_str)

        table.add_row(*row_data)

    # Display results table
    console.table(table)


def _format_quality_data(quality_result: ImageQualityResult | None) -> list[str]:
    """Format quality result data into separate column values.

    Args:
        quality_result: Quality analysis result or None

    Returns:
        List of formatted strings for [sharpness, brightness, contrast, composite_score]
    """
    if not quality_result:
        return ["N/A", "N/A", "N/A", "N/A"]

    # Format with color based on reasonable thresholds
    sharpness_str = f"{quality_result.sharpness_score:.0f}"
    brightness_str = f"{quality_result.brightness_score:.0f}"
    contrast_str = f"{quality_result.contrast_score:.0f}"
    quality_str = f"{quality_result.composite_score:.0f}"

    # Color code based on thresholds
    if quality_result.sharpness_score >= 150:
        sharpness_str = f"[green]{sharpness_str}[/green]"
    elif quality_result.sharpness_score >= 50:
        sharpness_str = f"[yellow]{sharpness_str}[/yellow]"
    else:
        sharpness_str = f"[red]{sharpness_str}[/red]"

    if 60 <= quality_result.brightness_score <= 200:
        brightness_str = f"[green]{brightness_str}[/green]"
    else:
        brightness_str = f"[yellow]{brightness_str}[/yellow]"

    if quality_result.contrast_score >= 40:
        contrast_str = f"[green]{contrast_str}[/green]"
    elif quality_result.contrast_score >= 20:
        contrast_str = f"[yellow]{contrast_str}[/yellow]"
    else:
        contrast_str = f"[red]{contrast_str}[/red]"

    if quality_result.composite_score >= 70:
        quality_str = f"[green]{quality_str}[/green]"
    elif quality_result.composite_score >= 50:
        quality_str = f"[yellow]{quality_str}[/yellow]"
    else:
        quality_str = f"[red]{quality_str}[/red]"

    return [sharpness_str, brightness_str, contrast_str, quality_str]


def _format_face_data(face_result: FaceDetectionResult | None) -> str:
    """Format face detection result data.

    Args:
        face_result: Face detection result or None

    Returns:
        Formatted string showing each face with size and confidence on separate lines
    """
    if not face_result:
        return "N/A"

    if face_result.face_count == 0:
        return "[red]none[/red]"

    # Format each face on its own line
    face_lines: list[str] = []
    for i, face in enumerate(face_result.faces, 1):
        # Calculate face size from bounding box (width x height)
        x1, y1, x2, y2 = face.bounding_box
        width = int(x2 - x1)
        height = int(y2 - y1)
        size_str = f"{width}x{height}"

        # Format confidence as percentage
        conf_pct = face.confidence * 100

        # Create line for this face
        face_line = f"[green]#{i}: {size_str} ({conf_pct:.1f}%)[/green]"
        face_lines.append(face_line)

    # Join all faces with newlines
    return "\n".join(face_lines)


def _format_dedup_data(dedup_result: DuplicateDetectionResult | None) -> str:
    """Format deduplication result data.

    Args:
        dedup_result: Deduplication result or None

    Returns:
        Formatted string showing hash info
    """
    if not dedup_result or not dedup_result.hash_value:
        return "N/A"

    # Show first 8 characters of hash with color
    return f"[cyan]{dedup_result.hash_value[:8]}[/cyan]"


def register_command(app: typer.Typer) -> None:
    """Register the analyze command with the given Typer app."""
    app.command(name="analyze")(analyze_command)
