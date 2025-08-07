"""Analyze command for CuLoRA CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from culora.analysis.analyzer import analyze_directory
from culora.models.analysis import (
    AnalysisResult,
    AnalysisStage,
    DirectoryAnalysis,
    StageConfig,
    StageResult,
)

console = Console()


def analyze_command(
    input_dir: Annotated[
        str, typer.Argument(help="Directory containing images to analyze")
    ],
    no_dedupe: Annotated[
        bool, typer.Option(help="Disable image deduplication")
    ] = False,
    no_quality: Annotated[
        bool, typer.Option(help="Disable image quality assessment")
    ] = False,
    no_face: Annotated[bool, typer.Option(help="Disable face detection")] = False,
    no_cache: Annotated[
        bool, typer.Option(help="Skip cache and force re-analysis of all images")
    ] = False,
) -> None:
    """Analyze images in a directory for curation.

    This command analyzes all images in the specified directory using multiple
    stages: deduplication, quality assessment, and face detection. Results are
    cached for future runs.

    Face detection uses YOLO11 to identify people in images. On first run, it will
    download the model weights (~12MB) which are cached for future use.
    """
    input_path = Path(input_dir)
    console.print(f"[bold green]Analyzing images in:[/bold green] {input_path}")

    # Determine enabled stages for display
    stages: list[str] = []
    if not no_dedupe:
        stages.append("deduplication")
    if not no_quality:
        stages.append("quality assessment")
    if not no_face:
        stages.append("face detection")

    if stages:
        console.print(f"[blue]Enabled stages:[/blue] {', '.join(stages)}")
    else:
        console.print("[yellow]No analysis stages enabled[/yellow]")
        return

    try:
        # Run the analysis
        analysis = analyze_directory(
            input_path,
            enable_deduplication=not no_dedupe,
            enable_quality=not no_quality,
            enable_face=not no_face,
            use_cache=not no_cache,
        )

        # Display results summary
        _display_analysis_summary(analysis)

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Directory not found: {input_path}")
        raise typer.Exit(1) from None
    except NotADirectoryError:
        console.print(f"[red]Error:[/red] Path is not a directory: {input_path}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error during analysis:[/red] {e}")
        raise typer.Exit(1) from e


def _display_analysis_summary(analysis: DirectoryAnalysis) -> None:
    """Display detailed analysis results for all images.

    Args:
        analysis: DirectoryAnalysis results to display.
    """
    console.print("\n[bold green]Analysis Complete![/bold green]")

    # Show enabled stages
    if analysis.enabled_stages:
        stages_str = ", ".join([stage.value for stage in analysis.enabled_stages])
        console.print(f"[blue]Stages analyzed:[/blue] {stages_str}")

    # Create detailed results table
    table = Table(title="Image Analysis Results")
    table.add_column("Image", style="cyan", width=25)
    table.add_column("Overall", justify="center", width=8)

    # Add a column for each enabled stage with appropriate widths
    for stage in analysis.enabled_stages:
        if stage == AnalysisStage.DEDUPLICATION:
            width = 18  # For "DUP:filename(dist)" or hash
            table.add_column(stage.value.title(), justify="center", width=width)
        elif stage == AnalysisStage.QUALITY:
            # Add separate columns for each quality metric
            table.add_column("Sharpness", justify="center", width=8, style="bold")
            table.add_column("Brightness", justify="center", width=10, style="bold")
            table.add_column("Contrast", justify="center", width=8, style="bold")
        elif stage == AnalysisStage.FACE:
            width = 12  # For "X people" or "none"
            table.add_column(stage.value.title(), justify="center", width=width)
        else:
            width = 12  # Default
            table.add_column(stage.value.title(), justify="center", width=width)

    # Sort images by filename for consistent output
    sorted_images = sorted(
        analysis.images, key=lambda img: Path(img.file_path).name.lower()
    )

    # Add results for each image
    for image in sorted_images:
        row_data = [
            Path(image.file_path).name,  # Just the filename
            _format_result_status(image.overall_result),
        ]

        # Add result for each enabled stage
        stage_results_map = {result.stage: result for result in image.stage_results}

        for stage in analysis.enabled_stages:
            if stage in stage_results_map:
                result = stage_results_map[stage]
                if stage == AnalysisStage.QUALITY:
                    # Add separate columns for quality metrics
                    # Get quality stage config
                    quality_config = None
                    for config in analysis.stage_configs:
                        if config.stage == AnalysisStage.QUALITY:
                            quality_config = config
                            break
                    sharpness, brightness, contrast = _format_quality_metrics(
                        result, quality_config
                    )
                    row_data.extend([sharpness, brightness, contrast])
                else:
                    row_data.append(_format_stage_result(result))
            else:
                if stage == AnalysisStage.QUALITY:
                    # Add N/A for all three quality columns
                    row_data.extend(
                        ["[dim]N/A[/dim]", "[dim]N/A[/dim]", "[dim]N/A[/dim]"]
                    )
                else:
                    row_data.append("[dim]N/A[/dim]")

        table.add_row(*row_data)

    console.print(table)

    # Show summary counts
    console.print(f"\n[bold]Summary:[/bold] {analysis.total_images} images analyzed")
    console.print(f"✅ [green]{len(analysis.passed_images)} passed all stages[/green]")
    console.print(
        f"❌ [red]{len(analysis.failed_images)} failed at least one stage[/red]"
    )
    console.print(f"⏭️  [yellow]{len(analysis.skipped_images)} skipped[/yellow]")

    # Show cache info
    console.print("\n[dim]Results cached for future runs[/dim]")


def _format_result_status(result: AnalysisResult) -> str:
    """Format analysis result with color and emoji.

    Args:
        result: AnalysisResult to format.

    Returns:
        Formatted string with color and status indicator.
    """
    if result.value == "pass":
        return "[green]✅[/green]"
    elif result.value == "fail":
        return "[red]❌[/red]"
    else:  # skip
        return "[yellow]⏭️[/yellow]"


def _format_stage_result(stage_result: StageResult) -> str:
    """Format stage result with meaningful data instead of just pass/fail.

    Args:
        stage_result: StageResult to format with data.

    Returns:
        Formatted string with stage-specific information.
    """
    if stage_result.stage == AnalysisStage.DEDUPLICATION:
        return _format_deduplication_result(stage_result)
    elif stage_result.stage == AnalysisStage.QUALITY:
        return _format_quality_result(stage_result)
    elif stage_result.stage == AnalysisStage.FACE:
        return _format_face_result(stage_result)
    else:
        # Fallback to emoji for unknown stages
        return _format_result_status(stage_result.result)


def _format_deduplication_result(stage_result: StageResult) -> str:
    """Format deduplication result with hash or duplicate info."""
    if stage_result.result == AnalysisResult.PASS:
        # Show shortened hash for passed images
        if stage_result.metadata and "hash" in stage_result.metadata:
            hash_str = stage_result.metadata["hash"][:8]  # First 8 characters
            return f"[green]{hash_str}[/green]"
        return "[green]PASS[/green]"
    elif stage_result.result == AnalysisResult.FAIL:
        # Show duplicate info for failed images
        if stage_result.metadata and "duplicate_of" in stage_result.metadata:
            duplicate_of = stage_result.metadata["duplicate_of"][
                :12
            ]  # Truncate long names
            hamming_dist = stage_result.metadata.get("hamming_distance", "?")
            return f"[red]DUP:{duplicate_of}({hamming_dist})[/red]"
        return "[red]FAIL[/red]"
    else:
        return "[yellow]SKIP[/yellow]"


def _format_quality_result(stage_result: StageResult) -> str:
    """Format quality result with metrics (legacy function for non-separate columns)."""
    if stage_result.result == AnalysisResult.PASS:
        # Show key metrics for passed images
        if stage_result.metadata:
            sharpness = stage_result.metadata.get("sharpness_laplacian", "?")
            brightness = stage_result.metadata.get("brightness_mean", "?")
            # Format as "sharpness/brightness"
            try:
                s = f"{float(sharpness):.0f}" if sharpness != "?" else "?"
                b = f"{float(brightness):.0f}" if brightness != "?" else "?"
                return f"[green]{s}/{b}[/green]"
            except (ValueError, TypeError):
                return "[green]PASS[/green]"
        return "[green]PASS[/green]"
    elif stage_result.result == AnalysisResult.FAIL:
        # Show what failed
        if stage_result.metadata:
            failed_parts: list[str] = []
            if stage_result.metadata.get("sharpness_pass") == "False":
                failed_parts.append("S")
            if stage_result.metadata.get("brightness_pass") == "False":
                failed_parts.append("B")
            if stage_result.metadata.get("contrast_pass") == "False":
                failed_parts.append("C")
            if failed_parts:
                return f"[red]{''.join(failed_parts)}[/red]"
        return "[red]FAIL[/red]"
    else:
        return "[yellow]SKIP[/yellow]"


def _format_quality_metrics(
    stage_result: StageResult, stage_config: StageConfig | None = None
) -> tuple[str, str, str]:
    """Format quality metrics into separate columns with human-readable descriptions.

    Returns:
        Tuple of (sharpness_formatted, brightness_formatted, contrast_formatted)
    """
    if not stage_result.metadata:
        return "[dim]?[/dim]", "[dim]?[/dim]", "[dim]?[/dim]"

    # Get raw values
    sharpness = stage_result.metadata.get("sharpness_laplacian", "?")
    brightness = stage_result.metadata.get("brightness_mean", "?")
    contrast = stage_result.metadata.get("contrast_std", "?")

    # Get pass/fail status
    sharpness_pass = stage_result.metadata.get("sharpness_pass", "unknown") == "True"
    brightness_pass = stage_result.metadata.get("brightness_pass", "unknown") == "True"
    contrast_pass = stage_result.metadata.get("contrast_pass", "unknown") == "True"

    # Extract thresholds from stage config or use defaults
    if stage_config and stage_config.config:
        sharpness_threshold = float(
            stage_config.config.get("sharpness_threshold", "150")
        )
        brightness_min = float(stage_config.config.get("brightness_min", "60"))
        brightness_max = float(stage_config.config.get("brightness_max", "200"))
        contrast_threshold = float(stage_config.config.get("contrast_threshold", "40"))
    else:
        # Fallback to defaults if no config provided
        sharpness_threshold = 150.0
        brightness_min = 60.0
        brightness_max = 200.0
        contrast_threshold = 40.0

    # Format each metric with textual description and score relative to actual config
    try:
        # Sharpness (Laplacian variance - higher is sharper)
        if sharpness != "?":
            s_val = float(sharpness)
            s_color = "green" if sharpness_pass else "red"

            # Determine textual description relative to configured threshold
            if s_val >= sharpness_threshold:  # Above threshold
                if s_val > sharpness_threshold * 3:  # Very sharp
                    s_text = "Sharp"
                else:
                    s_text = "Clear"
            else:  # Below threshold
                if s_val < sharpness_threshold * 0.33:  # Very blurry
                    s_text = "Blurry"
                else:
                    s_text = "Soft"

            sharpness_str = f"[{s_color}]{s_text}[/{s_color}]"
        else:
            sharpness_str = "[dim]?[/dim]"

        # Brightness (0-255 scale)
        if brightness != "?":
            b_val = float(brightness)
            b_color = "green" if brightness_pass else "red"

            # Determine textual description relative to configured range
            if brightness_min <= b_val <= brightness_max:  # In acceptable range
                b_text = "Good"
            elif b_val < brightness_min:
                b_text = "Dark"
            else:  # > brightness_max
                b_text = "Bright"

            brightness_str = f"[{b_color}]{b_text}[/{b_color}]"
        else:
            brightness_str = "[dim]?[/dim]"

        # Contrast (standard deviation - higher is more contrast)
        if contrast != "?":
            c_val = float(contrast)
            c_color = "green" if contrast_pass else "red"

            # Determine textual description relative to configured threshold
            if c_val >= contrast_threshold:  # Above threshold
                if c_val > contrast_threshold * 2:  # Very high contrast
                    c_text = "Vivid"
                else:
                    c_text = "Good"
            else:  # Below threshold
                if c_val < contrast_threshold * 0.5:  # Very low contrast
                    c_text = "Flat"
                else:
                    c_text = "Low"

            contrast_str = f"[{c_color}]{c_text}[/{c_color}]"
        else:
            contrast_str = "[dim]?[/dim]"

    except (ValueError, TypeError):
        # Fallback for parsing errors
        s_color = "green" if sharpness_pass else "red"
        b_color = "green" if brightness_pass else "red"
        c_color = "green" if contrast_pass else "red"
        sharpness_str = (
            f"[{s_color}]Unknown ({sharpness}/{sharpness_threshold:.0f}+)[/{s_color}]"
        )
        brightness_str = f"[{b_color}]Unknown ({brightness} of {brightness_min:.0f}-{brightness_max:.0f})[/{b_color}]"
        contrast_str = (
            f"[{c_color}]Unknown ({contrast}/{contrast_threshold:.0f}+)[/{c_color}]"
        )

    return sharpness_str, brightness_str, contrast_str


def _format_face_result(stage_result: StageResult) -> str:
    """Format face detection result with people count."""
    if stage_result.result == AnalysisResult.PASS:
        # Show face/people count only
        if stage_result.metadata:
            face_count = stage_result.metadata.get("face_count", "0")
            count = int(face_count) if face_count != "0" else 0
            if count == 1:
                return "[green]1 person[/green]"
            elif count > 1:
                return f"[green]{count} people[/green]"
            else:
                return "[green]detected[/green]"
        return "[green]detected[/green]"
    elif stage_result.result == AnalysisResult.FAIL:
        return "[red]none[/red]"  # No people detected
    else:
        return "[yellow]skip[/yellow]"


def register_command(app: typer.Typer) -> None:
    """Register the analyze command with the given Typer app."""
    app.command(name="analyze")(analyze_command)
