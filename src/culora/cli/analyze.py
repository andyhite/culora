"""Analyze command for CuLoRA CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from culora.analysis.analyzer import analyze_directory
from culora.models.analysis import AnalysisResult, DirectoryAnalysis

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
) -> None:
    """Analyze images in a directory for curation.

    This command analyzes all images in the specified directory using multiple
    stages: deduplication, quality assessment, and face detection. Results are
    cached for future runs.
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
    table.add_column("Image", style="cyan", width=30)
    table.add_column("Overall", justify="center", width=10)

    # Add a column for each enabled stage
    for stage in analysis.enabled_stages:
        table.add_column(stage.value.title(), justify="center", width=12)

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
                row_data.append(_format_result_status(result.result))
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


def register_command(app: typer.Typer) -> None:
    """Register the analyze command with the given Typer app."""
    app.command(name="analyze")(analyze_command)
