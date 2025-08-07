"""Main CLI application for CuLoRA."""

from pathlib import Path

import typer
from rich.console import Console

from culora import __version__

app = typer.Typer(
    name="culora",
    help="A command-line tool for intelligently curating image datasets for LoRA training.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """CuLoRA - Intelligently curate image datasets for LoRA training."""


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"CuLoRA version: {__version__}")


@app.command()
def analyze(
    input_dir: str = typer.Argument(help="Directory containing images to analyze"),
    no_dedupe: bool = typer.Option(False, help="Disable image deduplication"),
    no_quality: bool = typer.Option(False, help="Disable image quality assessment"),
    no_face: bool = typer.Option(False, help="Disable face detection"),
) -> None:
    """Analyze images in a directory for curation.

    This command analyzes all images in the specified directory using multiple
    stages: deduplication, quality assessment, and face detection. Results are
    cached for future runs.
    """
    input_path = Path(input_dir)
    console.print(f"[bold green]Analyzing images in:[/bold green] {input_path}")

    stages: list[str] = []
    if not no_dedupe:
        stages.append("deduplication")
    if not no_quality:
        stages.append("quality assessment")
    if not no_face:
        stages.append("face detection")

    console.print(f"[blue]Enabled stages:[/blue] {', '.join(stages)}")
    console.print("[yellow]Analysis not yet implemented[/yellow]")


@app.command()
def select(
    output_dir: str = typer.Argument(help="Directory to copy selected images to"),
    input_dir: str = typer.Option(
        ".",
        "--input",
        "-i",
        help="Directory containing analyzed images (defaults to current directory)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be selected without copying files"
    ),
) -> None:
    """Select and copy curated images to output directory.

    This command copies images that passed all enabled analysis stages from
    the analyzed directory to the specified output directory. Images are
    renamed sequentially for training use.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    console.print(f"[bold green]Selecting images from:[/bold green] {input_path}")
    console.print(f"[bold green]Output directory:[/bold green] {output_path}")

    if dry_run:
        console.print("[blue]Dry run mode - no files will be copied[/blue]")

    console.print("[yellow]Selection not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
