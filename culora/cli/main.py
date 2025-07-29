"""Main CLI entry point for CuLoRA."""

import typer
from rich.console import Console

from culora import VERSION

# Create Typer app and Rich console
app = typer.Typer(
    name="culora",
    help="CuLoRA - Advanced LoRA Dataset Curation Utility",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"CuLoRA version {VERSION}")


@app.command()
def curate(
    input_dir: str = typer.Argument(..., help="Input directory containing images"),
    output_dir: str = typer.Argument(..., help="Output directory for curated dataset"),
    target_count: int | None = typer.Option(
        None, "--count", "-c", help="Target number of images to select"
    ),
) -> None:
    """Curate an image dataset for LoRA training."""
    console.print(f"[bold green]CuLoRA v{VERSION}[/bold green]")
    console.print(f"Input directory: {input_dir}")
    console.print(f"Output directory: {output_dir}")
    if target_count:
        console.print(f"Target count: {target_count}")
    console.print("[yellow]Implementation coming soon...[/yellow]")


if __name__ == "__main__":
    app()
