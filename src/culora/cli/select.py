"""Select command for CuLoRA CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def select_command(
    output_dir: Annotated[
        str, typer.Argument(help="Directory to copy selected images to")
    ],
    input_dir: Annotated[
        str, typer.Option(help="Directory containing analyzed images")
    ] = ".",
    dry_run: Annotated[
        bool, typer.Option(help="Show what would be selected without copying files")
    ] = False,
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


def register_command(app: typer.Typer) -> None:
    """Register the select command with the given Typer app."""
    app.command(name="select")(select_command)
