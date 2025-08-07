"""Main CLI application for CuLoRA."""

import typer
from rich.console import Console

from culora import __version__
from culora.cli import analyze, select

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


analyze.register_command(app)
select.register_command(app)

if __name__ == "__main__":
    app()
