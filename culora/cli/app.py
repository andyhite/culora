"""Main CuLoRA CLI application."""

from typing import Annotated

import typer

from culora.cli.commands.config import config_app
from culora.cli.commands.device import device_app
from culora.cli.commands.images import images_app
from culora.cli.display.console import console
from culora.core import ConfigError, CuLoRAError

# Create main Typer app
app = typer.Typer(
    name="culora",
    help="CuLoRA - Advanced LoRA Dataset Curation Utility",
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)

# Add subcommands
app.add_typer(config_app, name="config", help="Configuration management")
app.add_typer(device_app, name="device", help="Device information and management")
app.add_typer(images_app, name="images", help="Image loading and processing")


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", "-v", help="Show version information")
    ] = False,
) -> None:
    """CuLoRA - Advanced LoRA Dataset Curation Utility.

    An intelligent command-line tool for curating image datasets specifically
    optimized for LoRA (Low-Rank Adaptation) training workflows.
    """
    if version:
        console.info("CuLoRA v0.1.0")
        raise typer.Exit(0)

    # If no subcommand was invoked, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def cli_main() -> None:
    """Main CLI entry point with error handling."""
    try:
        app()
    except ConfigError as e:
        console.error(f"Configuration Error: {e}")
        if hasattr(e, "error_code") and e.error_code:
            console.info(f"Error Code: {e.error_code}")
        raise typer.Exit(1) from e
    except CuLoRAError as e:
        console.error(f"CuLoRA Error: {e}")
        if hasattr(e, "error_code") and e.error_code:
            console.info(f"Error Code: {e.error_code}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt as e:
        console.warning("Operation cancelled by user")
        raise typer.Exit(130) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        console.info(
            "This may be a bug. Please report it at: https://github.com/andyhite/culora/issues"
        )
        raise typer.Exit(1) from e


if __name__ == "__main__":
    cli_main()
