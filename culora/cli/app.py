"""Main CuLoRA CLI application."""

from typing import Annotated

import typer

from culora.cli.commands.config import config_app
from culora.cli.commands.device import device_app
from culora.cli.commands.faces import faces_app
from culora.cli.commands.images import images_app
from culora.cli.display.console import console
from culora.core import ConfigError, CuLoRAError
from culora.services.config_service import get_config_service
from culora.services.device_service import get_device_service
from culora.services.face_analysis_service import get_face_analysis_service
from culora.services.image_service import get_image_service
from culora.services.memory_service import get_memory_service

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
app.add_typer(faces_app, name="faces", help="Face detection and analysis")


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

    # Initialize global services if a subcommand will be invoked
    if ctx.invoked_subcommand is not None:
        try:
            # Initialize all services using consistent get_*_service pattern
            get_config_service()
            get_device_service()
            get_memory_service()
            get_image_service()
            get_face_analysis_service()

        except Exception as e:
            console.error(f"Failed to initialize services: {e}")
            raise typer.Exit(1) from e
    else:
        # If no subcommand was invoked, show help
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
