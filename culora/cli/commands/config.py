"""Configuration management commands."""

from pathlib import Path
from typing import Annotated

import typer

from culora.cli.display.console import console
from culora.cli.display.tables import display_config_table
from culora.cli.validation.validators import (
    convert_config_value,
    validate_config_file,
    validate_config_file_for_write,
    validate_config_key,
    validate_output_file,
)
from culora.core import ConfigError, InvalidConfigError, MissingConfigError
from culora.services import get_config_service

# Create config subcommand app
config_app = typer.Typer(
    name="config",
    help="Configuration management commands",
    rich_markup_mode="rich",
)


@config_app.command("show")
def show_config(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            callback=validate_config_file,
            help="Configuration file to load",
        ),
    ] = None,
) -> None:
    """Display current configuration."""
    try:
        config_service = get_config_service()

        # Load configuration
        config = config_service.load_config(config_file=config_file)

        # Get configuration summary
        summary = config_service.get_config_summary()

        console.header("Configuration")
        display_config_table(config, summary.get("sources"))

    except MissingConfigError as e:
        console.error(
            "No configuration loaded. Use 'culora config show --config <file>' to load from file."
        )
        raise typer.Exit(1) from e
    except ConfigError as e:
        console.error(f"Configuration error: {e}")
        raise typer.Exit(1) from e


@config_app.command("get")
def get_config_value(
    key: Annotated[
        str,
        typer.Argument(
            callback=validate_config_key,
            help="Configuration key to retrieve (e.g., 'device.preferred_device')",
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            callback=validate_config_file,
            help="Configuration file to load",
        ),
    ] = None,
) -> None:
    """Get a specific configuration value."""
    try:
        config_service = get_config_service()

        # Load configuration if not already loaded
        try:
            config_service.get_config()
        except MissingConfigError:
            config_service.load_config(config_file=config_file)

        # Get the value
        value = config_service.get_config_value(key)

        console.key_value(key, value)

    except KeyError as e:
        console.error(f"Configuration key not found: {e}")
        raise typer.Exit(1) from e
    except ConfigError as e:
        console.error(f"Configuration error: {e}")
        raise typer.Exit(1) from e


@config_app.command("set")
def set_config_value(
    key: Annotated[
        str,
        typer.Argument(
            callback=validate_config_key,
            help="Configuration key to set (e.g., 'device.preferred_device')",
        ),
    ],
    value: Annotated[str, typer.Argument(help="Value to set")],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            callback=validate_config_file_for_write,
            help="Configuration file to update",
        ),
    ] = None,
) -> None:
    """Set a configuration value and save to file."""
    try:
        config_service = get_config_service()

        # Convert value to appropriate type
        converted_value = convert_config_value(value)

        # Load configuration if not already loaded
        try:
            config_service.get_config()
        except MissingConfigError:
            config_service.load_config(config_file=config_file)

        # Set the value
        config_service.set_config_value(key, converted_value, config_file)

        console.success(f"Set {key} = {converted_value}")

        # Show which file was updated
        updated_file = config_service.get_config_file()
        if updated_file:
            console.info(f"Configuration saved to: {updated_file}")

    except InvalidConfigError as e:
        console.error(f"Invalid configuration value: {e}")
        if hasattr(e, "details") and e.details:
            console.error(f"Validation errors: {e.details}")
        raise typer.Exit(1) from e
    except ConfigError as e:
        console.error(f"Configuration error: {e}")
        raise typer.Exit(1) from e


@config_app.command("validate")
def validate_config(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            callback=validate_config_file,
            help="Configuration file to validate",
        ),
    ] = None,
) -> None:
    """Validate configuration file."""
    try:
        config_service = get_config_service()

        # Load and validate configuration
        config = config_service.load_config(config_file=config_file)

        console.success("Configuration is valid")

        # Show basic info about the configuration
        console.info(f"Device: {config.device.preferred_device}")
        console.info(f"Log Level: {config.logging.log_level}")

        # Show config file location if loaded from file
        loaded_file = config_service.get_config_file()
        if loaded_file:
            console.info(f"Loaded from: {loaded_file}")

    except ConfigError as e:
        console.error(f"Configuration validation failed: {e}")
        raise typer.Exit(1) from e


@config_app.command("export")
def export_config(
    output_file: Annotated[
        Path,
        typer.Argument(
            callback=validate_output_file,
            help="Output file path for exported configuration",
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            callback=validate_config_file,
            help="Configuration file to export",
        ),
    ] = None,
    include_defaults: Annotated[
        bool,
        typer.Option(
            "--include-defaults/--exclude-defaults",
            help="Include default values in exported configuration",
        ),
    ] = True,
) -> None:
    """Export configuration to a file."""
    try:
        config_service = get_config_service()

        # Load configuration if not already loaded
        try:
            config_service.get_config()
        except MissingConfigError:
            config_service.load_config(config_file=config_file)

        # Export configuration
        config_service.export_config(output_file, include_defaults=include_defaults)

        console.success(f"Configuration exported to: {output_file}")

    except ConfigError as e:
        console.error(f"Export failed: {e}")
        raise typer.Exit(1) from e
