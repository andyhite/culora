"""Configuration management commands for CuLoRA CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from culora.config import AnalysisStage
from culora.managers.config_manager import ConfigManager
from culora.utils.console import get_console

console = get_console()


def config_init_command(
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save config file (defaults to app data directory)",
        ),
    ] = None,
) -> None:
    """Create a default configuration file.

    This command creates a configuration file with all default values that can
    be customized. The file is saved in TOML format.

    If no output path is specified, the config is saved to the app data directory
    where CuLoRA will automatically find and use it.
    """
    try:
        manager = ConfigManager.get_instance()
        config_path = manager.save_defaults_to_file(output_path)
        console.success(f"Default configuration saved to: {config_path}")
        console.info("You can now edit this file to customize CuLoRA's behavior.")
        console.info("Configuration values can be overridden with CLI flags.")
    except Exception as e:
        console.error(f"Failed to create configuration file: {e}")
        raise typer.Exit(1) from e


def config_show_command(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration file (defaults to app data directory)",
        ),
    ] = None,
) -> None:
    """Show current configuration values.

    This command loads and displays the current configuration, showing which
    values are being used for analysis. Useful for debugging configuration issues.
    """
    try:
        manager = ConfigManager.get_instance()
        manager.load_from_file(config_file)
        config = manager.config

        # Create a nice display of the configuration
        config_text = ""
        config_dict = config.model_dump()

        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict):
                config_text += f"[bold blue][{section_name}][/bold blue]\n"
                for key, value in section_data.items():  # type: ignore[misc]
                    key_str = str(key)  # type: ignore[arg-type]
                    if key_str != "version":  # Skip version fields
                        config_text += f"  {key_str} = {value!s}\n"
                config_text += "\n"

        panel = Panel(
            config_text.strip(),
            title="Current Configuration",
            title_align="left",
            border_style="blue",
        )
        console.print(panel)

        # Show config file location
        if config_file is None:
            default_config_path = manager.get_config_file_path()
            if manager.config_file_exists():
                console.info(f"Configuration loaded from: {default_config_path}")
            else:
                console.info("Using default configuration (no config file found)")
        else:
            if manager.config_file_exists(config_file):
                console.info(f"Configuration loaded from: {config_file}")
            else:
                console.warning(
                    f"Config file not found: {config_file} (using defaults)"
                )

    except Exception as e:
        console.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1) from e


def config_validate_command(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration file to validate",
        ),
    ] = None,
) -> None:
    """Validate a configuration file.

    This command checks if a configuration file is valid and can be loaded
    without errors. Useful for testing configuration files before use.
    """
    try:
        manager = ConfigManager.get_instance()
        manager.load_from_file(config_file)
        config = manager.config
        console.success("Configuration file is valid!")

        # Show some key validation info
        total_weights = config.scoring.quality_weight + config.scoring.face_weight
        if abs(total_weights - 1.0) > 0.01:
            console.warning(
                f"Quality and face weights should sum to 1.0, got {total_weights}"
            )
        else:
            console.info(
                f"✓ Scoring weights are balanced ({config.scoring.quality_weight:.1f} + {config.scoring.face_weight:.1f} = 1.0)"
            )

        enabled_stages = [
            stage for stage in AnalysisStage if getattr(config, stage.value).enabled
        ]
        console.info(f"✓ Enabled stages: {enabled_stages}")

    except Exception as e:
        console.error(f"Configuration validation failed: {e}")
        raise typer.Exit(1) from e


def config_get_command(
    key: Annotated[
        str,
        typer.Argument(
            help="Configuration key to retrieve (e.g., 'quality.sharpness_threshold')"
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration file (defaults to app data directory)",
        ),
    ] = None,
) -> None:
    """Get a specific configuration value.

    Use dot notation to specify nested keys, for example:
    - quality.sharpness_threshold
    - face.confidence_threshold
    - scoring.quality_weight
    - display.score_excellent_threshold
    """
    try:
        manager = ConfigManager.get_instance()
        manager.load_from_file(config_file)

        value = manager.get_config_value(key)
        console.info(f"{key} = {value}")

    except ValueError as e:
        console.error(str(e))
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Failed to get configuration value: {e}")
        raise typer.Exit(1) from e


def config_set_command(
    key: Annotated[
        str,
        typer.Argument(
            help="Configuration key to set (e.g., 'quality.sharpness_threshold')"
        ),
    ],
    value: Annotated[str, typer.Argument(help="Value to set")],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration file (defaults to app data directory)",
        ),
    ] = None,
) -> None:
    """Set a specific configuration value.

    Use dot notation to specify nested keys, for example:
    - quality.sharpness_threshold 150.0
    - face.confidence_threshold 0.7
    - scoring.quality_weight 0.6
    - display.score_excellent_threshold 0.8

    Values are automatically converted to the appropriate type.
    """
    try:
        manager = ConfigManager.get_instance()
        manager.load_from_file(config_file)

        # Set the value (with type conversion)
        manager.set_config_value(key, value)

        # Validate weights if they were changed
        key_parts = key.split(".")
        if (
            len(key_parts) == 2
            and key_parts[0] == "scoring"
            and key_parts[1] in ("quality_weight", "face_weight")
        ):
            config = manager.config
            total_weight = config.scoring.quality_weight + config.scoring.face_weight
            if abs(total_weight - 1.0) > 0.01:
                console.warning(
                    f"Quality and face weights should sum to 1.0, got {total_weight}"
                )

        # Save the updated config
        config_path = manager.save_to_file(config_file)
        converted_value = manager.get_config_value(key)

        console.success(f"Set {key} = {converted_value}")
        console.info(f"Configuration saved to: {config_path}")

    except ValueError as e:
        console.error(str(e))
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Failed to set configuration value: {e}")
        raise typer.Exit(1) from e


def config_clear_command(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to configuration file (defaults to app data directory)",
        ),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt",
        ),
    ] = False,
) -> None:
    """Clear/reset configuration to defaults.

    This removes the configuration file, causing CuLoRA to use default values.
    The file can be recreated with 'culora config init'.
    """
    try:
        manager = ConfigManager.get_instance()
        config_path = manager.get_config_file_path(config_file)

        if not manager.config_file_exists(config_file):
            console.info("No configuration file found - already using defaults")
            return

        if not confirm:
            response = typer.confirm(f"Are you sure you want to delete {config_path}?")
            if not response:
                console.info("Configuration not changed")
                return

        manager.delete_config_file(config_file)
        console.success(f"Configuration file deleted: {config_path}")
        console.info("CuLoRA will now use default configuration values")
        console.info("Run 'culora config init' to create a new configuration file")

    except Exception as e:
        console.error(f"Failed to clear configuration: {e}")
        raise typer.Exit(1) from e


def register_commands(app: typer.Typer) -> None:
    """Register config commands with the given Typer app."""
    config_app = typer.Typer(name="config", help="Configuration management commands")

    config_app.command(name="init")(config_init_command)
    config_app.command(name="show")(config_show_command)
    config_app.command(name="validate")(config_validate_command)
    config_app.command(name="get")(config_get_command)
    config_app.command(name="set")(config_set_command)
    config_app.command(name="clear")(config_clear_command)

    app.add_typer(config_app)
