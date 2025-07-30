"""CLI argument validators for CuLoRA."""

from pathlib import Path
from typing import Any

import typer


def validate_config_file(value: str | None) -> Path | None:
    """Validate configuration file path.

    Args:
        value: Configuration file path string

    Returns:
        Validated Path object or None

    Raises:
        typer.BadParameter: If file doesn't exist or isn't readable
    """
    if value is None:
        return None

    path = Path(value)

    if not path.exists():
        raise typer.BadParameter(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise typer.BadParameter(f"Configuration path is not a file: {path}")

    if path.suffix.lower() not in [".yaml", ".yml", ".json"]:
        raise typer.BadParameter(
            f"Configuration file must be .yaml/.yml or .json: {path}"
        )

    try:
        # Test if file is readable
        with open(path):
            pass
    except PermissionError as e:
        raise typer.BadParameter(f"Cannot read configuration file: {path}") from e
    except Exception as e:
        raise typer.BadParameter(
            f"Error accessing configuration file {path}: {e}"
        ) from e

    return path


def validate_config_file_for_write(value: str | None) -> Path | None:
    """Validate configuration file path for writing (allows creation).

    Args:
        value: Configuration file path string

    Returns:
        Validated Path object or None

    Raises:
        typer.BadParameter: If path is invalid
    """
    if value is None:
        return None

    path = Path(value)

    # Check if file extension is valid
    if path.suffix.lower() not in [".yaml", ".yml", ".json"]:
        raise typer.BadParameter(
            f"Configuration file must be .yaml/.yml or .json: {path}"
        )

    # If file exists, check that it's actually a file and readable
    if path.exists():
        if not path.is_file():
            raise typer.BadParameter(f"Configuration path is not a file: {path}")

        try:
            # Test if file is readable
            with open(path):
                pass
        except PermissionError as e:
            raise typer.BadParameter(f"Cannot read configuration file: {path}") from e
        except Exception as e:
            raise typer.BadParameter(
                f"Error accessing configuration file {path}: {e}"
            ) from e
    else:
        # File doesn't exist, check if parent directory exists or can be created
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise typer.BadParameter(
                f"Cannot create parent directory: {path.parent}"
            ) from e
        except Exception as e:
            raise typer.BadParameter(
                f"Error creating parent directory for {path}: {e}"
            ) from e

    return path


def validate_output_file(value: str) -> Path:
    """Validate output file path for export.

    Args:
        value: Output file path string

    Returns:
        Validated Path object

    Raises:
        typer.BadParameter: If path is invalid or not writable
    """
    path = Path(value)

    # Check if parent directory exists or can be created
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise typer.BadParameter(
            f"Cannot create parent directory: {path.parent}"
        ) from e
    except Exception as e:
        raise typer.BadParameter(
            f"Error creating parent directory for {path}: {e}"
        ) from e

    # Check if file can be written (if it exists)
    if path.exists():
        if not path.is_file():
            raise typer.BadParameter(f"Output path exists but is not a file: {path}")
        try:
            # Test write access
            with open(path, "a"):
                pass
        except PermissionError as e:
            raise typer.BadParameter(f"Cannot write to output file: {path}") from e
        except Exception as e:
            raise typer.BadParameter(f"Error accessing output file {path}: {e}") from e

    return path


def validate_config_key(value: str) -> str:
    """Validate configuration key path.

    Args:
        value: Configuration key path (e.g., 'device.preferred_device')

    Returns:
        Validated key path

    Raises:
        typer.BadParameter: If key path is invalid
    """
    if not value:
        raise typer.BadParameter("Configuration key cannot be empty")

    if not value.replace(".", "").replace("_", "").isalnum():
        raise typer.BadParameter(f"Invalid configuration key format: {value}")

    # Check for valid key patterns
    valid_prefixes = [
        "device.",
        "logging.",
        "quality_assessment.",
        "selection.",
        "export.",
    ]
    if "." in value and not any(value.startswith(prefix) for prefix in valid_prefixes):
        raise typer.BadParameter(
            f"Invalid configuration section in key: {value}. "
            f"Valid sections: {', '.join(prefix.rstrip('.') for prefix in valid_prefixes)}"
        )

    return value


def convert_config_value(value: str) -> Any:
    """Convert string configuration value to appropriate type.

    Args:
        value: String value from CLI

    Returns:
        Converted value (bool, int, float, or string)
    """
    # Handle boolean values
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False

    # Try integer conversion
    try:
        return int(value)
    except ValueError:
        pass

    # Try float conversion
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value
