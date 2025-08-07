"""Device information and management commands."""

import typer

from culora.cli.display.console import console
from culora.cli.display.tables import display_device_table, display_memory_table
from culora.core import DeviceError
from culora.services import get_device_service

# Create device subcommand app
device_app = typer.Typer(
    name="device",
    help="Device information and management commands",
    rich_markup_mode="rich",
)


@device_app.command("info")
def device_info() -> None:
    """Show detected device information."""
    try:
        device_service = get_device_service()

        # Get all detected devices
        devices = device_service.get_all_devices()
        available_devices = [d for d in devices if d.is_available]

        if not available_devices:
            console.warning("No available devices detected")
            return

        console.header("Device Information")
        display_device_table(available_devices)

        # Show selected device
        selected_device = device_service.get_selected_device()
        console.info(
            f"Selected device: {selected_device.name} ({selected_device.device_type.value})"
        )

    except DeviceError as e:
        console.error(f"Device detection error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@device_app.command("list")
def list_devices() -> None:
    """List all available devices."""
    try:
        device_service = get_device_service()

        # Get all devices (including unavailable ones)
        all_devices = device_service.get_all_devices()

        if not all_devices:
            console.warning("No devices found")
            return

        console.header("All Devices")
        display_device_table(all_devices)

        # Show counts
        available_count = sum(1 for d in all_devices if d.is_available)
        console.info(f"Available devices: {available_count}/{len(all_devices)}")

    except DeviceError as e:
        console.error(f"Device detection error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@device_app.command("memory")
def memory_info() -> None:
    """Show memory information for the selected device."""
    try:
        device_service = get_device_service()

        # Get selected device
        selected_device = device_service.get_selected_device()

        console.header(f"Memory Information - {selected_device.name}")

        if selected_device.memory:
            display_memory_table(selected_device.memory)
        else:
            console.warning("No memory information available for this device")

        # Additional memory details functionality not yet implemented
        # Could be added to MemoryService in the future if needed

    except DeviceError as e:
        console.error(f"Device or memory error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e
