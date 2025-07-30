"""Rich table displays for CuLoRA CLI."""

from typing import Any

from rich.table import Table

from culora.domain import CuLoRAConfig, Device, Memory

from .console import console


def display_config_table(
    config: CuLoRAConfig, sources: dict[str, str] | None = None
) -> None:
    """Display configuration in a formatted table.

    Args:
        config: Configuration to display
        sources: Configuration sources information
    """
    table = Table(title="CuLoRA Configuration", style="table.border")
    table.add_column("Section", style="table.header", no_wrap=True)
    table.add_column("Setting", style="key", no_wrap=True)
    table.add_column("Value", style="value")

    # Device configuration
    table.add_row("Device", "Preferred Device", str(config.device.preferred_device))

    # Logging configuration
    table.add_row("Logging", "Log Level", str(config.logging.log_level))
    if hasattr(config.logging, "log_file") and config.logging.log_file:
        table.add_row("", "Log File", str(config.logging.log_file))

    console.print(table)

    # Display sources if provided
    if sources:
        console.print()
        sources_table = Table(title="Configuration Sources", style="table.border")
        sources_table.add_column("Source", style="table.header")
        sources_table.add_column("Location", style="value")

        for source_type, location in sources.items():
            sources_table.add_row(source_type.title(), location)

        console.print(sources_table)


def display_device_table(devices: list[Device]) -> None:
    """Display device information in a formatted table.

    Args:
        devices: List of detected devices
    """
    table = Table(title="Detected Devices", style="table.border")
    table.add_column("Device", style="table.header")
    table.add_column("Type", style="key")
    table.add_column("Name", style="value")
    table.add_column("Available", style="key")
    table.add_column("Memory", style="value")

    for device in devices:
        # Format availability
        available = "✅ Yes" if device.is_available else "❌ No"

        # Format memory info
        memory_str = "N/A"
        if device.memory:
            if device.memory.total_mb:
                total_gb = device.memory.total_mb / 1024
                available_gb = (
                    device.memory.available_mb / 1024
                    if device.memory.available_mb
                    else 0
                )
                memory_str = f"{available_gb:.1f}GB / {total_gb:.1f}GB"
            else:
                memory_str = "System RAM"

        table.add_row(
            device.device_type.value,
            device.device_type.value,
            device.name,
            available,
            memory_str,
        )

    console.print(table)


def display_memory_table(memory_info: Memory) -> None:
    """Display memory information in a formatted table.

    Args:
        memory_info: Memory information to display
    """
    table = Table(title="Memory Information", style="table.border")
    table.add_column("Metric", style="table.header")
    table.add_column("Value", style="value")

    if memory_info.total_mb:
        total_gb = memory_info.total_mb / 1024
        available_gb = (
            memory_info.available_mb / 1024 if memory_info.available_mb else 0
        )
        table.add_row("Total Memory", f"{total_gb:.1f} GB")
        table.add_row("Available Memory", f"{available_gb:.1f} GB")
        if memory_info.usage_percent:
            table.add_row("Usage", f"{memory_info.usage_percent:.1f}%")
    else:
        table.add_row("Memory Type", "System RAM")
        if memory_info.available_mb:
            available_gb = memory_info.available_mb / 1024
            table.add_row("Available", f"{available_gb:.1f} GB")

    console.print(table)


def display_key_value_table(title: str, data: dict[str, Any]) -> None:
    """Display key-value data in a formatted table.

    Args:
        title: Table title
        data: Dictionary of key-value pairs to display
    """
    table = Table(title=title, style="table.border")
    table.add_column("Key", style="key")
    table.add_column("Value", style="value")

    def add_nested_rows(data_dict: dict[str, Any], prefix: str = "") -> None:
        """Recursively add nested dictionary data to table."""
        for key, value in data_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Add section header
                table.add_row(f"[table.header]{full_key}[/table.header]", "")
                add_nested_rows(value, full_key)
            else:
                table.add_row(full_key, str(value))

    add_nested_rows(data)
    console.print(table)
