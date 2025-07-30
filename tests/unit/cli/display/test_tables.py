"""Tests for CLI table displays."""

from unittest.mock import patch

from culora.cli.display.tables import (
    display_config_table,
    display_device_table,
    display_key_value_table,
    display_memory_table,
)
from culora.domain import Device, DeviceType, Memory
from tests.helpers import ConfigBuilder


class TestDisplayConfigTable:
    """Test configuration table display."""

    def test_display_config_table_basic(self) -> None:
        """Test basic configuration table display."""
        config = ConfigBuilder().build()

        with patch("culora.cli.display.tables.console") as mock_console:
            display_config_table(config)

            # Should call print twice - once for main table, once for sources
            assert mock_console.print.call_count >= 1

            # Check that a table was created
            call_args = mock_console.print.call_args_list[0][0]
            table = call_args[0]
            assert hasattr(table, "columns")

    def test_display_config_table_with_sources(self) -> None:
        """Test configuration table display with sources."""
        config = ConfigBuilder().build()
        sources = {"defaults": "Built-in defaults", "file": "/path/to/config.yaml"}

        with patch("culora.cli.display.tables.console") as mock_console:
            display_config_table(config, sources)

            # Should call print twice - main table and sources table
            assert mock_console.print.call_count >= 2

    def test_display_config_table_no_sources(self) -> None:
        """Test configuration table display without sources."""
        config = ConfigBuilder().build()

        with patch("culora.cli.display.tables.console") as mock_console:
            display_config_table(config, None)

            # Should call print once for main table only
            assert mock_console.print.call_count == 1


class TestDisplayDeviceTable:
    """Test device table display."""

    def test_display_device_table_single_device(self) -> None:
        """Test device table with single device."""
        memory = Memory(total_mb=8192, available_mb=4096)
        device = Device(
            device_type=DeviceType.CUDA,
            name="NVIDIA GeForce RTX 3080",
            memory=memory,
            is_available=True,
        )

        with patch("culora.cli.display.tables.console") as mock_console:
            display_device_table([device])

            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0]
            table = call_args[0]
            assert hasattr(table, "columns")

    def test_display_device_table_multiple_devices(self) -> None:
        """Test device table with multiple devices."""
        devices = [
            Device(
                device_type=DeviceType.CPU,
                name="CPU",
                memory=Memory(total_mb=None, available_mb=None),
                is_available=True,
            ),
            Device(
                device_type=DeviceType.CUDA,
                name="NVIDIA GPU",
                memory=Memory(total_mb=8192, available_mb=4096),
                is_available=True,
            ),
        ]

        with patch("culora.cli.display.tables.console") as mock_console:
            display_device_table(devices)

            mock_console.print.assert_called_once()

    def test_display_device_table_unavailable_device(self) -> None:
        """Test device table with unavailable device."""
        device = Device(
            device_type=DeviceType.CUDA,
            name="Unavailable GPU",
            memory=None,
            is_available=False,
            error_message="Driver not found",
        )

        with patch("culora.cli.display.tables.console") as mock_console:
            display_device_table([device])

            mock_console.print.assert_called_once()

    def test_display_device_table_empty_list(self) -> None:
        """Test device table with empty device list."""
        with patch("culora.cli.display.tables.console") as mock_console:
            display_device_table([])

            mock_console.print.assert_called_once()


class TestDisplayMemoryTable:
    """Test memory table display."""

    def test_display_memory_table_with_total(self) -> None:
        """Test memory table with total memory info."""
        memory = Memory(total_mb=8192, available_mb=4096)

        with patch("culora.cli.display.tables.console") as mock_console:
            display_memory_table(memory)

            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0]
            table = call_args[0]
            assert hasattr(table, "columns")

    def test_display_memory_table_unlimited(self) -> None:
        """Test memory table with unlimited memory (CPU)."""
        memory = Memory(total_mb=None, available_mb=None)

        with patch("culora.cli.display.tables.console") as mock_console:
            display_memory_table(memory)

            mock_console.print.assert_called_once()

    def test_display_memory_table_no_available(self) -> None:
        """Test memory table with no available memory info."""
        memory = Memory(total_mb=None, available_mb=4096)

        with patch("culora.cli.display.tables.console") as mock_console:
            display_memory_table(memory)

            mock_console.print.assert_called_once()


class TestDisplayKeyValueTable:
    """Test key-value table display."""

    def test_display_key_value_table_flat(self) -> None:
        """Test key-value table with flat data."""
        data = {"key1": "value1", "key2": "value2", "key3": 42}

        with patch("culora.cli.display.tables.console") as mock_console:
            display_key_value_table("Test Data", data)

            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0]
            table = call_args[0]
            assert hasattr(table, "columns")

    def test_display_key_value_table_nested(self) -> None:
        """Test key-value table with nested data."""
        data = {
            "section1": {"nested_key1": "value1", "nested_key2": "value2"},
            "section2": {"nested_key3": "value3"},
            "flat_key": "flat_value",
        }

        with patch("culora.cli.display.tables.console") as mock_console:
            display_key_value_table("Nested Data", data)

            mock_console.print.assert_called_once()

    def test_display_key_value_table_empty(self) -> None:
        """Test key-value table with empty data."""
        with patch("culora.cli.display.tables.console") as mock_console:
            display_key_value_table("Empty Data", {})

            mock_console.print.assert_called_once()

    def test_display_key_value_table_deeply_nested(self) -> None:
        """Test key-value table with deeply nested data."""
        data = {"level1": {"level2": {"level3": {"deep_key": "deep_value"}}}}

        with patch("culora.cli.display.tables.console") as mock_console:
            display_key_value_table("Deep Data", data)

            mock_console.print.assert_called_once()
