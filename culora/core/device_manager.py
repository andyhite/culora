"""Device management for CuLoRA.

This module provides high-level device management functionality including
device selection, configuration, and status reporting.
"""

from rich.console import Console
from rich.table import Table

from .config import CuLoRAConfig
from .device_detector import DeviceDetector
from .device_info import DeviceInfo
from .exceptions import DeviceError
from .logging import CuLoRALogger
from .types import DeviceType


class DeviceManager:
    """Manages device selection and configuration for AI model execution."""

    def __init__(self, config: CuLoRAConfig, logger: CuLoRALogger) -> None:
        """Initialize device manager.

        Args:
            config: CuLoRA configuration
            logger: Structured logger instance
        """
        self.config = config
        self.logger = logger
        self.detector = DeviceDetector()
        self._selected_device: DeviceInfo | None = None
        self._all_devices: list[DeviceInfo] | None = None

    def initialize(self) -> DeviceInfo:
        """Initialize device management and select optimal device.

        Returns:
            DeviceInfo for the selected device

        Raises:
            DeviceError: If device initialization fails critically
        """
        self.logger.info("Initializing device management")

        try:
            # Detect all available devices
            self._all_devices = self.detector.detect_all_devices()

            # Log device detection results
            self.logger.info(
                "Device detection completed",
                device_count=len(self._all_devices),
                available_count=len([d for d in self._all_devices if d.is_available]),
            )

            # Select device based on configuration
            self._selected_device = self._select_device()

            self.logger.info(
                "Device selected",
                device_type=self._selected_device.device_type.value,
                device_name=self._selected_device.name,
                has_memory_info=self._selected_device.memory_total is not None,
            )

            return self._selected_device

        except Exception as e:
            self.logger.error("Device initialization failed", error=str(e))
            raise DeviceError(f"Failed to initialize device management: {e}") from e

    def get_selected_device(self) -> DeviceInfo:
        """Get the currently selected device.

        Returns:
            DeviceInfo for selected device

        Raises:
            DeviceError: If no device has been selected
        """
        if self._selected_device is None:
            raise DeviceError("No device selected. Call initialize() first.")
        return self._selected_device

    def get_all_devices(self) -> list[DeviceInfo]:
        """Get all detected devices.

        Returns:
            List of all DeviceInfo objects

        Raises:
            DeviceError: If device detection hasn't been run
        """
        if self._all_devices is None:
            raise DeviceError("Device detection not run. Call initialize() first.")
        return self._all_devices

    def display_device_status(self, console: Console) -> None:
        """Display device status using Rich console.

        Args:
            console: Rich console for output
        """
        if self._all_devices is None:
            console.print("[red]Device detection not run[/red]")
            return

        table = Table(title="🖥️  Device Status")
        table.add_column("Device", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Memory", style="blue")
        table.add_column("Selected", style="yellow")

        for device in self._all_devices:
            # Status with emoji
            if device.is_available:
                if device.has_sufficient_memory:
                    status = "✅ Available"
                else:
                    status = "⚠️  Low Memory"
            else:
                status = f"❌ {device.error_message or 'Unavailable'}"

            # Memory info
            if device.memory_total is not None:
                memory_str = f"{device.memory_available}/{device.memory_total} MB"
                if device.memory_usage_percent is not None:
                    memory_str += f" ({device.memory_usage_percent:.1f}% used)"
            else:
                memory_str = "N/A"

            # Selected indicator
            is_selected = (
                self._selected_device
                and device.device_type == self._selected_device.device_type
                and device.name == self._selected_device.name
            )
            selected_str = "🎯 Yes" if is_selected else ""

            table.add_row(
                device.name,
                device.device_type.value.upper(),
                status,
                memory_str,
                selected_str,
            )

        console.print(table)

        if self._selected_device:
            console.print(
                f"\n[bold green]Selected device:[/bold green] {self._selected_device}"
            )

    def estimate_memory_usage(self, model_name: str) -> int | None:
        """Estimate memory usage for a given model.

        Args:
            model_name: Name of the model to estimate for

        Returns:
            Estimated memory usage in MB, or None if unknown
        """
        # Basic estimates for common models (can be expanded)
        model_estimates = {
            "insightface": 500,
            "clip": 800,
            "moondream": 1200,
            "mediapipe": 100,
            "brisque": 50,
        }

        return model_estimates.get(model_name.lower())

    def check_memory_availability(self, required_mb: int) -> bool:
        """Check if selected device has sufficient memory.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if sufficient memory available
        """
        if self._selected_device is None:
            return False

        # CPU always returns True (system memory)
        if self._selected_device.device_type == DeviceType.CPU:
            return True

        if self._selected_device.memory_available is None:
            # Unknown memory, assume sufficient for MPS
            return self._selected_device.device_type == DeviceType.MPS

        return self._selected_device.memory_available >= required_mb

    def _select_device(self) -> DeviceInfo:
        """Select optimal device based on configuration and availability."""
        # If user specified a device preference, try to honor it
        preferred_type = self.config.device.preferred_device

        if preferred_type != DeviceType.CPU:
            # Look for preferred device type
            assert self._all_devices is not None  # Should be set by initialize()
            for device in self._all_devices:
                if (
                    device.device_type == preferred_type
                    and device.is_available
                    and device.has_sufficient_memory
                ):
                    return device

            self.logger.warning(
                "Preferred device not available, using automatic selection",
                preferred_device=preferred_type.value,
            )

        # Fall back to automatic optimal selection
        return self.detector.get_optimal_device()

    def get_execution_provider(self) -> str:
        """Get the execution provider string for the selected device.

        Returns:
            Execution provider string compatible with model libraries
        """
        if self._selected_device is None:
            return "cpu"

        provider_map = {
            DeviceType.CUDA: "cuda",
            DeviceType.MPS: "mps",
            DeviceType.CPU: "cpu",
        }

        return provider_map[self._selected_device.device_type]
