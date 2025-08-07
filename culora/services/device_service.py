"""Device service for CuLoRA.

This module provides high-level device management functionality including
device selection, configuration, and status reporting.
"""

from typing import Any

from rich.console import Console
from rich.table import Table

from culora.core.exceptions import DeviceError
from culora.domain.models import CuLoRAConfig
from culora.domain.models.device import Device, DeviceType
from culora.domain.models.memory import Memory


class DeviceService:
    """Manages device selection and configuration for AI model execution."""

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize device service.

        Args:
            config: CuLoRA configuration
        """
        self.config = config
        self._selected_device: Device | None = None
        self._all_devices: list[Device] | None = None
        self._torch_available: bool | None = None

    def initialize(self) -> Device:
        """Initialize device management and select optimal device.

        Returns:
            Device for the selected device

        Raises:
            DeviceError: If device initialization fails critically
        """
        try:
            # Detect all available devices
            self._all_devices = self._detect_all_devices()

            # Select device based on configuration
            self._selected_device = self._select_optimal_device(self._all_devices)

            return self._selected_device

        except Exception as e:
            raise DeviceError(f"Failed to initialize device management: {e}") from e

    def get_selected_device(self) -> Device:
        """Get the currently selected device.

        Returns:
            Device for selected device

        Raises:
            DeviceError: If no device has been selected
        """
        if self._selected_device is None:
            raise DeviceError("No device selected. Call initialize() first.")
        return self._selected_device

    def get_all_devices(self) -> list[Device]:
        """Get all detected devices.

        Returns:
            List of all Device objects

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

        self._display_device_status(console, self._all_devices, self._selected_device)

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

    def get_device_summary(self) -> dict[str, Any]:
        """Get device summary for programmatic use.

        Returns:
            Dictionary with device summary information
        """
        if self._all_devices is None:
            return {"error": "Device detection not run"}

        return self._create_device_summary(self._all_devices, self._selected_device)

    def _detect_all_devices(self) -> list[Device]:
        """Detect all available devices.

        Returns:
            List of Device objects for all detected devices

        Raises:
            DeviceError: If critical device detection fails
        """
        devices: list[Device] = []

        # Always add CPU as fallback
        devices.append(self._detect_cpu())

        # Try to detect CUDA devices
        cuda_devices = self._detect_cuda_devices()
        devices.extend(cuda_devices)

        # Try to detect MPS
        mps_device = self._detect_mps()
        if mps_device:
            devices.append(mps_device)

        return devices

    def _get_optimal_device(self) -> Device:
        """Get the optimal device for AI model execution.

        Priority: CUDA > MPS > CPU

        Returns:
            Device for the best available device
        """
        devices = self._detect_all_devices()

        # Filter to available devices with sufficient memory
        available_devices = [
            d for d in devices if d.is_available and d.has_sufficient_memory
        ]

        if not available_devices:
            # Return CPU as ultimate fallback
            return next(
                (d for d in devices if d.device_type == DeviceType.CPU),
                self._detect_cpu(),
            )

        # Priority order: CUDA, MPS, CPU
        for device_type in [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]:
            for device in available_devices:
                if device.device_type == device_type:
                    return device

        # Should never reach here, but return first available as fallback
        return available_devices[0]

    def _select_optimal_device(self, available_devices: list[Device]) -> Device:
        """Select optimal device based on configuration and availability.

        Args:
            available_devices: List of available devices

        Returns:
            Selected optimal device
        """
        # If user specified a device preference, try to honor it
        preferred_type = self.config.device.preferred_device

        if preferred_type != DeviceType.CPU:
            # Look for preferred device type
            for device in available_devices:
                if (
                    device.device_type == preferred_type
                    and device.is_available
                    and device.has_sufficient_memory
                ):
                    return device

        # Fall back to automatic optimal selection
        optimal_device = self._get_optimal_device()
        return optimal_device

    def _detect_cpu(self) -> Device:
        """Detect CPU device (always available)."""
        return Device(
            device_type=DeviceType.CPU,
            name="CPU",
            memory=Memory(total_mb=None, available_mb=None),  # Unlimited system memory
            is_available=True,
        )

    def _detect_cuda_devices(self) -> list[Device]:
        """Detect CUDA GPU devices."""
        devices: list[Device] = []

        if not self.torch_available:
            return devices

        try:
            import torch

            if not torch.cuda.is_available():
                return devices

            device_count = torch.cuda.device_count()

            for i in range(device_count):
                try:
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)

                    # Get memory information
                    torch.cuda.set_device(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory // (
                        1024 * 1024
                    )  # MB
                    memory_free = torch.cuda.memory_reserved(i) // (1024 * 1024)  # MB
                    memory_available = memory_total - memory_free

                    device = Device(
                        device_type=DeviceType.CUDA,
                        name=f"CUDA:{i} ({props.name})",
                        memory=Memory(
                            total_mb=memory_total, available_mb=memory_available
                        ),
                        is_available=True,
                    )
                    devices.append(device)

                except Exception as e:
                    # Individual device failed, mark as unavailable
                    device = Device(
                        device_type=DeviceType.CUDA,
                        name=f"CUDA:{i}",
                        memory=Memory(total_mb=0, available_mb=0),
                        is_available=False,
                        error_message=str(e),
                    )
                    devices.append(device)

        except Exception:
            # CUDA detection failed entirely, but don't fail - just log
            pass

        return devices

    def _detect_mps(self) -> Device | None:
        """Detect Apple Silicon MPS device."""
        if not self.torch_available:
            return None

        try:
            import torch

            if not torch.backends.mps.is_available():
                return None

            # MPS doesn't provide memory information like CUDA
            return Device(
                device_type=DeviceType.MPS,
                name="Apple Silicon MPS",
                memory=Memory(total_mb=None, available_mb=None),  # Unknown memory
                is_available=True,
            )

        except Exception as e:
            # MPS detection failed
            return Device(
                device_type=DeviceType.MPS,
                name="Apple Silicon MPS",
                memory=Memory(total_mb=None, available_mb=None),
                is_available=False,
                error_message=str(e),
            )

    @property
    def torch_available(self) -> bool:
        """Check if PyTorch is available for device detection."""
        if self._torch_available is None:
            try:
                __import__("torch")
                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available

    def _display_device_status(
        self,
        console: Console,
        devices: list[Device],
        selected_device: Device | None,
    ) -> None:
        """Display device status table using Rich console.

        Args:
            console: Rich console for output
            devices: List of all devices
            selected_device: Currently selected device
        """
        table = Table(
            title="🖥️  Device Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("Device", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Memory", justify="right")
        table.add_column("Selected", justify="center")

        for device in devices:
            # Device name
            device_name = device.name

            # Device type
            device_type = device.device_type.value.upper()

            # Status with color coding
            if device.is_available:
                if device.has_sufficient_memory:
                    status = "[green]✓ Available[/green]"
                else:
                    status = "[yellow]⚠ Low Memory[/yellow]"
            else:
                status = f"[red]✗ {device.error_message or 'Unavailable'}[/red]"

            # Memory information
            memory_info = str(device.memory) if device.memory is not None else "N/A"

            # Selected indicator
            is_selected = (
                "[bold green]●[/bold green]"
                if selected_device and device.device_type == selected_device.device_type
                else " "
            )

            table.add_row(device_name, device_type, status, memory_info, is_selected)

        console.print(table)

        # Show selection summary
        if selected_device:
            console.print(
                f"\n[bold]Selected Device:[/bold] {selected_device.name} "
                f"({selected_device.device_type.value.upper()})"
            )

    def _create_device_summary(
        self, devices: list[Device], selected_device: Device | None
    ) -> dict[str, Any]:
        """Create device summary for programmatic use.

        Args:
            devices: List of all devices
            selected_device: Currently selected device

        Returns:
            Dictionary with device summary information
        """
        device_list = []
        for device in devices:
            device_info: dict[str, Any] = {
                "name": device.name,
                "type": device.device_type.value,
                "available": device.is_available,
                "sufficient_memory": device.has_sufficient_memory,
                "selected": (
                    selected_device is not None
                    and device.device_type == selected_device.device_type
                ),
            }

            if device.memory is not None and device.memory.is_limited:
                device_info["memory"] = {
                    "total_mb": device.memory.total_mb,
                    "available_mb": device.memory.available_mb,
                    "usage_percent": device.memory.usage_percent,
                }

            if device.error_message:
                device_info["error"] = device.error_message

            device_list.append(device_info)

        summary = {
            "devices": device_list,
            "selected_device": (
                {
                    "name": selected_device.name,
                    "type": selected_device.device_type.value,
                }
                if selected_device
                else None
            ),
            "total_devices": len(devices),
            "available_devices": len([d for d in devices if d.is_available]),
        }

        return summary

    # Backward compatibility method for testing
    def _select_device(self) -> Device:
        """Select optimal device (backward compatibility for tests)."""
        if self._all_devices is None:
            raise DeviceError("Device detection not run. Call initialize() first.")

        return self._select_optimal_device(self._all_devices)


# Global device service instance
_device_service: DeviceService | None = None


def get_device_service() -> DeviceService:
    """Get the global device service instance."""
    global _device_service
    if _device_service is None:
        from culora.services import get_config_service

        config_service = get_config_service()

        # Load default config if not already loaded
        try:
            config = config_service.get_config()
        except Exception:
            config = config_service.load_config()

        _device_service = DeviceService(config)
        _device_service.initialize()

    return _device_service
