"""Mock implementations for PyTorch/CUDA testing."""

from typing import Any
from unittest.mock import Mock, patch


class MockContext:
    """Context manager for common mock scenarios."""

    def __init__(self) -> None:
        self._patches: list[Any] = []
        self._mocks: dict[str, Mock] = {}

    def mock_torch_available(self, available: bool = True) -> "MockContext":
        """Mock torch availability."""
        if available:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.backends.mps.is_available.return_value = False
            self._patches.append(patch.dict("sys.modules", {"torch": mock_torch}))
        else:
            self._patches.append(
                patch(
                    "builtins.__import__",
                    side_effect=ImportError("No module named 'torch'"),
                )
            )
        return self

    def mock_cuda_devices(self, count: int = 1, memory_mb: int = 8192) -> "MockContext":
        """Mock CUDA devices."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = count > 0
        mock_torch.cuda.device_count.return_value = count

        # Mock device properties
        mock_props = Mock()
        mock_props.name = "GeForce RTX 3080"
        mock_props.total_memory = memory_mb * 1024 * 1024
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_reserved.return_value = (memory_mb // 4) * 1024 * 1024

        self._patches.append(patch.dict("sys.modules", {"torch": mock_torch}))
        return self

    def mock_structlog(self) -> "MockContext":
        """Mock structlog logger."""
        mock_logger = Mock()
        self._mocks["structlog_logger"] = mock_logger
        self._patches.append(patch("structlog.get_logger", return_value=mock_logger))
        return self

    def __enter__(self) -> "MockContext":
        """Enter context and start all patches."""
        for patch_obj in self._patches:
            patch_obj.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and stop all patches."""
        for patch_obj in self._patches:
            patch_obj.stop()

    def get_mock(self, name: str) -> Mock:
        """Get a mock by name."""
        return self._mocks[name]
