"""Shared test fixtures and configuration."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from culora.domain.models import CuLoRAConfig, DeviceConfig
from culora.domain.models.device import Device, DeviceType
from culora.domain.models.memory import Memory
from culora.services.config_service import ConfigService
from culora.services.device_service import DeviceService
from culora.services.memory_service import MemoryService

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> CuLoRAConfig:
    """Create a default CuLoRA configuration."""
    return CuLoRAConfig()


@pytest.fixture
def cuda_config() -> CuLoRAConfig:
    """Create a CuLoRA configuration with CUDA device preference."""
    return CuLoRAConfig(
        device=DeviceConfig(preferred_device=DeviceType.CUDA),
    )


@pytest.fixture
def mps_config() -> CuLoRAConfig:
    """Create a CuLoRA configuration with MPS device preference."""
    return CuLoRAConfig(
        device=DeviceConfig(preferred_device=DeviceType.MPS),
    )


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_config_service() -> Mock:
    """Create a mock ConfigService instance."""
    return Mock(spec=ConfigService)


# ============================================================================
# Device Fixtures
# ============================================================================


@pytest.fixture
def cpu_device() -> Device:
    """Create a CPU device instance."""
    return Device(device_type=DeviceType.CPU, name="CPU")


@pytest.fixture
def cuda_device() -> Device:
    """Create a CUDA device instance with memory."""
    memory = Memory(total_mb=8192, available_mb=4096)
    return Device(
        device_type=DeviceType.CUDA,
        name="CUDA:0",
        memory=memory,
        is_available=True,
    )


@pytest.fixture
def mps_device() -> Device:
    """Create an MPS device instance."""
    return Device(
        device_type=DeviceType.MPS,
        name="Apple Silicon MPS",
        is_available=True,
    )


@pytest.fixture
def unavailable_device() -> Device:
    """Create an unavailable device instance."""
    return Device(
        device_type=DeviceType.CUDA,
        name="CUDA:1",
        is_available=False,
        error_message="Device not found",
    )


@pytest.fixture
def device_list(
    cpu_device: Device, cuda_device: Device, mps_device: Device
) -> list[Device]:
    """Create a list of various devices."""
    return [cpu_device, cuda_device, mps_device]


# ============================================================================
# Memory Fixtures
# ============================================================================


@pytest.fixture
def unlimited_memory() -> Memory:
    """Create unlimited memory instance."""
    return Memory(total_mb=None, available_mb=None)


@pytest.fixture
def limited_memory() -> Memory:
    """Create limited memory instance."""
    return Memory(total_mb=8192, available_mb=4096)


@pytest.fixture
def low_memory() -> Memory:
    """Create low memory instance."""
    return Memory(total_mb=2048, available_mb=512)


# ============================================================================
# Service Fixtures
# ============================================================================


@pytest.fixture
def config_service() -> ConfigService:
    """Create a ConfigService instance."""
    return ConfigService()


@pytest.fixture
def device_service(default_config: CuLoRAConfig) -> DeviceService:
    """Create a DeviceService instance."""
    return DeviceService(default_config)


@pytest.fixture
def memory_service() -> MemoryService:
    """Create a MemoryService instance."""
    return MemoryService()


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def config_file(temp_dir: Path) -> Path:
    """Create a temporary configuration file."""
    config_path = temp_dir / "test_config.json"
    config_data = {
        "device": {"preferred_device": "cpu"},
    }

    import json

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_path


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Create sample configuration dictionary."""
    return {
        "device": {"preferred_device": "cuda"},
    }


@pytest.fixture
def invalid_config_dict() -> dict[str, Any]:
    """Create invalid configuration dictionary."""
    return {
        "device": {"preferred_device": "invalid_device"},
    }


@pytest.fixture
def environment_vars() -> dict[str, str]:
    """Create sample environment variables."""
    return {
        "CULORA_DEVICE_PREFERRED": "mps",
    }


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers and warning filters."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU hardware")
    config.addinivalue_line(
        "markers", "mps: marks tests that require Apple Silicon MPS"
    )
