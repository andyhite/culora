"""Configuration fixtures for testing."""

from pathlib import Path
from typing import Any

import pytest

from culora.core.config import CuLoRAConfig, DeviceConfig, FaceAnalysisConfig
from culora.core.types import DeviceType


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """Provide a valid configuration dictionary."""
    return {
        "device": {
            "preferred_device": "cpu",
            "fallback_device": "cpu",
            "batch_size": 16,
            "auto_detect": True,
        },
        "face_analysis": {
            "detection_threshold": 0.6,
            "similarity_threshold": 0.7,
            "max_faces_per_image": 5,
            "min_face_size": 64,
        },
        "quality_assessment": {
            "enable_brisque": True,
            "enable_technical_metrics": True,
            "quality_weights": {
                "brisque": 0.4,
                "sharpness": 0.2,
                "brightness": 0.1,
                "contrast": 0.1,
                "face_quality": 0.2,
            },
        },
        "selection": {
            "target_count": 100,
            "quality_weight": 0.6,
            "diversity_weight": 0.4,
            "min_quality_score": 0.5,
        },
        "logging": {
            "log_level": "info",
            "enable_console_output": False,
        },
    }


@pytest.fixture
def invalid_config_dict() -> dict[str, Any]:
    """Provide an invalid configuration dictionary."""
    return {
        "device": {
            "preferred_device": "invalid_device",  # Invalid device type
            "batch_size": -1,  # Invalid batch size
        },
        "face_analysis": {
            "detection_threshold": 1.5,  # Out of range
            "similarity_threshold": -0.1,  # Out of range
        },
        "quality_assessment": {
            "quality_weights": {
                "brisque": 0.5,
                "sharpness": 0.8,  # Weights sum > 1.0
            },
        },
        "selection": {
            "quality_weight": 0.8,
            "diversity_weight": 0.4,  # Sum > 1.0
        },
    }


@pytest.fixture
def sample_config() -> CuLoRAConfig:
    """Provide a sample CuLoRA configuration instance."""
    return CuLoRAConfig(
        device=DeviceConfig(
            preferred_device=DeviceType.CPU,
            batch_size=8,
        ),
        face_analysis=FaceAnalysisConfig(
            detection_threshold=0.7,
            similarity_threshold=0.8,
        ),
    )


@pytest.fixture
def temp_config_file(tmp_path: Path, valid_config_dict: dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    import json

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f, indent=2)

    return config_file


@pytest.fixture
def temp_yaml_config_file(tmp_path: Path, valid_config_dict: dict[str, Any]) -> Path:
    """Create a temporary YAML configuration file."""
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_dict, f, default_flow_style=False)

    return config_file


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_reference_images(tmp_path: Path) -> list[Path]:
    """Create mock reference image files."""
    ref_images = []
    for i in range(3):
        ref_image = tmp_path / f"reference_{i}.jpg"
        ref_image.write_text("mock image data")  # Mock file content
        ref_images.append(ref_image)

    return ref_images
