"""Tests for CuLoRAConfig model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from culora.domain.enums import LogLevel
from culora.domain.enums.device_types import DeviceType
from culora.domain.models.config import CuLoRAConfig, DeviceConfig, LoggingConfig

from .....helpers import AssertionHelpers, ConfigBuilder


class TestCuLoRAConfig:
    """Test cases for CuLoRAConfig model."""

    def test_culora_config_default_values(self) -> None:
        """Test CuLoRAConfig default initialization."""
        config = CuLoRAConfig()
        assert isinstance(config.device, DeviceConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.device.preferred_device == DeviceType.CPU
        assert config.logging.log_level == LogLevel.INFO

    def test_culora_config_with_custom_values(self) -> None:
        """Test CuLoRAConfig with custom values."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.DEBUG

    def test_culora_config_from_dict(self) -> None:
        """Test CuLoRAConfig creation from dictionary."""
        config_dict = {
            "device": {"preferred_device": "mps"},
            "logging": {"log_level": "warning"},
        }
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.MPS
        assert config.logging.log_level == LogLevel.WARNING

    def test_culora_config_from_dict_partial(self) -> None:
        """Test CuLoRAConfig creation from partial dictionary."""
        config_dict = {"device": {"preferred_device": "cuda"}}
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.CUDA
        assert config.logging.log_level == LogLevel.INFO  # Default

    def test_culora_config_from_dict_empty(self) -> None:
        """Test CuLoRAConfig creation from empty dictionary."""
        config = CuLoRAConfig.from_dict({})

        assert config.device.preferred_device == DeviceType.CPU
        assert config.logging.log_level == LogLevel.INFO

    def test_culora_config_model_dump(self) -> None:
        """Test CuLoRAConfig serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.ERROR)
            .build()
        )

        dumped = config.model_dump(mode="json")
        expected = {
            "device": {"preferred_device": "cuda"},
            "logging": {"log_level": "error"},
            "images": {
                "supported_formats": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".tiff",
                    ".tif",
                ],
                "max_batch_size": 32,
                "max_image_size": [4096, 4096],  # tuple becomes list in JSON mode
                "max_file_size": 52428800,
                "recursive_scan": True,
                "max_scan_depth": 10,
                "skip_hidden_files": True,
                "progress_update_interval": 10,
            },
            "faces": {
                "model_name": "buffalo_l",
                "model_cache_dir": str(
                    Path.home()
                    / "Library"
                    / "Application Support"
                    / "culora"
                    / "models"
                ),
                "confidence_threshold": 0.5,
                "max_faces_per_image": 10,
                "device_preference": "auto",
                "batch_size": 8,
                "extract_embeddings": True,
                "extract_landmarks": True,
                "extract_attributes": False,
                "embedding_size": 512,
                "normalize_embeddings": True,
                "enable_model_caching": True,
                "memory_optimization": True,
                "reference_similarity_threshold": 0.7,
                "reference_matching_method": "average",
                "use_reference_fallback": True,
            },
            "quality": {
                "sharpness_kernel_size": 3,
                "optimal_brightness_range": [0.3, 0.7],
                "high_contrast_threshold": 0.4,
                "min_saturation": 0.1,
                "max_saturation": 0.8,
                "noise_threshold": 50.0,
                "sharpness_weight": 0.35,
                "brightness_weight": 0.2,
                "contrast_weight": 0.25,
                "color_weight": 0.15,
                "noise_weight": 0.05,
                "min_quality_score": 0.3,
                "enable_brisque": True,
                "brisque_weight": 0.3,
                "brisque_lower_better": True,
                "brisque_score_range": [0.0, 100.0],
                "resize_for_analysis": True,
                "max_analysis_size": [1024, 1024],
                "enable_face_quality_bonus": True,
                "face_quality_bonus_weight": 0.1,
                "enable_reference_match_bonus": True,
                "reference_match_bonus_weight": 0.15,
            },
            "composition": {
                "model_name": "vikhyatk/moondream2",
                "model_cache_dir": str(
                    Path.home()
                    / "Library"
                    / "Application Support"
                    / "culora"
                    / "composition_models"
                ),
                "device_preference": "auto",
                "enable_shot_type_analysis": True,
                "enable_scene_analysis": True,
                "enable_lighting_analysis": True,
                "enable_background_analysis": True,
                "enable_expression_analysis": True,
                "enable_angle_analysis": True,
                "min_confidence_threshold": 0.7,
                "enable_confidence_scoring": True,
                "batch_size": 4,
                "max_image_size": [1024, 1024],
                "enable_model_caching": True,
                "memory_optimization": True,
                "max_retries": 3,
                "response_timeout": 30.0,
                "enable_fallback_parsing": True,
                "use_structured_prompts": True,
                "prompt_temperature": 0.1,
                "enable_example_prompts": True,
            },
        }
        assert dumped == expected

    def test_culora_config_model_dump_json(self) -> None:
        """Test CuLoRAConfig JSON serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.MPS)
            .with_log_level(LogLevel.WARNING)
            .build()
        )

        json_str = config.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["device"]["preferred_device"] == "mps"
        assert parsed["logging"]["log_level"] == "warning"

        # Check formatting (should be indented)
        assert "\n" in json_str

    def test_culora_config_get_section(self) -> None:
        """Test CuLoRAConfig get_section method."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        device_section = config.get_section("device")
        logging_section = config.get_section("logging")

        assert isinstance(device_section, DeviceConfig)
        assert isinstance(logging_section, LoggingConfig)
        assert device_section.preferred_device == DeviceType.CUDA
        assert logging_section.log_level == LogLevel.DEBUG

    def test_culora_config_get_section_invalid(self) -> None:
        """Test CuLoRAConfig get_section with invalid section name."""
        config = CuLoRAConfig()

        with pytest.raises(AttributeError):
            config.get_section("invalid_section")

    def test_culora_config_validate_assignment(self) -> None:
        """Test that CuLoRAConfig validates assignments."""
        config = CuLoRAConfig()

        # Valid assignment
        config.device = DeviceConfig(preferred_device=DeviceType.MPS)
        assert config.device.preferred_device == DeviceType.MPS

        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            config.device = "invalid"  # type: ignore[assignment]

    def test_culora_config_nested_validation(self) -> None:
        """Test CuLoRAConfig nested validation."""
        with pytest.raises(ValidationError) as exc_info:
            CuLoRAConfig(
                device={"preferred_device": "invalid_device"},  # type: ignore[arg-type]
                logging={"log_level": "debug"},  # type: ignore[arg-type]
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "device" in str(errors[0]["loc"])
        assert "preferred_device" in str(errors[0]["loc"])

    def test_culora_config_equality(self) -> None:
        """Test CuLoRAConfig equality comparison."""
        config1 = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )
        config2 = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )
        config3 = (
            ConfigBuilder()
            .with_device(DeviceType.MPS)
            .with_log_level(LogLevel.DEBUG)
            .build()
        )

        AssertionHelpers.assert_config_equal(config1, config2)
        assert config1 != config3

    def test_culora_config_repr(self) -> None:
        """Test CuLoRAConfig string representation."""
        config = CuLoRAConfig()
        repr_str = repr(config)
        assert "CuLoRAConfig" in repr_str

    def test_culora_config_json_encoders(self) -> None:
        """Test that Path objects are properly encoded."""
        # Create a config that would use Path encoding
        config = CuLoRAConfig()

        # The json_encoders should handle Path objects
        # This is tested indirectly through the ConfigDict
        config_dict = config.model_config
        if "json_encoders" in config_dict and config_dict["json_encoders"] is not None:
            assert Path in config_dict["json_encoders"]
            assert config_dict["json_encoders"][Path] is str

    def test_culora_config_use_enum_values(self) -> None:
        """Test that enum values are used in serialization."""
        config = (
            ConfigBuilder()
            .with_device(DeviceType.CUDA)
            .with_log_level(LogLevel.ERROR)
            .build()
        )

        # When serialized, should use enum values, not enum names
        dumped = config.model_dump()
        assert dumped["device"]["preferred_device"] == "cuda"  # value, not "CUDA"
        assert dumped["logging"]["log_level"] == "error"  # value, not "ERROR"

    @pytest.mark.parametrize(
        "device_type,log_level",
        [
            (DeviceType.CUDA, LogLevel.DEBUG),
            (DeviceType.MPS, LogLevel.INFO),
            (DeviceType.CPU, LogLevel.WARNING),
            (DeviceType.CUDA, LogLevel.ERROR),
            (DeviceType.MPS, LogLevel.CRITICAL),
        ],
    )
    def test_culora_config_combinations(
        self, device_type: DeviceType, log_level: LogLevel
    ) -> None:
        """Parametrized test for various device/log level combinations."""
        config = (
            ConfigBuilder().with_device(device_type).with_log_level(log_level).build()
        )

        assert config.device.preferred_device == device_type
        assert config.logging.log_level == log_level
