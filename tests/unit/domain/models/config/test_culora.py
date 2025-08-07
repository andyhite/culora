"""Tests for CuLoRAConfig model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from culora.domain.enums.device_types import DeviceType
from culora.domain.models.config import CuLoRAConfig, DeviceConfig
from culora.utils.app_dir import get_models_dir

from .....helpers import AssertionHelpers, ConfigBuilder


class TestCuLoRAConfig:
    """Test cases for CuLoRAConfig model."""

    def test_culora_config_default_values(self) -> None:
        """Test CuLoRAConfig default initialization."""
        config = CuLoRAConfig()
        assert isinstance(config.device, DeviceConfig)
        assert config.device.preferred_device == DeviceType.CPU

    def test_culora_config_with_custom_values(self) -> None:
        """Test CuLoRAConfig with custom values."""
        config = ConfigBuilder().with_device(DeviceType.CUDA).build()

        assert config.device.preferred_device == DeviceType.CUDA

    def test_culora_config_from_dict(self) -> None:
        """Test CuLoRAConfig creation from dictionary."""
        config_dict = {
            "device": {"preferred_device": "mps"},
        }
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.MPS

    def test_culora_config_from_dict_partial(self) -> None:
        """Test CuLoRAConfig creation from partial dictionary."""
        config_dict = {"device": {"preferred_device": "cuda"}}
        config = CuLoRAConfig.from_dict(config_dict)

        assert config.device.preferred_device == DeviceType.CUDA

    def test_culora_config_from_dict_empty(self) -> None:
        """Test CuLoRAConfig creation from empty dictionary."""
        config = CuLoRAConfig.from_dict({})

        assert config.device.preferred_device == DeviceType.CPU

    def test_culora_config_model_dump(self) -> None:
        """Test CuLoRAConfig serialization."""
        config = ConfigBuilder().with_device(DeviceType.CUDA).build()

        dumped = config.model_dump(mode="json")
        expected = {
            "device": {"preferred_device": "cuda"},
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
                "model_cache_dir": str(get_models_dir()),
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
                "model_cache_dir": str(get_models_dir() / "composition"),
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
            "clip": {
                "model_name": "openai/clip-vit-base-patch32",
                "model_cache_dir": str(get_models_dir() / "clip"),
                "device_preference": "auto",
                "normalize_embeddings": True,
                "embedding_precision": "float32",
                "enable_embedding_cache": True,
                "cache_compression": True,
                "similarity_metric": "cosine",
                "similarity_threshold": 0.8,
                "clustering_method": "kmeans",
                "max_clusters": 20,
                "min_cluster_size": 2,
                "enable_auto_clustering": True,
                "batch_size": 8,
                "max_image_size": [224, 224],
                "num_workers": 2,
                "memory_limit_mb": 2048,
                "diversity_weight": 0.7,
                "quality_weight": 0.3,
                "min_diversity_score": 0.3,
                "enable_similarity_analysis": True,
                "enable_clustering_analysis": True,
                "enable_diversity_analysis": True,
                "max_similarity_pairs": 10,
                "export_embeddings": False,
                "export_similarity_matrix": False,
                "export_clusters": True,
            },
            "pose": {
                "model_complexity": 1,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "enable_segmentation": False,
                "smooth_landmarks": True,
                "smooth_segmentation": True,
                "max_image_size": [1024, 1024],
                "batch_size": 4,
                "enable_pose_cache": True,
                "cache_compression": True,
                "feature_vector_dim": 66,
                "key_landmarks_only": True,
                "normalize_coordinates": True,
                "include_visibility": True,
                "min_pose_score": 0.3,
                "min_visible_landmarks": 20,
                "min_landmark_confidence": 0.3,
                "enable_pose_classification": True,
                "classification_confidence_threshold": 0.6,
                "max_clusters": 15,
                "min_cluster_size": 2,
                "enable_auto_clustering": True,
                "diversity_weight": 0.6,
                "quality_weight": 0.4,
                "min_diversity_score": 0.4,
                "enable_similarity_analysis": True,
                "enable_clustering_analysis": True,
                "enable_diversity_analysis": True,
                "max_similarity_pairs": 10,
                "export_landmarks": False,
                "export_pose_vectors": False,
                "export_pose_visualization": True,
                "export_clusters": True,
            },
            "duplicate": {
                "hash_algorithm": "perceptual",
                "similarity_threshold": 10,
                "group_threshold": 5,
                "removal_strategy": "highest_quality",
                "enable_exact_matching": True,
                "enable_near_matching": True,
                "max_group_size": 50,
                "progress_reporting": True,
            },
        }
        assert dumped == expected

    def test_culora_config_model_dump_json(self) -> None:
        """Test CuLoRAConfig JSON serialization."""
        config = ConfigBuilder().with_device(DeviceType.MPS).build()

        json_str = config.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["device"]["preferred_device"] == "mps"

        # Check formatting (should be indented)
        assert "\n" in json_str

    def test_culora_config_get_section(self) -> None:
        """Test CuLoRAConfig get_section method."""
        config = ConfigBuilder().with_device(DeviceType.CUDA).build()

        device_section = config.get_section("device")

        assert isinstance(device_section, DeviceConfig)
        assert device_section.preferred_device == DeviceType.CUDA

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
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "device" in str(errors[0]["loc"])
        assert "preferred_device" in str(errors[0]["loc"])

    def test_culora_config_equality(self) -> None:
        """Test CuLoRAConfig equality comparison."""
        config1 = ConfigBuilder().with_device(DeviceType.CUDA).build()
        config2 = ConfigBuilder().with_device(DeviceType.CUDA).build()
        config3 = ConfigBuilder().with_device(DeviceType.MPS).build()

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
        config = ConfigBuilder().with_device(DeviceType.CUDA).build()

        # When serialized, should use enum values, not enum names
        dumped = config.model_dump()
        assert dumped["device"]["preferred_device"] == "cuda"  # value, not "CUDA"
