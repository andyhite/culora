"""Tests for DeviceConfig model."""

import pytest
from pydantic import ValidationError

from culora.domain.models.config.device import DeviceConfig
from culora.domain.models.device import DeviceType


class TestDeviceConfig:
    """Test cases for DeviceConfig model."""

    def test_device_config_default_values(self) -> None:
        """Test DeviceConfig default initialization."""
        config = DeviceConfig()
        assert config.preferred_device == DeviceType.CPU

    def test_device_config_with_cuda(self) -> None:
        """Test DeviceConfig with CUDA device."""
        config = DeviceConfig(preferred_device=DeviceType.CUDA)
        assert config.preferred_device == DeviceType.CUDA

    def test_device_config_with_mps(self) -> None:
        """Test DeviceConfig with MPS device."""
        config = DeviceConfig(preferred_device=DeviceType.MPS)
        assert config.preferred_device == DeviceType.MPS

    def test_device_config_with_cpu(self) -> None:
        """Test DeviceConfig with CPU device."""
        config = DeviceConfig(preferred_device=DeviceType.CPU)
        assert config.preferred_device == DeviceType.CPU

    def test_device_config_from_dict(self) -> None:
        """Test DeviceConfig creation from dictionary."""
        config = DeviceConfig(preferred_device=DeviceType.CUDA)
        assert config.preferred_device == DeviceType.CUDA

    def test_device_config_model_dump(self) -> None:
        """Test DeviceConfig serialization."""
        config = DeviceConfig(preferred_device=DeviceType.MPS)
        dumped = config.model_dump()
        assert dumped == {"preferred_device": "mps"}

    def test_device_config_model_dump_json(self) -> None:
        """Test DeviceConfig JSON serialization."""
        config = DeviceConfig(preferred_device=DeviceType.CUDA)
        json_str = config.model_dump_json()
        assert '"preferred_device":"cuda"' in json_str.replace(" ", "")

    def test_device_config_invalid_device_type(self) -> None:
        """Test DeviceConfig with invalid device type."""
        with pytest.raises(ValidationError) as exc_info:
            DeviceConfig(preferred_device="invalid_device")  # type: ignore[arg-type]

        error = exc_info.value.errors()[0]
        assert error["type"] == "enum"
        assert "preferred_device" in str(error["loc"])

    def test_device_config_field_description(self) -> None:
        """Test that field has proper description."""
        schema = DeviceConfig.model_json_schema()
        field_info = schema["properties"]["preferred_device"]
        assert (
            field_info["description"] == "Preferred device type for AI model execution"
        )

    def test_device_config_equality(self) -> None:
        """Test DeviceConfig equality comparison."""
        config1 = DeviceConfig(preferred_device=DeviceType.CUDA)
        config2 = DeviceConfig(preferred_device=DeviceType.CUDA)
        config3 = DeviceConfig(preferred_device=DeviceType.MPS)

        assert config1 == config2
        assert config1 != config3

    def test_device_config_repr(self) -> None:
        """Test DeviceConfig string representation."""
        config = DeviceConfig(preferred_device=DeviceType.CUDA)
        repr_str = repr(config)
        assert "DeviceConfig" in repr_str
        assert "preferred_device" in repr_str

    @pytest.mark.parametrize(
        "device_type",
        [
            DeviceType.CUDA,
            DeviceType.MPS,
            DeviceType.CPU,
        ],
    )
    def test_device_config_all_device_types(self, device_type: DeviceType) -> None:
        """Parametrized test for all device types."""
        config = DeviceConfig(preferred_device=device_type)
        assert config.preferred_device == device_type

    @pytest.mark.parametrize(
        "device_string,expected_enum",
        [
            ("cuda", DeviceType.CUDA),
            ("mps", DeviceType.MPS),
            ("cpu", DeviceType.CPU),
        ],
    )
    def test_device_config_string_to_enum(
        self, device_string: str, expected_enum: DeviceType
    ) -> None:
        """Test DeviceConfig with string values that get converted to enums."""
        config = DeviceConfig(preferred_device=device_string)  # type: ignore[arg-type]
        assert config.preferred_device == expected_enum
