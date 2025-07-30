"""Tests for CLI validators."""

import json

import pytest
import typer

from culora.cli.validation.validators import (
    convert_config_value,
    validate_config_file,
    validate_config_key,
    validate_output_file,
)
from tests.helpers import TempFileHelper


class TestValidateConfigFile:
    """Test config file validation."""

    def test_validate_config_file_none(self) -> None:
        """Test validation with None input."""
        result = validate_config_file(None)
        assert result is None

    def test_validate_config_file_yaml_valid(self) -> None:
        """Test validation with valid YAML file."""
        with TempFileHelper.create_temp_file(".yaml") as temp_path:
            temp_path.write_text("device:\n  preferred_device: cpu\n")

            result = validate_config_file(str(temp_path))
            assert result == temp_path

    def test_validate_config_file_json_valid(self) -> None:
        """Test validation with valid JSON file."""
        with TempFileHelper.create_temp_file(".json") as temp_path:
            config = {"device": {"preferred_device": "cpu"}}
            temp_path.write_text(json.dumps(config))

            result = validate_config_file(str(temp_path))
            assert result == temp_path

    def test_validate_config_file_not_exists(self) -> None:
        """Test validation with non-existent file."""
        with pytest.raises(
            typer.BadParameter, match="Configuration file does not exist"
        ):
            validate_config_file("/nonexistent/path.yaml")

    def test_validate_config_file_is_directory(self) -> None:
        """Test validation with directory path."""
        with (
            TempFileHelper.create_temp_dir() as temp_dir,
            pytest.raises(typer.BadParameter, match="Configuration path is not a file"),
        ):
            validate_config_file(str(temp_dir))

    def test_validate_config_file_invalid_extension(self) -> None:
        """Test validation with invalid file extension."""
        with TempFileHelper.create_temp_file(".txt") as temp_path:
            temp_path.write_text("some content")

            with pytest.raises(
                typer.BadParameter,
                match="Configuration file must be .yaml/.yml or .json",
            ):
                validate_config_file(str(temp_path))

    def test_validate_config_file_unreadable(self) -> None:
        """Test validation with unreadable file."""
        with TempFileHelper.create_temp_file(".yaml") as temp_path:
            temp_path.write_text("test")
            temp_path.chmod(0o000)  # Make unreadable

            try:
                with pytest.raises(
                    typer.BadParameter, match="Cannot read configuration file"
                ):
                    validate_config_file(str(temp_path))
            finally:
                temp_path.chmod(0o644)  # Restore permissions for cleanup


class TestValidateOutputFile:
    """Test output file validation."""

    def test_validate_output_file_new_file(self) -> None:
        """Test validation with new file path."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            output_path = temp_dir / "output.yaml"

            result = validate_output_file(str(output_path))
            assert result == output_path

    def test_validate_output_file_existing_file(self) -> None:
        """Test validation with existing writable file."""
        with TempFileHelper.create_temp_file(".yaml") as temp_path:
            temp_path.write_text("existing content")

            result = validate_output_file(str(temp_path))
            assert result == temp_path

    def test_validate_output_file_create_parent_dir(self) -> None:
        """Test validation that creates parent directories."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            output_path = temp_dir / "nested" / "deep" / "output.yaml"

            result = validate_output_file(str(output_path))
            assert result == output_path
            assert output_path.parent.exists()

    def test_validate_output_file_existing_directory(self) -> None:
        """Test validation with existing directory at output path."""
        with TempFileHelper.create_temp_dir() as temp_dir:
            existing_dir = temp_dir / "existing"
            existing_dir.mkdir()

            with pytest.raises(
                typer.BadParameter, match="Output path exists but is not a file"
            ):
                validate_output_file(str(existing_dir))


class TestValidateConfigKey:
    """Test config key validation."""

    def test_validate_config_key_valid_device(self) -> None:
        """Test validation with valid device key."""
        result = validate_config_key("device.preferred_device")
        assert result == "device.preferred_device"

    def test_validate_config_key_valid_logging(self) -> None:
        """Test validation with valid logging key."""
        result = validate_config_key("logging.log_level")
        assert result == "logging.log_level"

    def test_validate_config_key_valid_simple(self) -> None:
        """Test validation with simple key (no dots)."""
        result = validate_config_key("simple_key")
        assert result == "simple_key"

    def test_validate_config_key_empty(self) -> None:
        """Test validation with empty key."""
        with pytest.raises(
            typer.BadParameter, match="Configuration key cannot be empty"
        ):
            validate_config_key("")

    def test_validate_config_key_invalid_characters(self) -> None:
        """Test validation with invalid characters."""
        with pytest.raises(
            typer.BadParameter, match="Invalid configuration key format"
        ):
            validate_config_key("device.invalid-key!")

    def test_validate_config_key_invalid_section(self) -> None:
        """Test validation with invalid section."""
        with pytest.raises(typer.BadParameter, match="Invalid configuration section"):
            validate_config_key("invalid_section.some_key")


class TestConvertConfigValue:
    """Test config value conversion."""

    def test_convert_config_value_boolean_true(self) -> None:
        """Test conversion of boolean true values."""
        assert convert_config_value("true") is True
        assert convert_config_value("True") is True
        assert convert_config_value("1") is True
        assert convert_config_value("yes") is True
        assert convert_config_value("on") is True

    def test_convert_config_value_boolean_false(self) -> None:
        """Test conversion of boolean false values."""
        assert convert_config_value("false") is False
        assert convert_config_value("False") is False
        assert convert_config_value("0") is False
        assert convert_config_value("no") is False
        assert convert_config_value("off") is False

    def test_convert_config_value_integer(self) -> None:
        """Test conversion of integer values."""
        assert convert_config_value("42") == 42
        assert convert_config_value("-10") == -10
        assert convert_config_value("0") == 0

    def test_convert_config_value_float(self) -> None:
        """Test conversion of float values."""
        assert convert_config_value("3.14") == 3.14
        assert convert_config_value("-2.5") == -2.5
        assert convert_config_value("0.0") == 0.0

    def test_convert_config_value_string(self) -> None:
        """Test conversion of string values."""
        assert convert_config_value("hello") == "hello"
        assert convert_config_value("device_name") == "device_name"
        assert convert_config_value("3.14.159") == "3.14.159"  # Not a valid float
