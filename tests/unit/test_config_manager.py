"""Unit tests for ConfigManager."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from culora.config import AnalysisStage, CuLoRAConfig
from culora.managers.config_manager import ConfigManager


class TestConfigManager:
    """Tests for ConfigManager singleton and core functionality."""

    def setup_method(self) -> None:
        """Reset singleton for each test."""
        ConfigManager._instance = None

    def test_singleton_behavior(self) -> None:
        """Test ConfigManager singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_get_instance(self) -> None:
        """Test get_instance class method."""
        manager1 = ConfigManager.get_instance()
        manager2 = ConfigManager.get_instance()
        assert manager1 is manager2
        assert isinstance(manager1.config, CuLoRAConfig)

    def test_get_config_by_stage(self) -> None:
        """Test getting config by AnalysisStage."""
        manager = ConfigManager()

        quality_config = manager.get_config(AnalysisStage.QUALITY)
        assert quality_config is manager.config.quality

        face_config = manager.get_config(AnalysisStage.FACE)
        assert face_config is manager.config.face

        dedup_config = manager.get_config(AnalysisStage.DEDUPLICATION)
        assert dedup_config is manager.config.deduplication

    def test_get_config_by_string(self) -> None:
        """Test getting config by string key."""
        manager = ConfigManager()

        quality_config = manager.get_config("quality")
        assert quality_config is manager.config.quality

    def test_get_config_invalid_key(self) -> None:
        """Test getting config with invalid key."""
        manager = ConfigManager()

        with pytest.raises(
            ValueError, match="Configuration for key 'invalid' not found"
        ):
            manager.get_config("invalid")

    @patch("culora.managers.config_manager.get_app_dir")
    def test_get_config_file_path_default(self, mock_get_app_dir: MagicMock) -> None:
        """Test getting default config file path."""
        mock_get_app_dir.return_value = Path("/test/app")
        manager = ConfigManager()

        path = manager.get_config_file_path()
        expected = Path("/test/app/config.toml")
        assert path == expected

    def test_get_config_file_path_custom(self) -> None:
        """Test getting custom config file path."""
        manager = ConfigManager()
        custom_path = Path("/custom/config.toml")

        path = manager.get_config_file_path(custom_path)
        assert path == custom_path

    def test_config_file_exists_true(self) -> None:
        """Test config file exists check when file exists."""
        manager = ConfigManager()

        with patch.object(Path, "exists", return_value=True):
            assert manager.config_file_exists() is True

    def test_config_file_exists_false(self) -> None:
        """Test config file exists check when file doesn't exist."""
        manager = ConfigManager()

        with patch.object(Path, "exists", return_value=False):
            assert manager.config_file_exists() is False

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=b"[quality]\nsharpness_threshold = 200.0",
    )
    @patch("culora.managers.config_manager.tomllib.load")
    def test_load_from_file_success(
        self, mock_toml_load: MagicMock, mock_file: MagicMock
    ) -> None:
        """Test successful config loading from file."""
        mock_toml_load.return_value = {"quality": {"sharpness_threshold": 200.0}}

        manager = ConfigManager()
        with patch.object(Path, "exists", return_value=True):
            manager.load_from_file(Path("/test/config.toml"))

        mock_toml_load.assert_called_once()

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_from_file_not_found(self, mock_file: MagicMock) -> None:
        """Test loading from non-existent file (should not raise)."""
        manager = ConfigManager()

        with patch.object(Path, "exists", return_value=False):
            # Should not raise an exception
            manager.load_from_file(Path("/nonexistent/config.toml"))

    @patch("builtins.open", new_callable=mock_open, read_data=b"invalid toml content")
    @patch(
        "culora.managers.config_manager.tomllib.load",
        side_effect=ValueError("Invalid TOML"),
    )
    @patch("rich.console.Console")
    def test_load_from_file_invalid_toml(
        self,
        mock_console_class: MagicMock,
        mock_toml_load: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """Test loading invalid TOML file shows warning."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        manager = ConfigManager()
        with patch.object(Path, "exists", return_value=True):
            manager.load_from_file(Path("/test/invalid.toml"))

        # Should print warning messages
        assert mock_console.print.call_count == 2

    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_file_success(self, mock_file: MagicMock) -> None:
        """Test successful config saving to file."""
        manager = ConfigManager()

        # Create a mock parent Path and patch mkdir on it
        mock_parent = MagicMock()
        with patch.object(Path, "parent", mock_parent):
            result_path = manager.save_to_file(Path("/test/config.toml"))

            mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_file.assert_called_once_with(Path("/test/config.toml"), "w")
            assert result_path == Path("/test/config.toml")

    @patch("culora.managers.config_manager.get_app_dir")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_defaults_to_file(
        self, mock_file: MagicMock, mock_get_app_dir: MagicMock
    ) -> None:
        """Test saving defaults to file."""
        mock_get_app_dir.return_value = Path("/test/app")
        manager = ConfigManager()

        with patch.object(Path, "mkdir"):
            result_path = manager.save_defaults_to_file()

            expected_path = Path("/test/app/config.toml")
            assert result_path == expected_path

    def test_delete_config_file_exists(self) -> None:
        """Test deleting existing config file."""
        manager = ConfigManager()

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink") as mock_unlink,
        ):

            result = manager.delete_config_file(Path("/test/config.toml"))

            mock_unlink.assert_called_once()
            assert result is True
            # Should reset to defaults
            assert isinstance(manager.config, CuLoRAConfig)

    def test_delete_config_file_not_exists(self) -> None:
        """Test deleting non-existent config file."""
        manager = ConfigManager()

        with patch.object(Path, "exists", return_value=False):
            result = manager.delete_config_file(Path("/test/config.toml"))

            assert result is False

    def test_get_config_value_success(self) -> None:
        """Test getting config value with dot notation."""
        manager = ConfigManager()

        value = manager.get_config_value("quality.sharpness_threshold")
        assert value == manager.config.quality.sharpness_threshold

    def test_get_config_value_invalid_format(self) -> None:
        """Test getting config value with invalid key format."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Key must be in format 'section.field'"):
            manager.get_config_value("invalid_key")

    def test_get_config_value_invalid_section(self) -> None:
        """Test getting config value with invalid section."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Section 'invalid' not found"):
            manager.get_config_value("invalid.field")

    def test_get_config_value_invalid_field(self) -> None:
        """Test getting config value with invalid field."""
        manager = ConfigManager()

        with pytest.raises(
            ValueError, match="Field 'invalid' not found in section 'quality'"
        ):
            manager.get_config_value("quality.invalid")

    def test_set_config_value_float(self) -> None:
        """Test setting float config value."""
        manager = ConfigManager()
        original_value = manager.config.quality.sharpness_threshold

        manager.set_config_value("quality.sharpness_threshold", "200.5")
        assert manager.config.quality.sharpness_threshold == 200.5
        assert manager.config.quality.sharpness_threshold != original_value

    def test_set_config_value_int(self) -> None:
        """Test setting integer config value."""
        manager = ConfigManager()
        original_value = manager.config.quality.brightness_min

        manager.set_config_value("quality.brightness_min", "100")
        assert manager.config.quality.brightness_min == 100
        assert manager.config.quality.brightness_min != original_value

    def test_set_config_value_bool_true(self) -> None:
        """Test setting boolean config value to True."""
        manager = ConfigManager()

        manager.set_config_value("quality.enabled", "true")
        assert manager.config.quality.enabled is True

    def test_set_config_value_bool_false(self) -> None:
        """Test setting boolean config value to False."""
        manager = ConfigManager()

        manager.set_config_value("quality.enabled", "false")
        assert manager.config.quality.enabled is False

    def test_set_config_value_bool_variants(self) -> None:
        """Test setting boolean config value with different string variants."""
        manager = ConfigManager()

        # Test various true values
        for true_val in ["yes", "1", "on", "True", "YES"]:
            manager.set_config_value("quality.enabled", true_val)
            assert manager.config.quality.enabled is True

        # Test false values
        for false_val in ["no", "0", "off", "False", "NO"]:
            manager.set_config_value("quality.enabled", false_val)
            assert manager.config.quality.enabled is False

    def test_set_config_value_string(self) -> None:
        """Test setting string config value."""
        manager = ConfigManager()
        original_value = manager.config.face.model_repo

        manager.set_config_value("face.model_repo", "new_repo")
        assert manager.config.face.model_repo == "new_repo"
        assert manager.config.face.model_repo != original_value

    def test_set_config_value_invalid_format(self) -> None:
        """Test setting config value with invalid key format."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Key must be in format 'section.field'"):
            manager.set_config_value("invalid_key", "value")

    def test_set_config_value_invalid_section(self) -> None:
        """Test setting config value with invalid section."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Section 'invalid' not found"):
            manager.set_config_value("invalid.field", "value")

    def test_set_config_value_invalid_field(self) -> None:
        """Test setting config value with invalid field."""
        manager = ConfigManager()

        with pytest.raises(
            ValueError, match="Field 'invalid' not found in section 'quality'"
        ):
            manager.set_config_value("quality.invalid", "value")
