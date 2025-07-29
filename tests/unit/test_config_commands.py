"""Unit tests for config command implementations."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer

from culora.commands.config import (
    config_clear_command,
    config_get_command,
    config_init_command,
    config_set_command,
    config_show_command,
    config_validate_command,
)


class TestConfigInitCommand:
    """Tests for config init command."""

    @patch("culora.commands.config.ConfigManager")
    def test_init_command_success(self, mock_config_manager_class: MagicMock) -> None:
        """Test successful config init."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.save_defaults_to_file.return_value = Path("/test/config.toml")

        # Should complete without errors
        config_init_command()

        mock_manager.save_defaults_to_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_init_command_with_output_path(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config init with custom output path."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        output_path = Path("/custom/config.toml")
        mock_manager.save_defaults_to_file.return_value = output_path

        config_init_command(output_path)

        mock_manager.save_defaults_to_file.assert_called_once_with(output_path)

    @patch("culora.commands.config.ConfigManager")
    def test_init_command_error_handling(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config init error handling."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.save_defaults_to_file.side_effect = RuntimeError(
            "Permission denied"
        )

        with pytest.raises(typer.Exit):
            config_init_command()


class TestConfigShowCommand:
    """Tests for config show command."""

    @patch("culora.commands.config.ConfigManager")
    def test_show_command_success(self, mock_config_manager_class: MagicMock) -> None:
        """Test successful config show."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager

        # Mock config structure
        mock_config = Mock()
        mock_config.model_dump.return_value = {
            "quality": {"sharpness_threshold": 150.0, "enabled": True},
            "face": {"confidence_threshold": 0.5, "enabled": True},
            "deduplication": {"threshold": 2, "enabled": True},
        }
        mock_manager.config = mock_config

        mock_manager.config_file_exists.return_value = True
        mock_manager.get_config_file_path.return_value = Path(
            "/test/.culora/config.toml"
        )

        # Should complete without errors
        config_show_command()

        mock_manager.load_from_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_show_command_with_custom_config(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config show with custom config file."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager

        mock_config = Mock()
        mock_config.model_dump.return_value = {"quality": {"enabled": True}}
        mock_manager.config = mock_config
        mock_manager.config_file_exists.return_value = True

        config_file = Path("/custom/config.toml")
        config_show_command(config_file)

        mock_manager.load_from_file.assert_called_once_with(config_file)
        mock_manager.config_file_exists.assert_called_with(config_file)

    @patch("culora.commands.config.ConfigManager")
    def test_show_command_error_handling(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config show error handling."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.load_from_file.side_effect = RuntimeError("Config error")

        with pytest.raises(typer.Exit):
            config_show_command()


class TestConfigValidateCommand:
    """Tests for config validate command."""

    @patch("culora.commands.config.ConfigManager")
    @patch("culora.commands.config.AnalysisStage")
    def test_validate_command_success(
        self, mock_analysis_stage: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test successful config validation."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager

        # Mock a valid config
        mock_config = Mock()
        mock_config.scoring.quality_weight = 0.5
        mock_config.scoring.face_weight = 0.5
        mock_manager.config = mock_config

        # Mock AnalysisStage enum
        mock_stage1 = Mock()
        mock_stage1.value = "quality"
        mock_stage2 = Mock()
        mock_stage2.value = "face"
        mock_analysis_stage.__iter__.return_value = [mock_stage1, mock_stage2]

        # Mock config attributes
        mock_config.quality.enabled = True
        mock_config.face.enabled = True

        config_validate_command()

        mock_manager.load_from_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    @patch("culora.commands.config.AnalysisStage")
    def test_validate_command_weight_warning(
        self, mock_analysis_stage: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config validation with unbalanced weights."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager

        # Mock config with unbalanced weights
        mock_config = Mock()
        mock_config.scoring.quality_weight = 0.7
        mock_config.scoring.face_weight = 0.2  # Sum = 0.9, not 1.0
        mock_manager.config = mock_config

        # Mock empty AnalysisStage iteration
        mock_analysis_stage.__iter__.return_value = []

        config_validate_command()

        mock_manager.load_from_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_validate_command_error_handling(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config validate error handling."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.load_from_file.side_effect = ValueError("Invalid config")

        with pytest.raises(typer.Exit):
            config_validate_command()


class TestConfigGetCommand:
    """Tests for config get command."""

    @patch("culora.commands.config.ConfigManager")
    def test_get_command_success(self, mock_config_manager_class: MagicMock) -> None:
        """Test successful config get."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_value.return_value = 150.0

        config_get_command("quality.sharpness_threshold")

        mock_manager.load_from_file.assert_called_once_with(None)
        mock_manager.get_config_value.assert_called_once_with(
            "quality.sharpness_threshold"
        )

    @patch("culora.commands.config.ConfigManager")
    def test_get_command_with_custom_config(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config get with custom config file."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_value.return_value = 0.5

        config_file = Path("/custom/config.toml")
        config_get_command("face.confidence_threshold", config_file)

        mock_manager.load_from_file.assert_called_once_with(config_file)

    @patch("culora.commands.config.ConfigManager")
    def test_get_command_value_error(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config get with invalid key."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_value.side_effect = ValueError("Invalid key")

        with pytest.raises(typer.Exit):
            config_get_command("invalid.key")

    @patch("culora.commands.config.ConfigManager")
    def test_get_command_general_error(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config get with general error."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.load_from_file.side_effect = RuntimeError("Config error")

        with pytest.raises(typer.Exit):
            config_get_command("quality.sharpness_threshold")


class TestConfigSetCommand:
    """Tests for config set command."""

    @patch("culora.commands.config.ConfigManager")
    def test_set_command_success(self, mock_config_manager_class: MagicMock) -> None:
        """Test successful config set."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_value.return_value = 200.0
        mock_manager.save_to_file.return_value = Path("/test/config.toml")

        config_set_command("quality.sharpness_threshold", "200.0")

        mock_manager.load_from_file.assert_called_once_with(None)
        mock_manager.set_config_value.assert_called_once_with(
            "quality.sharpness_threshold", "200.0"
        )
        mock_manager.save_to_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_set_command_scoring_weight_validation(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config set with scoring weight validation."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager

        # Mock config with unbalanced weights after setting
        mock_config = Mock()
        mock_config.scoring.quality_weight = 0.7
        mock_config.scoring.face_weight = 0.2  # Total = 0.9
        mock_manager.config = mock_config
        mock_manager.get_config_value.return_value = 0.7
        mock_manager.save_to_file.return_value = Path("/test/config.toml")

        config_set_command("scoring.quality_weight", "0.7")

        mock_manager.set_config_value.assert_called_once_with(
            "scoring.quality_weight", "0.7"
        )

    @patch("culora.commands.config.ConfigManager")
    def test_set_command_value_error(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config set with invalid value."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.set_config_value.side_effect = ValueError("Invalid value")

        with pytest.raises(typer.Exit):
            config_set_command("quality.sharpness_threshold", "invalid")

    @patch("culora.commands.config.ConfigManager")
    def test_set_command_general_error(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config set with general error."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.load_from_file.side_effect = RuntimeError("Config error")

        with pytest.raises(typer.Exit):
            config_set_command("quality.sharpness_threshold", "200.0")


class TestConfigClearCommand:
    """Tests for config clear command."""

    @patch("culora.commands.config.ConfigManager")
    def test_clear_command_success_with_confirm(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test successful config clear with confirmation."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_file_path.return_value = Path("/test/config.toml")
        mock_manager.config_file_exists.return_value = True

        config_clear_command(confirm=True)

        mock_manager.delete_config_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_clear_command_no_file_exists(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config clear when no file exists."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.config_file_exists.return_value = False

        config_clear_command()

        # Should not attempt to delete
        mock_manager.delete_config_file.assert_not_called()

    @patch("culora.commands.config.ConfigManager")
    @patch("typer.confirm")
    def test_clear_command_user_cancels(
        self, mock_confirm: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config clear when user cancels."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.config_file_exists.return_value = True
        mock_confirm.return_value = False  # User says no

        config_clear_command()

        # Should not delete the file
        mock_manager.delete_config_file.assert_not_called()

    @patch("culora.commands.config.ConfigManager")
    @patch("typer.confirm")
    def test_clear_command_user_confirms(
        self, mock_confirm: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config clear when user confirms."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.get_config_file_path.return_value = Path("/test/config.toml")
        mock_manager.config_file_exists.return_value = True
        mock_confirm.return_value = True  # User says yes

        config_clear_command()

        mock_manager.delete_config_file.assert_called_once_with(None)

    @patch("culora.commands.config.ConfigManager")
    def test_clear_command_error_handling(
        self, mock_config_manager_class: MagicMock
    ) -> None:
        """Test config clear error handling."""
        mock_manager = Mock()
        mock_config_manager_class.get_instance.return_value = mock_manager
        mock_manager.config_file_exists.side_effect = RuntimeError("Permission error")

        with pytest.raises(typer.Exit):
            config_clear_command()
