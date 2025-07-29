"""Unit tests for app data utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from culora.utils.app_data import get_app_dir, get_models_dir


class TestAppData:
    """Tests for app data utilities functionality."""

    @patch("typer.get_app_dir")
    def test_get_app_dir(self, mock_get_app_dir: MagicMock) -> None:
        """Test get_app_dir function."""
        mock_get_app_dir.return_value = "/test/.culora"

        result = get_app_dir()

        mock_get_app_dir.assert_called_once_with("culora", force_posix=True)
        assert result == Path("/test/.culora")

    @patch("culora.utils.app_data.get_app_dir")
    def test_get_models_dir_creates_directory(
        self, mock_get_app_dir: MagicMock
    ) -> None:
        """Test get_models_dir creates the models directory."""
        mock_app_dir = MagicMock()
        mock_models_dir = MagicMock()
        mock_app_dir.__truediv__.return_value = mock_models_dir
        mock_get_app_dir.return_value = mock_app_dir

        result = get_models_dir()

        mock_app_dir.__truediv__.assert_called_once_with("models")
        mock_models_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result is mock_models_dir
