"""Tests for app data utilities."""

from pathlib import Path
from unittest.mock import patch

from culora.utils.app_data import get_app_data_dir, get_models_dir


def test_get_app_data_dir():
    """Test getting the app data directory."""
    with patch("typer.get_app_dir") as mock_get_app_dir:
        mock_get_app_dir.return_value = "/home/user/.local/share/culora"

        result = get_app_data_dir()

        mock_get_app_dir.assert_called_once_with("culora")
        assert result == Path("/home/user/.local/share/culora")


def test_get_models_dir():
    """Test getting the models directory."""
    with patch("culora.utils.app_data.get_app_data_dir") as mock_get_app_data_dir:
        mock_app_dir = Path("/tmp/test_app")
        mock_get_app_data_dir.return_value = mock_app_dir

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = get_models_dir()

            expected = mock_app_dir / "models"
            assert result == expected
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
