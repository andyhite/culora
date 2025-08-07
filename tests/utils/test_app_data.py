"""Tests for app data utilities."""

from pathlib import Path
from unittest.mock import patch

from culora.utils.app_data import get_app_data_dir, get_cache_dir, get_cache_file_path


def test_get_app_data_dir():
    """Test getting the app data directory."""
    with patch("typer.get_app_dir") as mock_get_app_dir:
        mock_get_app_dir.return_value = "/home/user/.local/share/culora"

        result = get_app_data_dir()

        mock_get_app_dir.assert_called_once_with("culora")
        assert result == Path("/home/user/.local/share/culora")


def test_get_cache_dir():
    """Test getting the cache directory."""
    with patch("culora.utils.app_data.get_app_data_dir") as mock_get_app_data_dir:
        mock_app_dir = Path("/tmp/test_app")
        mock_get_app_data_dir.return_value = mock_app_dir

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = get_cache_dir()

            expected = mock_app_dir / "cache"
            assert result == expected
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_get_cache_file_path():
    """Test getting cache file path for a directory."""
    test_dir = Path("/home/user/images")

    with patch("culora.utils.app_data.get_cache_dir") as mock_get_cache_dir:
        mock_cache_dir = Path("/tmp/cache")
        mock_get_cache_dir.return_value = mock_cache_dir

        result = get_cache_file_path(test_dir)

        # Should return a path in the cache directory
        assert result.parent == mock_cache_dir
        assert result.suffix == ".json"

        # Filename should contain directory name and be consistent
        assert "images_" in result.name

        # Should be consistent for the same directory
        result2 = get_cache_file_path(test_dir)
        assert result == result2


def test_get_cache_file_path_different_dirs():
    """Test that different directories get different cache files."""
    dir1 = Path("/home/user/images1")
    dir2 = Path("/home/user/images2")

    with patch("culora.utils.app_data.get_cache_dir") as mock_get_cache_dir:
        mock_cache_dir = Path("/tmp/cache")
        mock_get_cache_dir.return_value = mock_cache_dir

        result1 = get_cache_file_path(dir1)
        result2 = get_cache_file_path(dir2)

        # Different directories should get different cache files
        assert result1 != result2
        assert result1.name != result2.name


def test_get_cache_file_path_root_directory():
    """Test cache file path for root directory."""
    root_dir = Path("/")

    with patch("culora.utils.app_data.get_cache_dir") as mock_get_cache_dir:
        mock_cache_dir = Path("/tmp/cache")
        mock_get_cache_dir.return_value = mock_cache_dir

        result = get_cache_file_path(root_dir)

        # Should handle root directory gracefully
        assert result.parent == mock_cache_dir
        assert "root_" in result.name
        assert result.suffix == ".json"
