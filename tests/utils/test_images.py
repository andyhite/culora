"""Tests for image utilities."""

import tempfile
from pathlib import Path

from culora.managers.image_manager import ImageManager


def test_is_image_file():
    """Test image file detection."""
    manager = ImageManager.get_instance()
    assert manager.is_image_file(Path("test.jpg"))
    assert manager.is_image_file(Path("test.JPG"))
    assert manager.is_image_file(Path("test.png"))
    assert manager.is_image_file(Path("test.jpeg"))
    assert manager.is_image_file(Path("test.bmp"))
    assert manager.is_image_file(Path("test.tiff"))
    assert manager.is_image_file(Path("test.tif"))
    assert manager.is_image_file(Path("test.webp"))

    assert not manager.is_image_file(Path("test.txt"))
    assert not manager.is_image_file(Path("test.pdf"))
    assert not manager.is_image_file(Path("test.doc"))
    assert not manager.is_image_file(Path("test"))


def test_find_images_empty_directory():
    """Test finding images in an empty directory."""
    manager = ImageManager.get_instance()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = list(manager.find_images_in_directory(temp_path))
        assert images == []


def test_find_images_with_images():
    """Test finding images in a directory with image files."""
    manager = ImageManager.get_instance()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        image_files = ["test1.jpg", "test2.png", "test3.bmp"]
        non_image_files = ["readme.txt", "config.json"]

        for filename in image_files + non_image_files:
            (temp_path / filename).touch()

        # Find images
        found_images = list(manager.find_images_in_directory(temp_path))
        found_names = {img.name for img in found_images}

        assert len(found_images) == 3
        assert found_names == set(image_files)


def test_find_images_recursive():
    """Test finding images recursively in subdirectories."""
    manager = ImageManager.get_instance()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create subdirectory structure
        subdir = temp_path / "subdir"
        subdir.mkdir()

        # Create image files in root and subdirectory
        (temp_path / "root.jpg").touch()
        (subdir / "sub.png").touch()

        # Find images
        found_images = list(manager.find_images_in_directory(temp_path))
        found_names = {img.name for img in found_images}

        assert len(found_images) == 2
        assert found_names == {"root.jpg", "sub.png"}


def test_find_images_nonexistent_directory():
    """Test finding images in a nonexistent directory."""
    manager = ImageManager.get_instance()
    nonexistent = Path("/nonexistent/directory")

    try:
        list(manager.find_images_in_directory(nonexistent))
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


def test_find_images_not_directory():
    """Test finding images when path is not a directory."""
    manager = ImageManager.get_instance()
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)

        try:
            list(manager.find_images_in_directory(temp_path))
            raise AssertionError("Should have raised NotADirectoryError")
        except NotADirectoryError:
            pass


def test_count_images():
    """Test counting images in a directory."""
    manager = ImageManager.get_instance()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        (temp_path / "test1.jpg").touch()
        (temp_path / "test2.png").touch()
        (temp_path / "readme.txt").touch()

        count = manager.count_images_in_directory(temp_path)
        assert count == 2
