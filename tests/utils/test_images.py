"""Tests for image utilities."""

import tempfile
from pathlib import Path

from culora.utils.images import count_images, find_images, is_image_file


def test_is_image_file():
    """Test image file detection."""
    assert is_image_file(Path("test.jpg"))
    assert is_image_file(Path("test.JPG"))
    assert is_image_file(Path("test.png"))
    assert is_image_file(Path("test.jpeg"))
    assert is_image_file(Path("test.bmp"))
    assert is_image_file(Path("test.tiff"))
    assert is_image_file(Path("test.tif"))
    assert is_image_file(Path("test.webp"))

    assert not is_image_file(Path("test.txt"))
    assert not is_image_file(Path("test.pdf"))
    assert not is_image_file(Path("test.doc"))
    assert not is_image_file(Path("test"))


def test_find_images_empty_directory():
    """Test finding images in an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = list(find_images(temp_path))
        assert images == []


def test_find_images_with_images():
    """Test finding images in a directory with image files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        image_files = ["test1.jpg", "test2.png", "test3.bmp"]
        non_image_files = ["readme.txt", "config.json"]

        for filename in image_files + non_image_files:
            (temp_path / filename).touch()

        # Find images
        found_images = list(find_images(temp_path))
        found_names = {img.name for img in found_images}

        assert len(found_images) == 3
        assert found_names == set(image_files)


def test_find_images_recursive():
    """Test finding images recursively in subdirectories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create subdirectory structure
        subdir = temp_path / "subdir"
        subdir.mkdir()

        # Create image files in root and subdirectory
        (temp_path / "root.jpg").touch()
        (subdir / "sub.png").touch()

        # Find images
        found_images = list(find_images(temp_path))
        found_names = {img.name for img in found_images}

        assert len(found_images) == 2
        assert found_names == {"root.jpg", "sub.png"}


def test_find_images_nonexistent_directory():
    """Test finding images in a nonexistent directory."""
    nonexistent = Path("/nonexistent/directory")

    try:
        list(find_images(nonexistent))
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


def test_find_images_not_directory():
    """Test finding images when path is not a directory."""
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)

        try:
            list(find_images(temp_path))
            raise AssertionError("Should have raised NotADirectoryError")
        except NotADirectoryError:
            pass


def test_count_images():
    """Test counting images in a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create some test files
        (temp_path / "test1.jpg").touch()
        (temp_path / "test2.png").touch()
        (temp_path / "readme.txt").touch()

        count = count_images(temp_path)
        assert count == 2
