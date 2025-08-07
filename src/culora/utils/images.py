"""Image file utilities for CuLoRA."""

from collections.abc import Iterator
from pathlib import Path

# Common image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def is_image_file(file_path: Path) -> bool:
    """Check if a file is a supported image file.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file has a supported image extension.
    """
    return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def find_images(directory: Path) -> Iterator[Path]:
    """Find all supported image files in a directory.

    Args:
        directory: Directory to search for images.

    Yields:
        Path objects for each image file found.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    for file_path in directory.rglob("*"):
        if file_path.is_file() and is_image_file(file_path):
            yield file_path


def count_images(directory: Path) -> int:
    """Count the number of supported image files in a directory.

    Args:
        directory: Directory to search for images.

    Returns:
        Number of image files found.
    """
    return sum(1 for _ in find_images(directory))
