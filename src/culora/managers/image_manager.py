"""Image management functionality for CuLoRA."""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

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


class ImageManager:
    """Singleton image manager for CuLoRA."""

    _instance: "ImageManager | None" = None

    def __new__(cls) -> "ImageManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

    def is_image_file(self, file_path: Path) -> bool:
        """Check if a file is a supported image file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file has a supported image extension.
        """
        return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def find_images_in_directory(self, directory: Path) -> Iterator[Path]:
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
            if file_path.is_file() and self.is_image_file(file_path):
                yield file_path

    def get_image_metadata(self, image_path: Path) -> dict[str, Any]:
        """Get metadata for an image file."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        stat = image_path.stat()
        return {
            "file_path": str(image_path.resolve()),
            "file_size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
        }

    def validate_directory(self, directory: Path) -> None:
        """Validate that the directory exists and is accessible."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

    def count_images_in_directory(self, directory: Path) -> int:
        """Count the number of supported image files in a directory.

        Args:
            directory: Directory to search for images.

        Returns:
            Number of image files found.
        """
        return sum(1 for _ in self.find_images_in_directory(directory))

    def load_image(self, image_path: Path) -> Image.Image:
        """Load a PIL Image from the given path.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            return Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {e}") from e

    @classmethod
    def get_instance(cls) -> "ImageManager":
        """Get the singleton instance of ImageManager."""
        return cls()
