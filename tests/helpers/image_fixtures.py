"""Image test fixtures and utilities."""

from pathlib import Path

from PIL import Image


class ImageFixtures:
    """Helper for creating test image fixtures."""

    @staticmethod
    def create_test_image(
        width: int = 800,
        height: int = 600,
        color: str = "red",
        format: str = "JPEG",
    ) -> Image.Image:
        """Create a test image with specified dimensions and color.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Image color (red, green, blue, etc.)
            format: Image format (JPEG, PNG, etc.)

        Returns:
            PIL Image object
        """
        image = Image.new("RGB", (width, height), color=color)
        # Store format information for later use
        # Store format in info dict to avoid dynamic attributes
        image.info["_test_format"] = format
        return image

    @staticmethod
    def save_test_image(
        image: Image.Image,
        path: Path,
        format: str | None = None,
    ) -> None:
        """Save a test image to file.

        Args:
            image: PIL Image to save
            path: Path to save image to
            format: Optional format override
        """
        if format is None:
            format = image.info.get("_test_format") or image.format or "JPEG"

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        image.save(path, format=format)

    @staticmethod
    def create_test_image_file(
        path: Path,
        width: int = 800,
        height: int = 600,
        color: str = "red",
        format: str = "JPEG",
    ) -> Path:
        """Create and save a test image file.

        Args:
            path: Path to save image to
            width: Image width in pixels
            height: Image height in pixels
            color: Image color
            format: Image format

        Returns:
            Path to the created image file
        """
        image = ImageFixtures.create_test_image(width, height, color, format)
        ImageFixtures.save_test_image(image, path, format)
        return path

    @staticmethod
    def create_test_directory_structure(temp_dir: Path) -> dict[str, Path]:
        """Create a test directory structure with various image files.

        Args:
            temp_dir: Temporary directory to create structure in

        Returns:
            Dictionary mapping file descriptions to paths
        """
        files = {}

        # Valid images in root
        files["jpg_image"] = ImageFixtures.create_test_image_file(
            temp_dir / "image1.jpg", 800, 600, "red", "JPEG"
        )
        files["png_image"] = ImageFixtures.create_test_image_file(
            temp_dir / "image2.png", 1024, 768, "green", "PNG"
        )
        files["webp_image"] = ImageFixtures.create_test_image_file(
            temp_dir / "image3.webp", 640, 480, "blue", "WebP"
        )

        # Images in subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        files["subdir_jpg"] = ImageFixtures.create_test_image_file(
            subdir / "sub_image1.jpg", 1920, 1080, "yellow", "JPEG"
        )
        files["subdir_png"] = ImageFixtures.create_test_image_file(
            subdir / "sub_image2.png", 512, 512, "purple", "PNG"
        )

        # Hidden files (should be skipped if configured)
        files["hidden_jpg"] = ImageFixtures.create_test_image_file(
            temp_dir / ".hidden.jpg", 400, 300, "orange", "JPEG"
        )

        # Non-image files (should be ignored)
        files["text_file"] = temp_dir / "readme.txt"
        files["text_file"].write_text("This is not an image")

        files["unsupported_file"] = temp_dir / "document.pdf"
        files["unsupported_file"].write_bytes(b"Fake PDF content")

        # Create a deep nested structure for depth testing
        deep_dir = temp_dir / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True, exist_ok=True)
        files["deep_image"] = ImageFixtures.create_test_image_file(
            deep_dir / "deep.jpg", 200, 150, "cyan", "JPEG"
        )

        return files

    @staticmethod
    def create_corrupted_image_file(path: Path) -> Path:
        """Create a corrupted image file for testing error handling.

        Args:
            path: Path to create corrupted file at

        Returns:
            Path to the corrupted file
        """
        # Create a file with JPEG header but invalid content
        path.parent.mkdir(parents=True, exist_ok=True)

        # JPEG magic bytes followed by garbage
        corrupted_data = b"\xff\xd8\xff\xe0" + b"This is not valid JPEG data" * 100
        path.write_bytes(corrupted_data)

        return path
