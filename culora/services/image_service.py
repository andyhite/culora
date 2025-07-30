"""Image loading and processing service."""

import mimetypes
import os
import time
from collections import defaultdict
from collections.abc import Generator, Iterator
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile

from culora.core import CuLoRAError
from culora.domain import (
    BatchLoadResult,
    CuLoRAConfig,
    DirectoryScanResult,
    ImageLoadResult,
    ImageMetadata,
)
from culora.utils import get_logger

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)


class ImageServiceError(CuLoRAError):
    """Base exception for image service errors."""

    def __init__(self, message: str, error_code: str = "IMAGE_SERVICE_ERROR") -> None:
        super().__init__(message, error_code)


class ImageValidationError(ImageServiceError):
    """Exception for image validation failures."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message, "IMAGE_VALIDATION_ERROR")
        self.path = path


class ImageService:
    """Service for image loading, validation, and directory processing.

    Provides comprehensive image handling capabilities including:
    - Directory traversal and image discovery
    - Image loading with validation
    - Batch processing with memory management
    - Format support and validation
    - Metadata extraction
    """

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize ImageService with configuration.

        Args:
            config: CuLoRA configuration with image settings
        """
        self.config = config
        self.image_config = config.images

        # Initialize MIME types for validation
        mimetypes.init()

        logger.info(
            "ImageService initialized",
            supported_formats=self.image_config.supported_formats,
            max_batch_size=self.image_config.max_batch_size,
        )

    def scan_directory(
        self, directory: Path, show_progress: bool = True
    ) -> DirectoryScanResult:
        """Scan directory for image files.

        Args:
            directory: Directory to scan
            show_progress: Whether to show progress during scanning

        Returns:
            DirectoryScanResult with scan statistics and file paths

        Raises:
            ImageServiceError: If directory scanning fails
        """
        start_time = time.time()

        if not directory.exists():
            raise ImageServiceError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise ImageServiceError(f"Path is not a directory: {directory}")

        logger.info("Starting directory scan", directory=str(directory))

        image_paths: list[Path] = []
        errors: list[str] = []
        total_files = 0
        total_size = 0
        format_counts: dict[str, int] = defaultdict(int)

        try:
            for file_path in self._walk_directory(directory):
                total_files += 1

                if (
                    show_progress
                    and total_files % self.image_config.progress_update_interval == 0
                ):
                    logger.debug(f"Scanned {total_files} files...")

                if self._is_supported_image(file_path):
                    try:
                        file_size = file_path.stat().st_size

                        # Check file size limits
                        if file_size > self.image_config.max_file_size:
                            errors.append(
                                f"File too large: {file_path} ({file_size} bytes)"
                            )
                            continue

                        image_paths.append(file_path)
                        total_size += file_size
                        format_counts[file_path.suffix.lower()] += 1

                    except (OSError, PermissionError) as e:
                        errors.append(f"Cannot access file {file_path}: {e}")
                        continue

        except Exception as e:
            logger.exception("Directory scan failed", directory=str(directory))
            raise ImageServiceError(f"Directory scan failed: {e}") from e

        scan_duration = time.time() - start_time
        valid_images = len(image_paths)
        invalid_images = total_files - valid_images

        result = DirectoryScanResult(
            total_files=total_files,
            valid_images=valid_images,
            invalid_images=invalid_images,
            supported_formats=dict(format_counts),
            total_size=total_size,
            scan_duration=scan_duration,
            errors=errors,
            image_paths=image_paths,
        )

        logger.info(
            "Directory scan completed",
            directory=str(directory),
            total_files=total_files,
            valid_images=valid_images,
            invalid_images=invalid_images,
            duration=scan_duration,
            errors=len(errors),
        )

        return result

    def load_image(self, path: Path) -> ImageLoadResult:
        """Load and validate a single image.

        Args:
            path: Path to image file

        Returns:
            ImageLoadResult with loaded image and metadata
        """
        try:
            # Get file metadata first
            metadata = self._extract_metadata(path)

            if not metadata.is_valid:
                return ImageLoadResult(
                    success=False,
                    metadata=metadata,
                    error=metadata.error_message,
                    error_code="INVALID_METADATA",
                )

            # Load the image
            image = Image.open(path)

            # Convert to RGB if needed (for consistent processing)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Validate image dimensions
            if (
                image.width > self.image_config.max_image_size[0]
                or image.height > self.image_config.max_image_size[1]
            ):
                return ImageLoadResult(
                    success=False,
                    metadata=metadata,
                    error=f"Image dimensions too large: {image.width}x{image.height}",
                    error_code="IMAGE_TOO_LARGE",
                )

            logger.debug(
                "Image loaded successfully",
                path=str(path),
                format=image.format,
                size=f"{image.width}x{image.height}",
                mode=image.mode,
            )

            return ImageLoadResult(
                success=True,
                metadata=metadata,
                image=image,
            )

        except Exception as e:
            logger.warning(
                "Failed to load image",
                path=str(path),
                error=str(e),
            )

            # Create metadata with error info
            try:
                metadata = self._extract_metadata(path, force_invalid=True)
            except Exception:
                # Fallback metadata if even basic info extraction fails
                metadata = ImageMetadata(
                    path=path,
                    format="unknown",
                    width=0,
                    height=0,
                    file_size=0,
                    created_at=datetime.now(),
                    modified_at=datetime.now(),
                    is_valid=False,
                    error_message=str(e),
                )

            return ImageLoadResult(
                success=False,
                metadata=metadata,
                error=str(e),
                error_code="LOAD_FAILED",
            )

    def load_batch(self, paths: list[Path]) -> BatchLoadResult:
        """Load multiple images in a batch.

        Args:
            paths: List of image paths to load

        Returns:
            BatchLoadResult with all loaded images and statistics
        """
        start_time = time.time()

        logger.info("Starting batch load", batch_size=len(paths))

        results: list[ImageLoadResult] = []
        successful_loads = 0
        failed_loads = 0
        total_size = 0

        for i, path in enumerate(paths):
            if (i + 1) % self.image_config.progress_update_interval == 0:
                logger.debug(f"Loaded {i + 1}/{len(paths)} images...")

            result = self.load_image(path)
            results.append(result)

            if result.success:
                successful_loads += 1
                total_size += result.metadata.file_size
            else:
                failed_loads += 1

        processing_duration = time.time() - start_time

        logger.info(
            "Batch load completed",
            total_images=len(paths),
            successful=successful_loads,
            failed=failed_loads,
            total_size=total_size,
            duration=processing_duration,
        )

        return BatchLoadResult(
            results=results,
            successful_loads=successful_loads,
            failed_loads=failed_loads,
            total_size=total_size,
            processing_duration=processing_duration,
        )

    def load_directory_batch(
        self, directory: Path
    ) -> Generator[BatchLoadResult, None, None]:
        """Load images from directory in batches.

        Args:
            directory: Directory to load images from

        Yields:
            BatchLoadResult for each batch processed

        Raises:
            ImageServiceError: If directory processing fails
        """
        # First scan the directory
        scan_result = self.scan_directory(directory)

        if not scan_result.image_paths:
            logger.warning("No images found in directory", directory=str(directory))
            return

        # Process in batches
        batch_size = self.image_config.max_batch_size
        total_batches = (len(scan_result.image_paths) + batch_size - 1) // batch_size

        logger.info(
            "Starting batch processing",
            total_images=len(scan_result.image_paths),
            batch_size=batch_size,
            total_batches=total_batches,
        )

        for i in range(0, len(scan_result.image_paths), batch_size):
            batch_paths = scan_result.image_paths[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            yield self.load_batch(batch_paths)

    def validate_image_path(self, path: Path) -> tuple[bool, str | None]:
        """Validate if a path points to a supported image file.

        Args:
            path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path.exists():
            return False, f"File does not exist: {path}"

        if not path.is_file():
            return False, f"Path is not a file: {path}"

        if not self._is_supported_image(path):
            return False, f"Unsupported image format: {path.suffix}"

        try:
            file_size = path.stat().st_size
            if file_size > self.image_config.max_file_size:
                return False, f"File too large: {file_size} bytes"
        except (OSError, PermissionError) as e:
            return False, f"Cannot access file: {e}"

        return True, None

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.

        Returns:
            List of supported file extensions
        """
        return self.image_config.supported_formats.copy()

    def _walk_directory(self, directory: Path) -> Iterator[Path]:
        """Walk directory recursively or non-recursively based on config.

        Args:
            directory: Directory to walk

        Yields:
            Path objects for each file found
        """
        if self.image_config.recursive_scan:
            for root, dirs, files in os.walk(directory):
                # Check depth limit
                depth = len(Path(root).relative_to(directory).parts)
                if depth > self.image_config.max_scan_depth:
                    continue

                # Skip hidden directories if configured
                if self.image_config.skip_hidden_files:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                root_path = Path(root)
                for file_name in files:
                    # Skip hidden files if configured
                    if self.image_config.skip_hidden_files and file_name.startswith(
                        "."
                    ):
                        continue

                    yield root_path / file_name
        else:
            # Non-recursive scan
            try:
                for item in directory.iterdir():
                    if item.is_file():
                        # Skip hidden files if configured
                        if (
                            self.image_config.skip_hidden_files
                            and item.name.startswith(".")
                        ):
                            continue
                        yield item
            except (OSError, PermissionError):
                pass  # Skip directories we can't read

    def _is_supported_image(self, path: Path) -> bool:
        """Check if file has supported image extension.

        Args:
            path: Path to check

        Returns:
            True if file extension is supported
        """
        return path.suffix.lower() in self.image_config.supported_formats

    def _extract_metadata(
        self, path: Path, force_invalid: bool = False
    ) -> ImageMetadata:
        """Extract metadata from image file.

        Args:
            path: Path to image file
            force_invalid: Force metadata to be marked as invalid

        Returns:
            ImageMetadata object

        Raises:
            ImageServiceError: If metadata extraction fails
        """
        try:
            stat = path.stat()

            if force_invalid:
                return ImageMetadata(
                    path=path,
                    format="unknown",
                    width=0,
                    height=0,
                    file_size=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                    is_valid=False,
                    error_message="Forced invalid",
                )

            # Try to get basic image info without fully loading
            with Image.open(path) as img:
                format_name = img.format or "unknown"
                width, height = img.size

            return ImageMetadata(
                path=path,
                format=format_name,
                width=width,
                height=height,
                file_size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                is_valid=True,
            )

        except Exception as e:
            logger.debug("Failed to extract metadata", path=str(path), error=str(e))

            # Get basic file info even if image is invalid
            try:
                stat = path.stat()
                return ImageMetadata(
                    path=path,
                    format="unknown",
                    width=0,
                    height=0,
                    file_size=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                    is_valid=False,
                    error_message=str(e),
                )
            except Exception as stat_error:
                raise ImageServiceError(
                    f"Failed to get file metadata for {path}: {stat_error}"
                ) from stat_error


# Global image service instance
_image_service: ImageService | None = None


def get_image_service() -> ImageService:
    """Get the global image service instance.

    Returns:
        Global ImageService instance

    Raises:
        ImageServiceError: If service has not been initialized
    """
    if _image_service is None:
        raise ImageServiceError(
            "ImageService not initialized. Call initialize_image_service() first."
        )
    return _image_service


def initialize_image_service(config: CuLoRAConfig) -> ImageService:
    """Initialize the global image service instance.

    Args:
        config: CuLoRA configuration

    Returns:
        Initialized ImageService instance
    """
    global _image_service
    _image_service = ImageService(config)
    return _image_service
