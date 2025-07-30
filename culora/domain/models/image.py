"""Image processing domain models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata for an image file.

    Contains essential information about an image file including
    path, dimensions, format, and file system metadata.
    """

    path: Path
    format: str
    width: int
    height: int
    file_size: int
    created_at: datetime
    modified_at: datetime
    is_valid: bool
    error_message: str | None = None


@dataclass(frozen=True)
class ImageLoadResult:
    """Result of loading an image file.

    Contains the loaded image data, metadata, and success/failure status.
    """

    success: bool
    metadata: ImageMetadata
    image: Image.Image | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class DirectoryScanResult:
    """Result of scanning a directory for images.

    Contains statistics and file listings from directory traversal.
    """

    total_files: int
    valid_images: int
    invalid_images: int
    supported_formats: dict[str, int]  # format -> count
    total_size: int
    scan_duration: float
    errors: list[str]
    image_paths: list[Path]


@dataclass(frozen=True)
class BatchLoadResult:
    """Result of loading a batch of images.

    Contains loaded images and processing statistics.
    """

    results: list[ImageLoadResult]
    successful_loads: int
    failed_loads: int
    total_size: int
    processing_duration: float
