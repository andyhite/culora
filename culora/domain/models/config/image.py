"""Image processing configuration models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ImageConfig(BaseModel):
    """Configuration for image processing and loading.

    Controls image format support, batch processing limits,
    validation settings, and directory scanning behavior.
    """

    # Format support
    supported_formats: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"],
        description="List of supported image file extensions",
    )

    # Processing limits
    max_batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Maximum number of images to process in a single batch",
    )

    max_image_size: tuple[int, int] = Field(
        default=(4096, 4096),
        description="Maximum image dimensions (width, height) in pixels",
    )

    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        ge=1024,  # 1KB minimum
        description="Maximum image file size in bytes",
    )

    # Directory scanning
    recursive_scan: bool = Field(
        default=True, description="Whether to scan directories recursively"
    )

    max_scan_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum directory depth for recursive scanning",
    )

    skip_hidden_files: bool = Field(
        default=True, description="Whether to skip hidden files and directories"
    )

    # Progress reporting
    progress_update_interval: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of files between progress updates",
    )

    @field_validator("supported_formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate and normalize supported formats."""
        if not v:
            raise ValueError("At least one supported format must be specified")

        normalized = []
        for fmt in v:
            if not fmt.startswith("."):
                fmt = f".{fmt}"
            normalized.append(fmt.lower())

        return normalized

    @field_validator("max_image_size")
    @classmethod
    def validate_image_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate maximum image dimensions."""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        if width > 65535 or height > 65535:
            raise ValueError("Image dimensions too large (max 65535x65535)")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageConfig":
        """Create ImageConfig from dictionary."""
        return cls(**data)
