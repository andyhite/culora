"""Image management CLI commands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from culora.cli.display.console import console
from culora.services.config_service import ConfigService
from culora.services.image_service import ImageService, ImageServiceError

# Create images sub-application
images_app = typer.Typer(
    name="images",
    help="Image loading and processing commands",
    add_completion=False,
    rich_markup_mode="rich",
)


def _get_image_service() -> ImageService:
    """Get configured ImageService instance."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        return ImageService(config)
    except Exception as e:
        console.error(f"Failed to initialize image service: {e}")
        raise typer.Exit(1) from e


@images_app.command("scan")
def scan_directory(
    directory: Annotated[Path, typer.Argument(help="Directory to scan for images")],
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show scanning progress")
    ] = True,
) -> None:
    """Scan directory and display image statistics."""
    try:
        if not directory.exists():
            console.error(f"Directory does not exist: {directory}")
            raise typer.Exit(1)

        console.info(f"Scanning directory: {directory}")

        service = _get_image_service()
        result = service.scan_directory(directory, show_progress=show_progress)

        # Create results table
        table = Table(title="Directory Scan Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Total Files", str(result.total_files))
        table.add_row("Valid Images", f"[green]{result.valid_images}[/green]")
        table.add_row("Invalid Files", f"[red]{result.invalid_images}[/red]")
        table.add_row("Total Size", f"{result.total_size / (1024*1024):.1f} MB")
        table.add_row("Scan Duration", f"{result.scan_duration:.2f} seconds")

        console.print(table)

        # Show format breakdown if we found images
        if result.supported_formats:
            format_table = Table(title="Image Formats Found")
            format_table.add_column("Format", style="cyan")
            format_table.add_column("Count", style="white")

            for fmt, count in result.supported_formats.items():
                format_table.add_row(fmt, str(count))

            console.print(format_table)

        # Show errors if any
        if result.errors:
            console.warning(f"Found {len(result.errors)} errors:")
            for error in result.errors[:10]:  # Show first 10 errors
                console.print(f"  • {error}")

            if len(result.errors) > 10:
                console.print(f"  ... and {len(result.errors) - 10} more errors")

    except ImageServiceError as e:
        console.error(f"Scan failed: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@images_app.command("validate")
def validate_directory(
    directory: Annotated[Path, typer.Argument(help="Directory to validate images in")],
) -> None:
    """Validate all images in directory with detailed report."""
    try:
        if not directory.exists():
            console.error(f"Directory does not exist: {directory}")
            raise typer.Exit(1)

        console.info(f"Validating images in: {directory}")

        service = _get_image_service()

        # First scan to get image paths
        scan_result = service.scan_directory(directory)

        if not scan_result.image_paths:
            console.warning("No images found to validate")
            return

        console.info(f"Found {len(scan_result.image_paths)} images to validate")

        # Process in batches and collect results
        all_results = []
        successful = 0
        failed = 0

        for batch_result in service.load_directory_batch(directory):
            all_results.extend(batch_result.results)
            successful += batch_result.successful_loads
            failed += batch_result.failed_loads

        # Create validation summary table
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Total Images", str(len(all_results)))
        table.add_row("Valid Images", f"[green]{successful}[/green]")
        table.add_row("Invalid Images", f"[red]{failed}[/red]")

        if successful > 0:
            success_rate = (successful / len(all_results)) * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")

        console.print(table)

        # Show failed images with details
        if failed > 0:
            failed_table = Table(title="Failed Images")
            failed_table.add_column("Image", style="red", no_wrap=True)
            failed_table.add_column("Error", style="white")

            for result in all_results:
                if not result.success:
                    failed_table.add_row(
                        str(result.metadata.path.name), result.error or "Unknown error"
                    )

            console.print(failed_table)

    except ImageServiceError as e:
        console.error(f"Validation failed: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@images_app.command("info")
def show_image_info(
    image_path: Annotated[Path, typer.Argument(help="Path to image file")],
) -> None:
    """Show detailed metadata for a single image."""
    try:
        if not image_path.exists():
            console.error(f"Image file does not exist: {image_path}")
            raise typer.Exit(1)

        service = _get_image_service()

        # Validate the path first
        is_valid, error = service.validate_image_path(image_path)
        if not is_valid:
            console.error(f"Invalid image: {error}")
            raise typer.Exit(1)

        # Load the image
        result = service.load_image(image_path)

        # Create info table
        table = Table(title=f"Image Information: {image_path.name}")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        metadata = result.metadata
        table.add_row("Path", str(metadata.path))
        table.add_row("Format", metadata.format)
        table.add_row("Dimensions", f"{metadata.width} x {metadata.height}")
        table.add_row("File Size", f"{metadata.file_size / 1024:.1f} KB")
        table.add_row("Created", metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Modified", metadata.modified_at.strftime("%Y-%m-%d %H:%M:%S"))

        if result.success:
            table.add_row("Status", "[green]Valid[/green]")
            if result.image:
                table.add_row("Color Mode", result.image.mode)
        else:
            table.add_row("Status", "[red]Invalid[/red]")
            table.add_row("Error", result.error or "Unknown error")
            table.add_row("Error Code", result.error_code or "N/A")

        console.print(table)

    except ImageServiceError as e:
        console.error(f"Failed to get image info: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@images_app.command("formats")
def list_supported_formats() -> None:
    """List all supported image formats."""
    try:
        service = _get_image_service()
        formats = service.get_supported_formats()

        table = Table(title="Supported Image Formats")
        table.add_column("Extension", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        format_descriptions = {
            ".jpg": "JPEG - Joint Photographic Experts Group",
            ".jpeg": "JPEG - Joint Photographic Experts Group",
            ".png": "PNG - Portable Network Graphics",
            ".webp": "WebP - Modern web image format",
            ".tiff": "TIFF - Tagged Image File Format",
            ".tif": "TIFF - Tagged Image File Format",
        }

        for fmt in sorted(formats):
            description = format_descriptions.get(fmt, "Supported image format")
            table.add_row(fmt, description)

        console.print(table)
        console.info(f"Total supported formats: {len(formats)}")

    except Exception as e:
        console.error(f"Failed to list formats: {e}")
        raise typer.Exit(1) from e
