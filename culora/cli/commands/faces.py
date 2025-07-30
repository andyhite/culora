"""Face analysis CLI commands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from culora.cli.display.console import console
from culora.services.face_analysis_service import (
    FaceAnalysisService,
    FaceAnalysisServiceError,
)
from culora.services.image_service import ImageService

# Create faces sub-application
faces_app = typer.Typer(
    name="faces",
    help="Face detection and analysis commands",
    add_completion=False,
    rich_markup_mode="rich",
)


def _get_face_analysis_service() -> FaceAnalysisService:
    """Get configured FaceAnalysisService instance."""
    try:
        from culora.services.face_analysis_service import get_face_analysis_service

        return get_face_analysis_service()
    except Exception as e:
        console.error(f"Failed to get face analysis service: {e}")
        raise typer.Exit(1) from e


def _get_image_service() -> ImageService:
    """Get configured ImageService instance."""
    try:
        from culora.services.image_service import get_image_service

        return get_image_service()
    except Exception as e:
        console.error(f"Failed to get image service: {e}")
        raise typer.Exit(1) from e


@faces_app.command("detect")
def detect_faces_in_directory(
    directory: Annotated[Path, typer.Argument(help="Directory to scan for faces")],
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show processing progress")
    ] = True,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for processing", min=1, max=64),
    ] = 8,
) -> None:
    """Detect faces in all images in a directory."""
    try:
        if not directory.exists():
            console.error(f"Directory does not exist: {directory}")
            raise typer.Exit(1)

        if not directory.is_dir():
            console.error(f"Path is not a directory: {directory}")
            raise typer.Exit(1)

        console.info(f"Detecting faces in directory: {directory}")

        face_service = _get_face_analysis_service()

        # Process directory in batches
        total_images = 0
        total_faces = 0
        successful_analyses = 0
        failed_analyses = 0
        images_with_faces = 0

        try:
            if show_progress:
                console.info("Processing images in batches...")

            for batch_result in face_service.analyze_directory_batch(
                directory, batch_size
            ):
                total_images += batch_result.total_images
                total_faces += batch_result.total_faces_detected
                successful_analyses += batch_result.successful_analyses
                failed_analyses += batch_result.failed_analyses
                images_with_faces += batch_result.images_with_faces

                if show_progress:
                    console.info(
                        f"Processed batch: {batch_result.total_images} images, "
                        f"{batch_result.total_faces_detected} faces detected"
                    )

        except FaceAnalysisServiceError as e:
            console.error(f"Face analysis failed: {e}")
            raise typer.Exit(1) from e

        # Create results table
        table = Table(title="Face Detection Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Total Images", str(total_images))
        table.add_row("Successful Analyses", f"[green]{successful_analyses}[/green]")
        table.add_row("Failed Analyses", f"[red]{failed_analyses}[/red]")
        table.add_row("Images with Faces", f"[blue]{images_with_faces}[/blue]")
        table.add_row(
            "Images without Faces", str(successful_analyses - images_with_faces)
        )
        table.add_row("Total Faces Detected", f"[yellow]{total_faces}[/yellow]")

        if successful_analyses > 0:
            success_rate = (successful_analyses / total_images) * 100
            face_detection_rate = (images_with_faces / successful_analyses) * 100
            avg_faces_per_image = total_faces / successful_analyses

            table.add_row("Success Rate", f"{success_rate:.1f}%")
            table.add_row("Face Detection Rate", f"{face_detection_rate:.1f}%")
            table.add_row("Avg Faces per Image", f"{avg_faces_per_image:.2f}")

        console.print(table)

        if failed_analyses > 0:
            console.warning(f"Analysis failed for {failed_analyses} images")

    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@faces_app.command("analyze")
def analyze_single_image(
    image_path: Annotated[Path, typer.Argument(help="Path to image file")],
) -> None:
    """Analyze faces in a single image with detailed results."""
    try:
        if not image_path.exists():
            console.error(f"Image file does not exist: {image_path}")
            raise typer.Exit(1)

        console.info(f"Analyzing faces in: {image_path.name}")

        # Load image first
        image_service = _get_image_service()
        image_result = image_service.load_image(image_path)

        if not image_result.success:
            console.error(f"Failed to load image: {image_result.error}")
            raise typer.Exit(1)

        # Analyze faces
        face_service = _get_face_analysis_service()
        face_result = face_service.analyze_image(image_result)

        if not face_result.success:
            console.error(f"Face analysis failed: {face_result.error}")
            raise typer.Exit(1)

        # Create image info table
        image_table = Table(title=f"Image Analysis: {image_path.name}")
        image_table.add_column("Property", style="cyan", no_wrap=True)
        image_table.add_column("Value", style="white")

        image_table.add_row(
            "Image Size", f"{face_result.image_width} x {face_result.image_height}"
        )
        image_table.add_row(
            "Processing Time", f"{face_result.processing_duration:.3f} seconds"
        )
        image_table.add_row(
            "Faces Detected", f"[yellow]{face_result.face_count}[/yellow]"
        )

        if face_result.has_faces:
            image_table.add_row("Max Confidence", f"{face_result.max_confidence:.3f}")
            image_table.add_row(
                "Avg Confidence", f"{face_result.average_confidence:.3f}"
            )
            image_table.add_row(
                "Total Face Area", f"{face_result.total_face_area_ratio:.1%} of image"
            )

        console.print(image_table)

        # Show detailed face information
        if face_result.has_faces:
            faces_table = Table(title="Detected Faces")
            faces_table.add_column("Face #", style="cyan", no_wrap=True)
            faces_table.add_column("Confidence", style="white")
            faces_table.add_column("Bounding Box", style="white")
            faces_table.add_column("Area %", style="white")
            faces_table.add_column("Features", style="white")

            for i, face in enumerate(face_result.faces, 1):
                bbox_str = f"({face.bbox[0]:.0f}, {face.bbox[1]:.0f}, {face.bbox[2]:.0f}, {face.bbox[3]:.0f})"
                area_str = f"{face.face_area_ratio:.1%}"

                features = []
                if face.has_embedding:
                    features.append("Embedding")
                if face.has_landmarks:
                    features.append("Landmarks")
                if face.age is not None:
                    features.append(f"Age: {face.age}")
                if face.gender is not None:
                    features.append(f"Gender: {face.gender}")

                features_str = ", ".join(features) if features else "None"

                faces_table.add_row(
                    str(i),
                    f"{face.confidence:.3f}",
                    bbox_str,
                    area_str,
                    features_str,
                )

            console.print(faces_table)
        else:
            console.info("No faces detected in this image")

    except FaceAnalysisServiceError as e:
        console.error(f"Face analysis failed: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@faces_app.command("models")
def list_face_models() -> None:
    """List available InsightFace models and current configuration."""
    try:
        face_service = _get_face_analysis_service()
        model_info = face_service.get_model_info()

        # Create current config table
        config_table = Table(title="Face Analysis Configuration")
        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="white")

        config_table.add_row("Model Name", model_info["model_name"])
        config_table.add_row("Cache Directory", model_info["cache_dir"])
        config_table.add_row("Status", model_info["status"])

        if model_info["status"] == "initialized":
            config_table.add_row("Device Context", model_info["device_context"])
            config_table.add_row(
                "Confidence Threshold", model_info["confidence_threshold"]
            )
            config_table.add_row(
                "Max Faces per Image", model_info["max_faces_per_image"]
            )

        console.print(config_table)

        # Create available models table
        models_table = Table(title="Available InsightFace Models")
        models_table.add_column("Model Name", style="cyan", no_wrap=True)
        models_table.add_column("Description", style="white")
        models_table.add_column("Size", style="white")
        models_table.add_column("Speed", style="white")

        models_table.add_row(
            "buffalo_l", "High accuracy, large model", "Large (~600MB)", "Slower"
        )
        models_table.add_row(
            "buffalo_m", "Medium accuracy, balanced", "Medium (~300MB)", "Medium"
        )
        models_table.add_row(
            "buffalo_s", "Fast, lightweight model", "Small (~100MB)", "Faster"
        )
        models_table.add_row(
            "antelopev2", "Alternative high-accuracy", "Large (~500MB)", "Slower"
        )

        console.print(models_table)

        # Show usage instructions
        console.info("\n[bold]To change model:[/bold]")
        console.info(
            "• Set environment variable: [cyan]CULORA_FACES_MODEL_NAME=buffalo_s[/cyan]"
        )
        console.info(
            "• Or update config file: [cyan]faces.model_name: buffalo_s[/cyan]"
        )
        console.info(
            "• Or use CLI: [cyan]culora config set faces.model_name buffalo_s[/cyan]"
        )

    except Exception as e:
        console.error(f"Failed to get model information: {e}")
        raise typer.Exit(1) from e
