"""Select command for CuLoRA CLI."""

import shutil
from pathlib import Path
from typing import Annotated

import cv2
import typer
from rich.console import Console
from rich.table import Table

from culora.models.analysis import AnalysisResult, AnalysisStage, ImageAnalysis
from culora.utils.cache import load_analysis_cache

console = Console()


def draw_face_bounding_boxes(
    image_path: Path, output_path: Path, face_metadata: dict[str, str]
) -> None:
    """Draw bounding boxes on image for detected faces with confidence scores.

    Args:
        image_path: Path to the source image
        output_path: Path where the annotated image should be saved
        face_metadata: Face analysis metadata containing bounding boxes and confidence scores
    """
    # Load the image
    try:
        image = cv2.imread(str(image_path))
        if image is None:  # type: ignore[reportUnnecessaryComparison]
            raise ValueError("Failed to load image")
    except Exception:
        # Fallback to copying original if image can't be loaded
        shutil.copy2(image_path, output_path)
        return

    # Parse bounding boxes and confidence scores
    bbox_str = face_metadata.get("bounding_boxes", "")
    confidence_str = face_metadata.get("confidence_scores", "")

    if not bbox_str or not confidence_str:
        # No face data, just copy original
        shutil.copy2(image_path, output_path)
        return

    # Parse bounding boxes: "x1,y1,x2,y2;x1,y1,x2,y2"
    bboxes: list[list[float]] = []
    for bbox_part in bbox_str.split(";"):
        if bbox_part.strip():
            coords = [float(x) for x in bbox_part.split(",")]
            if len(coords) == 4:
                bboxes.append(coords)

    # Parse confidence scores: "0.850,0.920"
    confidences: list[float] = []
    for conf_part in confidence_str.split(","):
        if conf_part.strip():
            confidences.append(float(conf_part))

    # Draw bounding boxes
    for i, (bbox, confidence) in enumerate(zip(bboxes, confidences, strict=False)):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Different colors for different faces (cycle through colors)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add confidence score label
        label = f"Face {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Position label above the bounding box
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Save the annotated image
    cv2.imwrite(str(output_path), image)


def select_command(
    output_dir: Annotated[
        str, typer.Argument(help="Directory to copy selected images to")
    ],
    input_dir: Annotated[
        str, typer.Option(help="Directory containing analyzed images")
    ] = ".",
    dry_run: Annotated[
        bool, typer.Option(help="Show what would be selected without copying files")
    ] = False,
    draw_boxes: Annotated[
        bool,
        typer.Option(
            "--draw-boxes",
            help="Draw bounding boxes on detected faces with confidence scores",
        ),
    ] = False,
) -> None:
    """Select and copy curated images to output directory.

    This command copies images that passed all enabled analysis stages from
    the analyzed directory to the specified output directory. Images are
    renamed sequentially for training use.

    Use --draw-boxes to annotate detected faces with bounding boxes and
    confidence scores on the copied images.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    console.print(f"[bold green]Selecting images from:[/bold green] {input_path}")
    console.print(f"[bold green]Output directory:[/bold green] {output_path}")

    if dry_run:
        console.print("[blue]Dry run mode - no files will be copied[/blue]")

    if draw_boxes:
        console.print("[blue]Bounding boxes will be drawn on detected faces[/blue]")

    try:
        # Load analysis results from cache
        analysis = load_analysis_cache(input_path)
        if not analysis:
            console.print(
                "[red]Error:[/red] No analysis results found. Run 'analyze' first."
            )
            raise typer.Exit(1)

        # Get images that passed all stages
        selected_images = analysis.passed_images
        skipped_images = analysis.failed_images + analysis.skipped_images

        if not selected_images:
            console.print("[yellow]No images passed all analysis stages[/yellow]")
            _display_selection_summary(
                selected_images, skipped_images, analysis.enabled_stages
            )
            return

        # Create output directory if it doesn't exist (unless dry run)
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)

        # Copy and rename selected images
        copied_count = 0
        for i, image in enumerate(selected_images, 1):
            source_path = Path(image.file_path)
            # Keep original extension, use sequential numbering
            extension = source_path.suffix
            target_filename = f"{i:03d}{extension}"  # 001.jpg, 002.jpg, etc.
            target_path = output_path / target_filename

            if dry_run:
                box_info = " (with boxes)" if draw_boxes else ""
                console.print(
                    f"[dim]Would copy:[/dim] {source_path.name} → {target_filename}{box_info}"
                )
            else:
                if draw_boxes:
                    # Check if this image has face detection results
                    face_metadata = None
                    for stage_result in image.stage_results:
                        if (
                            stage_result.stage == AnalysisStage.FACE
                            and stage_result.result == AnalysisResult.PASS
                            and stage_result.metadata
                        ):
                            face_metadata = stage_result.metadata
                            break

                    if face_metadata:
                        # Draw bounding boxes and save annotated image
                        draw_face_bounding_boxes(
                            source_path, target_path, face_metadata
                        )
                    else:
                        # No face detection data, just copy original
                        shutil.copy2(source_path, target_path)
                else:
                    # Normal copy without annotations
                    shutil.copy2(source_path, target_path)
                copied_count += 1

        # Display results
        _display_selection_summary(
            selected_images,
            skipped_images,
            analysis.enabled_stages,
            dry_run,
            copied_count,
        )

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Directory not found: {input_path}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error during selection:[/red] {e}")
        raise typer.Exit(1) from e


def _display_selection_summary(
    selected_images: list[ImageAnalysis],
    skipped_images: list[ImageAnalysis],
    enabled_stages: list[AnalysisStage],
    dry_run: bool = False,
    copied_count: int = 0,
) -> None:
    """Display selection results summary."""
    console.print("\n[bold green]Selection Complete![/bold green]")

    # Show enabled stages that were used for selection
    if enabled_stages:
        stages_str = ", ".join([stage.value for stage in enabled_stages])
        console.print(f"[blue]Selection based on stages:[/blue] {stages_str}")

    # Create results table showing selected vs skipped
    table = Table(title="Selection Results")
    table.add_column("Image", style="cyan", width=30)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Reason", style="dim", width=40)

    # Sort all images by filename for consistent display
    all_images = selected_images + skipped_images
    sorted_images = sorted(all_images, key=lambda img: Path(img.file_path).name.lower())

    # Add results for each image
    for image in sorted_images:
        filename = Path(image.file_path).name

        if image.overall_result == AnalysisResult.PASS:
            status = "[green]✅ SELECTED[/green]"
            reason = "Passed all enabled stages"
        else:
            status = "[red]❌ SKIPPED[/red]"
            # Show which stages failed
            failed_stages = [
                result.stage.value
                for result in image.stage_results
                if result.result == AnalysisResult.FAIL
            ]
            if failed_stages:
                reason = f"Failed: {', '.join(failed_stages)}"
            else:
                reason = "No analysis results"

        table.add_row(filename, status, reason)

    console.print(table)

    # Show summary counts
    console.print(f"\n[bold]Summary:[/bold] {len(all_images)} images processed")
    console.print(f"✅ [green]{len(selected_images)} selected for training[/green]")
    console.print(f"❌ [red]{len(skipped_images)} skipped[/red]")

    if not dry_run and copied_count > 0:
        console.print(f"\n[green]Successfully copied {copied_count} images[/green]")


def register_command(app: typer.Typer) -> None:
    """Register the select command with the given Typer app."""
    app.command(name="select")(select_command)
