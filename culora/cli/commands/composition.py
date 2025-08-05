"""Composition analysis CLI commands."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from culora.cli.display.console import console
from culora.services.composition_service import (
    CompositionService,
    CompositionServiceError,
)
from culora.services.config_service import ConfigService
from culora.services.image_service import ImageService, ImageServiceError

if TYPE_CHECKING:
    from culora.domain.models.composition import (
        BatchCompositionResult,
        CompositionResult,
    )

# Create composition sub-application
composition_app = typer.Typer(
    name="composition",
    help="Image composition analysis commands",
    add_completion=False,
    rich_markup_mode="rich",
)


def _get_services() -> tuple[ImageService, CompositionService]:
    """Get configured service instances."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        image_service = ImageService(config)
        composition_service = CompositionService(config)
        return image_service, composition_service
    except Exception as e:
        console.error(f"Failed to initialize services: {e}")
        raise typer.Exit(1) from e


@composition_app.command("analyze")
def analyze_composition(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze")],
    show_details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed analysis results"),
    ] = False,
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show analysis progress")
    ] = True,
    min_confidence: Annotated[
        float | None, typer.Option("--min-confidence", help="Minimum confidence filter")
    ] = None,
    shot_type_filter: Annotated[
        str | None, typer.Option("--shot-type", help="Filter by shot type")
    ] = None,
    scene_filter: Annotated[
        str | None, typer.Option("--scene", help="Filter by scene type")
    ] = None,
) -> None:
    """Analyze image composition using vision-language models.

    Analyzes composition characteristics including shot types, scene types,
    lighting quality, background complexity, facial expressions, and camera angles.
    """
    try:
        if not path.exists():
            console.error(f"Path does not exist: {path}")
            raise typer.Exit(1)

        image_service, composition_service = _get_services()

        if path.is_file():
            _analyze_single_image(
                path, image_service, composition_service, show_details
            )
        else:
            _analyze_directory(
                path,
                image_service,
                composition_service,
                show_details,
                show_progress,
                min_confidence,
                shot_type_filter,
                scene_filter,
            )

    except (ImageServiceError, CompositionServiceError) as e:
        console.error(f"Composition analysis failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.warning("Analysis interrupted by user")
        raise typer.Exit(130) from None


def _analyze_single_image(
    image_path: Path,
    image_service: ImageService,
    composition_service: CompositionService,
    show_details: bool,
) -> None:
    """Analyze composition for a single image."""
    console.info(f"Analyzing composition: {image_path}")

    # Load image
    result = image_service.load_image(image_path)
    if not result.success or result.image is None:
        console.error(f"Failed to load image: {result.error}")
        return

    # Analyze composition
    composition_result = composition_service.analyze_image(result.image, image_path)

    if not composition_result.success:
        console.error(f"Composition analysis failed: {composition_result.error}")
        return

    # Display results
    _display_single_result(composition_result, show_details)


def _analyze_directory(
    directory: Path,
    image_service: ImageService,
    composition_service: CompositionService,
    show_details: bool,
    show_progress: bool,
    min_confidence: float | None,
    shot_type_filter: str | None,
    scene_filter: str | None,
) -> None:
    """Analyze composition for all images in a directory."""
    console.info(f"Scanning directory: {directory}")

    # Scan directory for images
    scan_result = image_service.scan_directory(directory, show_progress=show_progress)

    if scan_result.valid_images == 0:
        console.warning("No valid images found in directory")
        return

    console.info(f"Found {scan_result.valid_images} images to analyze")

    # Load images
    images_and_paths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console.console,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task(
            "Loading images...", total=len(scan_result.image_paths)
        )

        for image_path in scan_result.image_paths:
            result = image_service.load_image(image_path)
            if result.success and result.image is not None:
                images_and_paths.append((result.image, image_path))
            progress.advance(task)

    if not images_and_paths:
        console.error("No images could be loaded successfully")
        return

    console.info(f"Loaded {len(images_and_paths)} images")

    # Analyze composition in batch
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console.console,
        disable=not show_progress,
    ) as progress:
        progress.add_task("Analyzing image composition...", total=None)
        batch_result = composition_service.analyze_batch(images_and_paths)

    # Display results
    _display_batch_results(
        batch_result, show_details, min_confidence, shot_type_filter, scene_filter
    )


def _display_single_result(
    composition_result: "CompositionResult", show_details: bool
) -> None:
    """Display results for a single image composition analysis."""
    if not composition_result.success or not composition_result.analysis:
        console.error("Analysis failed")
        return

    analysis = composition_result.analysis

    # Create summary table
    table = Table(title="Composition Analysis Results")
    table.add_column("Aspect", style="cyan")
    table.add_column("Classification", style="magenta")
    table.add_column("Details", style="dim")

    # Core composition aspects
    table.add_row(
        "Shot Type",
        analysis.shot_type.value.replace("_", " ").title(),
        "How the subject is framed",
    )
    table.add_row(
        "Scene Type",
        analysis.scene_type.value.replace("_", " ").title(),
        "Environment and setting",
    )
    table.add_row(
        "Lighting Quality",
        analysis.lighting_quality.value.replace("_", " ").title(),
        "Overall lighting assessment",
    )
    table.add_row(
        "Background",
        analysis.background_complexity.value.replace("_", " ").title(),
        "Background complexity level",
    )

    # Optional aspects
    if analysis.facial_expression:
        table.add_row(
            "Expression",
            analysis.facial_expression.value.replace("_", " ").title(),
            "Facial expression detected",
        )

    if analysis.camera_angle:
        table.add_row(
            "Camera Angle",
            analysis.camera_angle.value.replace("_", " ").title(),
            "Camera positioning relative to subject",
        )

    # Confidence and metadata
    if analysis.confidence_score is not None:
        confidence_str = f"{analysis.confidence_score:.2f}"
        confidence_status = (
            "High"
            if analysis.confidence_score >= 0.8
            else "Medium" if analysis.confidence_score >= 0.6 else "Low"
        )
        table.add_row(
            "Confidence",
            confidence_str,
            f"{confidence_status} confidence in analysis",
        )

    console.console.print(table)

    if show_details:
        if composition_result.analysis_duration:
            console.info(f"Analysis time: {composition_result.analysis_duration:.3f}s")

        if analysis.raw_description:
            console.info(f"Description: {analysis.raw_description}")

        if (
            composition_result.model_response
            and len(composition_result.model_response) < 200
        ):
            console.info(f"Model response: {composition_result.model_response}")


def _display_batch_results(
    batch_result: "BatchCompositionResult",
    show_details: bool,
    min_confidence: float | None,
    shot_type_filter: str | None,
    scene_filter: str | None,
) -> None:
    """Display results for batch composition analysis."""
    # Apply filters
    display_results = batch_result.results
    if min_confidence is not None:
        display_results = [
            r
            for r in display_results
            if r.success
            and r.analysis
            and r.analysis.confidence_score is not None
            and r.analysis.confidence_score >= min_confidence
        ]

    if shot_type_filter:
        shot_filter = shot_type_filter.lower().replace(" ", "_")
        display_results = [
            r
            for r in display_results
            if r.success and r.analysis and r.analysis.shot_type.value == shot_filter
        ]

    if scene_filter:
        scene_filter_val = scene_filter.lower().replace(" ", "_")
        display_results = [
            r
            for r in display_results
            if r.success
            and r.analysis
            and r.analysis.scene_type.value == scene_filter_val
        ]

    # Display summary
    console.success("Composition analysis completed:")
    console.info(f"  • Successful: {batch_result.successful_analyses}")
    console.info(f"  • Failed: {batch_result.failed_analyses}")
    console.info(f"  • Mean confidence: {batch_result.mean_confidence:.3f}")
    console.info(
        f"  • Processing rate: {batch_result.images_per_second:.1f} images/sec"
    )

    if show_details:
        # Show composition distributions
        console.info("\\nComposition Distribution:")

        if batch_result.shot_type_distribution:
            console.info("  Shot Types:")
            for shot_type, count in sorted(batch_result.shot_type_distribution.items()):
                console.info(
                    f"    • {shot_type.value.replace('_', ' ').title()}: {count}"
                )

        if batch_result.scene_type_distribution:
            console.info("  Scene Types:")
            for scene_type, count in sorted(
                batch_result.scene_type_distribution.items()
            ):
                console.info(
                    f"    • {scene_type.value.replace('_', ' ').title()}: {count}"
                )

        if batch_result.confidence_distribution:
            console.info("  Confidence Distribution:")
            for percentile, score in sorted(
                batch_result.confidence_distribution.items()
            ):
                console.info(f"    • {percentile}: {score:.3f}")

    # Show individual results table
    filter_desc = []
    if min_confidence:
        filter_desc.append(f"confidence ≥ {min_confidence}")
    if shot_type_filter:
        filter_desc.append(f"shot type: {shot_type_filter}")
    if scene_filter:
        filter_desc.append(f"scene: {scene_filter}")

    if filter_desc:
        table_title = f"Composition Results ({', '.join(filter_desc)})"
    else:
        table_title = "Composition Results"

    table = Table(title=table_title)
    table.add_column("Image", style="cyan", max_width=30)
    table.add_column("Shot Type", style="magenta")
    table.add_column("Scene", style="green")
    table.add_column("Lighting", style="yellow")
    table.add_column("Background", style="blue")

    if show_details:
        table.add_column("Expression", style="red")
        table.add_column("Angle", style="white")
        table.add_column("Confidence", style="dim")

    # Sort by confidence (highest first)
    sorted_results = [
        r for r in display_results if r.success and r.analysis is not None
    ]
    sorted_results.sort(
        key=lambda r: (
            r.analysis.confidence_score
            if r.analysis and r.analysis.confidence_score
            else 0.0
        ),
        reverse=True,
    )

    # Limit display to avoid overwhelming output
    max_display = 20
    if len(sorted_results) > max_display:
        console.info(f"Showing top {max_display} results (use filters to narrow down)")
        sorted_results = sorted_results[:max_display]

    for result in sorted_results:
        if not result.success or not result.analysis:
            continue

        analysis = result.analysis

        row_data = [
            result.path.name,
            analysis.shot_type.value.replace("_", " ").title()[:12],
            analysis.scene_type.value.replace("_", " ").title()[:10],
            analysis.lighting_quality.value.replace("_", " ").title()[:10],
            analysis.background_complexity.value.replace("_", " ").title()[:10],
        ]

        if show_details:
            expression = (
                analysis.facial_expression.value.replace("_", " ").title()[:10]
                if analysis.facial_expression
                else "-"
            )
            angle = (
                analysis.camera_angle.value.replace("_", " ").title()[:10]
                if analysis.camera_angle
                else "-"
            )
            confidence = (
                f"{analysis.confidence_score:.2f}" if analysis.confidence_score else "-"
            )

            row_data.extend([expression, angle, confidence])

        table.add_row(*row_data)

    console.console.print(table)

    # Show failed analyses if any
    failed_results = [r for r in batch_result.results if not r.success]
    if failed_results and show_details:
        console.warning(f"\\nFailed analyses ({len(failed_results)}):")
        for result in failed_results[:5]:  # Show first 5 failures
            console.info(f"  • {result.path.name}: {result.error}")
        if len(failed_results) > 5:
            console.info(f"  • ... and {len(failed_results) - 5} more")


@composition_app.command("config")
def show_composition_config() -> None:
    """Display current composition analysis configuration."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        composition_config = config.composition

        table = Table(title="Composition Analysis Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Description", style="dim")

        # Model settings
        table.add_row(
            "Model Name",
            composition_config.model_name,
            "Vision-language model identifier",
        )
        table.add_row(
            "Device Preference",
            composition_config.device_preference,
            "Preferred device for model inference",
        )
        table.add_row(
            "Batch Size",
            str(composition_config.batch_size),
            "Batch size for model inference",
        )

        # Analysis toggles
        table.add_row(
            "Shot Type Analysis",
            str(composition_config.enable_shot_type_analysis),
            "Enable shot type classification",
        )
        table.add_row(
            "Scene Analysis",
            str(composition_config.enable_scene_analysis),
            "Enable scene type analysis",
        )
        table.add_row(
            "Lighting Analysis",
            str(composition_config.enable_lighting_analysis),
            "Enable lighting quality assessment",
        )
        table.add_row(
            "Background Analysis",
            str(composition_config.enable_background_analysis),
            "Enable background complexity analysis",
        )
        table.add_row(
            "Expression Analysis",
            str(composition_config.enable_expression_analysis),
            "Enable facial expression analysis",
        )
        table.add_row(
            "Angle Analysis",
            str(composition_config.enable_angle_analysis),
            "Enable camera angle analysis",
        )

        # Confidence and quality settings
        table.add_row(
            "Min Confidence",
            f"{composition_config.min_confidence_threshold:.2f}",
            "Minimum confidence threshold",
        )
        table.add_row(
            "Confidence Scoring",
            str(composition_config.enable_confidence_scoring),
            "Enable confidence scoring",
        )

        # Performance settings
        max_w, max_h = composition_config.max_image_size
        table.add_row(
            "Max Image Size",
            f"{max_w}x{max_h}",
            "Maximum image size for analysis",
        )
        table.add_row(
            "Model Caching",
            str(composition_config.enable_model_caching),
            "Cache loaded models in memory",
        )
        table.add_row(
            "Memory Optimization",
            str(composition_config.memory_optimization),
            "Enable memory optimization",
        )

        # Prompt settings
        table.add_row(
            "Structured Prompts",
            str(composition_config.use_structured_prompts),
            "Use structured prompts for consistency",
        )
        table.add_row(
            "Temperature",
            f"{composition_config.prompt_temperature:.2f}",
            "Model generation temperature",
        )

        console.console.print(table)

    except Exception as e:
        console.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1) from e
