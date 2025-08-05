"""Quality assessment CLI commands."""

from pathlib import Path

# Import for type hints
from typing import TYPE_CHECKING, Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from culora.cli.display.console import console
from culora.services.config_service import ConfigService
from culora.services.image_service import ImageService, ImageServiceError
from culora.services.quality_service import QualityService, QualityServiceError

if TYPE_CHECKING:
    from culora.domain.models.quality import BatchQualityResult, ImageQualityResult

# Create quality sub-application
quality_app = typer.Typer(
    name="quality",
    help="Image quality assessment commands",
    add_completion=False,
    rich_markup_mode="rich",
)


def _get_services() -> tuple[ImageService, QualityService]:
    """Get configured service instances."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        image_service = ImageService(config)
        quality_service = QualityService(config)
        return image_service, quality_service
    except Exception as e:
        console.error(f"Failed to initialize services: {e}")
        raise typer.Exit(1) from e


@quality_app.command("analyze")
def analyze_quality(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze")],
    show_details: Annotated[
        bool, typer.Option("--details/--no-details", help="Show detailed metrics")
    ] = False,
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show analysis progress")
    ] = True,
    min_score: Annotated[
        float | None, typer.Option("--min-score", help="Minimum quality score filter")
    ] = None,
) -> None:
    """Analyze image quality metrics.

    Analyzes technical quality metrics including sharpness, brightness,
    contrast, color quality, and noise levels.
    """
    try:
        if not path.exists():
            console.error(f"Path does not exist: {path}")
            raise typer.Exit(1)

        image_service, quality_service = _get_services()

        if path.is_file():
            _analyze_single_image(path, image_service, quality_service, show_details)
        else:
            _analyze_directory(
                path,
                image_service,
                quality_service,
                show_details,
                show_progress,
                min_score,
            )

    except (ImageServiceError, QualityServiceError) as e:
        console.error(f"Quality analysis failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.warning("Analysis interrupted by user")
        raise typer.Exit(130) from None


def _analyze_single_image(
    image_path: Path,
    image_service: ImageService,
    quality_service: QualityService,
    show_details: bool,
) -> None:
    """Analyze quality for a single image."""
    console.info(f"Analyzing image: {image_path}")

    # Load image
    result = image_service.load_image(image_path)
    if not result.success or result.image is None:
        console.error(f"Failed to load image: {result.error}")
        return

    # Analyze quality
    quality_result = quality_service.analyze_image(result.image, image_path)

    if not quality_result.success:
        console.error(f"Quality analysis failed: {quality_result.error}")
        return

    # Display results
    _display_single_result(quality_result, show_details)


def _analyze_directory(
    directory: Path,
    image_service: ImageService,
    quality_service: QualityService,
    show_details: bool,
    show_progress: bool,
    min_score: float | None,
) -> None:
    """Analyze quality for all images in a directory."""
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

    # Analyze quality in batch
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console.console,
        disable=not show_progress,
    ) as progress:
        progress.add_task("Analyzing image quality...", total=None)
        batch_result = quality_service.analyze_batch(images_and_paths)

    # Display results
    _display_batch_results(batch_result, show_details, min_score)


def _display_single_result(
    quality_result: "ImageQualityResult", show_details: bool
) -> None:
    """Display results for a single image analysis."""
    if not quality_result.success or not quality_result.score:
        console.error("Analysis failed")
        return

    score = quality_result.score
    metrics = quality_result.metrics

    # Create summary table
    table = Table(title="Quality Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Details", style="dim")

    table.add_row(
        "Overall Quality",
        f"{score.overall_score:.3f}",
        "✓ Passes" if score.passes_threshold else "✗ Below threshold",
    )

    if show_details and metrics:
        table.add_row(
            "Sharpness",
            f"{metrics.sharpness:.3f}",
            f"Laplacian: {metrics.laplacian_variance:.1f}",
        )
        table.add_row(
            "Brightness",
            f"{metrics.brightness_score:.3f}",
            f"Mean: {metrics.mean_brightness:.3f}",
        )
        table.add_row(
            "Contrast",
            f"{metrics.contrast_score:.3f}",
            f"StdDev: {metrics.contrast_value:.3f}",
        )
        table.add_row(
            "Color Quality",
            f"{metrics.color_quality:.3f}",
            f"Saturation: {metrics.mean_saturation:.3f}",
        )
        table.add_row(
            "Noise Score",
            f"{metrics.noise_score:.3f}",
            f"Level: {metrics.noise_level:.1f}",
        )

    console.console.print(table)

    if show_details:
        console.info(f"Analysis time: {quality_result.analysis_duration:.3f}s")
        if metrics:
            console.info(
                f"Analysis size: {metrics.analysis_width}x{metrics.analysis_height}"
            )
            if metrics.was_resized:
                console.info("Image was resized for analysis")


def _display_batch_results(
    batch_result: "BatchQualityResult", show_details: bool, min_score: float | None
) -> None:
    """Display results for batch analysis."""
    # Filter results if min_score specified
    display_results = batch_result.results
    if min_score is not None:
        display_results = [
            r
            for r in batch_result.results
            if r.success and r.score and r.score.overall_score >= min_score
        ]

    console.success("Quality analysis completed:")
    console.info(f"  • Successful: {batch_result.successful_analyses}")
    console.info(f"  • Failed: {batch_result.failed_analyses}")
    console.info(f"  • Mean quality: {batch_result.mean_quality_score:.3f}")
    console.info(f"  • Median quality: {batch_result.median_quality_score:.3f}")
    console.info(f"  • Passing threshold: {batch_result.passing_threshold_count}")
    console.info(
        f"  • Processing rate: {batch_result.images_per_second:.1f} images/sec"
    )

    if show_details:
        # Show quality distribution
        console.info("\nQuality Score Distribution:")
        for percentile in [25, 50, 75, 90, 95]:
            if percentile in batch_result.scores_by_percentile:
                score = batch_result.scores_by_percentile[percentile]
                console.info(f"  • {percentile}th percentile: {score:.3f}")

    # Show individual results table
    if min_score is not None:
        table_title = f"Quality Results (minimum score: {min_score:.3f})"
    else:
        table_title = "Quality Results"

    table = Table(title=table_title)
    table.add_column("Image", style="cyan", max_width=40)
    table.add_column("Quality", style="magenta")
    table.add_column("Sharpness", style="green")
    table.add_column("Brightness", style="yellow")
    table.add_column("Contrast", style="blue")

    if show_details:
        table.add_column("Color", style="red")
        table.add_column("Noise", style="white")
        table.add_column("Status", style="dim")

    # Sort by quality score (highest first)
    sorted_results = [r for r in display_results if r.success and r.score is not None]
    sorted_results.sort(
        key=lambda r: r.score.overall_score if r.score else 0.0, reverse=True
    )

    # Limit to top results to avoid overwhelming output
    max_display = 20
    if len(sorted_results) > max_display:
        console.info(f"Showing top {max_display} results (use --min-score to filter)")
        sorted_results = sorted_results[:max_display]

    for result in sorted_results:
        if not result.success or not result.score or not result.metrics:
            continue

        assert result.score is not None  # For mypy
        assert result.metrics is not None  # For mypy
        result_score = result.score
        result_metrics = result.metrics

        status = "✓" if result_score.passes_threshold else "✗"

        row_data = [
            result.path.name,
            f"{result_score.overall_score:.3f}",
            f"{result_metrics.sharpness:.3f}",
            f"{result_metrics.brightness_score:.3f}",
            f"{result_metrics.contrast_score:.3f}",
        ]

        if show_details:
            row_data.extend(
                [
                    f"{result_metrics.color_quality:.3f}",
                    f"{result_metrics.noise_score:.3f}",
                    status,
                ]
            )

        table.add_row(*row_data)

    console.console.print(table)

    # Show failed analyses if any
    failed_results = [r for r in batch_result.results if not r.success]
    if failed_results and show_details:
        console.warning(f"\nFailed analyses ({len(failed_results)}):")
        for result in failed_results[:5]:  # Show first 5 failures
            console.info(f"  • {result.path.name}: {result.error}")
        if len(failed_results) > 5:
            console.info(f"  • ... and {len(failed_results) - 5} more")


@quality_app.command("config")
def show_quality_config() -> None:
    """Display current quality assessment configuration."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        quality_config = config.quality

        table = Table(title="Quality Assessment Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Description", style="dim")

        # Scoring weights
        table.add_row(
            "Sharpness Weight",
            f"{quality_config.sharpness_weight:.2f}",
            "Weight for sharpness in composite score",
        )
        table.add_row(
            "Brightness Weight",
            f"{quality_config.brightness_weight:.2f}",
            "Weight for brightness in composite score",
        )
        table.add_row(
            "Contrast Weight",
            f"{quality_config.contrast_weight:.2f}",
            "Weight for contrast in composite score",
        )
        table.add_row(
            "Color Weight",
            f"{quality_config.color_weight:.2f}",
            "Weight for color quality in composite score",
        )
        table.add_row(
            "Noise Weight",
            f"{quality_config.noise_weight:.2f}",
            "Weight for noise score in composite score",
        )

        # Thresholds
        table.add_row(
            "Min Quality Score",
            f"{quality_config.min_quality_score:.2f}",
            "Minimum score for selection",
        )
        table.add_row(
            "High Contrast Threshold",
            f"{quality_config.high_contrast_threshold:.2f}",
            "Threshold for high contrast score",
        )

        # Ranges
        bright_min, bright_max = quality_config.optimal_brightness_range
        table.add_row(
            "Optimal Brightness",
            f"{bright_min:.2f} - {bright_max:.2f}",
            "Optimal brightness range",
        )
        table.add_row(
            "Min Saturation",
            f"{quality_config.min_saturation:.2f}",
            "Minimum saturation for good color",
        )
        table.add_row(
            "Max Saturation",
            f"{quality_config.max_saturation:.2f}",
            "Maximum saturation before penalty",
        )

        # Performance settings
        table.add_row(
            "Resize for Analysis",
            str(quality_config.resize_for_analysis),
            "Resize large images for speed",
        )
        if quality_config.resize_for_analysis:
            max_w, max_h = quality_config.max_analysis_size
            table.add_row(
                "Max Analysis Size", f"{max_w}x{max_h}", "Maximum size for analysis"
            )

        console.console.print(table)

    except Exception as e:
        console.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1) from e
