"""CLI commands for duplicate detection and analysis."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from culora.domain.models import DuplicateConfig, HashAlgorithm
from culora.domain.models.duplicate import (
    DuplicateAnalysis,
    DuplicateGroup,
    DuplicateMatch,
    DuplicateRemovalStrategy,
    ImageHash,
)
from culora.services import (
    get_duplicate_service,
    get_image_service,
)

app = typer.Typer(
    name="duplicate",
    help="🔍 Duplicate detection and analysis commands",
    rich_markup_mode="rich",
)

console = Console()


def _create_duplicate_config(
    algorithm: HashAlgorithm,
    threshold: int,
    group_threshold: int,
    strategy: DuplicateRemovalStrategy,
) -> DuplicateConfig:
    """Create duplicate detection configuration."""
    from culora.domain.models.duplicate import DuplicateThreshold

    duplicate_threshold = DuplicateThreshold(
        hash_algorithm=algorithm,
        similarity_threshold=threshold,
        group_threshold=group_threshold,
    )

    return DuplicateConfig(threshold=duplicate_threshold, removal_strategy=strategy)


@app.command(name="analyze")
def analyze_duplicates(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing images to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    algorithm: HashAlgorithm = typer.Option(
        HashAlgorithm.PERCEPTUAL,
        "--algorithm",
        "-a",
        help="Perceptual hash algorithm to use",
    ),
    threshold: int = typer.Option(
        10,
        "--threshold",
        "-t",
        min=0,
        max=64,
        help="Hamming distance threshold for duplicates (0=identical, 64=different)",
    ),
    group_threshold: int = typer.Option(
        5,
        "--group-threshold",
        "-g",
        min=0,
        max=64,
        help="Hamming distance threshold for grouping similar images",
    ),
    strategy: DuplicateRemovalStrategy = typer.Option(
        DuplicateRemovalStrategy.KEEP_HIGHEST_QUALITY,
        "--strategy",
        "-s",
        help="Strategy for selecting representatives from duplicate groups",
    ),
    include_quality: bool = typer.Option(
        False,
        "--include-quality",
        "-q",
        help="Include quality analysis for better representative selection",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Export analysis results to JSON file"
    ),
    show_matches: bool = typer.Option(
        False, "--show-matches", help="Show detailed duplicate matches"
    ),
    show_groups: bool = typer.Option(
        False, "--show-groups", help="Show duplicate groups with representatives"
    ),
) -> None:
    """Analyze directory for duplicate images using perceptual hashing.

    This command scans a directory for images and identifies duplicates
    using perceptual hash algorithms. It can group similar images and
    select the best representative from each group.
    """
    console.print(
        Panel.fit(
            "[bold blue]🔍 Duplicate Detection Analysis[/bold blue]",
            border_style="blue",
        )
    )

    # Validate threshold relationships
    if group_threshold > threshold:
        console.print(
            "[red]Error: Group threshold must be <= similarity threshold[/red]"
        )
        raise typer.Exit(1)

    # Create configuration
    config = _create_duplicate_config(algorithm, threshold, group_threshold, strategy)

    # Get services
    image_service = get_image_service()
    duplicate_service = get_duplicate_service(config)

    try:
        # Scan directory for images
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            scan_task = progress.add_task(
                "Scanning directory for images...", total=None
            )
            scan_result = image_service.scan_directory(directory)
            progress.remove_task(scan_task)

        if not scan_result.image_paths:
            console.print("[yellow]No valid images found in directory[/yellow]")
            return

        console.print(
            f"Found [cyan]{len(scan_result.image_paths)}[/cyan] images to analyze"
        )

        # Get quality scores if requested
        quality_scores = None
        if include_quality:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                quality_task = progress.add_task(
                    "Calculating quality scores...", total=None
                )
                # TODO: Integrate with quality service properly
                # For now, skip quality integration
                # quality_service = get_quality_service()
                quality_scores = None
                progress.remove_task(quality_task)

        # Analyze duplicates
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            analysis_task = progress.add_task("Analyzing duplicates...", total=None)
            analysis = duplicate_service.analyze_duplicates(
                scan_result.image_paths, quality_scores=quality_scores
            )
            progress.remove_task(analysis_task)

        # Display results
        _display_analysis_summary(analysis)

        if show_matches and analysis.matches:
            _display_duplicate_matches(analysis.matches)

        if show_groups and analysis.duplicate_groups:
            _display_duplicate_groups(analysis.duplicate_groups)

        # Export results if requested
        if output_file:
            _export_analysis_results(analysis, output_file)
            console.print(f"Analysis results exported to [cyan]{output_file}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="hash")
def calculate_hashes(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing images to hash",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    algorithm: HashAlgorithm = typer.Option(
        HashAlgorithm.PERCEPTUAL,
        "--algorithm",
        "-a",
        help="Perceptual hash algorithm to use",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Export hash results to JSON file"
    ),
    show_hashes: bool = typer.Option(
        False, "--show-hashes", help="Display calculated hashes"
    ),
) -> None:
    """Calculate perceptual hashes for images in a directory.

    This command calculates perceptual hashes for all images in the
    specified directory using the chosen algorithm.
    """
    console.print(
        Panel.fit(
            "[bold green]🔐 Perceptual Hash Calculation[/bold green]",
            border_style="green",
        )
    )

    # Create minimal configuration for hashing
    from culora.domain.models.duplicate import DuplicateThreshold

    threshold = DuplicateThreshold(hash_algorithm=algorithm)
    config = DuplicateConfig(threshold=threshold)

    # Get services
    image_service = get_image_service()
    duplicate_service = get_duplicate_service(config)

    try:
        # Scan directory for images
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            scan_task = progress.add_task(
                "Scanning directory for images...", total=None
            )
            scan_result = image_service.scan_directory(directory)
            progress.remove_task(scan_task)

        if not scan_result.image_paths:
            console.print("[yellow]No valid images found in directory[/yellow]")
            return

        console.print(
            f"Found [cyan]{len(scan_result.image_paths)}[/cyan] images to hash"
        )

        # Calculate hashes
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            hash_task = progress.add_task("Calculating hashes...", total=None)
            hashes = duplicate_service.calculate_batch_hashes(scan_result.image_paths)
            progress.remove_task(hash_task)

        console.print(f"Successfully calculated [green]{len(hashes)}[/green] hashes")

        if show_hashes:
            _display_hashes(hashes)

        # Export hashes if requested
        if output_file:
            _export_hashes(hashes, output_file)
            console.print(f"Hash results exported to [cyan]{output_file}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during hash calculation: {e}[/red]")
        raise typer.Exit(1) from e


def _display_analysis_summary(analysis: DuplicateAnalysis) -> None:
    """Display duplicate analysis summary."""
    table = Table(title="📊 Duplicate Analysis Summary", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    table.add_column("Details", style="dim")

    table.add_row("Total Images", str(analysis.total_images), "Images processed")
    table.add_row(
        "Valid Hashes", str(analysis.total_hashes), "Successful hash calculations"
    )
    table.add_row(
        "Duplicate Matches", str(analysis.total_matches), "Similar image pairs found"
    )
    table.add_row(
        "Duplicate Groups", str(analysis.total_groups), "Groups of similar images"
    )
    table.add_row(
        "Exact Duplicates",
        str(analysis.exact_duplicates),
        "Identical images (distance=0)",
    )
    table.add_row(
        "Near Duplicates",
        str(analysis.near_duplicates),
        "Very similar images (distance≤5)",
    )
    table.add_row(
        "Unique Images", str(len(analysis.unique_images)), "Images with no duplicates"
    )
    table.add_row(
        "Duplicate Rate",
        f"{analysis.duplicate_rate:.1f}%",
        "Percentage of duplicate images",
    )
    table.add_row(
        "Reduction Rate",
        f"{analysis.reduction_rate:.1f}%",
        "Dataset size reduction possible",
    )
    table.add_row(
        "After Deduplication",
        str(analysis.images_after_deduplication),
        "Images remaining after cleanup",
    )

    console.print(table)


def _display_duplicate_matches(matches: list[DuplicateMatch]) -> None:
    """Display duplicate matches table."""
    table = Table(title="🔍 Duplicate Matches", show_header=True)
    table.add_column("Image 1", style="cyan", no_wrap=True)
    table.add_column("Image 2", style="cyan", no_wrap=True)
    table.add_column("Distance", style="yellow", justify="center")
    table.add_column("Similarity", style="green", justify="right")
    table.add_column("Type", style="magenta")

    for match in matches[:20]:  # Show first 20 matches
        match_type = (
            "Exact"
            if match.is_exact_duplicate
            else "Near" if match.is_near_duplicate else "Similar"
        )

        table.add_row(
            match.image1_path.name,
            match.image2_path.name,
            str(match.hamming_distance),
            f"{match.similarity_score:.3f}",
            match_type,
        )

    if len(matches) > 20:
        console.print(f"[dim]... and {len(matches) - 20} more matches[/dim]")

    console.print(table)


def _display_duplicate_groups(groups: list[DuplicateGroup]) -> None:
    """Display duplicate groups table."""
    table = Table(title="👥 Duplicate Groups", show_header=True)
    table.add_column("Group ID", style="cyan", no_wrap=True)
    table.add_column("Images", style="yellow", justify="center")
    table.add_column("Representative", style="green")
    table.add_column("Max Distance", style="red", justify="center")

    for group in groups:
        representative = (
            group.representative_path.name
            if group.has_representative and group.representative_path is not None
            else "None"
        )

        table.add_row(
            group.group_id,
            str(group.image_count),
            representative,
            str(group.max_distance),
        )

    console.print(table)


def _display_hashes(hashes: list[ImageHash]) -> None:
    """Display calculated hashes."""
    table = Table(title="🔐 Calculated Hashes", show_header=True)
    table.add_column("Image", style="cyan", no_wrap=True)
    table.add_column("Hash", style="yellow", no_wrap=False)
    table.add_column("Algorithm", style="green")
    table.add_column("Size", style="magenta", justify="right")

    for hash_obj in hashes[:20]:  # Show first 20 hashes
        table.add_row(
            hash_obj.image_path.name,
            hash_obj.hash_value,
            hash_obj.hash_algorithm.value,
            f"{hash_obj.hash_size} bits",
        )

    if len(hashes) > 20:
        console.print(f"[dim]... and {len(hashes) - 20} more hashes[/dim]")

    console.print(table)


def _export_analysis_results(analysis: DuplicateAnalysis, output_file: Path) -> None:
    """Export analysis results to JSON file."""
    data = {
        "summary": {
            "total_images": analysis.total_images,
            "total_hashes": analysis.total_hashes,
            "total_matches": analysis.total_matches,
            "total_groups": analysis.total_groups,
            "exact_duplicates": analysis.exact_duplicates,
            "near_duplicates": analysis.near_duplicates,
            "unique_images_count": len(analysis.unique_images),
            "duplicate_rate": analysis.duplicate_rate,
            "reduction_rate": analysis.reduction_rate,
            "images_after_deduplication": analysis.images_after_deduplication,
            "hash_algorithm": analysis.hash_algorithm.value,
            "similarity_threshold": analysis.threshold_config.similarity_threshold,
            "group_threshold": analysis.threshold_config.group_threshold,
        },
        "unique_images": [str(path) for path in analysis.unique_images],
        "duplicate_groups": [
            {
                "group_id": group.group_id,
                "image_paths": [str(path) for path in group.image_paths],
                "representative_path": (
                    str(group.representative_path) if group.has_representative else None
                ),
                "image_count": group.image_count,
                "max_distance": group.max_distance,
                "quality_scores": group.quality_scores,
            }
            for group in analysis.duplicate_groups
        ],
        "matches": [
            {
                "image1_path": str(match.image1_path),
                "image2_path": str(match.image2_path),
                "hamming_distance": match.hamming_distance,
                "similarity_score": match.similarity_score,
                "is_exact_duplicate": match.is_exact_duplicate,
                "is_near_duplicate": match.is_near_duplicate,
            }
            for match in analysis.matches
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def _export_hashes(hashes: list[ImageHash], output_file: Path) -> None:
    """Export hashes to JSON file."""
    data = {
        "hashes": [
            {
                "image_path": str(hash_obj.image_path),
                "hash_value": hash_obj.hash_value,
                "hash_algorithm": hash_obj.hash_algorithm.value,
                "hash_size": hash_obj.hash_size,
            }
            for hash_obj in hashes
        ],
        "total_hashes": len(hashes),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
