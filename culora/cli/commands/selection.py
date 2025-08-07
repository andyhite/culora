"""CLI commands for multi-criteria image selection operations."""

import json
from pathlib import Path
from typing import Any

import typer
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from culora.cli.utils import handle_cli_error
from culora.core.exceptions import SelectionError
from culora.domain.models.config.selection import SelectionConfig
from culora.domain.models.selection import SelectionCandidate
from culora.services.config_service import get_config_service
from culora.services.selection_service import get_selection_service

app = typer.Typer(name="selection", help="Multi-criteria image selection operations")
console = Console()


@app.command("analyze")
def analyze_selection_candidates(
    input_path: Path = typer.Argument(
        ..., help="Directory containing candidate images with analysis data"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="JSON file to save analysis results"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Selection configuration file"
    ),
    target_count: int = typer.Option(
        50, "--count", "-n", help="Target number of images to select"
    ),
    min_quality: float = typer.Option(
        0.4, "--min-quality", help="Minimum quality threshold (0.0-1.0)"
    ),
    enable_diversity: bool = typer.Option(
        True, "--diversity/--no-diversity", help="Enable diversity optimization"
    ),
    enable_distribution: bool = typer.Option(
        True, "--distribution/--no-distribution", help="Enable distribution balancing"
    ),
    strategy: str = typer.Option(
        "multi_stage",
        "--strategy",
        help="Selection strategy: multi_stage, quality_first, diversity_first, balanced",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    fast_mode: bool = typer.Option(
        False,
        "--fast",
        help="Use quality-only analysis (faster but less comprehensive)",
    ),
) -> None:
    """Analyze selection candidates and show what would be selected.

    This command performs selection analysis without actually copying files,
    showing detailed statistics about the selection process and results.
    """

    try:
        with handle_cli_error():
            # Initialize configuration service
            config_service = get_config_service()
            try:
                # Load existing config or create default
                if config_file and config_file.exists():
                    config_service.load_config(config_file)
                else:
                    config_service.get_config()
            except Exception:
                # Create default config if none exists
                config_service.load_config()

            console.print(
                f"[bold blue]Analyzing selection candidates in:[/bold blue] {input_path}"
            )

            # Load selection configuration
            if config_file and config_file.exists():
                with open(config_file) as f:
                    config_data = json.load(f)
                config = SelectionConfig(**config_data)
            else:
                config = SelectionConfig(
                    target_count=target_count,
                    selection_strategy=strategy,
                )

                # Update config with CLI parameters
                config.quality_thresholds.min_composite_quality = min_quality
                config.diversity_settings.enable_pose_diversity = enable_diversity
                config.diversity_settings.enable_semantic_diversity = enable_diversity
                config.enable_distribution_enforcement = enable_distribution

            # Load selection candidates with appropriate analysis depth
            candidates = _load_selection_candidates(input_path, fast_mode)

            if not candidates:
                console.print("[red]No candidate images found with analysis data[/red]")
                raise typer.Exit(1)

            console.print(f"Found {len(candidates)} candidate images")

            # Perform selection analysis
            selection_service = get_selection_service()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Performing selection analysis...", total=None)

                result = selection_service.select_images(candidates, config)

                progress.remove_task(task)

            # Display results
            _display_selection_results(result)

            # Save results if requested
            if output_file:
                _save_selection_analysis(result, output_file)
                console.print(f"[green]Analysis saved to:[/green] {output_file}")

    except SelectionError as e:
        console.print(f"[red]Selection error:[/red] {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("select")
def select_images(
    input_path: Path = typer.Argument(
        ..., help="Directory containing candidate images with analysis data"
    ),
    output_path: Path = typer.Argument(..., help="Directory to save selected images"),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Selection configuration file"
    ),
    target_count: int = typer.Option(
        50, "--count", "-n", help="Target number of images to select"
    ),
    copy_mode: str = typer.Option(
        "selected", "--copy-mode", help="Copy mode: selected, all_with_status"
    ),
    generate_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Generate selection metadata"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be selected without copying"
    ),
) -> None:
    """Select and copy images using multi-criteria selection algorithms.

    Performs actual selection and copies chosen images to the output directory,
    with optional metadata generation and progress reporting.
    """

    try:
        with handle_cli_error():
            console.print(f"[bold blue]Selecting images from:[/bold blue] {input_path}")
            console.print(f"[bold blue]Output directory:[/bold blue] {output_path}")

            if dry_run:
                console.print("[yellow]DRY RUN MODE - No files will be copied[/yellow]")

            # Load configuration
            if config_file and config_file.exists():
                with open(config_file) as f:
                    config_data = json.load(f)
                config = SelectionConfig(**config_data)
            else:
                config = SelectionConfig(target_count=target_count)

            # Load candidates (using fast mode by default for select command)
            candidates = _load_selection_candidates(input_path, fast_mode=True)

            if not candidates:
                console.print("[red]No candidate images found[/red]")
                raise typer.Exit(1)

            # Perform selection
            selection_service = get_selection_service()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Performing image selection...", total=None)

                result = selection_service.select_images(candidates, config)

                progress.remove_task(task)

            # Display selection summary
            _display_selection_summary(result)

            if not dry_run:
                # Copy selected images
                _copy_selected_images(result, output_path, copy_mode, generate_metadata)
                console.print(
                    f"[green]Selection completed![/green] {len(result.selected_candidates)} images copied"
                )
            else:
                console.print(
                    f"[yellow]DRY RUN:[/yellow] Would copy {len(result.selected_candidates)} images"
                )

    except SelectionError as e:
        console.print(f"[red]Selection error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command("config")
def manage_selection_config(
    action: str = typer.Argument(..., help="Action: create, validate, show"),
    config_file: Path = typer.Option(
        Path("selection_config.json"), "--file", "-f", help="Configuration file path"
    ),
    target_count: int = typer.Option(50, "--count", help="Target selection count"),
    strategy: str = typer.Option(
        "multi_stage", "--strategy", help="Selection strategy"
    ),
) -> None:
    """Manage selection configuration files."""
    try:
        with handle_cli_error():
            if action == "create":
                _create_selection_config(config_file, target_count, strategy)
            elif action == "validate":
                _validate_selection_config(config_file)
            elif action == "show":
                _show_selection_config(config_file)
            else:
                console.print(f"[red]Unknown action:[/red] {action}")
                console.print("Available actions: create, validate, show")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from e


def _load_selection_candidates(
    input_path: Path, fast_mode: bool = False
) -> list[SelectionCandidate]:
    """Load selection candidates from input directory.

    This function loads images and performs analysis to create SelectionCandidate objects.

    Args:
        input_path: Directory containing images to analyze
        fast_mode: If True, only performs quality analysis (faster).
                  If False, performs comprehensive analysis (quality, composition, pose, semantic).
    """
    from rich.progress import Progress, TaskID

    from culora.services.clip_service import get_clip_service
    from culora.services.composition_service import get_composition_service
    from culora.services.image_service import get_image_service
    from culora.services.pose_service import get_pose_service
    from culora.services.quality_service import get_quality_service

    candidates: list[SelectionCandidate] = []

    # Get service instances
    image_service = get_image_service()
    quality_service = get_quality_service()
    composition_service = get_composition_service()
    pose_service = get_pose_service()
    clip_service = get_clip_service()
    # Note: duplicate_service would be used for duplicate detection in future enhancement

    # Load images from directory
    scan_result = image_service.scan_directory(input_path)
    image_paths = scan_result.image_paths

    if not image_paths:
        return candidates

    analysis_type = "quality-only" if fast_mode else "comprehensive"
    console.print(
        f"Found {len(image_paths)} images, performing {analysis_type} analysis..."
    )

    # Use progress bar for analysis
    with Progress(console=console) as progress:
        task: TaskID = progress.add_task("Analyzing images...", total=len(image_paths))

        for image_path in image_paths:
            try:
                # Get basic image info
                file_size = image_path.stat().st_size

                # Load image
                image = Image.open(image_path)

                # Always perform quality analysis
                quality_result = quality_service.analyze_image(image, image_path)
                if not quality_result.success or not quality_result.score:
                    progress.console.print(
                        f"[yellow]Skipping {image_path.name}: Quality analysis failed[/yellow]"
                    )
                    progress.advance(task)
                    continue

                # Initialize optional analysis results
                composition_result = None
                pose_result = None
                semantic_result = None

                # Perform additional analysis if not in fast mode
                if not fast_mode:
                    # Perform composition analysis
                    try:
                        composition_result = composition_service.analyze_image(
                            image, image_path
                        )
                        if not composition_result.success:
                            composition_result = None
                    except Exception:
                        composition_result = None

                    # Perform pose analysis
                    try:
                        pose_analysis_result = pose_service.analyze_pose(
                            image, image_path
                        )
                        if (
                            pose_analysis_result.success
                            and pose_analysis_result.pose_analysis
                        ):
                            pose_result = pose_analysis_result.pose_analysis
                        else:
                            pose_result = None
                    except Exception:
                        pose_result = None

                    # Perform semantic embedding
                    try:
                        semantic_analysis = clip_service.extract_embedding(
                            image, image_path
                        )
                        if semantic_analysis.success and semantic_analysis.embedding:
                            semantic_result = semantic_analysis.embedding
                        else:
                            semantic_result = None
                    except Exception:
                        semantic_result = None

                # Create candidate with available analysis data
                candidate = SelectionCandidate(
                    path=image_path,
                    file_size=file_size,
                    quality_assessment=quality_result.score,
                    composite_quality_score=quality_result.score.overall_score,
                    composition_analysis=(
                        composition_result.analysis if composition_result else None
                    ),
                    pose_analysis=pose_result if pose_result else None,
                    semantic_embedding=semantic_result if semantic_result else None,
                    duplicate_group_id=None,
                    is_duplicate_representative=False,
                )

                candidates.append(candidate)
                progress.advance(task)

            except Exception as e:
                progress.console.print(
                    f"[red]ERROR analyzing {image_path.name}: {type(e).__name__}: {e}[/red]"
                )
                progress.advance(task)
                continue

    console.print(f"[green]Successfully analyzed {len(candidates)} images[/green]")
    return candidates


def _display_selection_results(result: Any) -> None:
    """Display detailed selection results."""
    console.print("\n[bold]Selection Analysis Results[/bold]")

    # Selection overview
    overview_table = Table(title="Selection Overview")
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green")

    overview_table.add_row("Total Processed", str(result.total_processed))
    overview_table.add_row("Selected", str(result.selection_count))
    overview_table.add_row("Target Count", str(result.criteria.target_count))
    overview_table.add_row("Fulfillment", f"{result.target_fulfillment_ratio:.1%}")
    overview_table.add_row(
        "Quality Improvement", f"{result.quality_improvement_ratio:.2f}x"
    )
    overview_table.add_row("Processing Time", f"{result.total_duration:.2f}s")

    console.print(overview_table)

    # Stage results
    if result.stage_results:
        console.print("\n[bold]Selection Pipeline Stages[/bold]")

        stage_table = Table()
        stage_table.add_column("Stage", style="cyan")
        stage_table.add_column("Input", justify="right")
        stage_table.add_column("Output", justify="right")
        stage_table.add_column("Filtered", justify="right")
        stage_table.add_column("Retention", justify="right")
        stage_table.add_column("Duration", justify="right")

        for stage in result.stage_results:
            retention_pct = f"{stage.retention_ratio:.1%}"
            duration_str = f"{stage.duration:.2f}s"

            stage_table.add_row(
                stage.stage_name,
                str(stage.input_count),
                str(stage.output_count),
                str(stage.filtered_count),
                retention_pct,
                duration_str,
            )

        console.print(stage_table)

    # Quality distribution
    if result.quality_distribution:
        console.print("\n[bold]Quality Distribution[/bold]")

        quality_table = Table()
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="green")

        for metric, value in result.quality_distribution.items():
            if isinstance(value, float):
                quality_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}")
            else:
                quality_table.add_row(metric.replace("_", " ").title(), str(value))

        console.print(quality_table)

    # Composition distribution
    if result.composition_distribution:
        console.print("\n[bold]Composition Distribution[/bold]")

        dist_table = Table()
        dist_table.add_column("Category", style="cyan")
        dist_table.add_column("Target", justify="right")
        dist_table.add_column("Actual", justify="right")
        dist_table.add_column("Fulfillment", justify="right")

        for (
            category,
            target_count,
        ) in result.composition_distribution.target_counts.items():
            actual_count = result.composition_distribution.actual_counts.get(
                category, 0
            )
            fulfillment = result.composition_distribution.fulfillment_ratios.get(
                category, 0.0
            )

            dist_table.add_row(
                category,
                str(target_count),
                str(actual_count),
                f"{fulfillment:.1%}",
            )

        console.print(dist_table)

        console.print(
            f"Overall Distribution Score: {result.composition_distribution.overall_distribution_score:.2f}"
        )


def _display_selection_summary(result: Any) -> None:
    """Display concise selection summary."""
    console.print("\n[bold]Selection Summary[/bold]")

    summary_table = Table()
    summary_table.add_column("Selected", justify="center", style="green")
    summary_table.add_column("Target", justify="center")
    summary_table.add_column("Fulfillment", justify="center")
    summary_table.add_column("Avg Quality", justify="center")
    summary_table.add_column("Duration", justify="center")

    summary_table.add_row(
        str(result.selection_count),
        str(result.criteria.target_count),
        f"{result.target_fulfillment_ratio:.1%}",
        f"{result.average_selected_quality:.3f}",
        f"{result.total_duration:.1f}s",
    )

    console.print(summary_table)


def _save_selection_analysis(result: Any, output_file: Path) -> None:
    """Save selection analysis results to JSON file."""
    # Convert result to serializable format
    analysis_data = {
        "selection_overview": {
            "total_processed": result.total_processed,
            "selected_count": result.selection_count,
            "target_count": result.criteria.target_count,
            "fulfillment_ratio": result.target_fulfillment_ratio,
            "quality_improvement_ratio": result.quality_improvement_ratio,
            "processing_duration": result.total_duration,
        },
        "selected_images": [
            {
                "path": str(candidate.path),
                "quality_score": candidate.effective_quality_score,
                "selection_score": candidate.selection_score,
                "diversity_score": candidate.diversity_score,
            }
            for candidate in result.selected_candidates
        ],
        "quality_distribution": result.quality_distribution,
        "stage_results": [
            {
                "stage_name": stage.stage_name,
                "input_count": stage.input_count,
                "output_count": stage.output_count,
                "retention_ratio": stage.retention_ratio,
                "duration": stage.duration,
            }
            for stage in result.stage_results
        ],
    }

    # Add composition distribution if available
    if result.composition_distribution:
        analysis_data["composition_distribution"] = {
            "target_counts": result.composition_distribution.target_counts,
            "actual_counts": result.composition_distribution.actual_counts,
            "fulfillment_ratios": result.composition_distribution.fulfillment_ratios,
            "overall_score": result.composition_distribution.overall_distribution_score,
        }

    # Add diversity analysis if available
    if result.diversity_analysis:
        analysis_data["diversity_analysis"] = result.diversity_analysis

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analysis_data, f, indent=2)


def _copy_selected_images(
    result: Any, output_path: Path, _copy_mode: str, _generate_metadata: bool
) -> None:
    """Copy selected images to output directory."""
    # TODO: Implement actual file copying logic
    # This would handle:
    # - Creating output directory structure
    # - Copying selected images with sequential naming
    # - Generating metadata files
    # - Creating selection reports

    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]File copying not yet implemented[/yellow]")
    console.print(
        f"Would copy {len(result.selected_candidates)} images to {output_path}"
    )


def _create_selection_config(
    config_file: Path, target_count: int, strategy: str
) -> None:
    """Create a new selection configuration file."""
    config = SelectionConfig(
        target_count=target_count,
        selection_strategy=strategy,
    )

    config_data = config.model_dump()

    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    console.print(f"[green]Configuration created:[/green] {config_file}")


def _validate_selection_config(config_file: Path) -> None:
    """Validate a selection configuration file."""
    if not config_file.exists():
        console.print(f"[red]Configuration file not found:[/red] {config_file}")
        raise typer.Exit(1)

    try:
        with open(config_file) as f:
            config_data = json.load(f)

        config = SelectionConfig(**config_data)
        console.print(f"[green]Configuration is valid:[/green] {config_file}")

        # Show validation summary
        console.print(f"Target count: {config.target_count}")
        console.print(f"Selection strategy: {config.selection_strategy}")
        console.print(
            f"Quality threshold: {config.quality_thresholds.min_composite_quality}"
        )

    except Exception as e:
        console.print(f"[red]Configuration validation failed:[/red] {e}")
        raise typer.Exit(1) from e


def _show_selection_config(config_file: Path) -> None:
    """Display selection configuration details."""
    if not config_file.exists():
        console.print(f"[red]Configuration file not found:[/red] {config_file}")
        raise typer.Exit(1)

    try:
        with open(config_file) as f:
            config_data = json.load(f)

        config = SelectionConfig(**config_data)

        console.print(f"[bold]Selection Configuration:[/bold] {config_file}")

        # Create configuration tree
        tree = Tree("Configuration")

        # Basic settings
        basic = tree.add("Basic Settings")
        basic.add(f"Target Count: {config.target_count}")
        basic.add(f"Strategy: {config.selection_strategy}")
        basic.add(f"Max Selection Ratio: {config.max_selection_ratio}")

        # Quality settings
        quality = tree.add("Quality Thresholds")
        quality.add(
            f"Min Composite Quality: {config.quality_thresholds.min_composite_quality}"
        )
        quality.add(
            f"Min Technical Quality: {config.quality_thresholds.min_technical_quality}"
        )
        quality.add(
            f"Enable Quality Distribution: {config.quality_thresholds.enable_quality_distribution}"
        )

        # Diversity settings
        diversity = tree.add("Diversity Settings")
        diversity.add(
            f"Enable Pose Diversity: {config.diversity_settings.enable_pose_diversity}"
        )
        diversity.add(
            f"Enable Semantic Diversity: {config.diversity_settings.enable_semantic_diversity}"
        )
        diversity.add(f"Diversity Weight: {config.diversity_settings.diversity_weight}")
        diversity.add(
            f"Quality vs Diversity Balance: {config.diversity_settings.quality_vs_diversity_balance}"
        )

        # Distribution settings
        distribution = tree.add("Distribution Settings")
        distribution.add(
            f"Enable Distribution Enforcement: {config.enable_distribution_enforcement}"
        )
        distribution.add(
            f"Shot Type Balancing: {config.shot_type_distribution.enable_balancing}"
        )
        distribution.add(
            f"Scene Type Balancing: {config.scene_type_distribution.enable_balancing}"
        )

        console.print(tree)

    except Exception as e:
        console.print(f"[red]Failed to display configuration:[/red] {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
