"""CLIP semantic embedding CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from culora.cli.display.console import console
from culora.domain.enums.clip import ClusteringMethod, SimilarityMetric
from culora.domain.models.clip import (
    BatchSemanticResult,
    ClusteringResult,
    DiversityAnalysis,
    SemanticAnalysisResult,
)
from culora.services.clip_service import CLIPService, CLIPServiceError
from culora.services.config_service import ConfigService
from culora.services.image_service import ImageService, ImageServiceError

# Create CLIP sub-application
clip_app = typer.Typer(
    name="clip",
    help="CLIP semantic embedding analysis commands",
    add_completion=False,
    rich_markup_mode="rich",
)


def _get_services() -> tuple[ImageService, CLIPService]:
    """Get configured service instances."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        image_service = ImageService(config)
        clip_service = CLIPService(config)
        return image_service, clip_service
    except Exception as e:
        console.error(f"Failed to initialize services: {e}")
        raise typer.Exit(1) from e


@clip_app.command("analyze")
def analyze_embeddings(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze")],
    show_details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed analysis results"),
    ] = False,
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show analysis progress")
    ] = True,
    enable_clustering: Annotated[
        bool,
        typer.Option(
            "--clustering/--no-clustering", help="Perform semantic clustering"
        ),
    ] = True,
    clustering_method: Annotated[
        ClusteringMethod | None,
        typer.Option("--method", help="Clustering method to use"),
    ] = None,
    similarity_threshold: Annotated[
        float | None,
        typer.Option(
            "--similarity", help="Similarity threshold for analysis", min=0.0, max=1.0
        ),
    ] = None,
    export_embeddings: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Export embeddings to JSON file"),
    ] = False,
) -> None:
    """Analyze semantic embeddings using CLIP models.

    Extracts CLIP embeddings from images and performs semantic analysis including
    similarity calculation, diversity scoring, and optional clustering.
    """
    try:
        if not path.exists():
            console.error(f"Path does not exist: {path}")
            raise typer.Exit(1)

        image_service, clip_service = _get_services()

        if path.is_file():
            _analyze_single_image(
                path, image_service, clip_service, show_details, export_embeddings
            )
        else:
            _analyze_directory(
                path,
                image_service,
                clip_service,
                show_details,
                show_progress,
                enable_clustering,
                clustering_method,
                similarity_threshold,
                export_embeddings,
            )

    except (ImageServiceError, CLIPServiceError) as e:
        console.error(f"CLIP analysis failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.warning("Analysis interrupted by user")
        raise typer.Exit(130) from None


@clip_app.command("similarity")
def calculate_similarity(
    image1: Annotated[Path, typer.Argument(help="First image file")],
    image2: Annotated[Path, typer.Argument(help="Second image file")],
    _metric: Annotated[
        SimilarityMetric,
        typer.Option("--metric", help="Similarity metric to use"),
    ] = SimilarityMetric.COSINE,
    show_embeddings: Annotated[
        bool,
        typer.Option("--embeddings/--no-embeddings", help="Show raw embeddings"),
    ] = False,
) -> None:
    """Calculate semantic similarity between two images.

    Uses CLIP embeddings to compute similarity scores between image pairs.
    """
    try:
        if not image1.exists() or not image2.exists():
            console.error("Both image files must exist")
            raise typer.Exit(1)

        image_service, clip_service = _get_services()

        # Load images
        result1 = image_service.load_image(image1)
        result2 = image_service.load_image(image2)

        if not (
            result1.success
            and result2.success
            and result1.image is not None
            and result2.image is not None
        ):
            console.error("Failed to load one or both images")
            raise typer.Exit(1)

        console.info("Extracting semantic embeddings...")

        # Extract embeddings
        embedding_result1 = clip_service.extract_embedding(result1.image, image1)
        embedding_result2 = clip_service.extract_embedding(result2.image, image2)

        if not (
            embedding_result1.success
            and embedding_result2.success
            and embedding_result1.embedding is not None
            and embedding_result2.embedding is not None
        ):
            console.error("Failed to extract embeddings from one or both images")
            raise typer.Exit(1)

        # Calculate similarity
        similarity = clip_service.calculate_similarity(
            embedding_result1.embedding,
            embedding_result2.embedding,
        )

        # Display results
        _display_similarity_result(similarity, show_embeddings)

    except (ImageServiceError, CLIPServiceError) as e:
        console.error(f"Similarity calculation failed: {e}")
        raise typer.Exit(1) from e


@clip_app.command("diversity")
def analyze_diversity(
    directory: Annotated[Path, typer.Argument(help="Directory containing images")],
    show_progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show analysis progress")
    ] = True,
    max_pairs: Annotated[
        int,
        typer.Option("--max-pairs", help="Maximum similarity pairs to show", min=1),
    ] = 10,
    export_results: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Export diversity analysis to JSON"),
    ] = False,
) -> None:
    """Analyze semantic diversity in a directory of images.

    Calculates pairwise similarities and diversity metrics for image collections.
    """
    try:
        if not directory.exists() or not directory.is_dir():
            console.error(f"Directory does not exist: {directory}")
            raise typer.Exit(1)

        image_service, clip_service = _get_services()

        console.info(f"Scanning directory: {directory}")

        # Scan directory for images
        scan_result = image_service.scan_directory(
            directory, show_progress=show_progress
        )

        if scan_result.valid_images == 0:
            console.warning("No valid images found in directory")
            return

        if scan_result.valid_images < 2:
            console.warning("Need at least 2 images for diversity analysis")
            return

        console.info(f"Found {scan_result.valid_images} images to analyze")

        # Load images and extract embeddings
        embeddings = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console.console,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task(
                "Extracting embeddings...", total=len(scan_result.image_paths)
            )

            for image_path in scan_result.image_paths:
                image_result = image_service.load_image(image_path)
                if image_result.success and image_result.image is not None:
                    embedding_result = clip_service.extract_embedding(
                        image_result.image, image_path
                    )
                    if (
                        embedding_result.success
                        and embedding_result.embedding is not None
                    ):
                        embeddings.append(embedding_result.embedding)
                progress.advance(task)

        if len(embeddings) < 2:
            console.error("Could not extract enough embeddings for diversity analysis")
            return

        console.info(f"Extracted {len(embeddings)} embeddings")

        # Analyze diversity
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console.console,
            disable=not show_progress,
        ) as progress:
            progress.add_task("Analyzing diversity...", total=None)
            diversity_analysis = clip_service.analyze_diversity(embeddings)

        # Display results
        _display_diversity_analysis(diversity_analysis, max_pairs)

        # Export if requested
        if export_results:
            export_path = directory / "diversity_analysis.json"
            with open(export_path, "w") as f:
                f.write(diversity_analysis.model_dump_json())
            console.success(f"Diversity analysis exported to {export_path}")

    except (ImageServiceError, CLIPServiceError) as e:
        console.error(f"Diversity analysis failed: {e}")
        raise typer.Exit(1) from e


@clip_app.command("config")
def show_clip_config() -> None:
    """Display current CLIP semantic embedding configuration."""
    try:
        config_service = ConfigService()
        config = config_service.load_config()
        clip_config = config.clip

        table = Table(title="CLIP Semantic Embedding Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Description", style="dim")

        # Model settings
        table.add_row(
            "Model Name",
            clip_config.model_name.value,
            "CLIP model variant",
        )
        table.add_row(
            "Device Preference",
            clip_config.device_preference,
            "Preferred device for model inference",
        )
        table.add_row(
            "Batch Size",
            str(clip_config.batch_size),
            "Batch size for embedding extraction",
        )

        # Embedding settings
        table.add_row(
            "Normalize Embeddings",
            str(clip_config.normalize_embeddings),
            "Normalize embeddings to unit vectors",
        )
        table.add_row(
            "Embedding Precision",
            clip_config.embedding_precision,
            "Floating point precision for embeddings",
        )
        table.add_row(
            "Enable Caching",
            str(clip_config.enable_embedding_cache),
            "Cache embeddings to avoid recomputation",
        )

        # Analysis settings
        table.add_row(
            "Similarity Metric",
            clip_config.similarity_metric.value,
            "Default similarity calculation method",
        )
        table.add_row(
            "Similarity Threshold",
            f"{clip_config.similarity_threshold:.2f}",
            "Default similarity threshold",
        )
        table.add_row(
            "Clustering Method",
            clip_config.clustering_method.value,
            "Default clustering algorithm",
        )

        # Performance settings
        max_w, max_h = clip_config.max_image_size
        table.add_row(
            "Max Image Size",
            f"{max_w}x{max_h}",
            "Maximum image size for processing",
        )
        table.add_row(
            "Memory Limit",
            f"{clip_config.memory_limit_mb} MB",
            "Memory limit for operations",
        )

        # Diversity settings
        table.add_row(
            "Diversity Weight",
            f"{clip_config.diversity_weight:.2f}",
            "Weight for diversity in selection",
        )
        table.add_row(
            "Quality Weight",
            f"{clip_config.quality_weight:.2f}",
            "Weight for quality in selection",
        )

        console.console.print(table)

    except Exception as e:
        console.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1) from e


def _analyze_single_image(
    image_path: Path,
    image_service: ImageService,
    clip_service: CLIPService,
    show_details: bool,
    export_embeddings: bool,
) -> None:
    """Analyze CLIP embedding for a single image."""
    console.info(f"Analyzing semantic embedding: {image_path}")

    # Load image
    result = image_service.load_image(image_path)
    if not result.success or result.image is None:
        console.error(f"Failed to load image: {result.error}")
        return

    # Extract embedding
    embedding_result = clip_service.extract_embedding(result.image, image_path)

    if not embedding_result.success:
        console.error(f"Embedding extraction failed: {embedding_result.error}")
        return

    # Display results
    _display_single_embedding_result(embedding_result, show_details)

    # Export if requested
    if export_embeddings and embedding_result.embedding:
        export_path = image_path.parent / f"{image_path.stem}_embedding.json"
        with open(export_path, "w") as f:
            f.write(embedding_result.embedding.model_dump_json())
        console.info(f"Embedding exported to {export_path}")


def _analyze_directory(
    directory: Path,
    image_service: ImageService,
    clip_service: CLIPService,
    show_details: bool,
    show_progress: bool,
    enable_clustering: bool,
    clustering_method: ClusteringMethod | None,
    _similarity_threshold: float | None,
    export_embeddings: bool,
) -> None:
    """Analyze CLIP embeddings for all images in a directory."""
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

    # Extract embeddings in batch
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console.console,
        disable=not show_progress,
    ) as progress:
        progress.add_task("Extracting semantic embeddings...", total=None)
        batch_result = clip_service.extract_batch_embeddings(images_and_paths)

    # Perform clustering if enabled
    clustering_result = None
    if enable_clustering and batch_result.successful_analyses >= 2:
        embeddings = [
            r.embedding
            for r in batch_result.results
            if r.success and r.embedding is not None
        ]

        if len(embeddings) >= 2:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console.console,
                disable=not show_progress,
            ) as progress:
                progress.add_task("Performing semantic clustering...", total=None)
                clustering_result = clip_service.cluster_embeddings(
                    embeddings, clustering_method
                )

    # Display results
    _display_batch_results(batch_result, clustering_result, show_details)

    # Export if requested
    if export_embeddings:
        export_path = directory / "semantic_embeddings.json"
        with open(export_path, "w") as f:
            f.write(batch_result.model_dump_json())
        console.success(f"Batch results exported to {export_path}")


def _display_single_embedding_result(
    result: SemanticAnalysisResult, show_details: bool
) -> None:
    """Display results for a single embedding analysis."""
    if not result.success or not result.embedding:
        console.error("Analysis failed")
        return

    embedding = result.embedding

    # Create summary table
    table = Table(title="Semantic Embedding Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Details", style="dim")

    table.add_row(
        "Model",
        embedding.model_name,
        "CLIP model used for extraction",
    )
    table.add_row(
        "Embedding Dimension",
        str(embedding.embedding_dimension),
        "Size of the embedding vector",
    )
    table.add_row(
        "Extraction Time",
        f"{embedding.extraction_time:.3f}s",
        "Time taken to extract embedding",
    )

    if embedding.confidence_score is not None:
        confidence_str = f"{embedding.confidence_score:.3f}"
        table.add_row(
            "Confidence",
            confidence_str,
            "Model confidence in embedding quality",
        )

    console.console.print(table)

    if show_details:
        # Show embedding statistics
        import numpy as np

        vec = np.array(embedding.embedding)
        console.info(f"Embedding norm: {np.linalg.norm(vec):.4f}")
        console.info(f"Mean value: {np.mean(vec):.4f}")
        console.info(f"Std deviation: {np.std(vec):.4f}")
        console.info(f"Sparsity: {np.mean(vec == 0.0):.2%}")


def _display_similarity_result(similarity: Any, _show_embeddings: bool) -> None:
    """Display similarity calculation results."""
    table = Table(title="Semantic Similarity Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Image 1", str(similarity.path1.name))
    table.add_row("Image 2", str(similarity.path2.name))
    table.add_row("Similarity Score", f"{similarity.similarity_score:.4f}")
    table.add_row("Distance", f"{similarity.distance:.4f}")
    table.add_row("Metric", similarity.metric.value)

    # Color-code similarity level
    if similarity.similarity_score >= 0.8:
        similarity_level = "[green]Very Similar[/green]"
    elif similarity.similarity_score >= 0.6:
        similarity_level = "[yellow]Moderately Similar[/yellow]"
    elif similarity.similarity_score >= 0.4:
        similarity_level = "[orange1]Somewhat Similar[/orange1]"
    else:
        similarity_level = "[red]Not Similar[/red]"

    table.add_row("Similarity Level", similarity_level)

    console.console.print(table)


def _display_diversity_analysis(analysis: DiversityAnalysis, max_pairs: int) -> None:
    """Display diversity analysis results."""
    # Summary table
    table = Table(title="Semantic Diversity Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="dim")

    table.add_row(
        "Total Images",
        str(analysis.total_images),
        "Number of images analyzed",
    )
    table.add_row(
        "Mean Similarity",
        f"{analysis.mean_pairwise_similarity:.4f}",
        "Average pairwise similarity",
    )
    table.add_row(
        "Diversity Score",
        f"{analysis.diversity_score:.4f}",
        "Overall diversity (1 - mean similarity)",
    )

    console.console.print(table)

    # Similarity distribution
    if analysis.similarity_distribution:
        console.info("\nSimilarity Distribution:")
        for percentile, score in sorted(analysis.similarity_distribution.items()):
            console.info(f"  • {percentile}: {score:.4f}")

    # Most similar pairs
    if analysis.most_similar_pairs:
        console.info(f"\nMost Similar Pairs (top {max_pairs}):")
        for sim in analysis.most_similar_pairs[-max_pairs:]:
            console.info(
                f"  • {sim.path1.name} ↔ {sim.path2.name}: {sim.similarity_score:.4f}"
            )

    # Most diverse pairs
    if analysis.most_diverse_pairs:
        console.info(f"\nMost Diverse Pairs (top {max_pairs}):")
        for sim in analysis.most_diverse_pairs[:max_pairs]:
            console.info(
                f"  • {sim.path1.name} ↔ {sim.path2.name}: {sim.similarity_score:.4f}"
            )


def _display_batch_results(
    batch_result: BatchSemanticResult,
    clustering_result: ClusteringResult | None,
    show_details: bool,
) -> None:
    """Display results for batch semantic analysis."""
    # Summary
    console.success("Semantic analysis completed:")
    console.info(f"  • Successful: {batch_result.successful_analyses}")
    console.info(f"  • Failed: {batch_result.failed_analyses}")
    console.info(
        f"  • Processing rate: {batch_result.embeddings_per_second:.1f} embeddings/sec"
    )
    console.info(f"  • Mean similarity: {batch_result.mean_similarity:.4f}")

    if show_details and batch_result.embedding_statistics:
        console.info("\nEmbedding Statistics:")
        stats = batch_result.embedding_statistics
        for key, value in stats.items():
            console.info(f"  • {key.replace('_', ' ').title()}: {value}")

    # Clustering results
    if clustering_result:
        console.info(f"\nClustering Results ({clustering_result.method.value}):")
        console.info(f"  • Number of clusters: {clustering_result.num_clusters}")
        console.info(f"  • Silhouette score: {clustering_result.silhouette_score:.4f}")
        console.info(f"  • Processing time: {clustering_result.processing_time:.3f}s")

        if show_details:
            console.info("  • Cluster sizes:")
            for cluster in clustering_result.clusters:
                console.info(
                    f"    - Cluster {cluster.cluster_id}: {cluster.size} images"
                )

    # Show failed analyses if any
    failed_results = [r for r in batch_result.results if not r.success]
    if failed_results and show_details:
        console.warning(f"\nFailed analyses ({len(failed_results)}):")
        for result in failed_results[:5]:  # Show first 5 failures
            console.info(f"  • {result.path.name}: {result.error}")
        if len(failed_results) > 5:
            console.info(f"  • ... and {len(failed_results) - 5} more")
