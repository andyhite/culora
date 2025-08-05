"""Pose analysis CLI commands."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from culora.cli.display.console import console
from culora.cli.validation.validators import validate_config_file, validate_output_file
from culora.core.exceptions import CuLoRAError
from culora.services.config_service import get_config_service
from culora.services.image_service import get_image_service
from culora.services.pose_service import get_pose_service

pose_app = typer.Typer(
    name="pose",
    help="Pose estimation and analysis commands",
    no_args_is_help=True,
)


@pose_app.command("analyze")
def analyze_pose(
    path: Annotated[
        Path,
        typer.Argument(help="Path to image file or directory to analyze"),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
            callback=validate_config_file,
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results (JSON format)",
            callback=validate_output_file,
        ),
    ] = None,
    min_pose_score: Annotated[
        float | None,
        typer.Option(
            "--min-score",
            help="Minimum pose score filter (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    category_filter: Annotated[
        str | None,
        typer.Option(
            "--category",
            help="Filter by pose category (standing, sitting, etc.)",
        ),
    ] = None,
    show_details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Show detailed pose analysis",
        ),
    ] = False,
    export_landmarks: Annotated[
        bool,
        typer.Option(
            "--landmarks",
            help="Export pose landmarks data",
        ),
    ] = False,
) -> None:
    """Analyze poses in images using MediaPipe."""
    try:
        # Load configuration
        config_service = get_config_service()
        config = (
            config_service.load_config(config_file)
            if config_file
            else config_service.get_config()
        )

        # Override pose settings if provided
        if min_pose_score is not None:
            config.pose.min_pose_score = min_pose_score

        # Get services
        image_service = get_image_service()
        pose_service = get_pose_service(config)

        console.header("🤸 Pose Analysis")

        if path.is_file():
            # Analyze single image
            console.info(f"Analyzing pose in: {path}")

            # Load image
            load_result = image_service.load_image(path)
            if not load_result.success or load_result.image is None:
                console.error(f"Failed to load image: {load_result.error}")
                raise typer.Exit(1)

            # Analyze pose
            result = pose_service.analyze_pose(load_result.image, path)

            if result.success and result.pose_analysis:
                _display_single_pose_result(result, show_details)

                # Apply filters
                pose_analysis = result.pose_analysis
                if (
                    category_filter
                    and pose_analysis.classification.category != category_filter.lower()
                ):
                    console.warning(
                        f"Pose category '{pose_analysis.classification.category}' doesn't match filter '{category_filter}'"
                    )
                    return

                # Export if requested
                if output_file:
                    _export_pose_results([result], output_file, export_landmarks)
                    console.success(f"Results exported to: {output_file}")
            else:
                console.error(f"Pose analysis failed: {result.error}")
                raise typer.Exit(1)

        elif path.is_dir():
            # Analyze directory
            console.info(f"Scanning directory: {path}")

            # Scan for images
            scan_result = image_service.scan_directory(path)
            if not scan_result.image_paths:
                console.warning("No images found in directory")
                return

            console.info(f"Found {len(scan_result.image_paths)} images")

            # Load images
            images_and_paths = []
            for img_path in scan_result.image_paths:
                load_result = image_service.load_image(img_path)
                if load_result.success and load_result.image is not None:
                    images_and_paths.append((load_result.image, img_path))

            if not images_and_paths:
                console.error("No images could be loaded")
                raise typer.Exit(1)

            # Analyze poses
            with console.status(
                f"Analyzing poses in {len(images_and_paths)} images..."
            ):
                batch_result = pose_service.analyze_batch_poses(images_and_paths)

            # Filter results
            filtered_results = []
            for result in batch_result.results:
                if not result.success or not result.pose_analysis:
                    continue

                pose_analysis = result.pose_analysis

                # Apply filters
                if (
                    category_filter
                    and pose_analysis.classification.category != category_filter.lower()
                ):
                    continue

                filtered_results.append(result)

            # Display results
            _display_batch_pose_results(batch_result, filtered_results, show_details)

            # Export if requested
            if output_file:
                export_results = (
                    filtered_results if filtered_results else batch_result.results
                )
                _export_pose_results(export_results, output_file, export_landmarks)
                console.success(f"Results exported to: {output_file}")

        else:
            console.error(f"Path does not exist: {path}")
            raise typer.Exit(1)

    except CuLoRAError as e:
        console.error(f"Pose analysis error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@pose_app.command("diversity")
def analyze_diversity(
    path: Annotated[
        Path,
        typer.Argument(help="Path to directory containing images"),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
            callback=validate_config_file,
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results (JSON format)",
            callback=validate_output_file,
        ),
    ] = None,
    min_pose_score: Annotated[
        float | None,
        typer.Option(
            "--min-score",
            help="Minimum pose score filter (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    show_pairs: Annotated[
        bool,
        typer.Option(
            "--pairs",
            help="Show most similar and diverse pose pairs",
        ),
    ] = False,
) -> None:
    """Analyze pose diversity in a dataset."""
    try:
        if not path.is_dir():
            console.error("Path must be a directory")
            raise typer.Exit(1)

        # Load configuration
        config_service = get_config_service()
        config = (
            config_service.load_config(config_file)
            if config_file
            else config_service.get_config()
        )

        # Override settings if provided
        if min_pose_score is not None:
            config.pose.min_pose_score = min_pose_score

        # Get services
        image_service = get_image_service()
        pose_service = get_pose_service(config)

        console.header("📊 Pose Diversity Analysis")

        # Scan and load images
        scan_result = image_service.scan_directory(path)
        if not scan_result.image_paths:
            console.warning("No images found in directory")
            return

        console.info(f"Found {len(scan_result.image_paths)} images")

        # Load images and analyze poses
        images_and_paths = []
        for img_path in scan_result.image_paths:
            load_result = image_service.load_image(img_path)
            if load_result.success and load_result.image is not None:
                images_and_paths.append((load_result.image, img_path))

        if not images_and_paths:
            console.error("No images could be loaded")
            raise typer.Exit(1)

        # Analyze poses
        with console.status(f"Analyzing poses in {len(images_and_paths)} images..."):
            batch_result = pose_service.analyze_batch_poses(images_and_paths)

        # Get successful poses
        successful_poses = [
            r.pose_analysis
            for r in batch_result.results
            if r.success and r.pose_analysis
        ]

        if len(successful_poses) < 2:
            console.error(
                "Need at least 2 successful pose analyses for diversity analysis"
            )
            raise typer.Exit(1)

        # Analyze diversity
        with console.status("Calculating pose diversity..."):
            diversity_result = pose_service.analyze_pose_diversity(successful_poses)

        # Display results
        _display_diversity_results(diversity_result, show_pairs)

        # Export if requested
        if output_file:
            _export_diversity_results(diversity_result, output_file)
            console.success(f"Diversity analysis exported to: {output_file}")

    except CuLoRAError as e:
        console.error(f"Diversity analysis error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@pose_app.command("cluster")
def cluster_poses(
    path: Annotated[
        Path,
        typer.Argument(help="Path to directory containing images"),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
            callback=validate_config_file,
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results (JSON format)",
            callback=validate_output_file,
        ),
    ] = None,
    max_clusters: Annotated[
        int | None,
        typer.Option(
            "--max-clusters",
            help="Maximum number of clusters",
            min=2,
            max=50,
        ),
    ] = None,
    min_pose_score: Annotated[
        float | None,
        typer.Option(
            "--min-score",
            help="Minimum pose score filter (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
) -> None:
    """Cluster poses based on similarity."""
    try:
        if not path.is_dir():
            console.error("Path must be a directory")
            raise typer.Exit(1)

        # Load configuration
        config_service = get_config_service()
        config = (
            config_service.load_config(config_file)
            if config_file
            else config_service.get_config()
        )

        # Override settings if provided
        if max_clusters is not None:
            config.pose.max_clusters = max_clusters
        if min_pose_score is not None:
            config.pose.min_pose_score = min_pose_score

        # Get services
        image_service = get_image_service()
        pose_service = get_pose_service(config)

        console.header("🎯 Pose Clustering")

        # Scan and load images
        scan_result = image_service.scan_directory(path)
        if not scan_result.image_paths:
            console.warning("No images found in directory")
            return

        console.info(f"Found {len(scan_result.image_paths)} images")

        # Load images and analyze poses
        images_and_paths = []
        for img_path in scan_result.image_paths:
            load_result = image_service.load_image(img_path)
            if load_result.success and load_result.image is not None:
                images_and_paths.append((load_result.image, img_path))

        if not images_and_paths:
            console.error("No images could be loaded")
            raise typer.Exit(1)

        # Analyze poses
        with console.status(f"Analyzing poses in {len(images_and_paths)} images..."):
            batch_result = pose_service.analyze_batch_poses(images_and_paths)

        # Get successful poses
        successful_poses = [
            r.pose_analysis
            for r in batch_result.results
            if r.success and r.pose_analysis
        ]

        if len(successful_poses) < 2:
            console.error("Need at least 2 successful pose analyses for clustering")
            raise typer.Exit(1)

        # Perform clustering
        with console.status("Clustering poses..."):
            clustering_result = pose_service.cluster_poses(successful_poses)

        # Display results
        _display_clustering_results(clustering_result)

        # Export if requested
        if output_file:
            _export_clustering_results(clustering_result, output_file)
            console.success(f"Clustering results exported to: {output_file}")

    except CuLoRAError as e:
        console.error(f"Clustering error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


@pose_app.command("config")
def show_pose_config(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path",
            callback=validate_config_file,
        ),
    ] = None,
) -> None:
    """Show pose analysis configuration."""
    try:
        config_service = get_config_service()
        config = (
            config_service.load_config(config_file)
            if config_file
            else config_service.get_config()
        )

        console.header("⚙️  Pose Configuration")

        table = Table(title="Pose Analysis Settings")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="bright_white")
        table.add_column("Description", style="dim")

        # MediaPipe settings
        table.add_row(
            "Model Complexity",
            str(config.pose.model_complexity),
            "MediaPipe model complexity (0-2)",
        )
        table.add_row(
            "Min Detection Confidence",
            f"{config.pose.min_detection_confidence:.2f}",
            "Minimum pose detection confidence",
        )
        table.add_row(
            "Min Tracking Confidence",
            f"{config.pose.min_tracking_confidence:.2f}",
            "Minimum pose tracking confidence",
        )
        table.add_row(
            "Enable Segmentation",
            str(config.pose.enable_segmentation),
            "Enable pose segmentation mask",
        )

        # Processing settings
        table.add_row(
            "Max Image Size",
            f"{config.pose.max_image_size[0]}x{config.pose.max_image_size[1]}",
            "Maximum image size for processing",
        )
        table.add_row(
            "Batch Size", str(config.pose.batch_size), "Batch size for pose analysis"
        )
        table.add_row(
            "Enable Cache",
            str(config.pose.enable_pose_cache),
            "Cache pose analysis results",
        )

        # Quality settings
        table.add_row(
            "Min Pose Score",
            f"{config.pose.min_pose_score:.2f}",
            "Minimum pose quality score",
        )
        table.add_row(
            "Min Visible Landmarks",
            str(config.pose.min_visible_landmarks),
            "Minimum visible landmarks required",
        )
        table.add_row(
            "Min Landmark Confidence",
            f"{config.pose.min_landmark_confidence:.2f}",
            "Minimum individual landmark confidence",
        )

        # Clustering settings
        table.add_row(
            "Max Clusters",
            str(config.pose.max_clusters),
            "Maximum number of pose clusters",
        )
        table.add_row(
            "Min Cluster Size",
            str(config.pose.min_cluster_size),
            "Minimum images per cluster",
        )
        table.add_row(
            "Auto Clustering",
            str(config.pose.enable_auto_clustering),
            "Automatically determine cluster count",
        )

        console.print(table)

    except CuLoRAError as e:
        console.error(f"Configuration error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.error(f"Unexpected error: {e}")
        raise typer.Exit(1) from e


# Helper functions for display


def _display_single_pose_result(result: Any, show_details: bool) -> None:
    """Display single pose analysis result."""
    if not result.pose_analysis:
        return

    pose = result.pose_analysis

    console.success(f"✅ Pose detected (score: {pose.pose_score:.3f})")

    # Basic info table
    table = Table(title="Pose Analysis Results")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="bright_white")

    table.add_row("Category", pose.classification.category.title())
    table.add_row("Orientation", pose.classification.orientation.title())
    table.add_row(
        "Arm Position", pose.classification.arm_position.replace("_", " ").title()
    )
    table.add_row(
        "Leg Position", pose.classification.leg_position.replace("_", " ").title()
    )
    table.add_row("Symmetry", pose.classification.symmetry.title())
    table.add_row("Dynamism", pose.classification.dynamism.title())
    table.add_row("Classification Confidence", f"{pose.classification.confidence:.3f}")
    table.add_row("Landmarks Detected", str(len(pose.landmarks)))
    table.add_row("Pose Vector Confidence", f"{pose.pose_vector.confidence:.3f}")
    table.add_row("Analysis Duration", f"{pose.analysis_duration:.3f}s")

    console.print(table)

    if show_details:
        # Landmark details
        landmark_table = Table(title="Top 10 Landmarks")
        landmark_table.add_column("ID", style="dim")
        landmark_table.add_column("X", style="cyan")
        landmark_table.add_column("Y", style="cyan")
        landmark_table.add_column("Visibility", style="green")

        for i, landmark in enumerate(pose.landmarks[:10]):
            landmark_table.add_row(
                str(i),
                f"{landmark.x:.3f}",
                f"{landmark.y:.3f}",
                f"{landmark.visibility:.3f}",
            )

        console.print(landmark_table)


def _display_batch_pose_results(
    batch_result: Any, filtered_results: Any, show_details: bool
) -> None:
    """Display batch pose analysis results."""
    console.info("📊 Batch Analysis Complete")
    console.print(f"• Total images: {len(batch_result.results)}")
    console.print(f"• Successful analyses: {batch_result.successful_analyses}")
    console.print(f"• Failed analyses: {batch_result.failed_analyses}")
    console.print(
        f"• Processing rate: {batch_result.poses_per_second:.2f} poses/second"
    )
    console.print(f"• Mean pose score: {batch_result.mean_pose_score:.3f}")

    if filtered_results:
        console.print(f"• Filtered results: {len(filtered_results)}")

    if show_details and filtered_results:
        # Category distribution
        categories: dict[str, int] = {}
        orientations: dict[str, int] = {}

        for result in filtered_results:
            if result.pose_analysis:
                cat = result.pose_analysis.classification.category
                categories[cat] = categories.get(cat, 0) + 1

                orient = result.pose_analysis.classification.orientation
                orientations[orient] = orientations.get(orient, 0) + 1

        if categories:
            cat_table = Table(title="Pose Category Distribution")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="bright_white")
            cat_table.add_column("Percentage", style="green")

            total = len(filtered_results)
            for cat, count in sorted(categories.items()):
                percentage = (count / total) * 100
                cat_table.add_row(cat.title(), str(count), f"{percentage:.1f}%")

            console.print(cat_table)


def _display_diversity_results(diversity_result: Any, show_pairs: bool) -> None:
    """Display pose diversity analysis results."""
    console.success(f"📈 Diversity Score: {diversity_result.diversity_score:.3f}")
    console.print(f"• Total images analyzed: {diversity_result.total_images}")
    console.print(
        f"• Mean pairwise similarity: {diversity_result.mean_pairwise_similarity:.3f}"
    )

    # Distribution tables
    cat_table = Table(title="Pose Category Distribution")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", style="bright_white")
    cat_table.add_column("Percentage", style="green")

    total = diversity_result.total_images
    for cat, count in sorted(diversity_result.category_distribution.items()):
        percentage = (count / total) * 100
        cat_table.add_row(cat.title(), str(count), f"{percentage:.1f}%")

    console.print(cat_table)

    # Similarity distribution
    sim_table = Table(title="Similarity Distribution")
    sim_table.add_column("Percentile", style="cyan")
    sim_table.add_column("Similarity", style="bright_white")

    for percentile, similarity in diversity_result.similarity_distribution.items():
        sim_table.add_row(percentile.upper(), f"{similarity:.3f}")

    console.print(sim_table)

    if show_pairs:
        # Most similar pairs
        if diversity_result.most_similar_pairs:
            similar_table = Table(title="Most Similar Pose Pairs")
            similar_table.add_column("Image 1", style="cyan")
            similar_table.add_column("Image 2", style="cyan")
            similar_table.add_column("Similarity", style="red")

            for pair in diversity_result.most_similar_pairs[:5]:
                similar_table.add_row(
                    pair.path1.name, pair.path2.name, f"{pair.similarity_score:.3f}"
                )

            console.print(similar_table)

        # Most diverse pairs
        if diversity_result.most_diverse_pairs:
            diverse_table = Table(title="Most Diverse Pose Pairs")
            diverse_table.add_column("Image 1", style="cyan")
            diverse_table.add_column("Image 2", style="cyan")
            diverse_table.add_column("Similarity", style="green")

            for pair in diversity_result.most_diverse_pairs[:5]:
                diverse_table.add_row(
                    pair.path1.name, pair.path2.name, f"{pair.similarity_score:.3f}"
                )

            console.print(diverse_table)


def _display_clustering_results(clustering_result: Any) -> None:
    """Display pose clustering results."""
    console.success(f"🎯 Found {clustering_result.num_clusters} pose clusters")
    console.print(f"• Silhouette score: {clustering_result.silhouette_score:.3f}")
    console.print(f"• Processing time: {clustering_result.processing_time:.2f}s")

    # Cluster details
    cluster_table = Table(title="Pose Clusters")
    cluster_table.add_column("Cluster", style="cyan")
    cluster_table.add_column("Size", style="bright_white")
    cluster_table.add_column("Dominant Category", style="green")
    cluster_table.add_column("Intra-Similarity", style="yellow")

    for cluster in clustering_result.clusters:
        cluster_table.add_row(
            str(cluster.cluster_id),
            str(cluster.size),
            cluster.dominant_category.title(),
            f"{cluster.intra_cluster_similarity:.3f}",
        )

    console.print(cluster_table)


def _export_pose_results(
    results: Any, output_file: Path, export_landmarks: bool
) -> None:
    """Export pose analysis results to JSON."""
    export_data: dict[str, Any] = {
        "results": [],
        "summary": {
            "total_images": len(results),
            "successful_analyses": sum(1 for r in results if r.success),
            "failed_analyses": sum(1 for r in results if not r.success),
        },
    }

    for result in results:
        result_data = {
            "path": str(result.path),
            "success": result.success,
            "analysis_duration": result.analysis_duration,
        }

        if result.success and result.pose_analysis:
            pose = result.pose_analysis
            result_data.update(
                {
                    "pose_score": pose.pose_score,
                    "classification": {
                        "category": pose.classification.category,
                        "orientation": pose.classification.orientation,
                        "arm_position": pose.classification.arm_position,
                        "leg_position": pose.classification.leg_position,
                        "symmetry": pose.classification.symmetry,
                        "dynamism": pose.classification.dynamism,
                        "confidence": pose.classification.confidence,
                    },
                    "pose_vector": {
                        "dimension": pose.pose_vector.vector_dimension,
                        "confidence": pose.pose_vector.confidence,
                    },
                    "bbox": pose.bbox,
                    "landmark_count": len(pose.landmarks),
                }
            )

            if export_landmarks:
                result_data["landmarks"] = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                        "presence": lm.presence,
                    }
                    for lm in pose.landmarks
                ]
        else:
            result_data.update(
                {
                    "error": result.error,
                    "error_code": result.error_code,
                }
            )

        export_data["results"].append(result_data)

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)


def _export_diversity_results(diversity_result: Any, output_file: Path) -> None:
    """Export diversity analysis results to JSON."""
    export_data = {
        "diversity_analysis": {
            "total_images": diversity_result.total_images,
            "diversity_score": diversity_result.diversity_score,
            "mean_pairwise_similarity": diversity_result.mean_pairwise_similarity,
            "category_distribution": diversity_result.category_distribution,
            "orientation_distribution": diversity_result.orientation_distribution,
            "similarity_distribution": diversity_result.similarity_distribution,
        },
        "most_similar_pairs": [
            {
                "path1": str(pair.path1),
                "path2": str(pair.path2),
                "similarity_score": pair.similarity_score,
            }
            for pair in diversity_result.most_similar_pairs
        ],
        "most_diverse_pairs": [
            {
                "path1": str(pair.path1),
                "path2": str(pair.path2),
                "similarity_score": pair.similarity_score,
            }
            for pair in diversity_result.most_diverse_pairs
        ],
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)


def _export_clustering_results(clustering_result: Any, output_file: Path) -> None:
    """Export clustering results to JSON."""
    export_data = {
        "clustering": {
            "num_clusters": clustering_result.num_clusters,
            "silhouette_score": clustering_result.silhouette_score,
            "processing_time": clustering_result.processing_time,
            "cluster_size_distribution": clustering_result.cluster_size_distribution,
            "category_distribution": clustering_result.category_distribution,
        },
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "dominant_category": cluster.dominant_category,
                "intra_cluster_similarity": cluster.intra_cluster_similarity,
                "image_paths": [str(path) for path in cluster.image_paths],
            }
            for cluster in clustering_result.clusters
        ],
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)
