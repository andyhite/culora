"""Main analysis engine for CuLoRA."""

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from culora.models.analysis import (
    AnalysisResult,
    AnalysisStage,
    DirectoryAnalysis,
    ImageAnalysis,
    StageResult,
)
from culora.utils.cache import is_cache_valid, load_analysis_cache, save_analysis_cache
from culora.utils.images import find_images

console = Console()


def analyze_directory(
    input_directory: Path,
    enable_deduplication: bool = True,
    enable_quality: bool = True,
    enable_face: bool = True,
) -> DirectoryAnalysis:
    """Analyze all images in a directory.

    Args:
        input_directory: Directory containing images to analyze.
        enable_deduplication: Whether to run deduplication analysis.
        enable_quality: Whether to run quality analysis.
        enable_face: Whether to run face detection analysis.

    Returns:
        Analysis results for the directory.

    Raises:
        FileNotFoundError: If input directory doesn't exist.
        NotADirectoryError: If input path is not a directory.
    """
    # Determine enabled stages
    enabled_stages: list[AnalysisStage] = []
    if enable_deduplication:
        enabled_stages.append(AnalysisStage.DEDUPLICATION)
    if enable_quality:
        enabled_stages.append(AnalysisStage.QUALITY)
    if enable_face:
        enabled_stages.append(AnalysisStage.FACE)

    # Try to load from cache first
    cached_analysis = load_analysis_cache(input_directory)
    if cached_analysis and is_cache_valid(cached_analysis, input_directory):
        # Check if enabled stages match
        if set(cached_analysis.enabled_stages) == set(enabled_stages):
            console.print("[blue]Using cached analysis results[/blue]")
            return cached_analysis

    # Find all images
    console.print(f"[blue]Scanning for images in:[/blue] {input_directory}")
    image_paths = list(find_images(input_directory))

    if not image_paths:
        console.print("[yellow]No images found in directory[/yellow]")
        return DirectoryAnalysis(
            input_directory=str(input_directory.resolve()),
            analysis_time=datetime.now(),
            enabled_stages=enabled_stages,
            images=[],
        )

    console.print(f"[green]Found {len(image_paths)} images[/green]")

    # Analyze each image
    analyzed_images: list[ImageAnalysis] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing images...", total=len(image_paths))

        for image_path in image_paths:
            progress.update(task, description=f"Analyzing {image_path.name}")

            image_analysis = analyze_image(image_path, enabled_stages)
            analyzed_images.append(image_analysis)

            progress.advance(task)

    # Create directory analysis
    analysis = DirectoryAnalysis(
        input_directory=str(input_directory.resolve()),
        analysis_time=datetime.now(),
        enabled_stages=enabled_stages,
        images=analyzed_images,
    )

    # Save to cache
    save_analysis_cache(analysis)

    return analysis


def analyze_image(
    image_path: Path, enabled_stages: list[AnalysisStage]
) -> ImageAnalysis:
    """Analyze a single image file.

    Args:
        image_path: Path to the image file.
        enabled_stages: List of analysis stages to run.

    Returns:
        Analysis results for the image.
    """
    # Get file metadata
    stat = image_path.stat()
    file_size = stat.st_size
    modified_time = datetime.fromtimestamp(stat.st_mtime)

    # Run each enabled stage
    stage_results: list[StageResult] = []

    for stage in enabled_stages:
        if stage == AnalysisStage.DEDUPLICATION:
            result = analyze_deduplication(image_path)
        elif stage == AnalysisStage.QUALITY:
            result = analyze_quality(image_path)
        elif stage == AnalysisStage.FACE:
            result = analyze_face(image_path)
        else:
            # Unknown stage, skip
            result = StageResult(
                stage=stage,
                result=AnalysisResult.SKIP,
                reason="Unknown analysis stage",
            )

        stage_results.append(result)

        # If this stage failed and we're doing sequential processing,
        # we could break here, but for now we'll run all enabled stages

    return ImageAnalysis(
        file_path=str(image_path.resolve()),
        file_size=file_size,
        modified_time=modified_time,
        stage_results=stage_results,
    )


def analyze_deduplication(image_path: Path) -> StageResult:
    """Analyze image for deduplication.

    This is a placeholder implementation that always passes.

    Args:
        image_path: Path to the image file.

    Returns:
        Deduplication analysis result.
    """
    # TODO: Implement actual deduplication logic
    return StageResult(
        stage=AnalysisStage.DEDUPLICATION,
        result=AnalysisResult.PASS,
        reason="Deduplication not yet implemented",
    )


def analyze_quality(image_path: Path) -> StageResult:
    """Analyze image quality.

    This is a placeholder implementation that always passes.

    Args:
        image_path: Path to the image file.

    Returns:
        Quality analysis result.
    """
    # TODO: Implement actual quality analysis
    return StageResult(
        stage=AnalysisStage.QUALITY,
        result=AnalysisResult.PASS,
        reason="Quality analysis not yet implemented",
    )


def analyze_face(image_path: Path) -> StageResult:
    """Analyze image for face detection.

    This is a placeholder implementation that always passes.

    Args:
        image_path: Path to the image file.

    Returns:
        Face detection analysis result.
    """
    # TODO: Implement actual face detection
    return StageResult(
        stage=AnalysisStage.FACE,
        result=AnalysisResult.PASS,
        reason="Face detection not yet implemented",
    )
