"""Main analysis engine for CuLoRA."""

import os
from datetime import datetime
from pathlib import Path

# Suppress ultralytics verbose output
os.environ.setdefault("YOLO_VERBOSE", "False")

from typing import Any, cast

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from ultralytics import YOLO

from culora.analysis.config import get_enabled_stage_configs
from culora.models.analysis import (
    AnalysisResult,
    AnalysisStage,
    DirectoryAnalysis,
    ImageAnalysis,
    StageConfig,
    StageResult,
)
from culora.utils.app_data import get_models_dir
from culora.utils.cache import (
    get_stages_needing_analysis,
    load_analysis_cache,
    merge_analysis_results,
    save_analysis_cache,
)
from culora.utils.images import find_images

console = Console()


def detect_optimal_device() -> str:
    """Detect the optimal device for YOLO inference.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except Exception:
        # Fallback to CPU if torch detection fails
        return "cpu"


def analyze_directory(
    input_directory: Path,
    enable_deduplication: bool = True,
    enable_quality: bool = True,
    enable_face: bool = True,
    use_cache: bool = True,
) -> DirectoryAnalysis:
    """Analyze all images in a directory.

    Args:
        input_directory: Directory containing images to analyze.
        enable_deduplication: Whether to run deduplication analysis.
        enable_quality: Whether to run quality analysis.
        enable_face: Whether to run face detection analysis.
        use_cache: Whether to use cached results if available.

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

    # Get stage configurations for enabled stages
    stage_configs = get_enabled_stage_configs(enabled_stages)

    # Determine which stages need analysis based on intelligent cache validation
    cached_analysis = None
    stages_to_analyze: list[AnalysisStage] = enabled_stages

    if use_cache:
        cached_analysis = load_analysis_cache(input_directory)
        stages_to_analyze = get_stages_needing_analysis(
            cached_analysis, enabled_stages, stage_configs, input_directory
        )

        if not stages_to_analyze and cached_analysis:
            console.print("[blue]Using cached analysis results[/blue]")
            return cached_analysis
        elif len(stages_to_analyze) < len(enabled_stages):
            console.print(
                f"[blue]Using cached results for {len(enabled_stages) - len(stages_to_analyze)} "
                f"stage(s), analyzing {len(stages_to_analyze)} stage(s)[/blue]"
            )

    # Find all images
    console.print(f"[blue]Scanning for images in:[/blue] {input_directory}")
    image_paths = list(find_images(input_directory))

    if not image_paths:
        console.print("[yellow]No images found in directory[/yellow]")
        return DirectoryAnalysis(
            input_directory=str(input_directory.resolve()),
            analysis_time=datetime.now(),
            enabled_stages=enabled_stages,
            stage_configs=stage_configs,
            images=[],
        )

    console.print(f"[green]Found {len(image_paths)} images[/green]")

    # Analyze each image
    analyzed_images: list[ImageAnalysis] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing images...", total=len(image_paths))

        for idx, image_path in enumerate(image_paths, 1):
            progress.update(
                task,
                description=f"Analyzing {image_path.name} ({idx}/{len(image_paths)})",
                completed=idx - 1,
            )

            image_analysis = analyze_image(image_path, stages_to_analyze)
            analyzed_images.append(image_analysis)

            progress.update(task, completed=idx)

    # Post-process deduplication if it was enabled
    if AnalysisStage.DEDUPLICATION in stages_to_analyze:
        analyzed_images = post_process_deduplication(analyzed_images, stage_configs)

    # Create new analysis with only the stages that were analyzed
    new_analysis = DirectoryAnalysis(
        input_directory=str(input_directory.resolve()),
        analysis_time=datetime.now(),
        enabled_stages=enabled_stages,
        stage_configs=stage_configs,
        images=analyzed_images,
    )

    # Merge with cached results if they exist
    if cached_analysis:
        final_analysis = merge_analysis_results(cached_analysis, new_analysis)
    else:
        final_analysis = new_analysis

    # Save merged results to cache
    save_analysis_cache(final_analysis)

    return final_analysis


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

    # Get stage configs for this analysis
    stage_configs = get_enabled_stage_configs(enabled_stages)
    stage_config_map = {config.stage: config for config in stage_configs}

    for stage in enabled_stages:
        stage_config = stage_config_map.get(stage)

        if stage == AnalysisStage.DEDUPLICATION:
            result = analyze_deduplication(image_path, stage_config)
        elif stage == AnalysisStage.QUALITY:
            result = analyze_quality(image_path, stage_config)
        elif stage == AnalysisStage.FACE:
            result = analyze_face(image_path, stage_config)
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


def post_process_deduplication(
    analyzed_images: list[ImageAnalysis], stage_configs: list[StageConfig]
) -> list[ImageAnalysis]:
    """Post-process deduplication results to mark duplicate images.

    After all images have been analyzed and hashes generated, this function
    compares hashes to identify duplicates and updates their analysis results.

    Args:
        analyzed_images: List of analyzed images with hash data.
        stage_configs: List of stage configurations.

    Returns:
        Updated list with duplicate images marked appropriately.
    """
    # Get deduplication config
    dedup_config = None
    for config in stage_configs:
        if config.stage == AnalysisStage.DEDUPLICATION:
            dedup_config = config
            break

    if not dedup_config or not dedup_config.config:
        return analyzed_images  # No config, return unchanged

    # Get threshold for duplicate detection (Hamming distance)
    threshold = int(dedup_config.config.get("threshold", "2"))

    # Extract images with successful hash generation
    images_with_hashes: list[tuple[ImageAnalysis, str]] = []
    for image in analyzed_images:
        # Find deduplication stage result
        for stage_result in image.stage_results:
            if (
                stage_result.stage == AnalysisStage.DEDUPLICATION
                and stage_result.result == AnalysisResult.PASS
                and stage_result.metadata
                and "hash" in stage_result.metadata
            ):

                hash_str = stage_result.metadata["hash"]
                images_with_hashes.append((image, hash_str))
                break

    # Group similar hashes (find duplicates)
    duplicates_found: set[int] = set()

    for i, (image1, hash1) in enumerate(images_with_hashes):
        if i in duplicates_found:
            continue  # Already marked as duplicate

        for j, (image2, hash2) in enumerate(images_with_hashes[i + 1 :], i + 1):
            if j in duplicates_found:
                continue  # Already marked as duplicate

            # Calculate Hamming distance between hashes
            try:
                # Convert hex hash strings to integers for comparison
                hash1_int = int(hash1, 16)
                hash2_int = int(hash2, 16)
                hamming_distance = bin(hash1_int ^ hash2_int).count("1")

                if hamming_distance <= threshold:
                    # Mark second image as duplicate (keep first as original)
                    duplicates_found.add(j)

                    # Update the stage result for the duplicate
                    image1_path = Path(image1.file_path).name

                    for stage_result in image2.stage_results:
                        if stage_result.stage == AnalysisStage.DEDUPLICATION:
                            stage_result.result = AnalysisResult.FAIL
                            stage_result.reason = f"Duplicate of {image1_path} (Hamming distance: {hamming_distance})"
                            if stage_result.metadata:
                                stage_result.metadata["duplicate_of"] = image1_path
                                stage_result.metadata["hamming_distance"] = str(
                                    hamming_distance
                                )
                            break

            except (ValueError, TypeError):
                # Invalid hash format, skip comparison
                continue

    console.print(
        f"[yellow]Deduplication: Found {len(duplicates_found)} duplicate(s)[/yellow]"
    )
    return analyzed_images


def analyze_deduplication(
    image_path: Path, stage_config: StageConfig | None = None
) -> StageResult:
    """Analyze image for deduplication using perceptual hashing.

    Generates a dHash for the image that will be used for duplicate detection
    in the post-processing phase. This function always passes but stores the hash
    for later comparison against other images in the directory.

    Args:
        image_path: Path to the image file.
        stage_config: Optional stage configuration (unused in this phase).

    Returns:
        Deduplication analysis result with hash metadata.
    """
    try:
        with Image.open(image_path) as image:
            # Use dHash for speed and effectiveness with photo duplicates
            # Hash size 8 provides good balance of speed vs accuracy
            dhash = imagehash.dhash(image, hash_size=8)
            hash_str = str(dhash)

            # Always pass in initial phase - duplicates are detected in post-processing
            return StageResult(
                stage=AnalysisStage.DEDUPLICATION,
                result=AnalysisResult.PASS,
                reason=f"Generated hash: {hash_str}",
                metadata={
                    "hash": hash_str,
                    "hash_type": "dhash",
                    "hash_size": "8",
                },
            )

    except Exception as e:
        return StageResult(
            stage=AnalysisStage.DEDUPLICATION,
            result=AnalysisResult.FAIL,
            reason=f"Failed to generate image hash: {str(e)}",
        )


def analyze_quality(
    image_path: Path, stage_config: StageConfig | None = None
) -> StageResult:
    """Analyze image quality using OpenCV metrics.

    Evaluates sharpness, brightness, and contrast of the image.
    Based on research documented in docs/analysis-libraries.md.

    Args:
        image_path: Path to the image file.

    Returns:
        Quality analysis result with detailed metrics.
    """
    try:
        # Load image using OpenCV
        image = cv2.imread(str(image_path))

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate brightness (mean pixel intensity)
        brightness = float(np.mean(cast(np.ndarray[Any, np.dtype[Any]], gray)))

        # Calculate contrast (standard deviation)
        contrast = float(np.std(cast(np.ndarray[Any, np.dtype[Any]], gray)))

        # Apply quality thresholds from configuration
        if stage_config and stage_config.config:
            sharpness_threshold = float(
                stage_config.config.get("sharpness_threshold", "150")
            )
            brightness_min = float(stage_config.config.get("brightness_min", "60"))
            brightness_max = float(stage_config.config.get("brightness_max", "200"))
            contrast_threshold = float(
                stage_config.config.get("contrast_threshold", "40")
            )
        else:
            # Fallback to defaults if no config provided
            sharpness_threshold = 150.0
            brightness_min = 60.0
            brightness_max = 200.0
            contrast_threshold = 40.0

        # Evaluate each metric
        sharpness_pass = laplacian_var >= sharpness_threshold
        brightness_pass = brightness_min <= brightness <= brightness_max
        contrast_pass = contrast >= contrast_threshold

        # Determine overall result
        if sharpness_pass and brightness_pass and contrast_pass:
            result = AnalysisResult.PASS
            reason = f"Quality metrics passed (sharpness: {laplacian_var:.1f}, brightness: {brightness:.1f}, contrast: {contrast:.1f})"
        else:
            result = AnalysisResult.FAIL
            failed_metrics: list[str] = []
            if not sharpness_pass:
                failed_metrics.append(
                    f"sharpness: {laplacian_var:.1f} < {sharpness_threshold}"
                )
            if not brightness_pass:
                failed_metrics.append(
                    f"brightness: {brightness:.1f} not in range [{brightness_min}-{brightness_max}]"
                )
            if not contrast_pass:
                failed_metrics.append(
                    f"contrast: {contrast:.1f} < {contrast_threshold}"
                )
            reason = f"Quality metrics failed: {', '.join(failed_metrics)}"

        return StageResult(
            stage=AnalysisStage.QUALITY,
            result=result,
            reason=reason,
            metadata={
                "sharpness_laplacian": f"{laplacian_var:.2f}",
                "brightness_mean": f"{brightness:.2f}",
                "contrast_std": f"{contrast:.2f}",
                "sharpness_pass": str(sharpness_pass),
                "brightness_pass": str(brightness_pass),
                "contrast_pass": str(contrast_pass),
            },
        )

    except Exception as e:
        return StageResult(
            stage=AnalysisStage.QUALITY,
            result=AnalysisResult.FAIL,
            reason=f"Failed to analyze image quality: {str(e)}",
        )


def analyze_face(
    image_path: Path, stage_config: StageConfig | None = None
) -> StageResult:
    """Analyze image for face detection using YOLO11.

    Uses YOLO11 to detect people in images as a proxy for face detection.
    Based on research documented in docs/analysis-libraries.md.

    Args:
        image_path: Path to the image file.
        stage_config: Configuration for face detection parameters.

    Returns:
        Face detection analysis result with people count and confidence.
    """
    try:
        # Extract configuration parameters
        if stage_config and stage_config.config:
            confidence_threshold = float(
                stage_config.config.get("confidence_threshold", "0.5")
            )
            model_name = stage_config.config.get("model_name", "yolo11n.pt")
            max_detections = int(stage_config.config.get("max_detections", "10"))
            iou_threshold = float(stage_config.config.get("iou_threshold", "0.5"))
            use_half_precision = (
                stage_config.config.get("use_half_precision", "true").lower() == "true"
            )
            device_setting = stage_config.config.get("device", "auto")
        else:
            # Fallback to defaults if no config provided
            confidence_threshold = 0.5
            model_name = "yolo11n.pt"
            max_detections = 10
            iou_threshold = 0.5
            use_half_precision = True
            device_setting = "auto"

        # Determine device to use
        if device_setting == "auto":
            device = detect_optimal_device()
        else:
            device = device_setting

        # Load YOLO11 model for person detection
        models_dir = get_models_dir()
        model_path = models_dir / model_name
        model = YOLO(str(model_path))

        # Run inference on the image with optimized parameters
        results: Any = model(  # pyright: ignore[reportUnknownVariableType]
            str(image_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            device=device,
            half=use_half_precision,
            verbose=False,
        )

        if not results:
            return StageResult(
                stage=AnalysisStage.FACE,
                result=AnalysisResult.FAIL,
                reason="No predictions returned from model",
            )

        # Extract detections from the first (and only) result
        result = results[0]  # type: ignore[misc]

        # Filter for person class (class_id = 0 in COCO dataset)
        person_detections: list[dict[str, Any]] = []
        if hasattr(result, "boxes") and result.boxes is not None:  # type: ignore[misc]
            for box in result.boxes:  # type: ignore[misc]
                # Check if this detection is a person (class 0)
                if hasattr(box, "cls") and hasattr(box, "conf"):  # type: ignore[misc]
                    cls_tensor = box.cls  # type: ignore[misc]
                    conf_tensor = box.conf  # type: ignore[misc]
                    if len(cls_tensor) > 0 and int(cls_tensor[0]) == 0:  # type: ignore[misc] # Person class
                        confidence = float(conf_tensor[0])  # type: ignore[misc]
                        person_detections.append(
                            {
                                "confidence": confidence,
                                "class_name": "person",
                            }
                        )

        face_count = len(person_detections)
        confidences: list[float] = [det["confidence"] for det in person_detections]
        avg_confidence = 0.0

        if face_count > 0:
            avg_confidence = sum(confidences) / face_count
            analysis_result = AnalysisResult.PASS
            reason = f"Detected {face_count} person(s) with average confidence {avg_confidence:.3f}"
        else:
            analysis_result = AnalysisResult.FAIL
            reason = "No people detected in image"

        return StageResult(
            stage=AnalysisStage.FACE,
            result=analysis_result,
            reason=reason,
            metadata={
                "face_count": str(face_count),
                "confidence_scores": ",".join(f"{conf:.3f}" for conf in confidences),
                "average_confidence": f"{avg_confidence:.3f}",
                "model": model_name,
                "detection_type": "person",
                "device_used": device,
                "confidence_threshold": str(confidence_threshold),
                "max_detections": str(max_detections),
                "iou_threshold": str(iou_threshold),
                "half_precision": str(use_half_precision),
            },
        )

    except Exception as e:
        return StageResult(
            stage=AnalysisStage.FACE,
            result=AnalysisResult.FAIL,
            reason=f"Failed to analyze face detection: {str(e)}",
        )
