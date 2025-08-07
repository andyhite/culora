# Analysis Pipeline Library Selection

This document records the research and decisions made for CuLoRA's analysis pipeline libraries, based on requirements from Epic 3.

## Overview

CuLoRA's analysis pipeline consists of three modular stages:

1. **Deduplication** - Remove duplicate/near-duplicate images via perceptual hashing
2. **Quality Assessment** - Filter images based on sharpness, brightness, and contrast
3. **Face Detection** - Identify images containing human faces

Each stage is enabled by default but can be disabled via CLI flags (`--no-dedupe`, `--no-quality`, `--no-face`).

## Library Selection Criteria

For each analysis stage, we evaluated libraries based on:

- **Performance**: Speed and memory efficiency for large datasets
- **Cross-platform compatibility**: Mac, Linux, Windows support
- **Installation simplicity**: Minimal dependencies and compilation requirements
- **Device acceleration**: GPU/CUDA, MPS (Apple Silicon), multi-core CPU support
- **API ease of use**: Simple integration with CLI architecture
- **Maintenance**: Active development and community support

## Selected Libraries

### Image Deduplication: ImageHash

**Selected Library**: `imagehash` 4.3.2+

- **Installation**: `pip install imagehash`
- **Dependencies**: PIL/Pillow, numpy, scipy
- **Algorithm**: dHash (default) for speed, pHash option for accuracy
- **Performance**: ~1000 images/second, very low memory usage
- **Configuration**:
  - Hash size: 8 (64-bit hash)
  - Similarity threshold: 2 (allows minor lighting/compression variations)

**Alternatives Considered**: imagededup, dhash, py-image-dedup
**Decision Rationale**: ImageHash provides the optimal balance of speed, accuracy, simplicity, and cross-platform reliability for CLI applications.

### Image Quality Assessment: OpenCV

**Selected Library**: `opencv-python` 4.9.0+ with custom implementations

- **Installation**: `pip install opencv-python`
- **Dependencies**: NumPy (minimal)
- **Metrics Implemented**:
  - **Sharpness**: Laplacian variance (primary), Sobel gradient (alternative)
  - **Brightness**: Mean pixel intensity
  - **Contrast**: Standard deviation of grayscale values
- **Performance**: 1-5ms per image, ~10-15MB memory usage
- **Quality Thresholds**:
  - Sharpness (Laplacian): >150 good, >500 excellent, <50 poor
  - Brightness: 60-200 optimal range (0-255 scale)
  - Contrast: >40 good, >60 excellent, <20 poor

**Alternatives Considered**: PyIQA, BRISQUE, custom NumPy implementations
**Decision Rationale**: OpenCV provides fast, reliable quality metrics with minimal dependencies, perfect for CLI batch processing.

### Face Detection: YOLO11 (Ultralytics)

**Selected Library**: `ultralytics` 8.3.0+

- **Installation**: `pip install ultralytics`
- **Dependencies**: PyTorch, torchvision (automatically managed)
- **Model**: YOLO11 nano (~12MB download on first use)
- **Performance**: High-speed person detection with automatic device optimization
- **Configuration**:
  - Model: yolo11n.pt (nano version for speed)
  - Detection: Person class (class ID 0) as proxy for face detection
  - Confidence threshold: Default YOLO confidence filtering
- **Output**: Person count, detection confidence per person, bounding boxes

**Alternatives Considered**: MediaPipe BlazeFace, OpenCV YuNet, YOLOv8, face_recognition, RetinaFace, MTCNN
**Decision Rationale**: YOLO11 provides excellent person detection accuracy, automatic device acceleration, and clean integration with the ultralytics ecosystem. Person detection serves as an effective proxy for face detection in LoRA training contexts.

## Performance Characteristics

Based on research and benchmarks:

| Stage          | Library     | Processing Time* | Memory Usage | Device Acceleration |
| -------------- | ----------- | ---------------- | ------------ | ------------------- |
| Deduplication  | ImageHash   | 1-3ms/image      | ~5MB         | CPU optimized       |
| Quality        | OpenCV      | 1-5ms/image      | ~10MB        | CPU/OpenCL          |
| Face Detection | YOLO11      | ~50ms/image      | ~200MB       | GPU/CPU adaptive    |

*Times for typical LoRA training images (512x512 to 1024x1024)

## Implementation Architecture

### Device Selection Strategy

1. **Automatic detection**: Check for CUDA, MPS (Apple Silicon), then fallback to CPU
2. **Per-stage optimization**: Each analysis stage uses optimal device configuration
3. **Memory management**: Process images individually to minimize memory footprint

### Pipeline Configuration

```python
# Default configuration
ANALYSIS_CONFIG = {
    "deduplication": {
        "algorithm": "dhash",
        "hash_size": 8,
        "threshold": 2
    },
    "quality": {
        "sharpness_threshold": 150,
        "brightness_min": 60,
        "brightness_max": 200,
        "contrast_threshold": 40
    },
    "face_detection": {
        "confidence_threshold": 0.5,
        "model_selection": 1
    }
}
```

### Caching Strategy

- All analysis results cached in per-directory JSON files
- Cache invalidation based on image modification time and file size
- Results include per-stage pass/fail status and confidence scores
- Cache location: `~/.local/share/culora/cache/` (via `typer.get_app_dir()`)

## Dependencies Added

These libraries will be added to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
imagehash = "^4.3.2"
opencv-python = "^4.9.0"
ultralytics = "^8.3.0"
```

## Future Considerations

### Optional Advanced Features

- **BRISQUE integration**: Add `--advanced-quality` flag for ML-based quality assessment
- **Custom thresholds**: Allow per-project quality threshold configuration
- **Batch optimization**: Multi-threading for CPU-bound operations
- **Progress reporting**: Detailed progress bars for each analysis stage

### Potential Upgrades

- **Dedicated face detection**: Consider specialized face detection models for higher precision
- **Custom quality models**: Train domain-specific quality assessment models
- **GPU memory management**: Optimize VRAM usage for large batch processing

---

**Research Completed**: 2025-01-XX
**Implementation Status**: Ready for development (Epic 3 User Stories)
**Review Date**: TBD (after initial implementation and user feedback)
