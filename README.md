# CuLoRA

A command-line tool for intelligently curating image datasets for LoRA training. CuLoRA uses a multi-stage analysis pipeline to automatically select the highest quality images from your dataset, combining deduplication, quality assessment, and face detection to ensure optimal training data.

## Features

- **🔍 Deduplication**: Remove near-duplicate images using perceptual hashing
- **✨ Quality Assessment**: Filter images based on sharpness, brightness, and contrast
- **👤 Face Detection**: Identify and evaluate images containing faces with confidence scoring
- **🎯 Intelligent Selection**: Two-tier ranking system with composite scoring
- **📊 Rich Output**: Beautiful progress bars and detailed results tables
- **⚡ GPU Acceleration**: Automatic device detection for optimal performance
- **🔧 Modular Pipeline**: Enable/disable analysis stages independently

## Installation

```bash
git clone <repository-url>
cd culora
poetry install
```

## Quick Start

### Basic Usage

Analyze images in a directory and display results:

```bash
culora analyze /path/to/images
```

Analyze and automatically select curated images to an output directory:

```bash
culora analyze /path/to/images --output /path/to/curated
```

### Advanced Options

Select only the top 50 images by composite score:

```bash
culora analyze /path/to/images --output /path/to/curated --max-images 50
```

Preview selection without copying files:

```bash
culora analyze /path/to/images --output /path/to/curated --dry-run
```

Disable specific analysis stages:

```bash
culora analyze /path/to/images --no-dedupe --no-quality
```

Draw bounding boxes on detected faces:

```bash
culora analyze /path/to/images --output /path/to/curated --draw-boxes
```

## Commands

### `culora analyze`

Analyze images in a directory using the multi-stage pipeline.

```bash
culora analyze <input_dir> [OPTIONS]
```

**Arguments:**

- `input_dir`: Directory containing images to analyze

**Options:**

- `--output, -o`: Automatically select and copy curated images to this directory
- `--max-images`: Maximum number of images to select (ranked by score)
- `--dry-run`: Preview selection without copying files
- `--draw-boxes`: Draw bounding boxes on detected faces with confidence scores
- `--no-dedupe`: Disable image deduplication
- `--no-quality`: Disable image quality assessment  
- `--no-face`: Disable face detection

### Analysis Pipeline

CuLoRA's analysis pipeline consists of three modular stages:

#### 1. **Deduplication** (enabled by default)

- Uses perceptual hashing (dHash algorithm) to identify near-duplicate images
- Configurable similarity threshold for handling minor variations
- Keeps only the highest-scoring image from each duplicate group

#### 2. **Quality Assessment** (enabled by default)

- **Sharpness**: Laplacian variance detection for blur assessment
- **Brightness**: Mean pixel intensity analysis (optimal range: 60-200)
- **Contrast**: Standard deviation of grayscale values (threshold: >40)
- Composite quality score combining all metrics

#### 3. **Face Detection** (enabled by default)

- Specialized YOLO11 face detection model (AdamCodd/YOLOv11n-face-detection)
- Confidence scoring and bounding box detection
- Face size and face-to-image ratio analysis
- Multi-face penalty system for optimal single-subject selection

### Selection Algorithm

CuLoRA uses a sophisticated two-tier selection system:

#### Tier 1: Threshold Culling

- Apply minimum quality thresholds as hard filters
- Remove images that don't meet basic requirements
- Eliminate duplicates keeping only the best version

#### Tier 2: Score-Based Ranking

- Calculate composite scores combining quality (50%) and face metrics (50%)
- Intelligent face area scoring with optimal 5-25% range (peak at 15%)
- Face confidence gating and multi-face penalties
- Select top N images by final composite score

### Score Components

The composite scoring algorithm considers:

- **Quality Score (50% weight)**
  - Sharpness, brightness, and contrast metrics
  - Normalized to 0-1 range

- **Face Score (50% weight)**
  - Face area relative to image size (sigmoid curve)
  - Face detection confidence as multiplier
  - Penalties for multiple faces (10% per additional face, max 50%)

Final scores are clamped to [0.0, 1.0] range for consistent ranking.

## Output

### Analysis Results Table

The results table shows detailed metrics for each image:

- **Image**: Filename
- **Sharpness/Brightness/Contrast**: Individual quality metrics (color-coded)
- **Quality**: Composite quality score
- **Faces**: Detected faces with sizes and confidence scores
- **Score**: Final composite score for ranking (color-coded: green ≥0.7, yellow ≥0.4, red <0.4)

### Selection Summary

After selection, CuLoRA displays:

- Total images processed
- Tier 1 qualified images (passed thresholds)
- Tier 2 selected images (top N by score)
- Selected image mapping with scores

## Performance

CuLoRA is optimized for large datasets:

- **Deduplication**: ~1000 images/second with minimal memory usage
- **Quality Assessment**: 1-5ms per image using OpenCV
- **Face Detection**: ~50ms per image with GPU acceleration
- **Device Support**: Automatic CUDA, MPS (Apple Silicon), or CPU selection

## Development

### Setup

```bash
poetry install
poetry run pre-commit install
```

### Testing

```bash
make test
```

### Quality Checks

```bash
make check
```

### Project Structure

```txt
src/culora/
├── cli/           # Command-line interface
├── orchestrators/ # High-level workflow coordination
├── services/      # Analysis and selection services
├── managers/      # Resource and configuration management
├── models/        # Pydantic data models
└── utils/         # Shared utilities
```

## Requirements

- Python 3.12+
- Modern dependencies (Poetry, Pydantic, Rich, OpenCV, Ultralytics)
- Optional: CUDA-capable GPU for accelerated face detection

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
