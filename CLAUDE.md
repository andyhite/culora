# CuLoRA - Advanced LoRA Dataset Curation Utility

CuLoRA is a sophisticated command-line utility for intelligently curating image datasets specifically for LoRA (Low-Rank Adaptation) training. The system combines multiple AI models with advanced selection algorithms to automatically identify and select the best images from large datasets for optimal training outcomes.

## Project Overview

- **Purpose**: Automated curation of high-quality image datasets for stable diffusion model training
- **Architecture**: Modern Python CLI application with comprehensive AI analysis pipeline
- **Target**: Process 100+ image datasets with 2-10 images/second processing speed

## Core Technologies

### AI Models & Analysis

- **InsightFace**: Face detection, recognition, and embedding extraction
- **Moondream**: Vision-language model for composition classification
- **CLIP**: Semantic embeddings for composition diversity
- **MediaPipe**: Pose estimation and analysis
- **BRISQUE**: Perceptual quality assessment

### Development Stack

- **CLI Framework**: Typer with Rich integration for beautiful terminal output
- **Configuration**: Pydantic models with full validation
- **Logging**: Structured logging with structlog
- **Quality Tools**: Black, isort, Ruff, mypy, pytest
- **Dependency Management**: Poetry

## Key Features

### Analysis Pipeline

- **Face Analysis**: Multi-face detection with identity matching using reference images
- **Quality Assessment**: Technical metrics (sharpness, contrast, brightness) + BRISQUE perceptual scoring
- **Composition Analysis**: Shot type classification, scene detection, lighting assessment
- **Pose Diversity**: Body landmark detection for pose variation optimization
- **Duplicate Detection**: Perceptual hash-based duplicate identification and removal

### Selection Algorithms

- **Multi-Criteria Selection**: Balances quality, diversity, and target distribution
- **Clustering-Based Diversity**: K-means clustering on pose vectors and CLIP embeddings
- **Quality-Weighted Filtering**: Configurable thresholds with fallback strategies
- **Target Distribution Management**: Flexible composition category distribution

### Export System

- **Training-Optimized Output**: Sequential naming (01.jpg, 02.jpg) for training workflows
- **Comprehensive Metadata**: JSON export with all analysis results
- **Visualization Options**: Face bounding box overlays
- **Multiple Copy Modes**: Selected images only or all images with selection status

## Hardware Support

- **CUDA GPUs**: Optimized for NVIDIA graphics cards with memory analysis
- **Apple Silicon**: MPS backend support for M1/M2 Macs
- **CPU Fallback**: Graceful degradation for systems without dedicated AI hardware
- **Device Auto-Detection**: Intelligent device selection with manual override options

## Development Standards

- **Type Safety**: Full type hints with mypy strict mode
- **Code Quality**: Black formatting, isort imports, Ruff linting
- **Testing**: Comprehensive pytest suite with >90% coverage
- **Documentation**: Google-style docstrings throughout

## Quality Checks Workflow

After each implementation task:

```bash
black .        # Format code
isort .        # Sort imports  
ruff check .   # Lint for issues
mypy .         # Type checking
pytest         # Run tests
```

## Project Structure

```txt
culora/
├── culora/
│   ├── cli/           # Typer-based CLI with Rich integration
│   ├── core/          # Configuration, logging, device management
│   ├── analysis/      # AI model integrations and analysis
│   ├── selection/     # Selection algorithms and clustering
│   ├── export/        # Export functionality and formatters
│   └── utils/         # Shared utilities and type definitions
├── tests/             # Comprehensive test suite
├── pyproject.toml     # Poetry configuration
└── README.md
```

## Detailed Implementation Plan

For comprehensive implementation details, task breakdowns, and technical specifications, see [@prompts/01-prototype.md](prompts/01-prototype.md).

The implementation plan covers:

- 8-week development timeline
- Task-by-task requirements and deliverables
- Testing strategies for each component
- Performance benchmarks and success criteria
- Modern Python development workflow integration
