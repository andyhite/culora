# CuLoRA - Advanced LoRA Dataset Curation Utility

CuLoRA is a sophisticated command-line utility for intelligently curating image datasets specifically for LoRA (Low-Rank Adaptation) training.

## Development Stack

- **CLI Framework**: Typer with Rich integration for beautiful terminal output
- **Configuration**: Pydantic models with full validation
- **Logging**: Structured logging with structlog
- **Quality Tools**: Black, isort, Ruff, mypy, pytest
- **Dependency Management**: Poetry

## Core Technologies

### AI Models & Analysis

- **InsightFace**: Face detection, recognition, and embedding extraction
- **Moondream**: Vision-language model for composition classification
- **CLIP**: Semantic embeddings for composition diversity
- **MediaPipe**: Pose estimation and analysis
- **BRISQUE**: Perceptual quality assessment using PIQ (PyTorch Image Quality) library

### Hardware Support

- **CUDA GPUs**: Optimized for NVIDIA graphics cards with memory analysis
- **Apple Silicon**: MPS backend support for M1/M2 Macs
- **CPU Fallback**: Graceful degradation for systems without dedicated AI hardware
- **Device Auto-Detection**: Intelligent device selection with manual override options

## Development Standards

- **Type Safety**: Full type hints with mypy strict mode
- **Code Quality**: Black formatting, isort imports, Ruff linting
- **Testing**: Comprehensive pytest suite with >90% coverage
- **Documentation**: Google-style docstrings throughout

## Development Workflow

### Quick Start

```bash
# Setup development environment (first time only)
make dev-setup

# Run complete development workflow
make all

# Show all available commands
make help
```

### Makefile Commands

The project includes a comprehensive Makefile for streamlined development. All commands use Poetry to manage dependencies and virtual environments.

#### **Setup Commands**

```bash
make install     # Install dependencies with Poetry
make dev-setup   # Complete development environment setup (includes install)
```

#### **Code Quality Commands**

```bash
make format      # Format code with Black and sort imports with isort
make lint        # Run Ruff linter for code issues
make typecheck   # Run mypy type checking
make check       # Run all quality checks (format + lint + typecheck)
```

#### **Testing Commands**

```bash
make test        # Run pytest test suite
make test-cov    # Run tests with HTML and terminal coverage reports
```

#### **Maintenance Commands**

```bash
make clean       # Remove cache files and build artifacts (__pycache__, .pytest_cache, etc.)
make all         # Complete workflow: format + check + test
```

#### **Recommended Development Workflow**

1. **Initial Setup** (one time):

   ```bash
   make dev-setup
   ```

2. **Before Committing Changes**:

   ```bash
   make all
   ```

   This runs formatting, linting, type checking, and tests in sequence.

3. **During Development** (iterative):

   ```bash
   make check    # Quick quality checks without tests
   make test     # Run tests when needed
   ```

4. **Troubleshooting**:

   ```bash
   make clean    # Clear caches if experiencing issues
   make help     # Show all available commands
   ```

### Direct Poetry Commands (if needed)

If you prefer to run tools directly or need custom options:

```bash
# Development tools
poetry run black .                    # Format code
poetry run isort .                    # Sort imports  
poetry run ruff check .               # Lint for issues
poetry run mypy .                     # Type checking
poetry run pytest                     # Run tests
poetry run pytest --cov=culora        # Run tests with coverage
poetry run pytest -v                 # Verbose test output
poetry run pytest tests/core/        # Run specific test directory
```

## Project Structure

```txt
culora/
├── culora/
│   ├── cli/           # Typer-based CLI with Rich integration
│   ├── core/          # Foundation: types, exceptions, logging, configuration
│   ├── analysis/      # AI model integrations and analysis
│   ├── selection/     # Selection algorithms and clustering
│   ├── export/        # Export functionality and formatters
│   └── utils/         # Shared utilities and type definitions
├── tests/             # Comprehensive test suite with fixtures
├── pyproject.toml     # Poetry configuration with all dependencies
├── Makefile          # Development workflow automation
└── README.md
```

## Current Implementation

### Core Foundation (`culora/core/`)

- **Types System** (`types.py`): Complete enum definitions and type aliases
- **Exception Hierarchy** (`exceptions.py`): Structured error handling with context
- **Structured Logging** (`logging.py`): Production-ready JSON logging separate from Rich UI
- **Configuration System** (`config.py`): Type-safe Pydantic models with comprehensive validation
- **Configuration Manager** (`config_manager.py`): Multi-source configuration with precedence handling
- **Device Management** (`device_info.py`, `device_detector.py`, `device_manager.py`): Intelligent hardware detection with CUDA/MPS/CPU support

### Testing Infrastructure (`tests/`)

- **Test Fixtures**: Configuration and logging fixtures for comprehensive testing
- **Test Suite**: 121 passing tests with 100% code coverage
- **Quality Assurance**: All tests verify configuration validation, exception handling, and logging functionality

### Development Tooling

- **Dependencies**: All AI model and development tool dependencies configured and updated to latest versions (Python 3.12, Ruff 0.12.5, Black 25.1.0, mypy 1.14.1)
- **Quality Tools**: Black, isort, Ruff, mypy, pytest with optimal configurations
- **Automation**: Comprehensive Makefile for development workflow

## Implementation Plan

For comprehensive implementation details and task breakdowns, see [@prompts/01-prototype.md](prompts/01-prototype.md).
