# CuLoRA - Advanced LoRA Dataset Curation Utility

Intelligently curate image datasets for stable diffusion LoRA training using multiple AI models and sophisticated selection algorithms. CuLoRA combines face detection, quality assessment, composition analysis, and duplicate detection to automatically select the best images from large datasets.

## Features

- **Face Detection & Analysis**: InsightFace integration for identity consistency
- **Quality Assessment**: Technical metrics + BRISQUE perceptual quality scoring
- **Composition Analysis**: Vision-language models for shot type classification
- **Duplicate Detection**: Perceptual hashing to remove near-duplicates
- **Smart Selection**: Multi-criteria algorithms balancing quality, diversity, and distribution
- **Modern CLI**: Beautiful Rich-powered interface with progress tracking

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Optional: CUDA GPU for faster processing

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd culora

# Install dependencies
poetry install

# Verify installation
poetry run culora --help
```

### Basic Usage

```bash
# Show configuration and device information
poetry run culora config show
poetry run culora device info

# Manage configuration values
poetry run culora config set device.preferred_device cuda
poetry run culora config get device.preferred_device

# Export configuration to file
poetry run culora config export config.yaml

# Image management and processing
poetry run culora images scan /path/to/images     # Scan directory for images
poetry run culora images validate /path/to/images # Validate all images
poetry run culora images info /path/to/image.jpg  # Show image metadata
poetry run culora images formats                  # List supported formats

# Face detection and analysis
poetry run culora faces detect /path/to/images    # Detect faces in directory
poetry run culora faces analyze /path/to/image.jpg # Analyze single image faces
poetry run culora faces models                    # List available face models

# Future: Curate images (in development)
# poetry run culora curate input_folder output_folder --count 100
```

## Development

### Development Setup

```bash
# Install development dependencies
poetry install

# Set up development environment (recommended)
make dev-setup

# Verify everything works
make check
```

### Development Workflow

This project uses modern Python development tools with automated quality checks:

```bash
# Complete development workflow
make all                 # Format, check, and test everything

# Individual commands
make format             # Format code with Black + isort
make lint               # Run Ruff linter
make typecheck          # Run mypy type checking
make test               # Run pytest suite
make test-cov          # Run tests with coverage report

# Quick quality check (no tests)
make check              # Format + lint + typecheck
```

### Code Quality Standards

- **Type Checking**: Full mypy strict mode compliance
- **Formatting**: Black (88 char line length) + isort
- **Linting**: Ruff for fast, comprehensive linting
- **Testing**: pytest with fixtures and 100% coverage
- **Documentation**: Google-style docstrings

### Project Structure

```text
culora/
├── culora/
│   ├── cli/           # CLI layer with Typer + Rich
│   │   ├── commands/  # Command implementations (config, device, images, faces)
│   │   ├── display/   # Rich console components and theming
│   │   └── validation/ # CLI argument validators
│   ├── core/          # Foundation: exception hierarchy
│   │   └── exceptions/ # Modular exception classes (config, device, culora)
│   ├── domain/        # Domain-driven design models and enums
│   │   ├── enums/     # Type-safe enums (device types, log levels)
│   │   └── models/    # Domain models (device, memory, config, images, faces)
│   ├── services/      # Service layer (config, device, memory, image, face services)
│   └── utils/         # Shared utilities (logging, app directories)
├── tests/             # Best-practice test organization
│   ├── conftest.py    # Shared pytest fixtures
│   ├── helpers/       # Test utilities (factories, assertions, file utils)
│   ├── mocks/         # Mock implementations (PyTorch, AI models)
│   ├── fixtures/      # Static test data and configurations
│   ├── unit/          # Unit tests organized by domain (services, CLI, etc.)
│   └── integration/   # Integration and workflow tests
├── prompts/           # Implementation planning documents
├── Makefile          # Development automation
└── pyproject.toml    # Poetry configuration with all tools
```

### Testing

The project follows industry best practices for test organization:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest tests/unit/                    # All unit tests
pytest tests/integration/             # Integration tests
pytest tests/unit/services/           # Service layer tests
pytest tests/unit/domain/             # Domain model tests

# Run with markers
pytest -m integration                 # Integration tests only
pytest -m "not slow"                 # Skip slow tests
```

**Test Structure Benefits:**

- **Modular Helpers**: Reusable test utilities in `tests/helpers/`
- **Centralized Mocks**: PyTorch and AI model mocks in `tests/mocks/`
- **Clean Organization**: Unit and integration tests properly separated
- **Easy Imports**: Convenient helper imports from `tests.helpers`

### Configuration

The project uses Pydantic for type-safe configuration with multiple sources:

1. Command line arguments (highest priority)
2. Environment variables (`CULORA_*`)
3. Configuration files (`culora.yaml`)
4. Default values (lowest priority)

See [CLAUDE.md](CLAUDE.md) for detailed development information and architecture.

### Contributing

Before submitting changes:

1. Run the complete workflow: `make all`
2. Ensure all tests pass and code quality checks succeed
3. Add tests for new functionality
4. Update documentation as needed

## Current Status

**Tasks 1.1, 1.2, 2.1, 2.2, 2.3 & 3.1 Completed**: Project foundation with domain-driven architecture, service layer, modern CLI, image processing infrastructure, face analysis system, and comprehensive testing.

**Recent Updates**:

- **Task 3.1 Completion**: Full face detection and analysis system with InsightFace integration
- **Face Analysis Service**: Production-ready FaceAnalysisService with device optimization and batch processing
- **Face CLI Commands**: Complete face analysis commands (detect, analyze, models) with Rich-formatted output
- **Service Pattern Standardization**: All services now use consistent `get_*_service()` auto-initializing pattern
- **Third-Party Integration**: Robust InsightFace and onnxruntime integration with output suppression
- **App Directory Management**: Cross-platform Typer app directory utilities for config and model storage
- **Configuration Simplification**: Single app directory location for all configuration and cache files
- **Task 2.3 Completion**: Complete image loading and directory processing service with batch processing, validation, and metadata extraction
- **Image CLI Commands**: Full image management commands (scan, validate, info, formats) with Rich-formatted output
- **Configuration Enhancement**: Added ImageConfig with comprehensive validation and environment variable support
- **Test Infrastructure Expansion**: Added ImageFixtures helper for comprehensive image testing scenarios  
- **Architecture Refactor**: Complete reorganization using domain-driven design with service layer pattern
- **Module Reorganization**: Removed unused modules and implemented clean architecture
- **Exception Hierarchy**: Modular exception classes organized by domain (config, device, culora)
- **CLI Implementation**: Complete Typer-based CLI with Rich integration, comprehensive validation, and beautiful output
- **Test Infrastructure Overhaul**: Restructured entire test suite following industry best practices
- **Modern Test Organization**: Separated helpers, mocks, fixtures, unit, and integration tests
- **Enhanced Maintainability**: Comprehensive test coverage with modular utilities and clean imports
- **ConfigService Refactoring**: Eliminated duplicate state tracking with property-based approach

**In Development**: Reference image matching, quality assessment, and selection algorithms (see `prompts/01-prototype.md` for detailed roadmap)

## License

[Add license information]

## Support

For issues, questions, or contributions, please [open an issue](link-to-issues) or refer to the development documentation.
