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
# Curate 100 best images from a dataset
poetry run culora curate input_folder output_folder --count 100

# With reference images for identity matching
poetry run culora curate input_folder output_folder --count 50 --reference-dir references/

# Show version and device information
poetry run culora version
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
│   ├── cli/           # Typer-based CLI with Rich integration
│   ├── core/          # Configuration, logging, device management
│   ├── analysis/      # AI model integrations (planned)
│   ├── selection/     # Selection algorithms (planned)
│   ├── export/        # Export functionality (planned)
│   └── utils/         # Shared utilities (planned)
├── tests/             # Comprehensive test suite
├── prompts/           # Implementation planning documents
├── Makefile          # Development automation
└── pyproject.toml    # Poetry configuration with all tools
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
pytest tests/core/                    # Core functionality
pytest -m unit                        # Unit tests only
pytest -m "not slow"                 # Skip slow tests
```

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

**Tasks 1.1, 1.2 & 2.1 Completed**: Project foundation with structured logging, configuration system, and device management (121 passing tests, 100% coverage)

**In Development**: CLI interface (Task 2.2), then face analysis, quality assessment, and selection algorithms (see `prompts/01-prototype.md` for detailed roadmap)

## License

[Add license information]

## Support

For issues, questions, or contributions, please [open an issue](link-to-issues) or refer to the development documentation.
