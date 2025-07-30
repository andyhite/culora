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
poetry run pytest tests/unit/services/   # Run service layer tests
poetry run pytest tests/unit/domain/     # Run domain model tests
```

## Project Structure

```txt
culora/
├── culora/
│   ├── core/          # Foundation: exception hierarchy
│   │   └── exceptions/ # Modular exception classes (config, device, culora)
│   ├── domain/        # Domain-driven design models and enums
│   │   ├── enums/     # Type-safe enums (device types, log levels)
│   │   └── models/    # Domain models (device, memory, config)
│   ├── services/      # Service layer (config, device, memory services)
│   └── utils/         # Shared utilities (logging)
├── tests/             # Best-practice test organization
│   ├── conftest.py    # Shared pytest fixtures
│   ├── helpers/       # Test utilities (factories, assertions, file utils)
│   ├── mocks/         # Mock implementations (PyTorch, AI models)
│   ├── fixtures/      # Static test data and configurations
│   ├── unit/          # Unit tests organized by domain
│   └── integration/   # Integration and workflow tests
├── pyproject.toml     # Poetry configuration with all dependencies
├── Makefile          # Development workflow automation
└── README.md
```

## Current Implementation

### Domain-Driven Architecture

**Domain Layer** (`culora/domain/`):

- **Enums** (`enums/`): Type-safe enumerations for device types and log levels
- **Models** (`models/`): Domain models for devices, memory, and configuration
- **Config Models** (`models/config/`): Pydantic configuration models with full validation

**Service Layer** (`culora/services/`):

- **Config Service** (`config_service.py`): Multi-source configuration management with precedence handling and property-based source tracking
- **Device Service** (`device_service.py`): Intelligent hardware detection with CUDA/MPS/CPU support
- **Memory Service** (`memory_service.py`): Memory management and tracking

**Core Foundation** (`culora/core/`):

- **Exception Hierarchy** (`exceptions/`): Modular exception classes organized by domain (config, device, culora)

**CLI Layer** (`culora/cli/`):

- **Application** (`app.py`): Main Typer application with global error handling
- **Commands** (`commands/`): Modular command implementations (config, device)
- **Display** (`display/`): Rich console components and theming
- **Validation** (`validation/`): CLI argument validators with proper error handling

**Utilities** (`culora/utils/`):

- **Structured Logging** (`logging.py`): Production-ready JSON logging separate from Rich UI

### Modern Test Infrastructure (`tests/`)

- **Test Organization**: Industry-standard structure with helpers, mocks, fixtures, unit, and integration directories
- **Test Helpers** (`helpers/`): Modular utilities including ConfigBuilder factory, AssertionHelpers, and TempFileHelper
- **Mock Implementations** (`mocks/`): Centralized PyTorch/CUDA mocking with MockContext utility
- **Static Fixtures** (`fixtures/`): Reusable test data and configuration files
- **Unit Tests** (`unit/`): Organized by domain (services, domain models, CLI) with 363 passing tests
- **Integration Tests** (`integration/`): End-to-end workflow testing including full CLI integration
- **Comprehensive Coverage**: 363 tests with modern organization and 100% type safety

**Test Structure Benefits:**

- **Maintainability**: Clear separation of test utilities, mocks, and test categories
- **Reusability**: Modular helpers easily shared across test files (e.g., `from tests.helpers import ConfigBuilder`)
- **Scalability**: Easy to add new test utilities and organize future AI model tests
- **Best Practices**: Follows Python testing industry standards with proper imports and organization

### Development Tooling

- **Dependencies**: All AI model and development tool dependencies configured and updated to latest versions (Python 3.12, Ruff 0.12.5, Black 25.1.0, mypy 1.14.1)
- **Quality Tools**: Black, isort, Ruff, mypy, pytest with optimal configurations
- **Architecture**: Clean domain-driven design with service layer pattern and modern CLI
- **CLI Implementation**: Typer-based CLI with Rich integration, comprehensive validation, and beautiful output
- **Test Infrastructure**: Industry-standard test organization with helpers, mocks, and fixtures
- **Automation**: Comprehensive Makefile for development workflow

### Recent Architecture Improvements

**ConfigService Refactoring**:

- Eliminated duplicate state tracking between `_config_sources` and `_config_file`
- Implemented property-based `config_sources` that derives source information dynamically
- Simplified internal state management while maintaining identical public API
- Reduced potential for synchronization bugs between redundant state variables

**CLI Implementation Architecture**:

- Clean separation between CLI layer and business logic services
- Global service instances with lazy initialization pattern
- Rich theming system for consistent visual styling
- Comprehensive argument validation with helpful error messages
- Proper exception chaining and error handling throughout CLI commands

## Implementation Plan

For comprehensive implementation details and task breakdowns, see [@prompts/01-prototype.md](prompts/01-prototype.md).
