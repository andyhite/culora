# CuLoRA - Advanced LoRA Dataset Curation Utility

Intelligently curate image datasets for LoRA training using multiple AI models and sophisticated selection algorithms.

## Quick Start

```bash
# Install with Poetry
poetry install

# Run basic curation
poetry run culora curate input_folder output_folder --count 100

# Show version
poetry run culora version
```

## Development

```bash
# Install dependencies
poetry install

# Run quality checks
poetry run black .
poetry run isort .
poetry run ruff check .
poetry run mypy .
poetry run pytest
```

For detailed implementation information, see [CLAUDE.md](CLAUDE.md).
