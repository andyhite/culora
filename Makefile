.PHONY: install format lint typecheck test test-cov clean dev-setup check all help

# Default target
help:
	@echo "CuLoRA Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies with Poetry"
	@echo "  dev-setup   Complete development environment setup"
	@echo ""
	@echo "Code Quality:"
	@echo "  format      Format code with Black and sort imports with isort"
	@echo "  lint        Run Ruff linter"
	@echo "  typecheck   Run mypy type checking"
	@echo "  check       Run all quality checks (format, lint, typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run pytest test suite"
	@echo "  test-cov    Run tests with coverage report"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       Remove cache files and build artifacts"
	@echo "  all         Run complete workflow (format, check, test)"

# Installation and setup
install:
	@echo "Installing dependencies with Poetry..."
	poetry install

dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything is working."

# Code formatting
format:
	@echo "Formatting code with Black..."
	poetry run black .
	@echo "Sorting imports with isort..."
	poetry run isort .
	@echo "Code formatting complete!"

# Linting
lint:
	@echo "Running Ruff linter..."
	poetry run ruff check .

# Type checking
typecheck:
	@echo "Running mypy type checking..."
	poetry run mypy .

# Combined quality checks
check: format lint typecheck
	@echo "All quality checks passed!"

# Testing
test:
	@echo "Running pytest test suite..."
	poetry run pytest

test-cov:
	@echo "Running tests with coverage..."
	poetry run pytest --cov=culora --cov-report=html --cov-report=term

# Maintenance
clean:
	@echo "Cleaning cache files and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Complete workflow
all: format check test
	@echo ""
	@echo "🎉 Complete workflow finished successfully!"
	@echo "   - Code formatted"
	@echo "   - Quality checks passed"
	@echo "   - Tests passed"

# CLI commands for quick access
run-version:
	@echo "Running CuLoRA version command..."
	poetry run culora version

run-help:
	@echo "Showing CuLoRA help..."
	poetry run culora --help