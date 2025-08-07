.PHONY: help format lint typecheck test pre-commit pre-commit-install install

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and setup pre-commit hooks
	@echo "Installing dependencies..."
	@poetry install
	@echo "Installing pre-commit hooks..."
	@poetry run pre-commit install

format:  ## Format code with black and isort
	@echo "Formatting code..."
	@poetry run black .
	@poetry run isort .

lint:  ## Lint code with ruff
	@echo "Linting code..."
	@poetry run ruff check --fix .

typecheck:  ## Type check code with pyright
	@echo "Type checking code..."
	@poetry run pyright .

test:  ## Run tests with pytest
	@echo "Running tests..."
	@poetry run pytest

pre-commit:  ## Run all pre-commit hooks
	@echo "Running pre-commit hooks..."
	@poetry run pre-commit run --all-files
