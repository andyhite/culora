.PHONY: help install format lint typecheck test check

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
	@poetry run pre-commit run black --all-files
	@echo "\nSorting imports..."
	@poetry run pre-commit run isort --all-files

lint:  ## Lint code with ruff
	@echo "Linting code..."
	@poetry run pre-commit run ruff --all-files

typecheck:  ## Type check code with pyright
	@echo "Type-checking code..."
	@poetry run pre-commit run pyright --all-files

test:  ## Run tests with pytest
	@echo "Testing code..."
	@poetry run pre-commit run pytest --all-files

check:  ## Run all checks
	@echo "Running all checks..."
	@poetry run pre-commit run --all-files
