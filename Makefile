.PHONY: format lint typecheck test pre-commit install

install:
	@echo "Installing dependencies..."
	@poetry install

format:
	@echo "Formatting code..."
	@poetry run black ./src ./tests
	@poetry run isort ./src ./tests

lint:
	@echo "Linting code..."
	@poetry run ruff check --fix ./src ./tests

typecheck:
	@echo "Type checking code..."
	@poetry run pyright ./src ./tests

test:
	@echo "Running tests..."
	@poetry run pytest

pre-commit: format lint typecheck test
	@echo "All checks passed!"
