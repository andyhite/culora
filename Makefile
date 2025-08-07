.PHONY: format lint typecheck test pre-commit install

install:
	@poetry install

format:
	@poetry run black ./src ./tests
	@poetry run ruff check --fix ./src ./tests

lint:
	@poetry run ruff check ./src ./tests

typecheck:
	@poetry run pyright ./src ./tests

test:
	@poetry run pytest

pre-commit: format lint typecheck test
	@echo "All checks passed!"
