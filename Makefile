.PHONY: help install format lint typecheck test check

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and setup pre-commit hooks
	@printf "Installing dependencies..."
	@poetry install > /dev/null 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| echo "\033[0;31mfail\033[0m"

	@printf "Installing pre-commit hooks..."
	@poetry run pre-commit install > /dev/null 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| echo "\033[0;31mfail\033[0m"

format:  ## Format code with black and isort
	@poetry run black ./src ./tests
	@poetry run isort --profile black ./src ./tests

lint:  ## Lint code with ruff
	@poetry run ruff check --fix ./src ./tests

typecheck:  ## Type check code with pyrightj
	@poetry run pyright ./src ./tests

test:  ## Run tests with pytest
	@poetry run pytest -qq --disable-warnings --no-cov-on-fail --color=yes .

check:  ## Run all checks
	@CHECK_LOG=$$(mktemp /tmp/check-log.XXXXXX); \
	FAIL=0; \
	pretty() { printf "%-25s" "$$1"; }; \
	pretty "Formatting code..."; \
	$(MAKE) format > $$CHECK_LOG 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| (echo "\033[0;31mfail\033[0m"; cat $$CHECK_LOG; FAIL=1); \
	pretty "Linting code..."; \
	$(MAKE) lint > $$CHECK_LOG 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| (echo "\033[0;31mfail\033[0m"; cat $$CHECK_LOG; FAIL=1); \
	pretty "Type checking code..."; \
	$(MAKE) typecheck > $$CHECK_LOG 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| (echo "\033[0;31mfail\033[0m"; cat $$CHECK_LOG; FAIL=1); \
	pretty "Running tests..."; \
	$(MAKE) test > $$CHECK_LOG 2>&1 \
		&& echo "\033[0;32mdone\033[0m" \
		|| (echo "\033[0;31mfail\033[0m"; cat $$CHECK_LOG; FAIL=1); \
	rm -f $$CHECK_LOG; \
	exit $$FAIL
