# CuLoRA - Agent Memory & Development Guidelines

**Purpose:**
CuLoRA is a command-line tool for intelligently curating image datasets for LoRA training. It combines deduplication, image quality assessment, and face detection to automatically select the best images from large datasets. All features must be exposed via the CLI. The analysis pipeline is modular—each stage (deduplication, quality, face) is enabled by default but can be toggled off via CLI flags.

---

## Development Discipline

- Work on one user story or research task at a time (from `prompts/01-prototype.md` or current user prompt).
- Plan before coding—ask if anything is unclear.
- Only write code required for the current task; no speculative or dead code.
- All code must be reachable from the CLI or tests.
- Immediately remove any unused code.
- **Before presenting any work for review,** run lint (`ruff`), typecheck (`pylance` or `mypy`), and tests (`pytest`).
  `make pre-commit` must also pass before any commit or merge.
- All functions, methods, and variables must use type hints.
- All user-facing output must use Rich; never use `print()` or logging for output.
- Mock all external dependencies in tests.
- Summarize research findings for each analysis stage in PRs or code comments.
- Pause for review after every task.

---

## Architecture & Design Principles

- Business logic is separated from CLI—CLI only parses args and displays results.
- Each analysis stage (deduplication, quality, face detection) is implemented as a modular, independently-toggleable service.
- Device selection (GPU/CUDA, MPS, CPU) must be automatic and optimized for each analysis stage.
- All analysis results are cached in a per-directory JSON file in the app data directory.
- Only add config options/flags when needed by the current user story or research task.
- Use Pydantic models for all configs and data validation.
- Use custom exceptions with user-friendly messages for all error handling.
- All user-facing errors must be displayed via Rich with actionable suggestions.
- No business logic in CLI modules.
- No global state except for documented service singletons.

---

## Testing Standards

- All new code must have unit tests and (where relevant) integration tests.
- Mock all file I/O, model loading/inference, device/hardware checks, and network calls in tests.
- Focus tests on interface and expected behavior, not internal implementation.
- Prioritize happy path and likely failure scenarios; avoid unnecessary edge cases.
- Use fixtures and helpers to avoid duplication in tests.

---

## Code Style

- Python 3.12+ with modern syntax (unions, annotations, etc).
- Black formatting, isort for imports, Ruff for linting, Pylance for type checking.
- Google-style docstrings for all public APIs.
- snake_case for functions and variables; PascalCase for classes; UPPER_SNAKE for constants.

---

## CLI Guidelines

- Each command and flag must be well-documented with clear descriptions and examples.
- Argument validation and helpful error messages are required.
- Progress bars and summary tables must use Rich for a consistent, beautiful experience.
- No logs or sidecar files for MVP—just the per-directory JSON cache.

---

## Analysis Pipeline Requirements

- Analysis pipeline stages (deduplication, quality, face detection) must run in order, with each stage only processing images that passed the prior stage.
- Each stage is enabled by default, with individual CLI flags to disable (`--no-dedupe`, `--no-quality`, `--no-face`).
- Pipeline must auto-detect and use the best available device for acceleration.
- All results must be saved in a single JSON file per analyzed directory.
- Output must always show which stages were enabled, the results of each, and any images skipped with reasons.

---

## Research Tasks

- Before implementing any analysis stage, research and select the best Python library for that task:
  - Deduplication: Shortlist and compare (e.g., imagehash, imagededup)
  - Quality assessment: Benchmark available tools (e.g., Pillow, OpenCV, imquality)
  - Face detection: Find a performant, installable, cross-platform library with device offload support

---

## Prohibited Anti-Patterns

- No business logic in CLI modules.
- No global state except documented service singletons.
- No real external calls in tests.
- No speculative or dead code.
- No skipping of required quality checks.
- No `print()` statements or direct stdout writes.

---

## Agent Behavior

- Work methodically, focusing on one task at a time.
- Ask for clarification if requirements are unclear.
- Update this file if new conventions or patterns emerge.
- Document major decisions and trade-offs in PRs or code comments.

---

## Initiative Tracking

- All progress, task status, and user stories are tracked in @prompts/01-prototype.md
