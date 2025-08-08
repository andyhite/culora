# CuLoRA - Agent Memory & Development Guidelines

**Purpose:**
CuLoRA is a command-line tool for intelligently curating image datasets for LoRA training. It combines deduplication, image quality assessment, and face detection to automatically select the best images from large datasets using a sophisticated two-tier selection system with composite scoring. All features are exposed via the CLI. The analysis pipeline is modular—each stage (deduplication, quality, face) is enabled by default but can be toggled off via CLI flags.

---

## Development Discipline

- Work on one user story or research task at a time (from `prompts/01-prototype.md` or current user prompt).
- Plan before coding—ask if anything is unclear.
- Only write code required for the current task; no speculative or dead code.
- All code must be reachable from the CLI or tests.
- Immediately remove any unused code.
- **Before presenting any work for review**, you must run `make check` and it must pass.
- All functions, methods, and variables must use type hints.
- All user-facing output must use Rich; never use `print()` or logging for output.
- Mock all external dependencies in tests.
- Summarize research findings for each analysis stage in PRs or code comments.
- Analysis pipeline library decisions are documented in @docs/analysis-libraries.md
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

- Analysis pipeline stages (deduplication, quality, face detection) run independently on all images.
- Each stage is enabled by default, with individual CLI flags to disable (`--no-dedupe`, `--no-quality`, `--no-face`).
- Pipeline must auto-detect and use the best available device for acceleration.
- All results must be saved in a single JSON file per analyzed directory.
- Output must always show which stages were enabled, detailed per-image results, and composite scores.
- Two-tier selection system: Tier 1 applies minimum thresholds (culling), Tier 2 ranks by composite score.
- Composite scoring combines quality (50%) and face metrics (50%) with intelligent weighting.
- Face scoring uses relative face-to-image area ratios with sigmoid curves for optimal sizing (5-25% range, peak at 15%).
- Face confidence acts as multiplier, multi-face penalties apply (10% per additional face, max 50%).
- All scores are clamped to [0.0, 1.0] range for consistent ranking.
- Support `--max-images` parameter to select top N images by composite score.

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

## Known Issues & Workarounds

- **YOLO11 Messages**: Face detection may output minimal technical messages from YOLO/TensorFlow during model initialization. These are harmless and indicate proper GPU/CPU setup. For completely clean output:

  ```bash
  culora analyze <input_dir> 2>/dev/null
  ```

## Current Implementation Status

### Completed Features (Production Ready)

- ✅ **Multi-stage Analysis Pipeline**: Deduplication, quality assessment, and face detection
- ✅ **Two-tier Selection System**: Threshold culling + score-based ranking  
- ✅ **Composite Scoring Algorithm**: Sophisticated scoring with relative face sizing
- ✅ **CLI Integration**: Complete `analyze` command with all options
- ✅ **Rich Output**: Progress bars, detailed results tables, and colored scoring
- ✅ **Device Acceleration**: Automatic CUDA/MPS/CPU detection and optimization
- ✅ **Caching System**: Per-directory JSON cache for analysis results
- ✅ **Face Bounding Boxes**: Draw detection boxes with confidence scores
- ✅ **Dry Run Mode**: Preview selection without file operations
- ✅ **Max Images Limiting**: Select top N images by composite score

### Key Technical Decisions

- **Face Detection**: Switched to YOLO11 specialized face model (AdamCodd/YOLOv11n-face-detection) for superior accuracy vs general person detection
- **Scoring Algorithm**: Rebalanced to 50%/50% quality/face weights with sophisticated face area scoring using sigmoid curves
- **Selection Architecture**: Two-tier system separates threshold culling from ranking for optimal results
- **Progress UI**: Transient progress bars with integrated directory info for clean output

---

## Initiative Tracking

- All progress, task status, and user stories are tracked in @prompts/01-prototype.md
