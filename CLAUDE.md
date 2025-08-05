# CuLoRA - Agent Memory & Development Guidelines

**Purpose:**
CuLoRA is a command-line tool for intelligently curating image datasets for LoRA training. It combines face detection, quality assessment, composition analysis, and duplicate detection to automatically select the best images from large datasets. All features must be exposed via the CLI.

---

## Development Discipline

### **Single Task Focus**

- Work on exactly one task at a time from `prompts/01-prototype.md` or active user prompt
- Plan your approach before coding - ask for clarification if requirements are ambiguous
- Only add code that is required for the current task
- All new code must be exposed via the public interface (CLI commands or tests)
- Remove any unused code immediately - no "dead" or speculative code

### **Quality Gates**

- `make pre-commit` must pass before any commit, PR, or task completion
- Type hints required on every function, method, and variable
- Use Rich for user output, structlog for machine logs - never print()
- Mock all external dependencies in tests (file I/O, network, AI models, hardware)

### **Task Completion Process**

1. Complete the implementation with tests
2. Run `make pre-commit` and ensure it passes
3. Update the task status in `prompts/01-prototype.md` with completion summary
4. Update this `CLAUDE.md` if any new patterns or conventions were established
5. Pause for review before moving to next task

---

## Architecture Rules

### **Service Layer Pattern**

- All business logic lives in `culora/services` - CLI only coordinates and displays
- Use auto-initializing global service functions: `get_*_service()`
- Services receive Pydantic config objects and handle all domain operations
- Services should be stateless and thread-safe

### **Configuration System**

- Pydantic models with strict validation in `culora/domain/models/config`
- Loading hierarchy: CLI args > environment vars > config files > defaults
- Environment variables follow `CULORA_<SECTION>_<FIELD>` pattern
- Only add config options when required by current task

### **Error Handling**

- Custom exceptions in `culora/core/exceptions` organized by domain
- Proper exception chaining with context for debugging
- User-friendly error messages via Rich console
- Graceful degradation when possible

---

## Directory Organization

```txt
culora/
├── cli/           # CLI layer only - no business logic
├── services/      # All business logic and external integrations
├── domain/        # Type definitions, Pydantic models, enums
├── core/          # Exception hierarchy, base classes
└── utils/         # Non-domain utilities (logging, app directories)

tests/
├── unit/          # Unit tests mirroring source structure
├── integration/   # End-to-end workflow tests
├── helpers/       # Reusable test utilities
├── mocks/         # Centralized mock implementations
└── fixtures/      # Static test data
```

---

## Testing Standards

### **Test Organization**

- Test files: `test_<unit>.py`, Test classes: `Test<Component>`
- Unit tests in `tests/unit/` mirroring source structure
- Integration tests in `tests/integration/` for complete workflows
- Use helpers from `tests/helpers/` for common operations

### **Mocking Requirements**

- Mock all file I/O - use `temp_dir` fixtures
- Mock all AI models - use `tests/mocks/ai_model_mocks.py`
- Mock all hardware - use `tests/mocks/pytorch_mocks.py`
- Mock all network operations
- Test the interface/contract, not implementation details

### **Coverage Expectations**

- Cover happy path and likely error scenarios
- Don't over-engineer edge cases unless specified
- Keep tests focused and maintainable
- Use fixtures and helpers to reduce duplication

---

## Code Style & Quality

### **Python Standards**

- Python 3.12+ with modern syntax (X | Y unions, updated annotations)
- Black formatting, isort imports, Ruff linting, mypy strict typing
- Google-style docstrings for public APIs
- snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants

### **Makefile Commands**

- `make pre-commit` - Complete workflow (format, check, test) - REQUIRED before review
- `make check` - Quick quality checks (format, lint, typecheck)
- `make dev-setup` - Initial environment setup
- Only use Poetry directly for exceptional cases

---

## CLI Development

### **Command Structure**

- Commands organized by domain in `culora/cli/commands/`
- Register new commands in `culora/cli/app.py`
- Use Rich for beautiful output with consistent theming
- Comprehensive argument validation with helpful error messages

### **User Experience**

- All user-facing output through Rich console
- Structured logs go to files, not terminal
- Progress bars for long operations
- Clear error messages with suggested solutions

---

## Current Project State

### **Service Architecture**

The project uses a clean service layer with these established patterns:

- `get_config_service()` - Configuration management
- `get_device_service()` - Hardware detection and optimization
- `get_image_service()` - Image loading and processing
- `get_face_analysis_service()` - Face detection with InsightFace

### **AI Model Integration**

- InsightFace for face detection with device optimization
- Output suppression for chatty third-party libraries
- Cross-platform model caching using Typer app directories
- Batch processing with memory-aware sizing

### **Implementation Status**

See `prompts/01-prototype.md` for complete roadmap and current progress. Update that file with task completion summaries, not this one.

---

## Anti-Patterns (Never Do)

- Business logic in CLI modules
- Global state (except documented service singletons)
- Real external calls in tests
- Code additions without current task justification
- Skipping quality checks to move faster
- print() statements or direct stdout writes

---

## Quick Reference

**Add CLI command:** Create in `culora/cli/commands/`, register in `app.py`, test interface
**Add service:** Implement in `culora/services/`, add domain models, create global accessor
**Add config:** Extend Pydantic models, add environment mapping, update CLI if needed
**Debug issues:** Check `make pre-commit`, verify mocks in tests, use Rich for user feedback

---

## Critical Dependencies

- **Typer + Rich:** CLI framework with beautiful terminal output
- **Pydantic:** Type-safe configuration with validation
- **InsightFace + ONNX:** Face detection with device optimization
- **Pillow + OpenCV:** Image processing foundation
- **pytest + mocks:** Testing with external dependency isolation

---

## Agent Behavior

- Act like a methodical, quality-focused Python developer
- Always ask for clarification when requirements are unclear
- Update this file when new patterns or conventions are established
- Document key decisions and trade-offs
- Maintain discipline around single-task focus and quality gates

---

## Initiatives

- [`prompts/01-prototype.md`](prompts/01-prototype.md) – Culora V1 Implementation Plan (update with task progress)
