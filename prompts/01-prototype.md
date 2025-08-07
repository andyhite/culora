# Culora MVP User Stories & Research Tasks

---

## Epic 1: Modern Python CLI Foundation

- ~~**US1.1:** As a developer, I need a modern Python project setup (Poetry, Black, Ruff, Pylance, Pytest) so that contributing is fast and code quality is always enforced.~~
- ~~**US1.2:** As a developer, I want a clear `src/` directory layout with distinct modules (`cli`, `analysis`, `utils`) so the codebase is maintainable.~~
- ~~**US1.3:** As a user, I want `culora --help` to display clear subcommands and usage info for `analyze` and `select`.~~
- **US1.4:** As a user, I want all CLI output styled and readable (via Rich), so every interaction is visually clear.
- **US1.5:** As a developer, I need a cross-platform app data directory for analysis caches.
- **US1.6:** As a developer, I want basic Pytest tests for the CLI so I know the baseline is solid.

---

## Epic 2: End-to-End CLI Command Flow

- **US2.1:** As a user, I want to analyze a folder of images with `culora analyze <input_dir>`, so I can curate my dataset.
- **US2.2:** As a user, I want to see a Rich progress bar during analysis so I know it’s working.
- **US2.3:** As a user, I want to run `culora select <output_dir>` and have only curated images copied to a new folder.
- **US2.4:** As a user, I want a summary table after selection, showing how many images were found, selected, skipped, and why.
- **US2.5:** As a user, I want all errors (missing folder, no images, permissions) to be shown clearly in the terminal—never as a stack trace.
- **US2.6:** As a user, I want analysis results to persist between runs, so re-analysis is only needed if images change.

---

## Epic 3: Efficient, Modular Analysis Pipeline & Caching

### Research Tasks (do these first)

- **RT3.1:** Research and shortlist the most reliable, dependency-light Python libraries for **image deduplication** via perceptual hashing. Compare options like `imagehash`, `imagededup`, `phash`, etc., considering performance, accuracy, and ease of use.
- **RT3.2:** Evaluate lightweight libraries and algorithms for **image quality assessment** (sharpness, brightness, contrast). Benchmark Pillow/OpenCV built-ins versus any specialized packages.
- **RT3.3:** Research current Python libraries for **face detection** that are easy to install, work cross-platform, and efficiently use available device acceleration (GPU/MPS/CPU).

### User Stories / Implementation Tasks

- **US3.1:** As a user, I want deduplication, quality filtering, and face detection all enabled by default, so my results are the best possible.
- **US3.2:** As a user, I want to optionally disable deduplication (`--no-dedupe`), quality filtering (`--no-quality`), or face detection (`--no-face`) via CLI flags.
- **US3.3:** As a user, I want each image’s analysis (per enabled stage) recorded in the per-directory JSON cache, including pass/fail/skipped reason.
- **US3.4:** As a user, I want every analysis stage to only run on images that passed the previous stage, so the pipeline is efficient.
- **US3.5:** As a user, I want all analysis and filtering progress and results displayed using Rich (no logs, no sidecars).
- **US3.6:** As a user, I want analysis to use and update the cache intelligently so that repeated runs are fast.
- **US3.7:** As a user, I want the analysis pipeline to automatically detect and use the best available device (GPU/CUDA, MPS, or CPU) for acceleration, so that performance is maximized regardless of my hardware.

---

## Epic 4: Intelligent Image Selection

- **US4.1:** As a user, I want only images that passed all *enabled* analysis stages to be copied to my output directory.
- **US4.2:** As a user, I want the selected images to be renamed/numbered sequentially, making them training-ready.
- **US4.3:** As a user, I want to see a summary table after selection, including counts by reason for selection or skipping.
- **US4.4:** As a user, I want to run with a `--dry-run` flag to preview the output without actually copying files.
- **US4.5:** As a developer, I want selection logic to always respect the set of analysis stages enabled at analysis time, so behavior is always predictable.

---

## Cross-cutting Stories

- **USX.1:** As a user, I want all output—progress, errors, and summaries—to use Rich, never raw prints or logs.
- **USX.2:** As a user, I want minimal and obvious config with sane defaults, so I can use the tool without reading a manual.
- **USX.3:** As a user, I want Culora to work on Mac, Linux, and Windows.
