# Culora V1 Implementation Plan - Advanced LoRA Dataset Curation Utility

## Project Overview

Build "Culora" - an advanced command-line utility for intelligently curating image datasets specifically for LoRA training. The system combines multiple AI models with sophisticated selection algorithms to automatically identify and select the best images from large datasets for optimal training outcomes.

## Modern Development Standards

### **Code Quality Requirements**

- **Type Checking**: Full type hints with mypy for static type analysis
- **Code Formatting**: Black for consistent code formatting
- **Import Sorting**: isort for organized imports
- **Linting**: Ruff for fast, modern Python linting
- **Testing**: pytest with comprehensive test coverage
- **Documentation**: Docstrings following Google style conventions

### **Development Workflow**

After each implementation task, run the following quality checks:

1. `black .` - Format all code
2. `isort .` - Sort imports
3. `ruff check .` - Lint code for issues
4. `mypy .` - Type check all modules
5. `pytest` - Run test suite

### **Modern Python Tooling Stack**

- **Dependency Management**: Poetry 1.7+ with lock file management
- **Type Checking**: mypy with strict configuration
- **Formatting**: Black with line length 88
- **Import Sorting**: isort with Black-compatible profile
- **Linting**: Ruff (replacing flake8, pylint, etc.)
- **Testing**: pytest with fixtures and parametrization
- **Logging**: structlog for structured, machine-readable logs
- **CLI**: Typer with Rich integration for beautiful terminal output

---

## WEEK 1: Project Foundation and Modern Tooling Setup

### **Task 1.1: Project Structure and Poetry Configuration**

**Goal**: Establish modern Python project with comprehensive tooling

**Requirements**:

- Create complete Poetry pyproject.toml with all dependencies and tool configurations
- Establish clean modular directory structure with proper package organization
- Configure all modern Python tools (black, isort, mypy, ruff, pytest) with consistent settings
- Set up Poetry script entry point for CLI
- Create comprehensive .gitignore for Python projects

**Project Structure**:

```txt
culora/
├── culora/
│   ├── core/          # Foundation: exception hierarchy
│   │   └── exceptions/ # Modular exception classes (config, device, culora)
│   ├── domain/        # Domain-driven design models and enums
│   │   ├── enums/     # Type-safe enums (device types, log levels)
│   │   └── models/    # Domain models (device, memory, config)
│   ├── services/      # Service layer (config, device, memory services)
│   └── utils/         # Shared utilities (logging)
├── tests/             # Best-practice test organization
│   ├── conftest.py    # Shared pytest fixtures
│   ├── helpers/       # Test utilities (factories, assertions, file utils)
│   ├── mocks/         # Mock implementations (PyTorch, AI models)
│   ├── fixtures/      # Static test data and configurations
│   ├── unit/          # Unit tests organized by domain
│   └── integration/   # Integration and workflow tests
├── pyproject.toml     # Poetry configuration with all tools
└── README.md
```

**Dependencies to Include**:

- Core: typer[rich], rich, pydantic, structlog
- Image processing: pillow, numpy, opencv-python
- AI models: insightface, mediapipe, piq, imagehash, transformers, torch, sentence-transformers, scikit-learn
- Development: black, isort, mypy, ruff, pytest, pytest-cov, pytest-mock

**Python Version**: Python 3.12 with modern syntax (X | Y unions, updated type annotations)

**✅ TASK 1.1 COMPLETED**: Successfully implemented modern Python project foundation with Python 3.12 and all dependencies updated to latest compatible versions (Ruff 0.12.5, Black 25.1.0, pytest 8.4.1, mypy 1.14.1, etc.). Complete modular directory structure established with comprehensive tooling configuration and modern Python syntax throughout.

### **Task 1.2: Structured Logging and Configuration Foundation**

**Goal**: Establish structured logging that separates user-facing output from machine logs

**Requirements**:

- Configure structlog to write structured JSON logs to file for debugging
- Separate user-facing Rich console output from internal logging
- Create Pydantic configuration models with full validation
- Implement proper error handling and custom exception classes
- Design type-safe configuration system for all components

**Key Components**:

- Logging configuration that routes to files, not console
- Pydantic models for face analysis, quality assessment, selection parameters
- Custom exception hierarchy for different error types
- Configuration validation with helpful error messages

**Testing Requirements**:

- Create pytest fixtures for common test scenarios
- Set up temporary directory fixtures for file operations
- Mock configurations and logger instances
- Test configuration validation edge cases

**✅ TASK 1.2 COMPLETED**: Successfully implemented comprehensive structured logging and configuration foundation with 65 passing tests and 100% code coverage. Key implementation notes:

- **PIQ Library**: Integrated PIQ (PyTorch Image Quality) for BRISQUE analysis with GPU acceleration
- **Environment Variables**: Comprehensive parsing with field mapping (e.g., `CULORA_DEVICE_BATCH_SIZE` → `device.batch_size`)
- **Exception Chaining**: Proper exception chaining throughout for debugging
- **Type Safety**: Full mypy compliance with strict mode
- **Configuration System**: Multi-source precedence (CLI > env > file > defaults)

---

## WEEK 2: Device Management and CLI Foundation

### **Task 2.1: Hardware Detection and Device Management**

**Goal**: Intelligent device detection and selection for optimal AI model execution

**Requirements**:

- Detect CUDA GPUs with memory analysis and availability checking
- Detect Apple Silicon MPS with compatibility checking
- Provide CPU fallback for universal compatibility
- Log device information to structured logs, display summary to user via Rich

**Device Management Features**:

- Automatic optimal device selection with manual override
- Memory usage monitoring and estimation for model loading
- Device-specific execution provider configuration (CUDA/MPS/CPU)
- Graceful handling of device detection failures

**Testing Requirements**:

- Mock torch.cuda and torch.backends.mps for testing
- Test device detection on different simulated hardware
- Verify fallback behavior when preferred devices unavailable
- Test device selection and configuration logic

**✅ TASK 2.1 COMPLETED**: Successfully implemented comprehensive device detection and management system with modern test infrastructure and full Python 3.12 compatibility. Key implementation details:

- **Device Detection**: Robust CUDA GPU detection with memory analysis, Apple Silicon MPS detection, and CPU fallback
- **Service Architecture**: Clean service layer with DeviceService and MemoryService separation
- **Domain Models**: Type-safe device and memory models in domain layer
- **Smart Selection**: Priority-based selection (CUDA > MPS > CPU) with user preference override support
- **Memory Management**: Memory availability checking and model-specific usage estimation
- **Rich Integration**: Beautiful device status tables with real-time information display
- **Comprehensive Testing**: Full test coverage with mocked hardware scenarios and edge cases

All device management functionality is production-ready with proper error handling and logging integration.

**✅ ARCHITECTURE REFACTOR COMPLETED**: Successfully reorganized entire codebase using domain-driven design principles:

- **Domain Layer**: Created `culora/domain/` with type-safe enums and domain models
- **Service Layer**: Implemented `culora/services/` with config, device, and memory services
- **Exception Hierarchy**: Modularized exceptions into `culora/core/exceptions/` by domain
- **Clean Architecture**: Removed unused modules (cli, analysis, selection, export) for focused development
- **Test Infrastructure**: Restructured tests with industry-standard organization (helpers, mocks, fixtures, unit, integration)
- **255 Tests**: All tests passing with comprehensive coverage of the new architecture

All core foundation components (types, exceptions, logging, configuration) are production-ready with best-practice test organization:

- **Test Helpers**: Modular utilities in `tests/helpers/` (factories, assertions, file utils)
- **Mock Implementations**: Centralized PyTorch/CUDA mocks in `tests/mocks/`
- **Organized Tests**: Unit tests in `tests/unit/` by domain, integration tests in `tests/integration/`
- **Shared Fixtures**: Common test setup in `tests/conftest.py`
- **255 Tests**: Comprehensive coverage with modern test organization

### **Task 2.2: Typer CLI with Rich Integration**

**Goal**: Beautiful, modern CLI with excellent user experience

**Requirements**:

- Create Typer-based CLI with comprehensive argument validation
- Integrate Rich for beautiful console output, progress bars, and tables
- Design custom Rich theme for consistent styling throughout application
- Implement clear error handling with helpful suggestions
- Create structured command interface with logical groupings

**CLI Features**:

- Rich console themes with colors for different message types
- Progress bars for long-running operations with time estimates
- Configuration summary tables displayed beautifully
- Error messages with suggested solutions
- Help text with examples and detailed descriptions

**User Experience Requirements**:

- All user-facing output through Rich console, never print()
- Structured logs go to files, Rich output to terminal
- Clear validation messages for invalid inputs
- Beautiful headers and section separators
- Consistent emoji and styling throughout

**Testing Requirements**:

- Use Typer's CliRunner for testing CLI interactions
- Test argument validation and error handling
- Mock Rich console for testing output formatting
- Verify help text and command structure

---

## WEEK 3: Face Analysis System with InsightFace

### **Task 3.1: Face Detection and Analysis Core**

**Goal**: Robust face detection with comprehensive analysis capabilities

**Requirements**:

- Integrate InsightFace for high-accuracy face detection and recognition
- Extract face embeddings for identity matching and similarity calculation
- Calculate face area ratios relative to image size
- Extract additional face attributes (age, gender, landmarks) when available
- Handle device-specific optimization (CUDA/MPS/CPU execution providers)

**Face Analysis Features**:

- Multiple face detection per image with confidence scoring
- Face bounding box calculation in standardized format
- Face quality assessment based on size, confidence, and clarity
- Robust error handling for corrupted or invalid images
- Memory-efficient processing for large datasets

**Data Structures**:

- Define typed data structures for face detection results
- Use NamedTuple or dataclass for face detection information
- Type-safe handling of numpy arrays and embeddings
- Structured results with success/failure status and error messages

### **Task 3.2: Reference Image Matching System**

**Goal**: Identity consistency through reference image matching

**Requirements**:

- Load and process reference images to extract face embeddings
- Implement cosine similarity calculation for face matching
- Support multiple reference images with averaged similarity scores
- Handle cases where reference images contain no faces or multiple faces
- Provide configurable similarity thresholds for identity matching

**Reference Matching Features**:

- Robust reference image validation and preprocessing
- Multi-reference averaging for better identity consistency
- Fallback strategies when reference matching fails
- Clear logging of reference processing success/failure
- Similarity score reporting for transparency

**Multi-Person Image Handling**:

- Intelligent primary face selection using reference matching
- Fallback to largest face when reference matching unavailable
- Option to skip multi-person images entirely
- Clear reporting of multi-person handling decisions

**Testing Requirements**:

- Create test images with known face characteristics
- Mock InsightFace model responses for consistent testing
- Test reference matching with various similarity scenarios
- Verify multi-person image handling logic
- Test error handling for corrupted images and model failures

---

## WEEK 4: Quality Assessment Pipeline

### **Task 4.1: Technical Quality Metrics**

**Goal**: Comprehensive technical image quality assessment

**Requirements**:

- Implement sharpness calculation using Laplacian variance
- Calculate brightness and contrast metrics from image histograms
- Assess color distribution and saturation levels
- Detect potential noise and compression artifacts
- Combine metrics into unified technical quality scores

**Quality Metrics Implementation**:

- Sharpness assessment optimized for face-focused images
- Brightness evaluation with optimal range detection
- Contrast analysis with dynamic range assessment
- Color quality scoring with saturation balance
- Noise detection using statistical analysis

### **Task 4.2: BRISQUE Perceptual Quality Assessment**

**Goal**: No-reference perceptual quality scoring

**Requirements**:

- Integrate BRISQUE algorithm through PIQ (PyTorch Image Quality) library
- Handle BRISQUE calculation errors gracefully
- Provide configurable quality thresholds for filtering
- Optimize BRISQUE processing for batch operations
- Combine BRISQUE scores with technical metrics

**BRISQUE Integration Features**:

- Efficient BRISQUE calculation with error handling
- Quality threshold validation and filtering
- Performance optimization for large image sets
- Clear reporting of quality assessment results

### **Task 4.3: Composite Quality Scoring System**

**Goal**: Unified quality assessment combining multiple factors

**Requirements**:

- Design weighted scoring system combining technical and perceptual metrics
- Include face-specific quality adjustments
- Provide bonus scoring for reference similarity matches
- Support configurable quality weights for different use cases
- Generate quality rankings and percentile scores

**Composite Scoring Features**:

- Configurable weight system for different quality factors
- Face quality bonus based on detection confidence and size
- Reference matching bonus for identity consistency
- Normalized scoring for consistent comparisons
- Quality distribution analysis and reporting

**Testing Requirements**:

- Create test images with known quality characteristics
- Test quality metric calculations with edge cases
- Verify composite scoring logic and weight applications
- Mock BRISQUE calculations for consistent testing
- Test quality threshold filtering behavior

---

## WEEK 5: Composition and Pose Analysis

### **Task 5.1: Vision-Language Composition Classification**

**Goal**: Intelligent composition analysis using Moondream

**Requirements**:

- Integrate Moondream vision-language model for image classification
- Design effective prompts for consistent composition analysis
- Classify shot types (portrait, medium shot, full body)
- Analyze scene characteristics (indoor, outdoor, studio)
- Assess lighting quality and background complexity
- Extract facial expressions and camera angles

**Moondream Integration Features**:

- Efficient model loading with device-specific optimization
- Robust prompt engineering for consistent responses
- Response parsing and validation with fallback handling
- Batch processing optimization for performance
- Error handling for model failures and timeouts

**Classification Categories**:

- Shot type classification with clear category boundaries
- Scene type detection for environmental context
- Lighting quality assessment for technical evaluation
- Background complexity scoring for subject focus
- Expression analysis for emotional variety
- Camera angle detection for pose diversity

### **Task 5.2: CLIP Semantic Embeddings**

**Goal**: Semantic understanding for composition diversity

**Requirements**:

- Integrate CLIP model for semantic image embeddings
- Extract composition embeddings for similarity analysis
- Support clustering and diversity optimization
- Handle device-specific CLIP execution
- Provide semantic similarity scoring between images

**CLIP Integration Features**:

- Efficient CLIP model loading and inference
- Semantic embedding extraction optimized for composition analysis
- Memory-efficient processing for large image sets
- Device-specific optimization (CUDA/MPS/CPU)
- Embedding comparison and similarity calculation

### **Task 5.3: MediaPipe Pose Estimation**

**Goal**: Pose analysis for diversity optimization

**Requirements**:

- Integrate MediaPipe for full body pose detection
- Extract key body landmarks for pose analysis
- Generate pose feature vectors for clustering
- Handle pose detection failures gracefully
- Support pose diversity scoring and comparison

**Pose Analysis Features**:

- Robust pose landmark detection with confidence filtering
- Key landmark selection for diversity analysis
- Pose vector generation for clustering algorithms
- Visibility-based landmark validation
- Pose diversity calculation and scoring

**Testing Requirements**:

- Mock vision-language model responses for consistent testing
- Test composition classification with various image types
- Verify CLIP embedding extraction and similarity calculation
- Test pose detection with different body positions
- Mock model failures and test error handling

---

## WEEK 6: Selection Algorithms and Duplicate Detection

### **Task 6.1: Perceptual Duplicate Detection**

**Goal**: Identify and handle duplicate or near-duplicate images

**Requirements**:

- Implement perceptual hash calculation using ImageHash library
- Design configurable distance thresholds for similarity detection
- Group duplicate images and select highest quality representatives
- Optimize duplicate detection for large datasets
- Provide clear reporting of duplicate removal decisions

**Duplicate Detection Features**:

- Efficient perceptual hash calculation with error handling
- Configurable similarity thresholds for different use cases
- Quality-based duplicate resolution with clear criteria
- Batch processing optimization for performance
- Detailed duplicate removal reporting

### **Task 6.2: Multi-Criteria Selection Algorithm**

**Goal**: Sophisticated selection balancing quality, diversity, and distribution

**Requirements**:

- Implement target distribution management for composition categories
- Design quality-based filtering with configurable thresholds
- Create diversity optimization using clustering techniques
- Handle edge cases where target distributions cannot be met
- Provide fallback selection for remaining slots

**Selection Algorithm Features**:

- Flexible target distribution specification (ratios or counts)
- Multi-stage filtering: quality → duplicates → diversity → distribution
- Intelligent cluster-based diversity selection
- Quality-weighted selection within diversity constraints
- Clear selection reasoning and reporting

### **Task 6.3: Clustering-Based Diversity Optimization**

**Goal**: Maximize visual diversity while maintaining quality

**Requirements**:

- Implement pose-based clustering when pose data available
- Provide CLIP-based semantic clustering as fallback
- Design greedy diversity selection algorithms
- Balance diversity optimization with quality requirements
- Support configurable diversity weights and parameters

**Diversity Optimization Features**:

- K-means clustering on pose vectors for pose diversity
- Semantic clustering using CLIP embeddings for composition variety
- Greedy selection minimizing similarity while preserving quality
- Multi-criteria optimization balancing competing objectives
- Configurable diversity parameters for different use cases

**Testing Requirements**:

- Test duplicate detection with known similar images
- Verify selection algorithm with various target distributions
- Test clustering algorithms with diverse image sets
- Mock clustering results for consistent testing
- Test edge cases where targets cannot be met

---

## WEEK 7: Export System and Final Integration

### **Task 7.1: Flexible Export System**

**Goal**: Comprehensive export with multiple output formats

**Requirements**:

- Implement sequential image naming optimized for training workflows
- Generate comprehensive JSON metadata with all analysis results
- Support face visualization with bounding box overlays
- Provide multiple copy modes (selected only vs. all with status)
- Maintain original filename mapping for traceability

**Export Features**:

- Training-optimized sequential naming (01.jpg, 02.jpg, etc.)
- Comprehensive JSON export with all analysis metadata
- Optional face bounding box visualization on output images
- Flexible copy modes based on user requirements
- Original filename preservation and mapping

### **Task 7.2: Analysis Reporting and Statistics**

**Goal**: Rich reporting with actionable insights

**Requirements**:

- Generate selection summary with distribution analysis
- Provide quality statistics and score distributions
- Report processing performance and success rates
- Analyze duplicate detection and removal statistics
- Create composition and scene distribution reports

**Reporting Features**:

- Beautiful Rich tables and charts for statistics display
- Quality score distributions and percentile analysis
- Processing time and efficiency reporting
- Duplicate removal summary with quality improvements
- Composition balance analysis and recommendations

### **Task 7.3: End-to-End Integration and Performance Optimization**

**Goal**: Complete system integration with optimal performance

**Requirements**:

- Integrate all analysis components into unified processing pipeline
- Implement efficient memory management for large datasets
- Optimize processing order for early filtering and performance
- Handle edge cases and error conditions gracefully
- Provide comprehensive progress reporting throughout process

**Integration Features**:

- Streamlined processing pipeline with intelligent early filtering
- Memory-efficient batch processing with garbage collection
- Progress reporting with time estimates and detailed status
- Robust error handling with graceful degradation
- Performance optimization for different hardware configurations

**Testing Requirements**:

- End-to-end integration tests with complete workflows
- Performance testing with large datasets
- Memory usage testing and optimization validation
- Error handling testing with various failure scenarios
- Cross-platform compatibility testing

---

## WEEK 8: Testing, Documentation, and Production Readiness

### **Task 8.1: Comprehensive Test Suite**

**Goal**: Thorough testing ensuring reliability and robustness

**Requirements**:

- Create comprehensive unit tests for all components
- Implement integration tests for complete workflows
- Add performance tests for scalability validation
- Test error handling and edge cases thoroughly
- Ensure cross-platform compatibility

**Testing Coverage**:

- Unit tests for face analysis, quality assessment, composition classification
- Integration tests for selection algorithms and export functionality
- Performance benchmarks for processing speed and memory usage
- Error simulation tests for robustness validation
- Mock-based testing for AI model components

### **Task 8.2: Documentation and User Guides**

**Goal**: Complete documentation for users and developers

**Requirements**:

- Create comprehensive README with installation and usage instructions
- Document CLI commands with detailed examples
- Provide configuration guides with recommended settings
- Create troubleshooting guide for common issues
- Document API with type information and examples

**Documentation Requirements**:

- Clear installation instructions for different platforms
- Example workflows for common use cases
- Configuration parameter explanations with recommendations
- Performance optimization guides for different hardware
- Troubleshooting section with common solutions

### **Task 8.3: Production Optimization and Release Preparation**

**Goal**: Final optimization and release readiness

**Requirements**:

- Optimize performance for different hardware configurations
- Implement final error handling and user experience improvements
- Create packaging and distribution configuration
- Validate cross-platform compatibility
- Prepare release documentation and changelog

**Production Features**:

- Optimized model loading and inference for all supported devices
- Enhanced error messages with actionable solutions
- Memory usage optimization and monitoring
- Performance profiling and bottleneck identification
- Release packaging with proper version management

## Success Criteria

### **Functional Requirements**

- Process datasets of 100+ images efficiently with clear progress reporting
- Achieve significant improvement in dataset quality through intelligent selection
- Maintain optimal distribution across composition categories
- Successfully identify and remove duplicates while preserving highest quality
- Generate training-ready datasets with comprehensive metadata

### **Performance Requirements**

- Process 2-10 images per second depending on hardware capabilities
- Memory usage optimized for large datasets without system strain
- Graceful performance scaling across different hardware configurations
- Clear progress reporting with accurate time estimates

### **User Experience Requirements**

- Intuitive CLI with helpful error messages and suggestions
- Beautiful progress reporting and statistics display via Rich
- Comprehensive configuration options with sensible defaults
- Clear documentation with examples for common use cases
- Reliable operation across different platforms and environments

### **Code Quality Requirements**

- 100% type coverage with mypy strict mode
- Comprehensive test suite with >90% coverage
- All code formatted with Black and linted with Ruff
- Structured logging for debugging and monitoring
- Clean, maintainable architecture with proper separation of concerns

This implementation plan delivers a sophisticated, production-ready LoRA dataset curation utility that combines multiple AI models with intelligent selection algorithms to significantly improve training dataset quality while maintaining the highest standards of modern Python development.
