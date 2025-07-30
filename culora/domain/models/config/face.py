"""Face analysis configuration model."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from culora.utils.app_dir import get_models_dir


class FaceAnalysisConfig(BaseModel):
    """Configuration for face detection and analysis.

    Controls all aspects of face analysis including model selection,
    detection thresholds, device preferences, and performance settings.
    """

    # Model configuration
    model_name: str = Field(
        default="buffalo_l",
        description="InsightFace model to use for detection and analysis",
    )
    model_cache_dir: Path = Field(
        default_factory=get_models_dir,
        description="Directory to cache downloaded models",
    )

    # Detection parameters
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for face detection",
    )
    max_faces_per_image: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of faces to detect per image",
    )

    # Device and performance
    device_preference: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto", description="Preferred device for model execution"
    )
    batch_size: int = Field(
        default=8, ge=1, le=64, description="Batch size for processing multiple images"
    )

    # Model features
    extract_embeddings: bool = Field(
        default=True,
        description="Whether to extract face embeddings for similarity matching",
    )
    extract_landmarks: bool = Field(
        default=True, description="Whether to extract facial landmark points"
    )
    extract_attributes: bool = Field(
        default=False,
        description="Whether to extract age/gender attributes (if model supports)",
    )

    # Output and visualization
    embedding_size: int = Field(
        default=512, ge=128, le=2048, description="Expected face embedding dimension"
    )
    normalize_embeddings: bool = Field(
        default=True, description="Whether to normalize face embeddings to unit length"
    )

    # Performance tuning
    enable_model_caching: bool = Field(
        default=True,
        description="Whether to cache loaded models for faster subsequent runs",
    )
    memory_optimization: bool = Field(
        default=True,
        description="Enable memory optimization for large batch processing",
    )

    @field_validator("model_cache_dir")
    @classmethod
    def validate_model_cache_dir(cls, v: Path | str) -> Path:
        """Validate and ensure model cache directory exists."""
        # Convert string to Path if needed
        if isinstance(v, str):
            v = Path(v)

        # Expand user path
        v = v.expanduser().resolve()

        # Create directory if it doesn't exist
        try:
            v.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create model cache directory {v}: {e}") from e

        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate InsightFace model name."""
        # List of known InsightFace models
        known_models = {
            "buffalo_l",  # Most accurate, largest model
            "buffalo_m",  # Medium accuracy and size
            "buffalo_s",  # Fastest, smallest model
            "antelopev2",  # Alternative high-accuracy model
        }

        if v not in known_models:
            # Allow custom model names but show warning in logs
            # This validator doesn't have access to logger, so we just accept it
            pass

        return v

    @property
    def requires_gpu(self) -> bool:
        """Check if configuration requires GPU acceleration."""
        return self.device_preference in ("cuda", "mps")

    @property
    def model_cache_path(self) -> Path:
        """Get full path to cached model directory."""
        return self.model_cache_dir / self.model_name

    def get_device_context_providers(self) -> list[str]:
        """Get appropriate execution providers for the configured device."""
        if self.device_preference == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device_preference == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif self.device_preference == "cpu":
            return ["CPUExecutionProvider"]
        else:  # auto
            # Auto selection - will be determined by DeviceService
            return []

    def get_memory_optimized_batch_size(self, available_memory_gb: float) -> int:
        """Calculate optimal batch size based on available memory.

        Args:
            available_memory_gb: Available memory in gigabytes

        Returns:
            Optimized batch size for the available memory
        """
        if not self.memory_optimization:
            return self.batch_size

        # Rough estimate: each face analysis uses ~100MB of memory
        # This is conservative and can be tuned based on actual measurements
        memory_per_batch_item_gb = 0.1
        max_batch_from_memory = int(available_memory_gb / memory_per_batch_item_gb)

        # Use the smaller of configured batch size or memory-constrained size
        return min(self.batch_size, max(1, max_batch_from_memory))
