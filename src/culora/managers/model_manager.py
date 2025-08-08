"""Generic model management service for CuLoRA analysis pipeline."""

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.file_download import (
    hf_hub_download,  # type: ignore[import-untyped]
)
from ultralytics import YOLO

from culora.utils.app_data import get_models_dir

# Suppress ultralytics verbose output
os.environ.setdefault("YOLO_VERBOSE", "False")


class ModelManager:
    """Singleton model manager for CuLoRA analysis pipeline."""

    _instance: "ModelManager | None" = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model_cache: dict[str, Any] = {}
        self._device: str | None = None
        self._initialized = True

    def detect_optimal_device(self) -> str:
        """Detect the optimal device for model inference.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if self._device is None:
            try:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self._device = "mps"
                else:
                    self._device = "cpu"
            except Exception:
                # Fallback to CPU if torch detection fails
                self._device = "cpu"

        return self._device

    def get_cached_model(
        self, analysis_type: str, model_identifier: str, model_class: type = YOLO
    ) -> Any:
        """Get or create a cached model for the specified analysis type.

        Args:
            analysis_type: Type of analysis (e.g., 'face_detection', 'image_quality')
            model_identifier: Either a model filename or HF repo format ('repo:filename')
            model_class: Model class to instantiate (defaults to YOLO)

        Returns:
            Cached or newly created model instance
        """
        cache_key = (
            f"{analysis_type}_{model_identifier.replace('/', '_').replace(':', '_')}"
        )

        if cache_key not in self._model_cache:
            model_path = self._resolve_model_path(analysis_type, model_identifier)
            self._model_cache[cache_key] = model_class(str(model_path))

        return self._model_cache[cache_key]

    def _resolve_model_path(self, analysis_type: str, model_identifier: str) -> Path:
        """Resolve model identifier to actual file path.

        Args:
            analysis_type: Type of analysis for organizing cache
            model_identifier: Either filename or HF repo format

        Returns:
            Path to model file
        """
        if ":" in model_identifier:
            # Hugging Face format: repo_id:filename
            repo_id, filename = model_identifier.split(":", 1)
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        else:
            # Local model file in organized cache
            models_dir = get_models_dir() / analysis_type
            models_dir.mkdir(parents=True, exist_ok=True)
            return models_dir / model_identifier

    def clear_model_cache(self, analysis_type: str | None = None) -> None:
        """Clear model cache for specific analysis type or all models.

        Args:
            analysis_type: Specific analysis type to clear, or None for all
        """
        if analysis_type is None:
            self._model_cache.clear()
        else:
            # Clear only models for the specified analysis type
            keys_to_remove = [
                key
                for key in self._model_cache.keys()
                if key.startswith(f"{analysis_type}_")
            ]
            for key in keys_to_remove:
                del self._model_cache[key]

    def get_cache_info(self) -> dict[str, int]:
        """Get information about cached models.

        Returns:
            Dictionary with analysis types and their cached model counts
        """
        cache_info: dict[str, int] = {}
        for key in self._model_cache.keys():
            analysis_type = key.split("_", 1)[0]
            cache_info[analysis_type] = cache_info.get(analysis_type, 0) + 1
        return cache_info

    def reset_device_detection(self) -> None:
        """Reset device detection cache for testing purposes."""
        self._device = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Get singleton instance of ModelManager."""
        return cls()
