"""Unit tests for ModelManager singleton."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from culora.managers.model_manager import ModelManager


class TestModelManager:
    """Tests for ModelManager singleton functionality."""

    def setup_method(self) -> None:
        """Reset singleton state before each test."""
        ModelManager._instance = None

    def test_singleton_behavior(self) -> None:
        """Test that ModelManager follows singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2
        assert ModelManager.get_instance() is manager1

    def test_initialization_once(self) -> None:
        """Test that initialization only happens once for singleton."""
        manager = ModelManager()
        initial_cache = manager._model_cache

        # Create another instance
        manager2 = ModelManager()

        # Should have same cache reference (same instance)
        assert manager2._model_cache is initial_cache

    @patch("torch.cuda.is_available")
    def test_detect_optimal_device_cuda(self, mock_cuda_available: MagicMock) -> None:
        """Test device detection prioritizes CUDA when available."""
        mock_cuda_available.return_value = True

        manager = ModelManager()
        device = manager.detect_optimal_device()

        assert device == "cuda"
        assert manager._device == "cuda"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_detect_optimal_device_mps(
        self, mock_cuda_available: MagicMock, mock_mps_available: MagicMock
    ) -> None:
        """Test device detection falls back to MPS when CUDA unavailable."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True

        with patch("torch.backends", create=True) as mock_backends:
            mock_backends.mps = Mock()
            mock_backends.mps.is_available = mock_mps_available

            manager = ModelManager()
            device = manager.detect_optimal_device()

            assert device == "mps"
            assert manager._device == "mps"

    @patch("torch.cuda.is_available")
    def test_detect_optimal_device_cpu_fallback(
        self, mock_cuda_available: MagicMock
    ) -> None:
        """Test device detection falls back to CPU when GPU unavailable."""
        mock_cuda_available.return_value = False

        # Mock MPS as not having the attribute (older PyTorch)
        with patch("torch.backends", spec=[]):
            manager = ModelManager()
            device = manager.detect_optimal_device()

            assert device == "cpu"
            assert manager._device == "cpu"

    @patch("torch.cuda.is_available")
    def test_detect_optimal_device_exception_fallback(
        self, mock_cuda_available: MagicMock
    ) -> None:
        """Test device detection falls back to CPU on exceptions."""
        mock_cuda_available.side_effect = RuntimeError("CUDA initialization error")

        manager = ModelManager()
        device = manager.detect_optimal_device()

        assert device == "cpu"
        assert manager._device == "cpu"

    def test_detect_optimal_device_caching(self) -> None:
        """Test that device detection is cached."""
        with patch("torch.cuda.is_available", return_value=True) as mock_cuda:
            manager = ModelManager()

            # First call
            device1 = manager.detect_optimal_device()
            # Second call
            device2 = manager.detect_optimal_device()

            assert device1 == device2 == "cuda"
            # Should only check once due to caching
            mock_cuda.assert_called_once()

    def test_get_cached_model_local_file(self) -> None:
        """Test getting cached model with local file identifier."""
        mock_model = MagicMock()
        mock_yolo_class = MagicMock(return_value=mock_model)

        with patch.object(
            ModelManager, "_resolve_model_path", return_value=Path("/models/test.pt")
        ) as mock_resolve:
            manager = ModelManager()

            result = manager.get_cached_model(
                "face_detection", "model.pt", mock_yolo_class
            )

            mock_resolve.assert_called_once_with("face_detection", "model.pt")
            mock_yolo_class.assert_called_once_with("/models/test.pt")
            assert result is mock_model

    def test_get_cached_model_caching_behavior(self) -> None:
        """Test that models are cached and reused."""
        mock_model = MagicMock()
        mock_yolo_class = MagicMock(return_value=mock_model)

        with patch.object(
            ModelManager, "_resolve_model_path", return_value=Path("/models/test.pt")
        ):
            manager = ModelManager()

            # First call
            result1 = manager.get_cached_model(
                "face_detection", "model.pt", mock_yolo_class
            )
            # Second call with same parameters
            result2 = manager.get_cached_model(
                "face_detection", "model.pt", mock_yolo_class
            )

            # Should return same cached instance
            assert result1 is result2
            # YOLO class should only be called once
            mock_yolo_class.assert_called_once()

    def test_get_cached_model_different_models(self) -> None:
        """Test caching behavior with different model identifiers."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_yolo_class = MagicMock(side_effect=[mock_model1, mock_model2])

        with patch.object(ModelManager, "_resolve_model_path") as mock_resolve:
            mock_resolve.side_effect = [
                Path("/models/model1.pt"),
                Path("/models/model2.pt"),
            ]

            manager = ModelManager()

            result1 = manager.get_cached_model(
                "face_detection", "model1.pt", mock_yolo_class
            )
            result2 = manager.get_cached_model(
                "face_detection", "model2.pt", mock_yolo_class
            )

            assert result1 is mock_model1
            assert result2 is mock_model2
            assert result1 is not result2

    @patch("culora.managers.model_manager.YOLO")
    def test_get_cached_model_custom_class(self, mock_yolo_class: MagicMock) -> None:
        """Test using custom model class."""
        mock_custom_class = MagicMock()
        mock_model = MagicMock()
        mock_custom_class.return_value = mock_model

        with patch.object(
            ModelManager, "_resolve_model_path", return_value=Path("/models/test.pt")
        ):
            manager = ModelManager()

            result = manager.get_cached_model("quality", "model.pt", mock_custom_class)

            mock_custom_class.assert_called_once_with("/models/test.pt")
            assert result is mock_model
            # Default YOLO class should not be called
            mock_yolo_class.assert_not_called()

    @patch("culora.managers.model_manager.hf_hub_download")
    def test_resolve_model_path_huggingface(self, mock_hf_download: MagicMock) -> None:
        """Test resolving Hugging Face model identifier."""
        mock_hf_download.return_value = "/cache/models/model.pt"

        manager = ModelManager()
        result = manager._resolve_model_path("face_detection", "repo/model:weights.pt")

        mock_hf_download.assert_called_once_with(
            repo_id="repo/model", filename="weights.pt"
        )
        assert result == Path("/cache/models/model.pt")

    @patch("culora.managers.model_manager.get_models_dir")
    def test_resolve_model_path_local_file(
        self, mock_get_models_dir: MagicMock
    ) -> None:
        """Test resolving local model file path."""
        mock_models_dir = MagicMock()
        mock_analysis_dir = MagicMock()
        mock_models_dir.__truediv__.return_value = mock_analysis_dir
        mock_get_models_dir.return_value = mock_models_dir

        manager = ModelManager()
        result = manager._resolve_model_path("face_detection", "model.pt")

        mock_models_dir.__truediv__.assert_called_once_with("face_detection")
        mock_analysis_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_analysis_dir.__truediv__.assert_called_once_with("model.pt")
        assert result is mock_analysis_dir.__truediv__.return_value

    def test_clear_model_cache_all(self) -> None:
        """Test clearing all models from cache."""
        manager = ModelManager()

        # Populate cache
        manager._model_cache = {
            "face_detection_model1": MagicMock(),
            "quality_model2": MagicMock(),
            "detection_model3": MagicMock(),
        }

        manager.clear_model_cache()

        assert manager._model_cache == {}

    def test_clear_model_cache_specific_type(self) -> None:
        """Test clearing models for specific analysis type."""
        manager = ModelManager()

        # Populate cache with different types
        face_model1 = MagicMock()
        face_model2 = MagicMock()
        quality_model = MagicMock()

        manager._model_cache = {
            "face_detection_model1": face_model1,
            "face_detection_model2": face_model2,
            "quality_model": quality_model,
        }

        manager.clear_model_cache("face_detection")

        # Only face_detection models should be removed
        assert manager._model_cache == {"quality_model": quality_model}

    def test_clear_model_cache_nonexistent_type(self) -> None:
        """Test clearing cache for nonexistent analysis type."""
        manager = ModelManager()

        original_cache = {
            "face_detection_model": MagicMock(),
            "quality_model": MagicMock(),
        }
        manager._model_cache = original_cache.copy()

        manager.clear_model_cache("nonexistent_type")

        # Cache should remain unchanged
        assert manager._model_cache == original_cache

    def test_get_cache_info_empty(self) -> None:
        """Test getting cache info when cache is empty."""
        manager = ModelManager()

        cache_info = manager.get_cache_info()

        assert cache_info == {}

    def test_get_cache_info_with_models(self) -> None:
        """Test getting cache info with multiple models."""
        manager = ModelManager()

        manager._model_cache = {
            "face_detection_model1": MagicMock(),
            "face_detection_model2": MagicMock(),
            "quality_model1": MagicMock(),
            "detection_model1": MagicMock(),
        }

        cache_info = manager.get_cache_info()

        expected = {
            "face": 2,
            "quality": 1,
            "detection": 1,
        }
        assert cache_info == expected

    def test_get_cache_info_single_model_per_type(self) -> None:
        """Test cache info with single model per analysis type."""
        manager = ModelManager()

        manager._model_cache = {
            "face_model": MagicMock(),
            "quality_model": MagicMock(),
        }

        cache_info = manager.get_cache_info()

        expected = {
            "face": 1,
            "quality": 1,
        }
        assert cache_info == expected

    def test_reset_device_detection(self) -> None:
        """Test resetting device detection cache."""
        manager = ModelManager()

        # Set initial device
        manager._device = "cuda"

        manager.reset_device_detection()

        assert manager._device is None

    def test_reset_device_detection_forces_redetection(self) -> None:
        """Test that resetting device forces re-detection on next call."""
        with patch("torch.cuda.is_available", side_effect=[True, False]) as mock_cuda:
            manager = ModelManager()

            # First detection
            device1 = manager.detect_optimal_device()
            assert device1 == "cuda"

            # Reset and detect again
            manager.reset_device_detection()

            with patch("torch.backends", spec=[]):  # Mock no MPS support
                device2 = manager.detect_optimal_device()
                assert device2 == "cpu"

            # Should have been called twice
            assert mock_cuda.call_count == 2

    def test_model_cache_key_generation(self) -> None:
        """Test that cache keys are generated consistently."""
        manager = ModelManager()
        mock_model_class = MagicMock(return_value=MagicMock())

        with patch.object(
            manager, "_resolve_model_path", return_value=Path("/test.pt")
        ):

            # Test various model identifiers
            manager.get_cached_model("face_detection", "model.pt", mock_model_class)
            manager.get_cached_model(
                "face_detection", "repo/name:model.pt", mock_model_class
            )
            manager.get_cached_model("quality", "model-v2.pt", mock_model_class)

            expected_keys = {
                "face_detection_model.pt",
                "face_detection_repo_name_model.pt",
                "quality_model-v2.pt",
            }

            assert set(manager._model_cache.keys()) == expected_keys

    def test_get_cached_model_default_yolo_class(self) -> None:
        """Test that YOLO is used as default model class when no custom class provided."""
        # This test verifies that the default parameter works, but we can't test
        # actual YOLO instantiation without mocking it completely
        manager = ModelManager()

        # Test that the method signature includes YOLO as default
        from inspect import signature

        from ultralytics import YOLO

        sig = signature(manager.get_cached_model)
        assert sig.parameters["model_class"].default is YOLO
