"""Tests for ModelManager service."""

from unittest.mock import MagicMock, patch

from culora.managers.model_manager import ModelManager


class TestModelManager:
    """Tests for ModelManager class."""

    def test_singleton_instance(self) -> None:
        """Test that ModelManager.get_instance() returns singleton instance."""
        manager1 = ModelManager.get_instance()
        manager2 = ModelManager.get_instance()
        assert manager1 is manager2

    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_optimal_device_cuda(self, mock_cuda: MagicMock) -> None:
        """Test device detection with CUDA available."""
        manager = ModelManager()
        manager.reset_device_detection()  # Reset for test isolation
        device = manager.detect_optimal_device()
        assert device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_detect_optimal_device_mps(
        self, mock_mps: MagicMock, mock_cuda: MagicMock
    ) -> None:
        """Test device detection with MPS available."""
        manager = ModelManager()
        manager.reset_device_detection()  # Reset for test isolation
        device = manager.detect_optimal_device()
        assert device == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_detect_optimal_device_cpu(
        self, mock_mps: MagicMock, mock_cuda: MagicMock
    ) -> None:
        """Test device detection fallback to CPU."""
        manager = ModelManager()
        manager.reset_device_detection()  # Reset for test isolation
        device = manager.detect_optimal_device()
        assert device == "cpu"

    @patch("torch.cuda.is_available", side_effect=Exception("Torch error"))
    def test_detect_optimal_device_exception(self, mock_cuda: MagicMock) -> None:
        """Test device detection with torch exception."""
        manager = ModelManager()
        manager.reset_device_detection()  # Reset for test isolation
        device = manager.detect_optimal_device()
        assert device == "cpu"

    def test_device_caching(self) -> None:
        """Test that device detection result is cached."""
        manager = ModelManager()
        manager.reset_device_detection()  # Reset for test isolation

        with patch("torch.cuda.is_available", return_value=True) as mock_cuda:
            device1 = manager.detect_optimal_device()
            device2 = manager.detect_optimal_device()

            assert device1 == device2 == "cuda"
            # Should only call torch.cuda.is_available once due to caching
            assert mock_cuda.call_count == 1

    def test_cache_operations_basic(self) -> None:
        """Test basic cache operations without external dependencies."""
        manager = ModelManager()

        # Test initial empty cache
        cache_info = manager.get_cache_info()
        assert len(cache_info) == 0

        # Manually add to cache for testing (this simulates what would happen)
        manager._model_cache = {  # pyright: ignore[reportPrivateUsage]
            "face_detection_model1": "mock_model1",
            "face_detection_model2": "mock_model2",
            "image_quality_model3": "mock_model3",
        }

        # Test cache info
        cache_info = manager.get_cache_info()
        expected = {"face": 2, "image": 1}
        assert cache_info == expected

        # Test clearing specific type
        manager.clear_model_cache("face_detection")
        cache_info_after = manager.get_cache_info()
        assert cache_info_after == {"image": 1}

        # Test clearing all
        manager.clear_model_cache()
        cache_info_final = manager.get_cache_info()
        assert len(cache_info_final) == 0
