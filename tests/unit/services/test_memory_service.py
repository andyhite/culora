"""Tests for MemoryService."""

from unittest.mock import Mock

import pytest

from culora.domain.models.memory import Memory
from culora.services.memory_service import MemoryService


class TestMemoryService:
    """Test cases for MemoryService."""

    def test_memory_service_initialization(self, mock_logger: Mock) -> None:
        """Test MemoryService initialization."""
        service = MemoryService(mock_logger)

        assert service.logger == mock_logger
        assert isinstance(service._model_estimates, dict)
        assert len(service._model_estimates) > 0

    def test_estimate_memory_usage_known_model(
        self, memory_service: MemoryService
    ) -> None:
        """Test memory estimation for known models."""
        # Test various known models
        assert memory_service.estimate_memory_usage("insightface") == 500
        assert memory_service.estimate_memory_usage("clip") == 800
        assert memory_service.estimate_memory_usage("moondream") == 1200
        assert memory_service.estimate_memory_usage("mediapipe") == 100
        assert memory_service.estimate_memory_usage("brisque") == 50

    def test_estimate_memory_usage_case_insensitive(
        self, memory_service: MemoryService
    ) -> None:
        """Test that memory estimation is case insensitive."""
        assert memory_service.estimate_memory_usage("INSIGHTFACE") == 500
        assert memory_service.estimate_memory_usage("Clip") == 800
        assert memory_service.estimate_memory_usage("MoonDream") == 1200

    def test_estimate_memory_usage_unknown_model(
        self, memory_service: MemoryService
    ) -> None:
        """Test memory estimation for unknown models."""
        result = memory_service.estimate_memory_usage("unknown_model")
        assert result is None

    def test_estimate_memory_usage_logging_known(
        self, memory_service: MemoryService, mock_logger: Mock
    ) -> None:
        """Test logging for known model estimation."""
        memory_service.estimate_memory_usage("insightface")

        mock_logger.debug.assert_called_once_with(
            "Estimated model memory usage",
            model_name="insightface",
            estimated_mb=500,
        )

    def test_estimate_memory_usage_logging_unknown(
        self, memory_service: MemoryService, mock_logger: Mock
    ) -> None:
        """Test logging for unknown model estimation."""
        memory_service.estimate_memory_usage("unknown_model")

        mock_logger.debug.assert_called_once_with(
            "Unknown model for memory estimation",
            model_name="unknown_model",
        )

    def test_check_memory_availability_sufficient(
        self, memory_service: MemoryService, limited_memory: Memory
    ) -> None:
        """Test memory availability check with sufficient memory."""
        assert memory_service.check_memory_availability(limited_memory, 2048) is True
        assert memory_service.check_memory_availability(limited_memory, 4096) is True

    def test_check_memory_availability_insufficient(
        self, memory_service: MemoryService
    ) -> None:
        """Test memory availability check with insufficient memory."""
        memory = Memory(total_mb=4096, available_mb=2048)

        assert memory_service.check_memory_availability(memory, 4096) is False
        assert memory_service.check_memory_availability(memory, 8192) is False

    def test_check_memory_availability_unlimited(
        self, memory_service: MemoryService, unlimited_memory: Memory
    ) -> None:
        """Test memory availability check with unlimited memory."""
        assert memory_service.check_memory_availability(unlimited_memory, 1000) is True
        assert (
            memory_service.check_memory_availability(unlimited_memory, 1000000) is True
        )

    def test_get_memory_summary_limited_memory(
        self, memory_service: MemoryService, limited_memory: Memory
    ) -> None:
        """Test getting memory summary for limited memory."""
        summary = memory_service.get_memory_summary(limited_memory)

        assert summary["total_mb"] == 8192
        assert summary["available_mb"] == 4096
        assert summary["usage_percent"] == 50.0
        assert summary["is_limited"] is True
        assert summary["has_sufficient"] is True  # 4096 > 2048

    def test_get_memory_summary_unlimited_memory(
        self, memory_service: MemoryService, unlimited_memory: Memory
    ) -> None:
        """Test getting memory summary for unlimited memory."""
        summary = memory_service.get_memory_summary(unlimited_memory)

        assert summary["total_mb"] is None
        assert summary["available_mb"] is None
        assert summary["usage_percent"] is None
        assert summary["is_limited"] is False
        assert summary["has_sufficient"] is True  # Unlimited memory

    def test_get_memory_summary_insufficient_memory(
        self, memory_service: MemoryService
    ) -> None:
        """Test getting memory summary for insufficient memory."""
        memory = Memory(total_mb=2048, available_mb=1024)

        summary = memory_service.get_memory_summary(memory)

        assert summary["total_mb"] == 2048
        assert summary["available_mb"] == 1024
        assert summary["usage_percent"] == 50.0
        assert summary["is_limited"] is True
        assert summary["has_sufficient"] is False  # 1024 < 2048

    def test_create_memory_with_values(
        self, memory_service: MemoryService, mock_logger: Mock
    ) -> None:
        """Test creating memory with specific values."""
        memory = memory_service.create_memory(total_mb=8192, available_mb=4096)

        assert isinstance(memory, Memory)
        assert memory.total_mb == 8192
        assert memory.available_mb == 4096

        mock_logger.debug.assert_called_once_with(
            "Creating memory object",
            total_mb=8192,
            available_mb=4096,
        )

    def test_create_memory_unlimited(
        self, memory_service: MemoryService, mock_logger: Mock
    ) -> None:
        """Test creating unlimited memory."""
        memory = memory_service.create_memory()

        assert isinstance(memory, Memory)
        assert memory.total_mb is None
        assert memory.available_mb is None
        assert not memory.is_limited

        mock_logger.debug.assert_called_once_with(
            "Creating memory object",
            total_mb=None,
            available_mb=None,
        )

    def test_create_memory_partial_values(self, memory_service: MemoryService) -> None:
        """Test creating memory with partial values."""
        # Only total specified
        memory1 = memory_service.create_memory(total_mb=8192)
        assert memory1.total_mb == 8192
        assert memory1.available_mb is None

        # Only available specified
        memory2 = memory_service.create_memory(available_mb=4096)
        assert memory2.total_mb is None
        assert memory2.available_mb == 4096

    @pytest.mark.parametrize(
        "model_name,expected_mb",
        [
            ("insightface", 500),
            ("clip", 800),
            ("moondream", 1200),
            ("mediapipe", 100),
            ("brisque", 50),
        ],
    )
    def test_model_estimates_parametrized(
        self, memory_service: MemoryService, model_name: str, expected_mb: int
    ) -> None:
        """Parametrized test for model memory estimates."""
        result = memory_service.estimate_memory_usage(model_name)
        assert result == expected_mb

    @pytest.mark.parametrize(
        "total,available,required,expected",
        [
            (8192, 4096, 2048, True),  # Sufficient
            (4096, 2048, 4096, False),  # Insufficient
            (None, None, 1000000, True),  # Unlimited
            (2048, 1024, 2048, False),  # Exactly insufficient
            (4096, 2048, 2048, True),  # Exactly sufficient
        ],
    )
    def test_check_memory_availability_parametrized(
        self,
        memory_service: MemoryService,
        total: int | None,
        available: int | None,
        required: int,
        expected: bool,
    ) -> None:
        """Parametrized test for memory availability checking."""
        memory = Memory(total_mb=total, available_mb=available)
        result = memory_service.check_memory_availability(memory, required)
        assert result == expected

    def test_model_estimates_structure(self, memory_service: MemoryService) -> None:
        """Test that model estimates have expected structure."""
        estimates = memory_service._model_estimates

        # Check that all values are positive integers
        for model_name, estimate in estimates.items():
            assert isinstance(model_name, str)
            assert isinstance(estimate, int)
            assert estimate > 0

        # Check that all expected models are present
        expected_models = {"insightface", "clip", "moondream", "mediapipe", "brisque"}
        actual_models = set(estimates.keys())
        assert expected_models.issubset(actual_models)

    def test_memory_service_with_real_memory_objects(
        self, memory_service: MemoryService
    ) -> None:
        """Test MemoryService with real Memory objects."""
        # Create memory objects using the service
        unlimited_memory = memory_service.create_memory()
        limited_memory = memory_service.create_memory(total_mb=8192, available_mb=4096)

        # Test availability checks
        assert memory_service.check_memory_availability(unlimited_memory, 1000) is True
        assert memory_service.check_memory_availability(limited_memory, 2048) is True
        assert memory_service.check_memory_availability(limited_memory, 8192) is False

        # Test summaries
        unlimited_summary = memory_service.get_memory_summary(unlimited_memory)
        limited_summary = memory_service.get_memory_summary(limited_memory)

        assert unlimited_summary["is_limited"] is False
        assert limited_summary["is_limited"] is True
        assert unlimited_summary["has_sufficient"] is True
        assert limited_summary["has_sufficient"] is True  # 4096 > 2048
