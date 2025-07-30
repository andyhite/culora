"""Memory management service."""

from typing import Any

from culora.domain.models.memory import Memory
from culora.utils.logging import LoggingService


class MemoryService:
    """Service for managing memory estimation and availability checks."""

    def __init__(self, logger: LoggingService) -> None:
        """Initialize memory service.

        Args:
            logger: Structured logger instance
        """
        self.logger = logger

        # Basic estimates for common models (can be expanded)
        self._model_estimates = {
            "insightface": 500,
            "clip": 800,
            "moondream": 1200,
            "mediapipe": 100,
            "brisque": 50,
        }

    def estimate_memory_usage(self, model_name: str) -> int | None:
        """Estimate memory usage for a given model.

        Args:
            model_name: Name of the model to estimate for

        Returns:
            Estimated memory usage in MB, or None if unknown
        """
        estimate = self._model_estimates.get(model_name.lower())

        if estimate:
            self.logger.debug(
                "Estimated model memory usage",
                model_name=model_name,
                estimated_mb=estimate,
            )
        else:
            self.logger.debug(
                "Unknown model for memory estimation",
                model_name=model_name,
            )

        return estimate

    def check_memory_availability(self, memory: Memory, required_mb: int) -> bool:
        """Check if memory has sufficient capacity.

        Args:
            memory: Memory object to check
            required_mb: Required memory in MB

        Returns:
            True if sufficient memory available
        """
        return memory.has_sufficient_memory(required_mb)

    def get_memory_summary(self, memory: Memory) -> dict[str, Any]:
        """Get memory summary.

        Args:
            memory: Memory object to get summary for

        Returns:
            Dictionary with memory information
        """
        return {
            "total_mb": memory.total_mb,
            "available_mb": memory.available_mb,
            "usage_percent": memory.usage_percent,
            "is_limited": memory.is_limited,
            "has_sufficient": memory.has_sufficient_memory(2048),  # Default threshold
        }

    def create_memory(
        self, total_mb: int | None = None, available_mb: int | None = None
    ) -> Memory:
        """Create a Memory object with specified parameters.

        Args:
            total_mb: Total memory in MB (None for unlimited)
            available_mb: Available memory in MB (None for unlimited)

        Returns:
            Memory object with specified parameters
        """
        self.logger.debug(
            "Creating memory object",
            total_mb=total_mb,
            available_mb=available_mb,
        )

        return Memory(total_mb=total_mb, available_mb=available_mb)


# Global memory service instance
_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        from culora.utils import get_logger

        logger = get_logger(__name__)
        _memory_service = MemoryService(logger)

    return _memory_service
