"""Memory domain model for CuLoRA."""

from dataclasses import dataclass


@dataclass
class Memory:
    """Represents memory information for a device.

    Attributes:
        total_mb: Total memory in MB (None for unlimited/unknown)
        available_mb: Available memory in MB (None for unlimited/unknown)
    """

    total_mb: int | None = None
    available_mb: int | None = None

    @property
    def usage_percent(self) -> float | None:
        """Calculate memory usage percentage."""
        if self.total_mb is None or self.available_mb is None:
            return None

        if self.total_mb == 0:
            return 100.0

        used = self.total_mb - self.available_mb
        return (used / self.total_mb) * 100.0

    @property
    def used_mb(self) -> int | None:
        """Calculate used memory in MB."""
        if self.total_mb is None or self.available_mb is None:
            return None

        return self.total_mb - self.available_mb

    @property
    def is_limited(self) -> bool:
        """Check if memory is limited (has known total)."""
        return self.total_mb is not None

    def has_sufficient_memory(self, required_mb: int) -> bool:
        """Check if there is sufficient memory available.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if sufficient memory is available
        """
        if not self.is_limited:
            # Unlimited memory (e.g., system RAM for CPU)
            return True

        if self.available_mb is None:
            # Unknown available memory but limited total - be conservative
            return False

        return self.available_mb >= required_mb

    def __str__(self) -> str:
        """Human-readable memory description."""
        if not self.is_limited:
            return "Unlimited"

        if self.available_mb is not None:
            result = f"{self.available_mb:,}/{self.total_mb:,} MB"
            if self.usage_percent is not None:
                result += f" ({self.usage_percent:.1f}% used)"
            return result

        return f"{self.total_mb:,} MB (usage unknown)"
