"""Tests for Memory model."""

import pytest

from culora.domain.models.memory import Memory


class TestMemory:
    """Test cases for Memory model."""

    def test_memory_default_initialization(self) -> None:
        """Test Memory default initialization."""
        memory = Memory()
        assert memory.total_mb is None
        assert memory.available_mb is None

    def test_memory_with_values(self) -> None:
        """Test Memory with specific values."""
        memory = Memory(total_mb=8192, available_mb=4096)
        assert memory.total_mb == 8192
        assert memory.available_mb == 4096

    def test_memory_unlimited(self) -> None:
        """Test Memory with unlimited values."""
        memory = Memory(total_mb=None, available_mb=None)
        assert memory.total_mb is None
        assert memory.available_mb is None

    def test_memory_usage_percent_with_values(self) -> None:
        """Test usage_percent calculation with values."""
        memory = Memory(total_mb=8192, available_mb=4096)
        assert memory.usage_percent == 50.0

    def test_memory_usage_percent_full_usage(self) -> None:
        """Test usage_percent when memory is full."""
        memory = Memory(total_mb=8192, available_mb=0)
        assert memory.usage_percent == 100.0

    def test_memory_usage_percent_no_usage(self) -> None:
        """Test usage_percent when no memory is used."""
        memory = Memory(total_mb=8192, available_mb=8192)
        assert memory.usage_percent == 0.0

    def test_memory_usage_percent_zero_total(self) -> None:
        """Test usage_percent when total memory is zero."""
        memory = Memory(total_mb=0, available_mb=0)
        assert memory.usage_percent == 100.0

    def test_memory_usage_percent_unlimited(self) -> None:
        """Test usage_percent for unlimited memory."""
        memory = Memory(total_mb=None, available_mb=None)
        assert memory.usage_percent is None

    def test_memory_usage_percent_partial_unknown(self) -> None:
        """Test usage_percent when some values are unknown."""
        memory1 = Memory(total_mb=8192, available_mb=None)
        memory2 = Memory(total_mb=None, available_mb=4096)

        assert memory1.usage_percent is None
        assert memory2.usage_percent is None

    def test_memory_used_mb_with_values(self) -> None:
        """Test used_mb calculation with values."""
        memory = Memory(total_mb=8192, available_mb=4096)
        assert memory.used_mb == 4096

    def test_memory_used_mb_unlimited(self) -> None:
        """Test used_mb for unlimited memory."""
        memory = Memory(total_mb=None, available_mb=None)
        assert memory.used_mb is None

    def test_memory_used_mb_partial_unknown(self) -> None:
        """Test used_mb when some values are unknown."""
        memory1 = Memory(total_mb=8192, available_mb=None)
        memory2 = Memory(total_mb=None, available_mb=4096)

        assert memory1.used_mb is None
        assert memory2.used_mb is None

    def test_memory_is_limited_true(self) -> None:
        """Test is_limited property when memory is limited."""
        memory = Memory(total_mb=8192, available_mb=4096)
        assert memory.is_limited is True

    def test_memory_is_limited_false(self) -> None:
        """Test is_limited property when memory is unlimited."""
        memory = Memory(total_mb=None, available_mb=None)
        assert memory.is_limited is False

    def test_memory_is_limited_partial(self) -> None:
        """Test is_limited property with partial values."""
        memory1 = Memory(total_mb=8192, available_mb=None)
        memory2 = Memory(total_mb=None, available_mb=4096)

        assert memory1.is_limited is True
        assert memory2.is_limited is False

    def test_memory_has_sufficient_memory_unlimited(self) -> None:
        """Test has_sufficient_memory for unlimited memory."""
        memory = Memory(total_mb=None, available_mb=None)
        assert memory.has_sufficient_memory(1000) is True
        assert memory.has_sufficient_memory(1000000) is True

    def test_memory_has_sufficient_memory_sufficient(self) -> None:
        """Test has_sufficient_memory when memory is sufficient."""
        memory = Memory(total_mb=8192, available_mb=4096)
        assert memory.has_sufficient_memory(2048) is True
        assert memory.has_sufficient_memory(4096) is True

    def test_memory_has_sufficient_memory_insufficient(self) -> None:
        """Test has_sufficient_memory when memory is insufficient."""
        memory = Memory(total_mb=8192, available_mb=2048)
        assert memory.has_sufficient_memory(4096) is False
        assert memory.has_sufficient_memory(8192) is False

    def test_memory_has_sufficient_memory_unknown_available(self) -> None:
        """Test has_sufficient_memory when available memory is unknown."""
        memory = Memory(total_mb=8192, available_mb=None)
        assert memory.has_sufficient_memory(1000) is False

    def test_memory_str_unlimited(self) -> None:
        """Test string representation for unlimited memory."""
        memory = Memory(total_mb=None, available_mb=None)
        assert str(memory) == "Unlimited"

    def test_memory_str_with_values(self) -> None:
        """Test string representation with values."""
        memory = Memory(total_mb=8192, available_mb=4096)
        expected = "4,096/8,192 MB (50.0% used)"
        assert str(memory) == expected

    def test_memory_str_unknown_available(self) -> None:
        """Test string representation with unknown available memory."""
        memory = Memory(total_mb=8192, available_mb=None)
        expected = "8,192 MB (usage unknown)"
        assert str(memory) == expected

    def test_memory_str_large_numbers(self) -> None:
        """Test string representation with large numbers."""
        memory = Memory(total_mb=16384, available_mb=12288)
        expected = "12,288/16,384 MB (25.0% used)"
        assert str(memory) == expected

    def test_memory_equality(self) -> None:
        """Test Memory equality comparison."""
        memory1 = Memory(total_mb=8192, available_mb=4096)
        memory2 = Memory(total_mb=8192, available_mb=4096)
        memory3 = Memory(total_mb=8192, available_mb=2048)

        assert memory1 == memory2
        assert memory1 != memory3

    def test_memory_repr(self) -> None:
        """Test Memory string representation."""
        memory = Memory(total_mb=8192, available_mb=4096)
        repr_str = repr(memory)
        assert "Memory" in repr_str
        assert "total_mb=8192" in repr_str
        assert "available_mb=4096" in repr_str

    @pytest.mark.parametrize(
        "total,available,expected_percent",
        [
            (1000, 500, 50.0),
            (2048, 1024, 50.0),
            (8192, 2048, 75.0),
            (4096, 4096, 0.0),
            (1000, 0, 100.0),
            (0, 0, 100.0),
        ],
    )
    def test_memory_usage_percent_parametrized(
        self, total: int, available: int, expected_percent: float
    ) -> None:
        """Parametrized test for usage percent calculations."""
        memory = Memory(total_mb=total, available_mb=available)
        assert memory.usage_percent == expected_percent

    @pytest.mark.parametrize(
        "total,available,required,expected",
        [
            (8192, 4096, 2048, True),
            (8192, 4096, 4096, True),
            (8192, 4096, 8192, False),
            (8192, 2048, 4096, False),
            (None, None, 1000000, True),  # Unlimited
        ],
    )
    def test_memory_has_sufficient_memory_parametrized(
        self, total: int | None, available: int | None, required: int, expected: bool
    ) -> None:
        """Parametrized test for has_sufficient_memory method."""
        memory = Memory(total_mb=total, available_mb=available)
        assert memory.has_sufficient_memory(required) == expected
