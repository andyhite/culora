"""Custom assertion helpers for tests."""

from typing import Any

from culora.domain.models import CuLoRAConfig


class AssertionHelpers:
    """Common assertion helpers for tests."""

    @staticmethod
    def assert_config_equal(actual: CuLoRAConfig, expected: CuLoRAConfig) -> None:
        """Assert that two configurations are equal."""
        assert actual.device.preferred_device == expected.device.preferred_device
        assert actual.logging.log_level == expected.logging.log_level

    @staticmethod
    def assert_device_summary_valid(summary: dict[str, Any]) -> None:
        """Assert that a device summary has the expected structure."""
        assert "devices" in summary
        assert "selected_device" in summary
        assert "total_devices" in summary
        assert "available_devices" in summary

        assert isinstance(summary["devices"], list)
        assert isinstance(summary["total_devices"], int)
        assert isinstance(summary["available_devices"], int)

    @staticmethod
    def assert_memory_info_valid(memory_info: dict[str, Any]) -> None:
        """Assert that memory info has the expected structure."""
        required_keys = [
            "total_mb",
            "available_mb",
            "used_mb",
            "usage_percent",
            "is_limited",
        ]
        for key in required_keys:
            assert key in memory_info
