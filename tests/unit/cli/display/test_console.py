"""Tests for CLI console wrapper."""

from unittest.mock import patch

import pytest

from culora.cli.display.console import CuLoRAConsole


class TestCuLoRAConsole:
    """Test CuLoRA console wrapper."""

    @pytest.fixture
    def console(self) -> CuLoRAConsole:
        """Create console instance for testing."""
        return CuLoRAConsole()

    def test_init(self, console: CuLoRAConsole) -> None:
        """Test console initialization."""
        assert console.console is not None
        assert hasattr(console.console, "print")
        assert hasattr(console.console, "options")

    def test_print(self, console: CuLoRAConsole) -> None:
        """Test basic print method."""
        with patch.object(console.console, "print") as mock_print:
            console.print("test message")
            mock_print.assert_called_once_with("test message")

    def test_success(self, console: CuLoRAConsole) -> None:
        """Test success message method."""
        with patch.object(console.console, "print") as mock_print:
            console.success("Operation completed")
            mock_print.assert_called_once_with(
                "✅ Operation completed", style="success"
            )

    def test_warning(self, console: CuLoRAConsole) -> None:
        """Test warning message method."""
        with patch.object(console.console, "print") as mock_print:
            console.warning("Warning message")
            mock_print.assert_called_once_with("⚠️  Warning message", style="warning")

    def test_error(self, console: CuLoRAConsole) -> None:
        """Test error message method."""
        with patch.object(console.console, "print") as mock_print:
            console.error("Error occurred")
            mock_print.assert_called_once_with("❌ Error occurred", style="error")

    def test_info(self, console: CuLoRAConsole) -> None:
        """Test info message method."""
        with patch.object(console.console, "print") as mock_print:
            console.info("Information")
            mock_print.assert_called_once_with("💡 Information", style="info")

    def test_header(self, console: CuLoRAConsole) -> None:
        """Test header method."""
        with patch.object(console.console, "print") as mock_print:
            console.header("Section Title")
            mock_print.assert_called_once_with("\nSection Title", style="header")

    def test_panel(self, console: CuLoRAConsole) -> None:
        """Test panel method."""
        with patch.object(console.console, "print") as mock_print:
            console.panel("Panel content", "Panel Title")

            # Verify that print was called once with a Panel object
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0]
            assert len(call_args) == 1

            # Check panel properties (Panel object)
            panel = call_args[0]
            assert hasattr(panel, "renderable")  # Panel has content
            assert hasattr(panel, "title")  # Panel has title

    def test_panel_no_title(self, console: CuLoRAConsole) -> None:
        """Test panel method without title."""
        with patch.object(console.console, "print") as mock_print:
            console.panel("Panel content")
            mock_print.assert_called_once()

    def test_key_value(self, console: CuLoRAConsole) -> None:
        """Test key-value display method."""
        with patch.object(console.console, "print") as mock_print:
            console.key_value("config_key", "config_value")

            mock_print.assert_called_once()
            call_args = mock_print.call_args[0]
            assert "config_key" in call_args[0]
            assert "config_value" in call_args[0]
            assert "[key]" in call_args[0]
            assert "[value]" in call_args[0]

    def test_key_value_with_special_characters(self, console: CuLoRAConsole) -> None:
        """Test key-value display with special characters that need escaping."""
        with patch.object(console.console, "print") as mock_print:
            console.key_value("key<>&", "value<>&")
            mock_print.assert_called_once()

            # The method should handle escaping special characters
            call_args = mock_print.call_args[0]
            assert "key" in call_args[0]
            assert "value" in call_args[0]

    def test_rule_with_title(self, console: CuLoRAConsole) -> None:
        """Test rule method with title."""
        with patch.object(console.console, "rule") as mock_rule:
            console.rule("Section Divider")
            mock_rule.assert_called_once_with("Section Divider", style="muted")

    def test_rule_without_title(self, console: CuLoRAConsole) -> None:
        """Test rule method without title."""
        with patch.object(console.console, "rule") as mock_rule:
            console.rule()
            mock_rule.assert_called_once_with(style="muted")

    def test_rule_with_none_title(self, console: CuLoRAConsole) -> None:
        """Test rule method with None title."""
        with patch.object(console.console, "rule") as mock_rule:
            console.rule(None)
            mock_rule.assert_called_once_with(style="muted")
