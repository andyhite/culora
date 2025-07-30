"""Tests for CLI theme configuration."""

from rich.theme import Theme

from culora.cli.display.theme import CULORA_THEME


class TestCuLoRATheme:
    """Test CuLoRA CLI theme."""

    def test_theme_is_instance(self) -> None:
        """Test that theme is a Rich Theme instance."""
        assert isinstance(CULORA_THEME, Theme)

    def test_theme_has_required_styles(self) -> None:
        """Test that theme contains all required style definitions."""
        required_styles = [
            "primary",
            "secondary",
            "success",
            "warning",
            "error",
            "info",
            "muted",
            "progress",
            "header",
            "key",
            "value",
            "table.header",
            "table.border",
        ]

        theme_styles = CULORA_THEME.styles
        for style in required_styles:
            assert style in theme_styles, f"Required style '{style}' not found in theme"

    def test_theme_style_values(self) -> None:
        """Test specific theme style values."""
        styles = CULORA_THEME.styles

        # Test primary colors are cyan/blue based
        assert "cyan" in str(styles["primary"]) or "blue" in str(styles["primary"])

        # Test semantic colors
        assert "green" in str(styles["success"])
        assert "yellow" in str(styles["warning"])
        assert "red" in str(styles["error"])
        assert "magenta" in str(styles["progress"])

        # Test table styles exist
        assert styles["table.header"] is not None
        assert styles["table.border"] is not None

    def test_theme_consistency(self) -> None:
        """Test theme consistency and completeness."""
        styles = CULORA_THEME.styles

        # Ensure no None values
        for style_name, style_value in styles.items():
            assert style_value is not None, f"Style '{style_name}' has None value"

        # Test that key styles are distinct
        assert styles["key"] != styles["value"]
        assert styles["success"] != styles["error"]
        assert styles["primary"] != styles["secondary"]
