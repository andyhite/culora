"""Rich console wrapper for CuLoRA CLI."""

from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status

from .theme import CULORA_THEME


class CuLoRAConsole:
    """Themed console wrapper for CuLoRA CLI output."""

    def __init__(self) -> None:
        self.console = Console(theme=CULORA_THEME)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to console with theme support."""
        self.console.print(*args, **kwargs)

    def success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"✅ {message}", style="success")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"⚠️  {message}", style="warning")

    def error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"❌ {message}", style="error")

    def info(self, message: str) -> None:
        """Print informational message."""
        self.console.print(f"💡 {message}", style="info")

    def header(self, title: str) -> None:
        """Print section header."""
        self.console.print(f"\n{title}", style="header")

    def panel(self, content: str, title: str | None = None) -> None:
        """Print content in a bordered panel."""
        panel = Panel(content, title=title, border_style="primary")
        self.console.print(panel)

    def key_value(self, key: str, value: Any) -> None:
        """Print key-value pair with consistent formatting."""
        escaped_key = escape(str(key))
        escaped_value = escape(str(value))
        self.console.print(f"[key]{escaped_key}:[/key] [value]{escaped_value}[/value]")

    def rule(self, title: str | None = None) -> None:
        """Print a horizontal rule with optional title."""
        if title is None:
            self.console.rule(style="muted")
        else:
            self.console.rule(title, style="muted")

    def status(self, text: str) -> Status:
        """Create a status context manager."""
        return self.console.status(text)


# Global console instance
console = CuLoRAConsole()
