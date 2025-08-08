"""Console utilities for CuLoRA."""

from typing import Any

from rich.console import Console
from rich.table import Table


class ConsoleUtils:
    """Singleton utility class for consistent console output formatting."""

    _instance: "ConsoleUtils | None" = None
    _console: Console

    def __new__(cls) -> "ConsoleUtils":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._console = Console()
        return cls._instance

    def info(self, message: str) -> None:
        """Print an informational message in blue."""
        self._console.print(f"[blue]{message}[/blue]")

    def success(self, message: str) -> None:
        """Print a success message in green."""
        self._console.print(f"[green]{message}[/green]")

    def error(self, message: str) -> None:
        """Print an error message in red with 'Error:' prefix."""
        self._console.print(f"[red]Error:[/red] {message}")

    def warning(self, message: str) -> None:
        """Print a warning message in yellow."""
        self._console.print(f"[yellow]{message}[/yellow]")

    def progress(self, message: str) -> None:
        """Print a progress/status message in dim style."""
        self._console.print(f"[dim]{message}[/dim]")

    def header(self, message: str) -> None:
        """Print a bold header message."""
        self._console.print(f"[bold]{message}[/bold]")

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Direct access to console.print() for advanced formatting."""
        self._console.print(*args, **kwargs)

    def table(self, table: Table) -> None:
        """Print a Rich table."""
        self._console.print(table)

    def summary(self, message: str) -> None:
        """Print a summary message in bold style."""
        self._console.print(f"\n[bold]{message}[/bold]")

    @property
    def rich_console(self) -> Console:
        """Get the underlying Rich console for advanced usage."""
        return self._console


def get_console() -> ConsoleUtils:
    """Get the singleton console utilities instance.

    Returns:
        Singleton ConsoleUtils instance.
    """
    return ConsoleUtils()
