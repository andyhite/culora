"""Tests for console utilities module."""

from unittest.mock import MagicMock, patch

from rich.table import Table

from culora.utils.console import ConsoleUtils, get_console


class TestConsoleUtils:
    """Tests for ConsoleUtils singleton class."""

    def test_singleton_instance(self) -> None:
        """Test that ConsoleUtils follows singleton pattern."""
        # Create two instances
        console1 = ConsoleUtils()
        console2 = ConsoleUtils()
        console3 = get_console()

        # All should be the same instance
        assert console1 is console2
        assert console2 is console3
        assert console1 is console3

    @patch("culora.utils.console.Console")
    def test_console_creation(self, mock_console_class: MagicMock) -> None:
        """Test that console is created correctly."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        # Clear any existing instance
        ConsoleUtils._instance = None  # type: ignore[reportPrivateUsage]

        console_utils = ConsoleUtils()
        mock_console_class.assert_called_once()

        # Verify the console instance is stored
        assert console_utils.rich_console is mock_console

    def test_info_message(self) -> None:
        """Test info message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.info("Test info message")
            mock_print.assert_called_once_with("[blue]Test info message[/blue]")

    def test_success_message(self) -> None:
        """Test success message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.success("Operation completed")
            mock_print.assert_called_once_with("[green]Operation completed[/green]")

    def test_error_message(self) -> None:
        """Test error message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.error("Something went wrong")
            mock_print.assert_called_once_with("[red]Error:[/red] Something went wrong")

    def test_warning_message(self) -> None:
        """Test warning message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.warning("This is a warning")
            mock_print.assert_called_once_with("[yellow]This is a warning[/yellow]")

    def test_progress_message(self) -> None:
        """Test progress message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.progress("Processing...")
            mock_print.assert_called_once_with("[dim]Processing...[/dim]")

    def test_header_message(self) -> None:
        """Test header message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.header("Important Section")
            mock_print.assert_called_once_with("[bold]Important Section[/bold]")

    def test_summary_message(self) -> None:
        """Test summary message formatting."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.summary("Final Results")
            mock_print.assert_called_once_with("\n[bold]Final Results[/bold]")

    def test_direct_print(self) -> None:
        """Test direct print access."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            console_utils.print("Direct message", style="cyan")
            mock_print.assert_called_once_with("Direct message", style="cyan")

    def test_table_print(self) -> None:
        """Test table printing."""
        console_utils = ConsoleUtils()
        with patch.object(console_utils.rich_console, "print") as mock_print:
            test_table = Table()
            console_utils.table(test_table)
            mock_print.assert_called_once_with(test_table)

    def test_get_console_function(self) -> None:
        """Test the get_console helper function."""
        console = get_console()
        assert isinstance(console, ConsoleUtils)

        # Verify it returns the same instance each time
        console2 = get_console()
        assert console is console2

    @patch("culora.utils.console.Console")
    def test_rich_console_property(self, mock_console_class: MagicMock) -> None:
        """Test the rich_console property provides access to underlying Console."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        # Clear any existing instance
        ConsoleUtils._instance = None  # type: ignore[reportPrivateUsage]

        console_utils = ConsoleUtils()
        rich_console = console_utils.rich_console
        assert rich_console is mock_console
        assert rich_console is console_utils.rich_console
