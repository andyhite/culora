"""Rich theme configuration for CuLoRA CLI."""

from rich.theme import Theme

# CuLoRA CLI color scheme
CULORA_THEME = Theme(
    {
        "primary": "cyan bold",
        "secondary": "blue",
        "success": "green bold",
        "warning": "yellow bold",
        "error": "red bold",
        "info": "white",
        "muted": "bright_black",
        "progress": "magenta",
        "header": "cyan bold underline",
        "key": "cyan",
        "value": "white",
        "table.header": "cyan bold",
        "table.border": "bright_black",
    }
)
