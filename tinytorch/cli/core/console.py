"""
Console management for consistent CLI output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from typing import Optional
import sys

# Global console instance
_console: Optional[Console] = None

def get_console() -> Console:
    """Get the global console instance."""
    global _console
    if _console is None:
        _console = Console(stderr=False)
    return _console

def print_banner():
    """Print the TinyTorch banner using Rich."""
    console = get_console()
    banner_text = Text("Tiny🔥Torch: Build ML Systems from Scratch", style="bold red")
    console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

def print_error(message: str, title: str = "Error"):
    """Print an error message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[red]❌ {message}[/red]", title=title, border_style="red"))

def print_success(message: str, title: str = "Success"):
    """Print a success message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[green]✅ {message}[/green]", title=title, border_style="green"))

def print_warning(message: str, title: str = "Warning"):
    """Print a warning message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[yellow]⚠️ {message}[/yellow]", title=title, border_style="yellow"))

def print_info(message: str, title: str = "Info"):
    """Print an info message with consistent formatting."""
    console = get_console()
    console.print(Panel(f"[cyan]ℹ️ {message}[/cyan]", title=title, border_style="cyan")) 