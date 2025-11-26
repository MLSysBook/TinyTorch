"""
Version command for TinyTorch CLI: show version information for TinyTorch and dependencies.
"""

from argparse import ArgumentParser, Namespace
import sys
import os
from pathlib import Path
from datetime import datetime
from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand

class VersionCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "version"

    @property
    def description(self) -> str:
        return "Show version information for TinyTorch and dependencies"

    def add_arguments(self, parser: ArgumentParser) -> None:
        # No arguments needed
        pass

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print()
        console.print(Panel(
            "📦 TinyTorch Version Information",
            title="Version Info",
            border_style="bright_magenta"
        ))
        console.print()

        # Main Version Table
        version_table = Table(title="TinyTorch", show_header=True, header_style="bold cyan")
        version_table.add_column("Component", style="yellow", width=25)
        version_table.add_column("Version", style="white", width=50)

        # TinyTorch Version
        try:
            import tinytorch
            tinytorch_version = getattr(tinytorch, '__version__', '0.1.0-dev')
            tinytorch_path = Path(tinytorch.__file__).parent

            version_table.add_row("TinyTorch", f"v{tinytorch_version}")
            version_table.add_row("  └─ Installation", "Development Mode")
            version_table.add_row("  └─ Location", str(tinytorch_path))

            # Check if it's a git repo
            git_dir = Path.cwd() / ".git"
            if git_dir.exists():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    commit_hash = result.stdout.strip()
                    version_table.add_row("  └─ Git Commit", commit_hash)
                except Exception:
                    pass

        except ImportError:
            version_table.add_row("TinyTorch", "[red]Not Installed[/red]")

        # Python Version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        version_table.add_row("Python", python_version)

        console.print(version_table)
        console.print()

        # Dependencies Version Table
        deps_table = Table(title="Core Dependencies", show_header=True, header_style="bold magenta")
        deps_table.add_column("Package", style="cyan", width=20)
        deps_table.add_column("Version", style="white", width=20)
        deps_table.add_column("Status", width=30)

        dependencies = [
            ('numpy', 'NumPy'),
            ('pytest', 'pytest'),
            ('yaml', 'PyYAML'),
            ('rich', 'Rich'),
            ('jupyterlab', 'JupyterLab'),
            ('jupytext', 'Jupytext'),
            ('nbformat', 'nbformat'),
        ]

        for import_name, display_name in dependencies:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                deps_table.add_row(display_name, version, "[green]✅ Installed[/green]")
            except ImportError:
                deps_table.add_row(display_name, "—", "[red]❌ Not Installed[/red]")

        console.print(deps_table)
        console.print()

        # System Info (brief)
        system_table = Table(title="System", show_header=True, header_style="bold blue")
        system_table.add_column("Component", style="yellow", width=20)
        system_table.add_column("Value", style="white", width=50)

        import platform
        system_table.add_row("OS", f"{platform.system()} {platform.release()}")
        system_table.add_row("Architecture", platform.machine())
        system_table.add_row("Python Implementation", platform.python_implementation())

        console.print(system_table)
        console.print()

        # Helpful info panel
        console.print(Panel(
            "[dim]💡 For complete system information, run:[/dim] [cyan]tito system info[/cyan]\n"
            "[dim]💡 To check environment health, run:[/dim] [cyan]tito system health[/cyan]",
            border_style="blue"
        ))

        return 0
