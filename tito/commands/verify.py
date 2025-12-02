"""
TinyTorch Verify Command

Checks that the environment is set up correctly and ready to use.
On success, prompts to join the community map.

This is essentially `tito system health` + package import check + postcard.
"""

import sys
import os
import webbrowser
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich import box

from .base import BaseCommand


class VerifyCommand(BaseCommand):
    """Verify TinyTorch setup is ready, then join the community."""

    @property
    def name(self) -> str:
        return "verify"

    @property
    def description(self) -> str:
        return "Verify setup is ready, then join the community map"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--skip-registration",
            action="store_true",
            help="Skip registration prompt after verification"
        )

    def run(self, args: Namespace) -> int:
        """Run verification checks and prompt for registration."""
        
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]🔬 Verifying TinyTorch Setup[/bold cyan]",
            border_style="cyan"
        ))
        self.console.print()
        
        all_passed = True
        
        # 1. Environment checks
        all_passed &= self._check_environment()
        
        # 2. Project structure checks
        all_passed &= self._check_structure()
        
        # 3. Package import checks
        all_passed &= self._check_package()
        
        # Result
        self.console.print()
        if all_passed:
            self._show_success()
            if not args.skip_registration:
                self._prompt_registration()
            return 0
        else:
            self._show_failure()
            return 1

    def _check_environment(self) -> bool:
        """Check Python environment and dependencies."""
        self.console.print("[bold]Environment[/bold]")
        
        all_ok = True
        
        # Python
        self.console.print(f"  [green]✓[/green] Python {sys.version.split()[0]}")
        
        # Virtual environment
        venv_exists = self.venv_path.exists()
        in_venv = (
            os.environ.get('VIRTUAL_ENV') is not None or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            hasattr(sys, 'real_prefix')
        )
        
        if venv_exists and in_venv:
            self.console.print("  [green]✓[/green] Virtual environment active")
        elif venv_exists:
            self.console.print("  [yellow]![/yellow] Virtual environment exists but not active")
            self.console.print("    [dim]Run: source activate.sh[/dim]")
        else:
            self.console.print("  [yellow]![/yellow] No virtual environment")
        
        # Required dependencies
        required = [
            ('numpy', 'NumPy'),
            ('rich', 'Rich'),
            ('yaml', 'PyYAML'),
        ]
        
        for module, name in required:
            try:
                __import__(module)
                self.console.print(f"  [green]✓[/green] {name}")
            except ImportError:
                self.console.print(f"  [red]✗[/red] {name} [dim](pip install {module})[/dim]")
                all_ok = False
        
        self.console.print()
        return all_ok

    def _check_structure(self) -> bool:
        """Check project structure exists."""
        self.console.print("[bold]Project Structure[/bold]")
        
        all_ok = True
        
        paths = [
            ('tinytorch/', 'Package'),
            ('tinytorch/core/', 'Core modules'),
            ('src/', 'Source modules'),
        ]
        
        for path, desc in paths:
            if Path(path).exists():
                self.console.print(f"  [green]✓[/green] {path}")
            else:
                self.console.print(f"  [red]✗[/red] {path} [dim]({desc})[/dim]")
                all_ok = False
        
        self.console.print()
        return all_ok

    def _check_package(self) -> bool:
        """Check that tinytorch package is importable."""
        self.console.print("[bold]Package[/bold]")
        
        all_ok = True
        
        # Import tinytorch
        try:
            import tinytorch
            self.console.print("  [green]✓[/green] import tinytorch")
        except ImportError as e:
            self.console.print(f"  [red]✗[/red] import tinytorch")
            self.console.print(f"    [dim red]{e}[/dim red]")
            return False
        
        # Check core components
        try:
            from tinytorch import Tensor
            self.console.print("  [green]✓[/green] Tensor available")
        except ImportError:
            self.console.print("  [red]✗[/red] Tensor not available")
            all_ok = False
        
        try:
            from tinytorch import Linear, ReLU
            self.console.print("  [green]✓[/green] Layers available")
        except ImportError:
            self.console.print("  [red]✗[/red] Layers not available")
            all_ok = False
        
        try:
            from tinytorch import SGD
            self.console.print("  [green]✓[/green] Optimizer available")
        except ImportError:
            self.console.print("  [red]✗[/red] Optimizer not available")
            all_ok = False
        
        return all_ok

    def _show_success(self) -> None:
        """Show success message."""
        self.console.print(Panel.fit(
            "[bold green]✅ TinyTorch is ready![/bold green]\n\n"
            "Your environment is set up correctly.\n"
            "You can start working on modules.",
            border_style="green",
            box=box.ROUNDED
        ))

    def _show_failure(self) -> None:
        """Show failure message."""
        self.console.print(Panel.fit(
            "[bold red]❌ Setup incomplete[/bold red]\n\n"
            "Some checks failed. See above for details.\n\n"
            "[dim]Run 'tito setup' to fix common issues[/dim]",
            border_style="red",
            box=box.ROUNDED
        ))

    def _prompt_registration(self) -> None:
        """Prompt user to join the community."""
        from rich.prompt import Confirm
        
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]🌍 Join the TinyTorch Community[/bold cyan]\n\n"
            "Add yourself to the map at [link=https://tinytorch.ai/map]tinytorch.ai/map[/link]\n\n"
            "[dim]• See learners worldwide\n"
            "• Country & institution (optional)\n"
            "• No account required[/dim]",
            border_style="cyan"
        ))
        
        join = Confirm.ask("\n[bold]Join the community?[/bold]", default=True)
        
        if join:
            self._open_registration()
        else:
            self.console.print("[dim]No problem! Run 'tito verify' anytime to join later.[/dim]")

    def _open_registration(self) -> None:
        """Open registration page."""
        url = "https://tinytorch.ai/join"
        
        self.console.print(f"\n[cyan]Opening registration...[/cyan]")
        
        try:
            webbrowser.open(url)
            self.console.print(f"[green]✓[/green] Browser opened")
            self.console.print(f"[dim]  {url}[/dim]")
        except Exception:
            self.console.print(f"[yellow]Could not open browser.[/yellow]")
            self.console.print(f"Please visit: [cyan]{url}[/cyan]")
        
        self.console.print("\n[green]Welcome to the community! 🎉[/green]")
