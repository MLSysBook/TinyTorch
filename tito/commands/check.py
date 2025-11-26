"""
Check command for TinyTorch CLI: comprehensive environment validation.

Runs 60+ automated tests to validate the entire TinyTorch environment.
Perfect for students to share with TAs when something isn't working.
"""

import sys
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand

class CheckCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "check"

    @property
    def description(self) -> str:
        return "Run comprehensive environment validation (60+ tests)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='Show detailed test output'
        )

    def run(self, args: Namespace) -> int:
        """Run comprehensive validation tests with rich output."""
        console = self.console

        console.print()
        console.print(Panel(
            "🧪 Running Comprehensive Environment Validation\n\n"
            "This will test 60+ aspects of your TinyTorch environment.\n"
            "Perfect for sharing with TAs if something isn't working!",
            title="TinyTorch Environment Check",
            border_style="bright_cyan"
        ))
        console.print()

        # Check if tests directory exists
        tests_dir = Path("tests/environment")
        if not tests_dir.exists():
            console.print(Panel(
                "[red]❌ Validation tests not found![/red]\n\n"
                f"Expected location: {tests_dir.absolute()}\n\n"
                "Please ensure you're running this from the TinyTorch root directory.",
                title="Error",
                border_style="red"
            ))
            return 1

        # Run the validation tests with pytest
        test_files = [
            "tests/environment/test_setup_validation.py",
            "tests/environment/test_all_requirements.py"
        ]

        console.print("[bold cyan]Running validation tests...[/bold cyan]")
        console.print()

        # Build pytest command
        pytest_args = [
            sys.executable, "-m", "pytest"
        ] + test_files + [
            "-v" if args.verbose else "-q",
            "--tb=short",
            "--color=yes",
            "-p", "no:warnings"  # Suppress warnings for cleaner output
        ]

        # Run pytest and capture output
        result = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True
        )

        # Parse test results from output
        output_lines = result.stdout.split('\n')

        # Count results
        passed = failed = skipped = 0

        for line in output_lines:
            if 'passed' in line.lower():
                # Extract numbers from pytest summary
                import re
                match = re.search(r'(\d+) passed', line)
                if match:
                    passed = int(match.group(1))
                match = re.search(r'(\d+) failed', line)
                if match:
                    failed = int(match.group(1))
                match = re.search(r'(\d+) skipped', line)
                if match:
                    skipped = int(match.group(1))

        # Display results with rich formatting
        console.print()

        # Summary table
        results_table = Table(title="Test Results Summary", show_header=True, header_style="bold magenta")
        results_table.add_column("Category", style="cyan", width=30)
        results_table.add_column("Count", justify="right", width=10)
        results_table.add_column("Status", width=20)

        if passed > 0:
            results_table.add_row("Tests Passed", str(passed), "[green]✅ OK[/green]")
        if failed > 0:
            results_table.add_row("Tests Failed", str(failed), "[red]❌ Issues Found[/red]")
        if skipped > 0:
            results_table.add_row("Tests Skipped", str(skipped), "[yellow]⏭️  Optional[/yellow]")

        console.print(results_table)
        console.print()

        # Overall health status
        if failed == 0:
            status_panel = Panel(
                "[bold green]✅ Environment is HEALTHY![/bold green]\n\n"
                f"All {passed} required checks passed.\n"
                f"{skipped} optional checks skipped.\n\n"
                "Your TinyTorch environment is ready to use! 🎉\n\n"
                "[dim]Next: [/dim][cyan]tito module 01[/cyan]",
                title="Environment Status",
                border_style="green"
            )
        else:
            status_panel = Panel(
                f"[bold red]❌ Found {failed} issue(s)[/bold red]\n\n"
                f"{passed} checks passed, but some components need attention.\n\n"
                "[bold]What to share with your TA:[/bold]\n"
                "1. Copy the output above\n"
                "2. Include the error messages below\n"
                "3. Mention what you were trying to do\n\n"
                "[dim]Or try:[/dim] [cyan]tito setup[/cyan] [dim]to reinstall[/dim]",
                title="Environment Status",
                border_style="red"
            )

        console.print(status_panel)

        # Show detailed output if verbose or if there are failures
        if args.verbose or failed > 0:
            console.print()
            console.print(Panel("📋 Detailed Test Output", border_style="blue"))
            console.print()
            console.print(result.stdout)

            if result.stderr:
                console.print()
                console.print(Panel("⚠️  Error Messages", border_style="yellow"))
                console.print()
                console.print(result.stderr)

        # Add helpful hints for common failures
        if failed > 0:
            console.print()
            console.print(Panel(
                "[bold]Common Solutions:[/bold]\n\n"
                "• Missing packages: [cyan]pip install -r requirements.txt[/cyan]\n"
                "• Jupyter issues: [cyan]pip install --upgrade jupyterlab[/cyan]\n"
                "• Import errors: [cyan]pip install -e .[/cyan] [dim](reinstall TinyTorch)[/dim]\n"
                "• Still stuck: Run [cyan]tito system check --verbose[/cyan]\n\n"
                "[dim]Then share the full output with your TA[/dim]",
                title="💡 Quick Fixes",
                border_style="yellow"
            ))

        console.print()

        # Return appropriate exit code
        return 0 if failed == 0 else 1
