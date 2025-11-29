"""
Module Test Command for TinyTorch CLI.

Provides comprehensive module testing functionality:
- Run individual module tests
- Run all module tests in sequence
- Display detailed test results
- Track test failures and successes

This enables students to verify their implementations are correct.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..base import BaseCommand


class ModuleTestCommand(BaseCommand):
    """Command to test module implementations."""

    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run module tests to verify implementation"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add test command arguments."""
        parser.add_argument(
            "module_number",
            nargs="?",
            default=None,
            help="Module number to test (01, 02, etc.)",
        )
        parser.add_argument(
            "--all", action="store_true", help="Test all modules sequentially"
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Show detailed test output",
        )
        parser.add_argument(
            "--stop-on-fail",
            action="store_true",
            help="Stop testing if a module fails (only with --all)",
        )
        parser.add_argument(
            "--summary",
            action="store_true",
            help="Show only summary without running tests",
        )

    def get_module_mapping(self) -> Dict[str, str]:
        """Get mapping from numbers to module names."""
        return {
            "01": "01_tensor",
            "02": "02_activations",
            "03": "03_layers",
            "04": "04_losses",
            "05": "05_autograd",
            "06": "06_optimizers",
            "07": "07_training",
            "08": "08_dataloader",
            "09": "09_spatial",
            "10": "10_tokenization",
            "11": "11_embeddings",
            "12": "12_attention",
            "13": "13_transformers",
            "14": "14_profiling",
            "15": "15_quantization",
            "16": "16_compression",
            "17": "17_memoization",
            "18": "18_acceleration",
            "19": "19_benchmarking",
            "20": "20_capstone",
        }

    def normalize_module_number(self, module_input: str) -> str:
        """Normalize module input to 2-digit format."""
        if module_input.isdigit():
            return f"{int(module_input):02d}"
        return module_input

    def test_module(
        self, module_name: str, module_number: str, verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Test a single module.

        Returns:
            (success, output) tuple
        """
        console = self.console
        src_dir = self.config.project_root / "src"
        module_file = src_dir / module_name / f"{module_name}.py"

        if not module_file.exists():
            return False, f"Module file not found: {module_file}"

        console.print(f"[cyan]Testing Module {module_number}: {module_name}[/cyan]")

        try:
            # Run the module as a script (this triggers the if __name__ == "__main__" block)
            result = subprocess.run(
                [sys.executable, str(module_file)],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                timeout=300,  # 5 minute timeout per module
            )

            if verbose:
                if result.stdout:
                    console.print("[dim]" + result.stdout + "[/dim]")
                if result.stderr:
                    console.print("[yellow]" + result.stderr + "[/yellow]")

            if result.returncode == 0:
                console.print(f"[green]✓ Module {module_number} tests PASSED[/green]")
                return True, result.stdout
            else:
                console.print(f"[red]✗ Module {module_number} tests FAILED (exit code: {result.returncode})[/red]")
                if not verbose and result.stderr:
                    console.print(f"[red]{result.stderr}[/red]")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Test timeout (>5 minutes)"
            console.print(f"[red]✗ Module {module_number} TIMEOUT[/red]")
            return False, error_msg
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            console.print(f"[red]✗ Module {module_number} ERROR: {e}[/red]")
            return False, error_msg

    def test_all_modules(
        self, verbose: bool = False, stop_on_fail: bool = False
    ) -> int:
        """Test all modules sequentially."""
        console = self.console
        module_mapping = self.get_module_mapping()

        console.print()
        console.print(
            Panel(
                f"[bold cyan]Running All Module Tests[/bold cyan]\n\n"
                f"[bold]Testing {len(module_mapping)} modules sequentially[/bold]\n"
                f"  • Verbose: {'Yes' if verbose else 'No'}\n"
                f"  • Stop on failure: {'Yes' if stop_on_fail else 'No'}\n\n"
                f"[dim]This will take several minutes...[/dim]",
                title="🧪 Test All Modules",
                border_style="cyan",
            )
        )
        console.print()

        passed = []
        failed = []
        errors = {}

        for module_num, module_name in sorted(module_mapping.items()):
            success, output = self.test_module(module_name, module_num, verbose)

            if success:
                passed.append((module_num, module_name))
            else:
                failed.append((module_num, module_name))
                errors[module_num] = output

                if stop_on_fail:
                    console.print()
                    console.print(
                        Panel(
                            f"[red]Testing stopped due to failure in Module {module_num}[/red]\n\n"
                            f"[dim]Use --verbose to see full error details[/dim]",
                            title="Stopped on Failure",
                            border_style="red",
                        )
                    )
                    break

            console.print()

        # Display summary
        console.print()
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print("[bold cyan]Test Summary[/bold cyan]")
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print()

        # Create results table
        table = Table(title="Module Test Results", show_header=True)
        table.add_column("Module", style="cyan")
        table.add_column("Name", style="dim")
        table.add_column("Status", justify="center")

        for module_num, module_name in sorted(module_mapping.items()):
            if (module_num, module_name) in passed:
                status = "[green]✓ PASS[/green]"
            elif (module_num, module_name) in failed:
                status = "[red]✗ FAIL[/red]"
            else:
                status = "[dim]⏭ SKIPPED[/dim]"

            table.add_row(f"Module {module_num}", module_name, status)

        console.print(table)
        console.print()

        # Summary stats
        total = len(module_mapping)
        pass_count = len(passed)
        fail_count = len(failed)
        skip_count = total - pass_count - fail_count

        if fail_count == 0:
            console.print(
                Panel(
                    f"[bold green]✅ ALL TESTS PASSED![/bold green]\n\n"
                    f"[green]Passed: {pass_count}/{total} modules[/green]\n\n"
                    f"[bold]All TinyTorch modules are working correctly![/bold]",
                    title="🎉 Success",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    f"[bold red]❌ SOME TESTS FAILED[/bold red]\n\n"
                    f"[green]Passed: {pass_count} modules[/green]\n"
                    f"[red]Failed: {fail_count} modules[/red]\n"
                    + (f"[dim]Skipped: {skip_count} modules[/dim]\n" if skip_count > 0 else "")
                    + f"\n[bold]Failed modules:[/bold]\n"
                    + "\n".join([f"  • Module {num}: {name}" for num, name in failed]),
                    title="⚠️  Test Failures",
                    border_style="red",
                )
            )

            # Show error details for failed modules
            if errors and not verbose:
                console.print()
                console.print("[yellow]Failure details (run with --verbose for full output):[/yellow]")
                console.print()
                for module_num in sorted(errors.keys()):
                    console.print(f"[red]Module {module_num}:[/red]")
                    console.print(f"[dim]{errors[module_num][:500]}...[/dim]")
                    console.print()

            return 1

    def run(self, args: Namespace) -> int:
        """Execute the test command."""
        console = self.console

        # Handle --all (test all modules)
        if getattr(args, "all", False):
            return self.test_all_modules(
                verbose=args.verbose, stop_on_fail=args.stop_on_fail
            )

        # Require module number for single module test
        if not args.module_number:
            console.print(
                Panel(
                    "[red]Error: Module number required[/red]\n\n"
                    "[dim]Examples:[/dim]\n"
                    "[dim]  tito module test 01        # Test module 01[/dim]\n"
                    "[dim]  tito module test 01 -v     # Test with verbose output[/dim]\n"
                    "[dim]  tito module test --all     # Test all modules[/dim]",
                    title="Module Number Required",
                    border_style="red",
                )
            )
            return 1

        # Normalize and validate module number
        module_mapping = self.get_module_mapping()
        normalized = self.normalize_module_number(args.module_number)

        if normalized not in module_mapping:
            console.print(f"[red]Invalid module number: {args.module_number}[/red]")
            console.print("Available modules: 01-20")
            return 1

        module_name = module_mapping[normalized]

        # Test single module
        console.print()
        success, output = self.test_module(module_name, normalized, args.verbose)
        console.print()

        if success:
            console.print(
                Panel(
                    f"[bold green]✅ Module {normalized} tests passed![/bold green]\n\n"
                    f"[green]All tests completed successfully[/green]",
                    title=f"✓ {module_name}",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    f"[bold red]❌ Module {normalized} tests failed[/bold red]\n\n"
                    f"[dim]Use -v flag for detailed output[/dim]",
                    title=f"✗ {module_name}",
                    border_style="red",
                )
            )
            return 1
