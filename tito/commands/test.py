"""
Test command for TinyTorch CLI: runs module tests using pytest.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .base import BaseCommand

class TestCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run module tests"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--module", help="Module to test")
        parser.add_argument("--all", action="store_true", help="Run all module tests")

    def validate_args(self, args: Namespace) -> None:
        """Validate test command arguments."""
        if not args.all and not args.module:
            raise ValueError("Must specify either --module or --all")

    def run(self, args: Namespace) -> int:
        console = self.console
        valid_modules = ["setup", "tensor", "activations", "layers", "networks", "cnn", "data", "training", 
                         "profiling", "compression", "kernels", "benchmarking", "mlops"]
        
        if args.all:
            # Run all tests with progress bar
            failed_modules = []
            
            # Count existing test files in modules/{module}/tests/
            existing_tests = []
            for module in valid_modules:
                test_path = Path(f"modules/{module}/tests/test_{module}.py")
                if test_path.exists():
                    existing_tests.append(module)
            
            console.print(Panel(f"🧪 Running tests for {len(existing_tests)} modules", 
                              title="Test Suite", border_style="bright_cyan"))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task("Running tests...", total=len(existing_tests))
                
                for module in existing_tests:
                    progress.update(task, description=f"Testing {module}...")
                    
                    test_file = f"modules/{module}/tests/test_{module}.py"
                    try:
                        result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                                              capture_output=True, text=True, timeout=300)
                        
                        if result.returncode != 0:
                            failed_modules.append(module)
                            console.print(f"[red]❌ {module} tests failed[/red]")
                        else:
                            console.print(f"[green]✅ {module} tests passed[/green]")
                    except subprocess.TimeoutExpired:
                        failed_modules.append(module)
                        console.print(f"[red]❌ {module} tests timed out (5 minutes)[/red]")
                    except Exception as e:
                        failed_modules.append(module)
                        console.print(f"[red]❌ {module} tests failed with error: {e}[/red]")
                    
                    progress.advance(task)
            
            # Results summary
            if failed_modules:
                console.print(Panel(f"[red]❌ Failed modules: {', '.join(failed_modules)}[/red]", 
                                  title="Test Results", border_style="red"))
                return 1
            else:
                console.print(Panel("[green]✅ All tests passed![/green]", 
                                  title="Test Results", border_style="green"))
                return 0
        
        elif args.module in valid_modules:
            # Run specific module tests
            test_file = f"modules/{args.module}/tests/test_{args.module}.py"
            
            console.print(Panel(f"🧪 Running tests for module: [bold cyan]{args.module}[/bold cyan]", 
                              title="Single Module Test", border_style="bright_cyan"))
            
            if not Path(test_file).exists():
                console.print(Panel(f"[yellow]⏳ Test file not found: {test_file}\n"
                                  f"Module '{args.module}' may not be implemented yet.[/yellow]", 
                                  title="Test Not Found", border_style="yellow"))
                return 1
            
            console.print(f"Running: pytest {test_file} -v")
            
            try:
                result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                                      capture_output=True, text=True, timeout=300)
                
                # Print test output
                if result.stdout:
                    console.print(result.stdout)
                if result.stderr:
                    console.print(result.stderr)
                
                if result.returncode == 0:
                    console.print(Panel("[green]✅ All tests passed for {}![/green]".format(args.module), 
                                      title="Test Results", border_style="green"))
                else:
                    console.print(Panel("[red]❌ Some tests failed for {}[/red]".format(args.module), 
                                      title="Test Results", border_style="red"))
                
                return result.returncode
            except subprocess.TimeoutExpired:
                console.print(Panel("[red]❌ Tests timed out after 5 minutes for {}[/red]".format(args.module), 
                                  title="Test Results", border_style="red"))
                return 1
            except Exception as e:
                console.print(Panel("[red]❌ Test execution failed: {}[/red]".format(str(e)), 
                                  title="Test Results", border_style="red"))
                return 1
        
        else:
            console.print(Panel(f"[red]❌ Invalid module: {args.module}\n"
                              f"Valid modules: {', '.join(valid_modules)}[/red]", 
                              title="Invalid Module", border_style="red"))
            return 1 