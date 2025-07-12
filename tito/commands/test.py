"""
Test command for TinyTorch CLI: runs individual module tests with detailed output.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class TestCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run individual module tests with detailed output"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--module", help="Module to test")
        parser.add_argument("--all", action="store_true", help="Run all module tests (redirects to 'tito modules --test')")

    def validate_args(self, args: Namespace) -> None:
        """Validate test command arguments."""
        # Allow running without arguments to show helpful error message
        pass

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if args.all:
            # Redirect to modules command for better overview
            console.print(Panel(
                "[yellow]💡 For testing all modules, use:[/yellow]\n\n"
                "[bold cyan]tito modules --test[/bold cyan]\n\n"
                "[dim]This provides a better overview of all module status and test results.[/dim]",
                title="Recommendation", border_style="yellow"
            ))
            
            # Still run the tests, but suggest the better command
            from .modules import ModulesCommand
            modules_cmd = ModulesCommand(self.config)
            modules_args = ArgumentParser()
            modules_cmd.add_arguments(modules_args)
            modules_args = modules_args.parse_args(['--test'])
            return modules_cmd.run(modules_args)
        
        elif args.module:
            # Run specific module tests with detailed output
            test_file = f"modules/{args.module}/tests/test_{args.module}.py"
            
            console.print(Panel(f"🧪 Running tests for module: [bold cyan]{args.module}[/bold cyan]", 
                              title="Single Module Test", border_style="bright_cyan"))
            
            if not Path(test_file).exists():
                console.print(Panel(f"[yellow]⏳ Test file not found: {test_file}\n"
                                  f"Module '{args.module}' may not be implemented yet.[/yellow]", 
                                  title="Test Not Found", border_style="yellow"))
                return 1
            
            console.print(f"Running: pytest {test_file} -v")
            console.print()
            
            try:
                result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                                      capture_output=True, text=True, timeout=300)
                
                # Print test output with syntax highlighting
                if result.stdout:
                    console.print("[dim]--- Test Output ---[/dim]")
                    console.print(result.stdout)
                if result.stderr:
                    console.print("[dim]--- Error Output ---[/dim]")
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
            # List available modules
            modules_dir = Path("modules")
            if modules_dir.exists():
                available_modules = []
                for module_dir in modules_dir.iterdir():
                    if module_dir.is_dir() and (module_dir / f"{module_dir.name}_dev.py").exists():
                        available_modules.append(module_dir.name)
                
                console.print(Panel(f"[red]❌ Please specify a module to test[/red]\n\n"
                                  f"Available modules: {', '.join(sorted(available_modules))}\n\n"
                                  f"[dim]Example: tito test --module tensor[/dim]\n"
                                  f"[dim]For all modules: tito modules --test[/dim]", 
                                  title="Module Required", border_style="red"))
            else:
                console.print(Panel("[red]❌ No modules directory found[/red]", 
                                  title="Error", border_style="red"))
            return 1 