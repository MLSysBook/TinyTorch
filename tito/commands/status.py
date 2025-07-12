"""
Status command for TinyTorch CLI: checks status of all modules in modules/ directory.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import BaseCommand

class StatusCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "status"

    @property
    def description(self) -> str:
        return "Check status of all modules"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--details", action="store_true", help="Show detailed file structure")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Scan modules directory
        modules_dir = Path("modules")
        if not modules_dir.exists():
            console.print(Panel("[red]❌ modules/ directory not found[/red]", 
                              title="Error", border_style="red"))
            return 1
        
        # Find all module directories (exclude special directories)
        exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
        module_dirs = [d for d in modules_dir.iterdir() 
                      if d.is_dir() and d.name not in exclude_dirs]
        
        if not module_dirs:
            console.print(Panel("[yellow]⚠️  No modules found in modules/ directory[/yellow]", 
                              title="Warning", border_style="yellow"))
            return 0
        
        console.print(Panel(f"📋 Found {len(module_dirs)} modules in modules/ directory", 
                          title="Module Status Check", border_style="bright_cyan"))
        
        # Create status table
        status_table = Table(title="Module Status Overview", show_header=True, header_style="bold blue")
        status_table.add_column("Module", style="bold cyan", width=15)
        status_table.add_column("Dev File", width=12, justify="center")
        status_table.add_column("Tests", width=12, justify="center")
        status_table.add_column("README", width=12, justify="center")
        
        # Check each module
        modules_status = []
        for module_dir in sorted(module_dirs):
            module_name = module_dir.name
            status = self._check_module_status(module_dir)
            modules_status.append((module_name, status))
            
            # Add to table
            row = [
                module_name,
                "✅" if status['dev_file'] else "❌",
                "✅" if status['tests'] else "❌", 
                "✅" if status['readme'] else "❌"
            ]
            
            status_table.add_row(*row)
        
        console.print(status_table)
        
        # Summary
        total_modules = len(modules_status)
        complete_modules = sum(1 for _, status in modules_status 
                             if status['dev_file'] and status['tests'] and status['readme'])
        
        console.print(f"\n📊 Summary: {complete_modules}/{total_modules} modules complete")
        console.print(f"💡 To run tests: [bold cyan]tito test --all[/bold cyan]")
        
        # Detailed view
        if args.details:
            console.print("\n" + "="*60)
            console.print("📁 Detailed Module Structure")
            console.print("="*60)
            
            for module_name, status in modules_status:
                self._print_module_details(module_name, status)
        
        return 0
    
    def _check_module_status(self, module_dir: Path) -> dict:
        """Check the status of a single module."""
        module_name = module_dir.name
        
        # Check for required files
        dev_file = module_dir / f"{module_name}_dev.py"
        tests_dir = module_dir / "tests"
        test_file = tests_dir / f"test_{module_name}.py"
        readme_file = module_dir / "README.md"
        
        status = {
            'dev_file': dev_file.exists(),
            'tests': test_file.exists(),
            'readme': readme_file.exists(),
        }
        
        return status
    
    def _print_module_details(self, module_name: str, status: dict) -> None:
        """Print detailed information about a module."""
        console = self.console
        
        # Module header
        console.print(f"\n📦 {module_name.upper()}", style="bold cyan")
        console.print("-" * 40)
        
        # File structure
        files_table = Table(show_header=False, box=None, padding=(0, 2))
        files_table.add_column("File", style="dim")
        files_table.add_column("Status")
        
        files_table.add_row(f"{module_name}_dev.py", "✅ Found" if status['dev_file'] else "❌ Missing")
        files_table.add_row("tests/test_*.py", "✅ Found" if status['tests'] else "❌ Missing")
        files_table.add_row("README.md", "✅ Found" if status['readme'] else "❌ Missing")
        
        console.print(files_table) 