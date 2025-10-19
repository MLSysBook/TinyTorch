"""
Notebooks command for building Jupyter notebooks from Python files using Jupytext.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand
from ..core.exceptions import ExecutionError, ModuleNotFoundError

class NotebooksCommand(BaseCommand):
    """Command to build Jupyter notebooks from Python files using Jupytext."""
    
    @property
    def name(self) -> str:
        return "notebooks"
    
    @property
    def description(self) -> str:
        return "Build notebooks from Python files"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add notebooks command arguments."""
        parser.add_argument(
            '--module', 
            help='Build notebook for specific module'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force rebuild even if notebook exists'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be built without actually building'
        )
    
    def validate_args(self, args: Namespace) -> None:
        """Validate notebooks command arguments."""
        if args.module:
            # Look in modules/source/ subdirectory
            source_dir = self.config.modules_dir / 'source'
            if not source_dir.exists():
                source_dir = self.config.modules_dir
            module_file = source_dir / args.module / f"{args.module}_dev.py"
            if not module_file.exists():
                raise ModuleNotFoundError(
                    f"Module '{args.module}' not found or no {args.module}_dev.py file"
                )
    
    def _find_dev_files(self) -> List[Path]:
        """Find all *_dev.py files in modules directory."""
        dev_files = []
        # Look in modules/source/ subdirectory
        source_dir = self.config.modules_dir / 'source'
        if not source_dir.exists():
            # Fallback to modules_dir directly
            source_dir = self.config.modules_dir
        
        for module_dir in source_dir.iterdir():
            if module_dir.is_dir():
                dev_py = module_dir / f"{module_dir.name}_dev.py"
                if dev_py.exists():
                    dev_files.append(dev_py)
        return dev_files
    
    def _convert_file(self, dev_file: Path) -> Tuple[bool, str]:
        """Convert a single Python file to notebook using Jupytext."""
        try:
            # Use Jupytext to convert Python file to notebook
            result = subprocess.run([
                "jupytext", "--to", "notebook", str(dev_file)
            ], capture_output=True, text=True, timeout=30, cwd=dev_file.parent)
            
            if result.returncode == 0:
                notebook_file = dev_file.with_suffix('.ipynb')
                return True, f"{dev_file.name} → {notebook_file.name}"
            else:
                error_msg = result.stderr.strip() if result.stderr.strip() else "Conversion failed"
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Conversion timed out"
        except FileNotFoundError:
            return False, "Jupytext not found. Install with: pip install jupytext"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def run(self, args: Namespace) -> int:
        """Execute the notebooks command."""
        self.console.print(Panel(
            "📓 Building Notebooks from Python Files (using Jupytext)", 
            title="Notebook Generation", 
            border_style="bright_cyan"
        ))
        
        # Find files to convert
        if args.module:
            # Look in modules/source/ subdirectory
            source_dir = self.config.modules_dir / 'source'
            if not source_dir.exists():
                source_dir = self.config.modules_dir
            dev_files = [source_dir / args.module / f"{args.module}_dev.py"]
            self.console.print(f"🔄 Building notebook for module: {args.module}")
        else:
            dev_files = self._find_dev_files()
            if not dev_files:
                self.console.print(Panel(
                    "[yellow]⚠️  No *_dev.py files found in modules/[/yellow]", 
                    title="Nothing to Convert", 
                    border_style="yellow"
                ))
                return 0
            self.console.print(f"🔄 Building notebooks for {len(dev_files)} modules...")
        
        # Dry run mode
        if args.dry_run:
            self.console.print("\n[cyan]Dry run mode - would convert:[/cyan]")
            for dev_file in dev_files:
                module_name = dev_file.parent.name
                self.console.print(f"  • {module_name}: {dev_file.name}")
            return 0
        
        # Convert files
        success_count = 0
        error_count = 0
        
        for dev_file in dev_files:
            success, message = self._convert_file(dev_file)
            module_name = dev_file.parent.name
            
            if success:
                success_count += 1
                self.console.print(f"  ✅ {module_name}: {message}")
            else:
                error_count += 1
                self.console.print(f"  ❌ {module_name}: {message}")
        
        # Summary
        self._print_summary(success_count, error_count)
        
        return 0 if error_count == 0 else 1
    
    def _print_summary(self, success_count: int, error_count: int) -> None:
        """Print command execution summary."""
        summary_text = Text()
        
        if success_count > 0:
            summary_text.append(f"✅ Successfully built {success_count} notebook(s)\n", style="bold green")
        if error_count > 0:
            summary_text.append(f"❌ Failed to build {error_count} notebook(s)\n", style="bold red")
        
        if success_count > 0:
            summary_text.append("\n💡 Next steps:\n", style="bold yellow")
            summary_text.append("  • Open notebooks with: jupyter lab\n", style="white")
            summary_text.append("  • Work interactively in the notebooks\n", style="white")
            summary_text.append("  • Export code with: tito package export\n", style="white")
            summary_text.append("  • Run tests with: tito module test\n", style="white")
        
        border_style = "green" if error_count == 0 else "yellow"
        self.console.print(Panel(
            summary_text, 
            title="Notebook Generation Complete", 
            border_style=border_style
        )) 