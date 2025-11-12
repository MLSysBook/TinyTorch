"""
View command for TinyTorch CLI: generates notebooks and opens Jupyter Lab.
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

class ViewCommand(BaseCommand):
    """Command to generate notebooks and open Jupyter Lab."""
    
    @property
    def name(self) -> str:
        return "view"
    
    @property
    def description(self) -> str:
        return "Generate notebooks and open Jupyter Lab"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add view command arguments."""
        parser.add_argument(
            'module', 
            nargs='?',
            help='View specific module (optional)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force rebuild even if notebook exists'
        )
    
    def validate_args(self, args: Namespace) -> None:
        """Validate view command arguments."""
        if args.module:
            module_dir = self.config.modules_dir / args.module
            if not module_dir.exists():
                raise ModuleNotFoundError(f"Module directory '{args.module}' not found")
            
            # Look for the specific dev file for this module
            # Extract module name (e.g., "tensor" from "01_tensor")
            module_name = args.module.split('_', 1)[1] if '_' in args.module else args.module
            dev_file = module_dir / f"{module_name}.py"
            
            if not dev_file.exists():
                # Fallback: look for any *.py file
                dev_files = list(module_dir.glob("*.py"))
                if not dev_files:
                    raise ModuleNotFoundError(
                        f"No dev file found in module '{args.module}'. Expected: {dev_file}"
                    )
    
    def _find_dev_files(self) -> List[Path]:
        """Find all *.py files in modules directory."""
        dev_files = []
        for module_dir in self.config.modules_dir.iterdir():
            if module_dir.is_dir():
                # Look for any *.py file in the directory
                for dev_py in module_dir.glob("*.py"):
                    dev_files.append(dev_py)
        return dev_files
    
    def _convert_file(self, dev_file: Path, force: bool = False) -> Tuple[bool, str]:
        """Convert a single Python file to notebook using Jupytext."""
        try:
            notebook_file = dev_file.with_suffix('.ipynb')
            
            # Check if notebook exists and we're not forcing
            if notebook_file.exists() and not force:
                return True, f"{dev_file.name} → {notebook_file.name} (already exists)"
            
            # Use Jupytext to convert Python file to notebook
            result = subprocess.run([
                "jupytext", "--to", "notebook", str(dev_file)
            ], capture_output=True, text=True, timeout=30, cwd=dev_file.parent)
            
            if result.returncode == 0:
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
    
    def _launch_jupyter_lab(self, target_dir: Path) -> bool:
        """Launch Jupyter Lab in the specified directory."""
        try:
            # Change to target directory and launch Jupyter Lab
            subprocess.Popen([
                "jupyter", "lab", "--no-browser"
            ], cwd=target_dir)
            return True
        except FileNotFoundError:
            self.console.print(Panel(
                "[red]❌ Jupyter Lab not found. Install with: pip install jupyterlab[/red]", 
                title="Error", 
                border_style="red"
            ))
            return False
        except Exception as e:
            self.console.print(Panel(
                f"[red]❌ Failed to launch Jupyter Lab: {e}[/red]", 
                title="Error", 
                border_style="red"
            ))
            return False
    
    def run(self, args: Namespace) -> int:
        """Execute the view command."""
        self.console.print(Panel(
            "📓 View: Generate Notebooks and Open Jupyter Lab", 
            title="Interactive Development", 
            border_style="bright_cyan"
        ))
        
        # Determine target directory for Jupyter Lab
        if args.module:
            target_dir = self.config.modules_dir / args.module
            # Find the specific dev file for this module
            module_name = args.module.split('_', 1)[1] if '_' in args.module else args.module
            dev_file = target_dir / f"{module_name}.py"
            
            if dev_file.exists():
                dev_files = [dev_file]
            else:
                # Fallback: find any dev files
                dev_files = list(target_dir.glob("*.py"))
            
            self.console.print(f"🔄 Generating notebook for module: {args.module}")
        else:
            target_dir = self.config.modules_dir
            dev_files = self._find_dev_files()
            if not dev_files:
                self.console.print(Panel(
                    "[yellow]⚠️  No *.py files found in modules/[/yellow]", 
                    title="Nothing to Convert", 
                    border_style="yellow"
                ))
                # Still launch Jupyter Lab even if no notebooks to generate
                self.console.print("🚀 Opening Jupyter Lab anyway...")
                if self._launch_jupyter_lab(target_dir):
                    self._print_launch_info(target_dir)
                return 0
            self.console.print(f"🔄 Generating notebooks for {len(dev_files)} modules...")
        
        # Generate notebooks
        success_count = 0
        error_count = 0
        
        for dev_file in dev_files:
            success, message = self._convert_file(dev_file, args.force)
            module_name = dev_file.parent.name
            
            if success:
                success_count += 1
                self.console.print(f"  ✅ {module_name}: {message}")
            else:
                error_count += 1
                self.console.print(f"  ❌ {module_name}: {message}")
        
        # Launch Jupyter Lab
        self.console.print("\n🚀 Opening Jupyter Lab...")
        if not self._launch_jupyter_lab(target_dir):
            return 1
        
        # Print summary and instructions
        self._print_summary(success_count, error_count, target_dir)
        
        return 0 if error_count == 0 else 1
    
    def _print_launch_info(self, target_dir: Path) -> None:
        """Print Jupyter Lab launch information."""
        info_text = Text()
        info_text.append("🌟 Jupyter Lab launched successfully!\n\n", style="bold green")
        info_text.append("📍 Working directory: ", style="white")
        info_text.append(f"{target_dir}\n", style="cyan")
        info_text.append("🌐 Open your browser and navigate to the URL shown in the terminal\n", style="white")
        info_text.append("📁 Your notebooks will be available in the file browser\n", style="white")
        info_text.append("🔄 Press Ctrl+C in the terminal to stop Jupyter Lab", style="white")
        
        self.console.print(Panel(
            info_text, 
            title="Jupyter Lab Ready", 
            border_style="green"
        ))
    
    def _print_summary(self, success_count: int, error_count: int, target_dir: Path) -> None:
        """Print command execution summary."""
        summary_text = Text()
        
        if success_count > 0:
            summary_text.append(f"✅ Successfully generated {success_count} notebook(s)\n", style="bold green")
        if error_count > 0:
            summary_text.append(f"❌ Failed to generate {error_count} notebook(s)\n", style="bold red")
        
        summary_text.append("\n🌟 Jupyter Lab launched successfully!\n\n", style="bold green")
        summary_text.append("📍 Working directory: ", style="white")
        summary_text.append(f"{target_dir}\n", style="cyan")
        summary_text.append("🌐 Open your browser and navigate to the URL shown above\n", style="white")
        summary_text.append("📁 Your notebooks are ready for interactive development\n", style="white")
        summary_text.append("🔄 Press Ctrl+C in the terminal to stop Jupyter Lab", style="white")
        
        border_style = "green" if error_count == 0 else "yellow"
        self.console.print(Panel(
            summary_text, 
            title="View Command Complete", 
            border_style=border_style
        ))