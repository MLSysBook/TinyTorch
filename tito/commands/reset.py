"""
Reset command for TinyTorch CLI: resets tinytorch package to clean state.
"""

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class ResetCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset tinytorch package to clean state"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        console.print(Panel("🔄 Resetting TinyTorch Package", 
                           title="Package Reset", border_style="bright_yellow"))
        
        tinytorch_path = Path("tinytorch")
        
        if not tinytorch_path.exists():
            console.print(Panel("[yellow]⚠️  TinyTorch package directory not found. Nothing to reset.[/yellow]", 
                              title="Nothing to Reset", border_style="yellow"))
            return 0
        
        # Ask for confirmation unless --force is used
        if not (hasattr(args, 'force') and args.force):
            console.print("\n[yellow]This will remove all exported Python files from the tinytorch package.[/yellow]")
            console.print("[yellow]Notebooks in modules/ will be preserved.[/yellow]\n")
            
            try:
                response = input("Are you sure you want to reset? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]", 
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("[cyan]Reset cancelled.[/cyan]", 
                                  title="Cancelled", border_style="cyan"))
                return 0
        
        reset_text = Text()
        reset_text.append("🗑️  Removing all exported files:\n", style="bold red")
        
        # Simple approach: remove all .py files except __init__.py
        files_removed = 0
        for py_file in tinytorch_path.rglob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    rel_path = py_file.relative_to(tinytorch_path)
                    reset_text.append(f"  🗑️  tinytorch/{rel_path}\n", style="red")
                    py_file.unlink()
                    files_removed += 1
                except Exception as e:
                    reset_text.append(f"  ❌ Failed to remove {py_file}: {e}\n", style="red")
        
        # Remove __pycache__ directories
        for pycache in tinytorch_path.rglob("__pycache__"):
            if pycache.is_dir():
                reset_text.append(f"  🗑️  {pycache}/\n", style="red")
                shutil.rmtree(pycache)
        
        # Remove .pytest_cache if it exists
        pytest_cache = Path(".pytest_cache")
        if pytest_cache.exists():
            reset_text.append(f"  🗑️  .pytest_cache/\n", style="red")
            shutil.rmtree(pytest_cache)
        
        if files_removed > 0:
            reset_text.append(f"\n✅ Reset complete! Removed {files_removed} generated files.\n", style="bold green")
            reset_text.append("\n💡 Next steps:\n", style="bold yellow")
            reset_text.append("  • Run: tito module export --all           - Re-export modules\n", style="white")
            reset_text.append("  • Run: tito module export --module setup  - Export specific module\n", style="white")
            reset_text.append("  • Run: tito module test --all           - Test everything\n", style="white")
            
            console.print(Panel(reset_text, title="Reset Complete", border_style="green"))
        else:
            console.print(Panel("[yellow]No generated files found to remove.[/yellow]", 
                              title="Nothing to Reset", border_style="yellow"))
        
        return 0 