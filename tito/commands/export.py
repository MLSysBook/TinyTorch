"""
Sync command for TinyTorch CLI: exports notebook code to Python package using nbdev.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class ExportCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "export"

    @property
    def description(self) -> str:
        return "Export notebook code to Python package"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--module", help="Export specific module (e.g., setup, tensor)")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Determine what to export
        if hasattr(args, 'module') and args.module:
            module_path = f"modules/{args.module}"
            if not Path(module_path).exists():
                console.print(Panel(f"[red]❌ Module '{args.module}' not found at {module_path}[/red]", 
                                  title="Module Not Found", border_style="red"))
                return 1
            
            console.print(Panel(f"🔄 Exporting Module: {args.module}", 
                               title="nbdev Export", border_style="bright_cyan"))
            console.print(f"🔄 Exporting {args.module} notebook to tinytorch package...")
            
            # Use nbdev_export with --path for specific module
            cmd = ["nbdev_export", "--path", module_path]
        else:
            console.print(Panel("🔄 Exporting All Notebooks to Package", 
                               title="nbdev Export", border_style="bright_cyan"))
            console.print("🔄 Exporting all notebook code to tinytorch package...")
            
            # Use nbdev_export for all modules  
            cmd = ["nbdev_export"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                console.print(Panel("[green]✅ Successfully exported notebook code to tinytorch package![/green]", 
                                  title="Export Success", border_style="green"))
                
                # Show what was exported
                exports_text = Text()
                exports_text.append("📦 Exported modules:\n", style="bold cyan")
                
                # Check for exported files
                tinytorch_path = Path("tinytorch")
                if tinytorch_path.exists():
                    for py_file in tinytorch_path.rglob("*.py"):
                        if py_file.name != "__init__.py" and py_file.stat().st_size > 100:  # Non-empty files
                            rel_path = py_file.relative_to(tinytorch_path)
                            exports_text.append(f"  ✅ tinytorch/{rel_path}\n", style="green")
                
                exports_text.append("\n💡 Next steps:\n", style="bold yellow")
                exports_text.append("  • Run: tito test --module setup\n", style="white")
                exports_text.append("  • Or: tito test --all\n", style="white")
                
                console.print(Panel(exports_text, title="Export Summary", border_style="bright_green"))
                
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                console.print(Panel(f"[red]❌ Export failed:\n{error_msg}[/red]", 
                                  title="Export Error", border_style="red"))
                
                # Helpful error guidance
                help_text = Text()
                help_text.append("💡 Common issues:\n", style="bold yellow")
                help_text.append("  • Missing #| default_exp directive in notebook\n", style="white")
                help_text.append("  • Syntax errors in exported code\n", style="white")
                help_text.append("  • Missing settings.ini configuration\n", style="white")
                help_text.append("\n🔧 Run 'tito doctor' for detailed diagnosis", style="cyan")
                
                console.print(Panel(help_text, title="Troubleshooting", border_style="yellow"))
                
            return result.returncode
            
        except FileNotFoundError:
            console.print(Panel("[red]❌ nbdev not found. Install with: pip install nbdev[/red]", 
                              title="Missing Dependency", border_style="red"))
            return 1 