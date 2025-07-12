"""
Export command for TinyTorch CLI: exports notebook code to Python package using nbdev.
"""

import subprocess
import sys
import re
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
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("module", nargs="?", help="Export specific module (e.g., setup, tensor)")
        group.add_argument("--all", action="store_true", help="Export all modules")

    def _get_export_target(self, module_path: Path) -> str:
        """
        Read the actual export target from the dev file's #| default_exp directive.
        This is the source of truth, not the YAML file.
        """
        # Extract the short name from the full module name
        module_name = module_path.name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        
        dev_file = module_path / f"{short_name}_dev.py"
        if not dev_file.exists():
            return "unknown"
        
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for #| default_exp directive with more flexible regex
                match = re.search(r'#\|\s*default_exp\s+([^\n\r]+)', content)
                if match:
                    return match.group(1).strip()
        except Exception as e:
            # Debug: print the error for troubleshooting
            print(f"Debug: Error reading {dev_file}: {e}")
        
        return "unknown"

    def _show_export_details(self, console, module_name: str = None):
        """Show detailed export information including where each module exports to."""
        exports_text = Text()
        exports_text.append("📦 Export Details:\n", style="bold cyan")
        
        if module_name:
            # Single module export
            module_path = Path(f"assignments/source/{module_name}")
            export_target = self._get_export_target(module_path)
            if export_target != "unknown":
                target_file = export_target.replace('.', '/') + '.py'
                exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
                
                # Extract the short name for display
                short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
                exports_text.append(f"     Source: assignments/source/{module_name}/{short_name}_dev.py\n", style="dim")
                exports_text.append(f"     Target: tinytorch/{target_file}\n", style="dim")
            else:
                exports_text.append(f"  ❓ {module_name} → export target not found\n", style="yellow")
        else:
            # All modules export
            source_dir = Path("assignments/source")
            if source_dir.exists():
                exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
                for module_dir in source_dir.iterdir():
                    if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                        export_target = self._get_export_target(module_dir)
                        if export_target != "unknown":
                            target_file = export_target.replace('.', '/') + '.py'
                            exports_text.append(f"  🔄 {module_dir.name} → tinytorch/{target_file}\n", style="green")
        
        # Show what was actually created
        exports_text.append("\n📁 Generated Files:\n", style="bold cyan")
        tinytorch_path = Path("tinytorch")
        if tinytorch_path.exists():
            for py_file in tinytorch_path.rglob("*.py"):
                if py_file.name != "__init__.py" and py_file.stat().st_size > 100:  # Non-empty files
                    rel_path = py_file.relative_to(tinytorch_path)
                    exports_text.append(f"  ✅ tinytorch/{rel_path}\n", style="green")
        
        exports_text.append("\n💡 Next steps:\n", style="bold yellow")
        exports_text.append("  • Run: tito module test --all\n", style="white")
        exports_text.append("  • Or: tito module test <module_name>\n", style="white")
        
        console.print(Panel(exports_text, title="Export Summary", border_style="bright_green"))

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Determine what to export
        if hasattr(args, 'module') and args.module:
            module_path = f"assignments/source/{args.module}"
            if not Path(module_path).exists():
                console.print(Panel(f"[red]❌ Module '{args.module}' not found at {module_path}[/red]", 
                                  title="Module Not Found", border_style="red"))
                return 1
            
            console.print(Panel(f"🔄 Exporting Module: {args.module}", 
                               title="nbdev Export", border_style="bright_cyan"))
            console.print(f"🔄 Exporting {args.module} notebook to tinytorch package...")
            
            # Use nbdev_export with --path for specific module
            cmd = ["nbdev_export", "--path", module_path]
        elif hasattr(args, 'all') and args.all:
            console.print(Panel("🔄 Exporting All Notebooks to Package", 
                               title="nbdev Export", border_style="bright_cyan"))
            console.print("🔄 Exporting all notebook code to tinytorch package...")
            
            # Use nbdev_export for all modules  
            cmd = ["nbdev_export"]
        else:
            console.print(Panel("[red]❌ Must specify either a module name or --all[/red]", 
                              title="Missing Arguments", border_style="red"))
            return 1
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                console.print(Panel("[green]✅ Successfully exported notebook code to tinytorch package![/green]", 
                                  title="Export Success", border_style="green"))
                
                # Show detailed export information
                self._show_export_details(console, args.module if hasattr(args, 'module') else None)
                
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
                help_text.append("\n🔧 Run 'tito system doctor' for detailed diagnosis", style="cyan")
                
                console.print(Panel(help_text, title="Troubleshooting", border_style="yellow"))
                
            return result.returncode
            
        except FileNotFoundError:
            console.print(Panel("[red]❌ nbdev not found. Install with: pip install nbdev[/red]", 
                              title="Missing Dependency", border_style="red"))
            return 1 