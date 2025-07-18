"""
Export command for TinyTorch CLI: exports notebook code to Python package using nbdev.
"""

import subprocess
import sys
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional
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
        parser.add_argument("--from-release", action="store_true", help="Export from release directory (student version) instead of source")

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

    def _discover_modules(self) -> list:
        """Discover available modules from modules/source directory."""
        source_dir = Path("modules/source")
        modules = []
        
        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    modules.append(module_dir.name)
        
        return sorted(modules)

    def _show_export_details(self, console, module_name: Optional[str] = None):
        """Show detailed export information including where each module exports to."""
        exports_text = Text()
        exports_text.append("📦 Export Details:\n", style="bold cyan")
        
        if module_name:
            # Single module export
            module_path = Path(f"modules/source/{module_name}")
            export_target = self._get_export_target(module_path)
            if export_target != "unknown":
                target_file = export_target.replace('.', '/') + '.py'
                exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
                
                # Extract the short name for display
                short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
                exports_text.append(f"     Source: modules/source/{module_name}/{short_name}_dev.py\n", style="dim")
                exports_text.append(f"     Target: tinytorch/{target_file}\n", style="dim")
            else:
                exports_text.append(f"  ❓ {module_name} → export target not found\n", style="yellow")
        else:
            # All modules export
            modules = self._discover_modules()
            for module_name in modules:
                module_path = Path(f"modules/source/{module_name}")
                export_target = self._get_export_target(module_path)
                if export_target != "unknown":
                    target_file = export_target.replace('.', '/') + '.py'
                    exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
        
        # Show what was actually created
        exports_text.append("\n📁 Generated Files:\n", style="bold cyan")
        tinytorch_path = Path("tinytorch")
        if tinytorch_path.exists():
            for py_file in tinytorch_path.rglob("*.py"):
                if py_file.name != "__init__.py" and py_file.stat().st_size > 100:  # Non-empty files
                    rel_path = py_file.relative_to(tinytorch_path)
                    exports_text.append(f"  ✅ tinytorch/{rel_path}\n", style="green")
        
        exports_text.append("\n💡 Next steps:\n", style="bold yellow")
        exports_text.append("  • Run: tito test --all\n", style="white")
        exports_text.append("  • Or: tito test <module_name>\n", style="white")
        
        console.print(Panel(exports_text, title="Export Summary", border_style="bright_green"))

    def _convert_py_to_notebook(self, module_path: Path) -> bool:
        """Convert .py dev file to .ipynb using Jupytext."""
        module_name = module_path.name
        short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
        
        dev_file = module_path / f"{short_name}_dev.py"
        if not dev_file.exists():
            return False
        
        notebook_file = module_path / f"{short_name}_dev.ipynb"
        
        # Check if notebook is newer than .py file
        if notebook_file.exists():
            py_mtime = dev_file.stat().st_mtime
            nb_mtime = notebook_file.stat().st_mtime
            if nb_mtime > py_mtime:
                return True  # Notebook is up to date
        
        try:
            result = subprocess.run([
                "jupytext", "--to", "ipynb", str(dev_file)
            ], capture_output=True, text=True, cwd=module_path)
            
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _convert_all_modules(self) -> list:
        """Convert all modules' .py files to .ipynb files."""
        modules = self._discover_modules()
        converted = []
        
        for module_name in modules:
            module_path = Path(f"modules/source/{module_name}")
            if self._convert_py_to_notebook(module_path):
                converted.append(module_name)
        
        return converted

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Determine what to export
        if hasattr(args, 'module') and args.module:
            # Validate module exists
            module_path = Path(f"modules/source/{args.module}")
            if not module_path.exists():
                console.print(Panel(f"[red]❌ Module '{args.module}' not found in modules/source/[/red]", 
                                  title="Module Not Found", border_style="red"))
                
                # Show available modules
                available_modules = self._discover_modules()
                if available_modules:
                    help_text = Text()
                    help_text.append("Available modules:\n", style="bold yellow")
                    for module in available_modules:
                        help_text.append(f"  • {module}\n", style="white")
                    console.print(Panel(help_text, title="Available Modules", border_style="yellow"))
                
                return 1
            
            console.print(Panel(f"🔄 Exporting Module: {args.module}", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            # Step 1: Convert .py to .ipynb
            console.print(f"📝 Converting {args.module} Python file to notebook...")
            if not self._convert_py_to_notebook(module_path):
                console.print(Panel("[red]❌ Failed to convert .py file to notebook. Is jupytext installed?[/red]", 
                                  title="Conversion Error", border_style="red"))
                return 1
            
            console.print(f"🔄 Exporting {args.module} notebook to tinytorch package...")
            
            # Step 2: Use nbdev_export with --path for specific module
            cmd = ["nbdev_export", "--path", str(module_path)]
        elif hasattr(args, 'all') and args.all:
            console.print(Panel("🔄 Exporting All Modules to Package", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            # Step 1: Convert all .py files to .ipynb
            console.print("📝 Converting all Python files to notebooks...")
            converted = self._convert_all_modules()
            if not converted:
                console.print(Panel("[red]❌ No modules converted. Check if jupytext is installed and .py files exist.[/red]", 
                                  title="Conversion Error", border_style="red"))
                return 1
            
            console.print(f"✅ Converted {len(converted)} modules: {', '.join(converted)}")
            console.print("🔄 Exporting all notebook code to tinytorch package...")
            
            # Step 2: Use nbdev_export for all modules  
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
                module_name = args.module if hasattr(args, 'module') and args.module else None
                self._show_export_details(console, module_name)
                
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