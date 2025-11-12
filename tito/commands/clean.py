"""
Clean command for TinyTorch CLI: cleans up module directories to start fresh.
"""

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class CleanCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "clean"

    @property
    def description(self) -> str:
        return "Clean up module directories (notebooks, cache, etc.)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("module", nargs="?", help="Clean specific module only")
        parser.add_argument("--notebooks", action="store_true", help="Remove generated notebook files")
        parser.add_argument("--cache", action="store_true", help="Remove Python cache files")
        parser.add_argument("--all", action="store_true", help="Clean all modules")
        parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        console.print(Panel("🧹 Cleaning Module Directories", 
                           title="Module Cleanup", border_style="bright_yellow"))
        
        modules_dir = Path("modules")
        if not modules_dir.exists():
            console.print(Panel("[red]❌ modules/ directory not found[/red]", 
                              title="Error", border_style="red"))
            return 1
        
        # Determine what to clean (file types)
        clean_notebooks = args.notebooks or (not args.notebooks and not args.cache)
        clean_cache = args.cache or (not args.notebooks and not args.cache)
        
        # Determine which modules to clean
        if args.module:
            module_path = modules_dir / args.module
            if not module_path.exists():
                console.print(Panel(f"[red]❌ Module '{args.module}' not found[/red]", 
                                  title="Module Not Found", border_style="red"))
                return 1
            module_dirs = [module_path]
        elif args.all:
            # Find all module directories (exclude special directories)
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache', 'sidebar.yml', 'nbdev.yml'}
            module_dirs = [d for d in modules_dir.iterdir() 
                          if d.is_dir() and d.name not in exclude_dirs]
        else:
            # No module specified and no --all flag
            console.print(Panel("[red]❌ Please specify a module name or use --all to clean all modules[/red]\n\n"
                              "[dim]Examples:[/dim]\n"
                              "[dim]  tito module clean tensor     - Clean specific module[/dim]\n"
                              "[dim]  tito module clean --all      - Clean all modules[/dim]", 
                              title="Module Required", border_style="red"))
            return 1
        
        if not module_dirs:
            console.print(Panel("[yellow]⚠️  No modules found to clean[/yellow]", 
                              title="Nothing to Clean", border_style="yellow"))
            return 0
        
        # Show what will be cleaned
        clean_text = Text()
        clean_text.append("📋 Cleanup Plan:\n\n", style="bold cyan")
        
        files_to_remove = []
        for module_dir in module_dirs:
            module_name = module_dir.name
            clean_text.append(f"📁 {module_name}:\n", style="bold white")
            
            if clean_notebooks:
                # Find .ipynb files
                for ipynb_file in module_dir.glob("*.ipynb"):
                    files_to_remove.append(ipynb_file)
                    clean_text.append(f"  🗑️  {ipynb_file.name}\n", style="yellow")
            
            if clean_cache:
                # Find __pycache__ directories
                pycache_dirs = []
                for pycache in module_dir.rglob("__pycache__"):
                    if pycache.is_dir():
                        pycache_dirs.append(pycache)
                        files_to_remove.append(pycache)
                        clean_text.append(f"  🗑️  {pycache.relative_to(module_dir)}/\n", style="yellow")
                
                # Find .pyc files that are NOT inside __pycache__ directories
                for pyc_file in module_dir.rglob("*.pyc"):
                    # Check if this pyc file is inside any __pycache__ directory
                    is_in_pycache = any(pycache in pyc_file.parents for pycache in pycache_dirs)
                    if not is_in_pycache:
                        files_to_remove.append(pyc_file)
                        clean_text.append(f"  🗑️  {pyc_file.relative_to(module_dir)}\n", style="yellow")
        
        if not files_to_remove:
            console.print(Panel("[green]✅ No files found to clean - modules are already clean![/green]", 
                              title="Already Clean", border_style="green"))
            return 0
        
        clean_text.append(f"\n📊 Total: {len(files_to_remove)} files/directories to remove\n", style="bold cyan")
        
        console.print(Panel(clean_text, title="Cleanup Preview", border_style="bright_yellow"))
        
        # Ask for confirmation unless --force is used
        if not args.force:
            console.print("\n[yellow]This will permanently remove the files listed above.[/yellow]")
            console.print("[yellow]Python source files (*.py) will be preserved.[/yellow]\n")
            
            try:
                response = input("Are you sure you want to proceed? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Cleanup cancelled.[/cyan]", 
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("[cyan]Cleanup cancelled.[/cyan]", 
                                  title="Cancelled", border_style="cyan"))
                return 0
        
        # Perform cleanup
        removed_count = 0
        error_count = 0
        
        for file_path in files_to_remove:
            try:
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                removed_count += 1
            except Exception as e:
                console.print(f"  ❌ Failed to remove {file_path}: {e}")
                error_count += 1
        
        # Show results
        result_text = Text()
        if removed_count > 0:
            result_text.append(f"✅ Successfully removed {removed_count} files/directories\n", style="bold green")
        if error_count > 0:
            result_text.append(f"❌ Failed to remove {error_count} files/directories\n", style="bold red")
        
        if removed_count > 0:
            result_text.append("\n💡 Next steps:\n", style="bold yellow")
            result_text.append("  • Run: tito module notebooks      - Regenerate notebooks\n", style="white")
            result_text.append("  • Run: tito module test --all     - Test all modules\n", style="white")
            result_text.append("  • Run: tito module export --all   - Export to package\n", style="white")
        
        border_style = "green" if error_count == 0 else "yellow"
        console.print(Panel(result_text, title="Cleanup Complete", border_style=border_style))
        
        return 0 if error_count == 0 else 1 