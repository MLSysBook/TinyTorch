"""
nbdev command for TinyTorch CLI: runs nbdev commands for notebook development.
"""

import subprocess
from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from .base import BaseCommand

class NbdevCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "nbdev"

    @property
    def description(self) -> str:
        return "nbdev notebook development commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--export", action="store_true", help="Export notebooks to Python package")
        parser.add_argument("--build-docs", action="store_true", help="Build documentation from notebooks")
        parser.add_argument("--test", action="store_true", help="Run notebook tests")
        parser.add_argument("--clean", action="store_true", help="Clean notebook outputs")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        console.print(Panel("📓 nbdev Notebook Development", 
                           title="Notebook Tools", border_style="bright_cyan"))
        
        if args.export:
            # Use the sync command logic
            from .sync import SyncCommand
            sync_cmd = SyncCommand(self.config)
            sync_args = ArgumentParser()
            sync_cmd.add_arguments(sync_args)
            sync_args = sync_args.parse_args([])  # Empty args for sync
            return sync_cmd.run(sync_args)
        
        elif args.build_docs:
            console.print("📚 Building documentation from notebooks...")
            result = subprocess.run(["nbdev_docs"], capture_output=True, text=True)
            if result.returncode == 0:
                console.print(Panel("[green]✅ Documentation built successfully![/green]", 
                                  title="Docs Success", border_style="green"))
            else:
                console.print(Panel(f"[red]❌ Docs build failed: {result.stderr}[/red]", 
                                  title="Docs Error", border_style="red"))
            return result.returncode
        
        elif args.test:
            console.print("🧪 Running notebook tests...")
            result = subprocess.run(["nbdev_test"], capture_output=True, text=True)
            if result.returncode == 0:
                console.print(Panel("[green]✅ Notebook tests passed![/green]", 
                                  title="Test Success", border_style="green"))
            else:
                console.print(Panel(f"[red]❌ Notebook tests failed: {result.stderr}[/red]", 
                                  title="Test Error", border_style="red"))
            return result.returncode
        
        elif args.clean:
            console.print("🧹 Cleaning notebook outputs...")
            result = subprocess.run(["nbdev_clean"], capture_output=True, text=True)
            if result.returncode == 0:
                console.print(Panel("[green]✅ Notebook outputs cleaned![/green]", 
                                  title="Clean Success", border_style="green"))
            else:
                console.print(Panel(f"[red]❌ Clean failed: {result.stderr}[/red]", 
                                  title="Clean Error", border_style="red"))
            return result.returncode
        
        else:
            console.print(Panel("[yellow]⚠️  No nbdev action specified. Use --export, --build-docs, --test, or --clean[/yellow]", 
                              title="No Action", border_style="yellow"))
            return 1 