"""
Clean command for TinyTorch CLI: clean up generated files and caches.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

from .base import BaseCommand

class CleanWorkspaceCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "clean"

    @property
    def description(self) -> str:
        return "Clean up generated files, caches, and temporary files"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--all',
            action='store_true',
            help='Clean everything including build artifacts'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )
        parser.add_argument(
            '-y', '--yes',
            action='store_true',
            help='Skip confirmation prompt'
        )

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print()
        console.print(Panel(
            "🧹 Cleaning TinyTorch Workspace",
            title="Workspace Cleanup",
            border_style="bright_yellow"
        ))
        console.print()

        # Define patterns to clean
        patterns = {
            '__pycache__': ('__pycache__/', 'Python bytecode cache'),
            '.pytest_cache': ('.pytest_cache/', 'Pytest cache'),
            '.ipynb_checkpoints': ('.ipynb_checkpoints/', 'Jupyter checkpoints'),
            '*.pyc': ('*.pyc', 'Compiled Python files'),
            '*.pyo': ('*.pyo', 'Optimized Python files'),
            '*.pyd': ('*.pyd', 'Python extension modules'),
        }

        if args.all:
            # Additional patterns for --all
            patterns.update({
                '.coverage': ('.coverage', 'Coverage data'),
                'htmlcov': ('htmlcov/', 'Coverage HTML report'),
                '.tox': ('.tox/', 'Tox environments'),
                'dist': ('dist/', 'Distribution files'),
                'build': ('build/', 'Build files'),
                '*.egg-info': ('*.egg-info/', 'Egg info directories'),
            })

        # Scan for files to delete
        console.print("[bold cyan]🔍 Scanning for files to clean...[/bold cyan]")
        console.print()

        files_to_delete = []
        total_size = 0

        # Find __pycache__ directories
        for pycache_dir in Path.cwd().rglob('__pycache__'):
            for file in pycache_dir.iterdir():
                if file.is_file():
                    files_to_delete.append(file)
                    total_size += file.stat().st_size
            files_to_delete.append(pycache_dir)

        # Find .pytest_cache directories
        for cache_dir in Path.cwd().rglob('.pytest_cache'):
            for file in cache_dir.rglob('*'):
                if file.is_file():
                    files_to_delete.append(file)
                    total_size += file.stat().st_size
            files_to_delete.append(cache_dir)

        # Find .ipynb_checkpoints directories
        for checkpoint_dir in Path.cwd().rglob('.ipynb_checkpoints'):
            for file in checkpoint_dir.rglob('*'):
                if file.is_file():
                    files_to_delete.append(file)
                    total_size += file.stat().st_size
            files_to_delete.append(checkpoint_dir)

        # Find .pyc, .pyo, .pyd files
        for ext in ['*.pyc', '*.pyo', '*.pyd']:
            for file in Path.cwd().rglob(ext):
                if file.is_file():
                    files_to_delete.append(file)
                    total_size += file.stat().st_size

        if args.all:
            # Additional cleanups for --all flag
            for pattern in ['.coverage', 'htmlcov', '.tox', 'dist', 'build']:
                target = Path.cwd() / pattern
                if target.exists():
                    if target.is_file():
                        files_to_delete.append(target)
                        total_size += target.stat().st_size
                    elif target.is_dir():
                        for file in target.rglob('*'):
                            if file.is_file():
                                files_to_delete.append(file)
                                total_size += file.stat().st_size
                        files_to_delete.append(target)

            # Find .egg-info directories
            for egg_info in Path.cwd().rglob('*.egg-info'):
                if egg_info.is_dir():
                    for file in egg_info.rglob('*'):
                        if file.is_file():
                            files_to_delete.append(file)
                            total_size += file.stat().st_size
                    files_to_delete.append(egg_info)

        if not files_to_delete:
            console.print(Panel(
                "[green]✅ Workspace is already clean![/green]\n\n"
                "No temporary files or caches found.",
                title="Clean Workspace",
                border_style="green"
            ))
            return 0

        # Count file types
        file_count = len([f for f in files_to_delete if f.is_file()])
        dir_count = len([f for f in files_to_delete if f.is_dir()])

        # Summary table
        summary_table = Table(title="Files Found", show_header=True, header_style="bold yellow")
        summary_table.add_column("Type", style="cyan", width=30)
        summary_table.add_column("Count", justify="right", width=15)
        summary_table.add_column("Size", width=20)

        # Count by type
        pycache_count = len([f for f in files_to_delete if '__pycache__' in str(f)])
        pytest_count = len([f for f in files_to_delete if '.pytest_cache' in str(f)])
        checkpoint_count = len([f for f in files_to_delete if '.ipynb_checkpoints' in str(f)])
        pyc_count = len([f for f in files_to_delete if str(f).endswith(('.pyc', '.pyo', '.pyd'))])

        if pycache_count > 0:
            summary_table.add_row("__pycache__/", str(pycache_count), "—")
        if pytest_count > 0:
            summary_table.add_row(".pytest_cache/", str(pytest_count), "—")
        if checkpoint_count > 0:
            summary_table.add_row(".ipynb_checkpoints/", str(checkpoint_count), "—")
        if pyc_count > 0:
            summary_table.add_row("*.pyc/*.pyo/*.pyd", str(pyc_count), f"{total_size / 1024 / 1024:.2f} MB")

        if args.all:
            other_count = file_count - pycache_count - pytest_count - checkpoint_count - pyc_count
            if other_count > 0:
                summary_table.add_row("Build artifacts", str(other_count), "—")

        summary_table.add_row("[bold]Total[/bold]", f"[bold]{file_count} files, {dir_count} dirs[/bold]", f"[bold]{total_size / 1024 / 1024:.2f} MB[/bold]")

        console.print(summary_table)
        console.print()

        # Dry run mode
        if args.dry_run:
            console.print(Panel(
                "[yellow]🔍 DRY RUN MODE[/yellow]\n\n"
                f"Would delete {file_count} files and {dir_count} directories ({total_size / 1024 / 1024:.2f} MB)\n\n"
                "[dim]Remove --dry-run flag to actually delete these files[/dim]",
                title="Dry Run",
                border_style="yellow"
            ))
            return 0

        # Confirm deletion
        if not args.yes:
            confirmed = Confirm.ask(
                f"\n[yellow]⚠️  Delete {file_count} files and {dir_count} directories ({total_size / 1024 / 1024:.2f} MB)?[/yellow]",
                default=False
            )
            if not confirmed:
                console.print("\n[dim]Cleanup cancelled.[/dim]")
                return 0

        # Perform cleanup
        console.print()
        console.print("[bold cyan]🗑️  Cleaning workspace...[/bold cyan]")

        deleted_files = 0
        deleted_dirs = 0
        freed_space = 0

        # Delete files first, then directories
        for item in files_to_delete:
            try:
                if item.is_file():
                    size = item.stat().st_size
                    item.unlink()
                    deleted_files += 1
                    freed_space += size
                elif item.is_dir() and not any(item.iterdir()):  # Only delete if empty
                    shutil.rmtree(item, ignore_errors=True)
                    deleted_dirs += 1
            except Exception as e:
                console.print(f"[dim red]  ✗ Failed to delete {item}: {e}[/dim red]")

        # Success message
        console.print()
        console.print(Panel(
            f"[bold green]✅ Workspace Cleaned![/bold green]\n\n"
            f"🗑️  Deleted: {deleted_files} files, {deleted_dirs} directories\n"
            f"💾 Freed: {freed_space / 1024 / 1024:.2f} MB\n"
            f"⏱️  Time: < 1 second",
            title="Cleanup Complete",
            border_style="green"
        ))

        return 0
