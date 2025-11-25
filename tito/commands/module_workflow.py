"""
Enhanced Module Workflow for TinyTorch CLI.

Implements the natural workflow:
1. tito module 01 → Opens module 01 in Jupyter
2. Student works and saves
3. tito module complete 01 → Tests, exports, updates progress
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import BaseCommand
from .view import ViewCommand
from .test import TestCommand
from .export import ExportCommand
from .module_reset import ModuleResetCommand
from ..core.exceptions import ModuleNotFoundError

class ModuleWorkflowCommand(BaseCommand):
    """Enhanced module command with natural workflow."""
    
    @property
    def name(self) -> str:
        return "module"
    
    @property
    def description(self) -> str:
        return "Module development workflow - open, work, complete"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add module workflow arguments."""
        # Add subcommands - clean lifecycle workflow
        subparsers = parser.add_subparsers(
            dest='module_command',
            help='Module lifecycle operations'
        )
        
        # START command - begin working on a module
        start_parser = subparsers.add_parser(
            'start',
            help='Start working on a module (first time)'
        )
        start_parser.add_argument(
            'module_number',
            help='Module number to start (01, 02, 03, etc.)'
        )
        
        # RESUME command - continue working on a module
        resume_parser = subparsers.add_parser(
            'resume',
            help='Resume working on a module (continue previous work)'
        )
        resume_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to resume (01, 02, 03, etc.) - defaults to last worked'
        )
        
        # COMPLETE command - finish and validate a module
        complete_parser = subparsers.add_parser(
            'complete',
            help='Complete module: run tests, export if passing, update progress'
        )
        complete_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to complete (01, 02, 03, etc.) - defaults to current'
        )
        complete_parser.add_argument(
            '--skip-tests',
            action='store_true',
            help='Skip integration tests'
        )
        complete_parser.add_argument(
            '--skip-export',
            action='store_true',
            help='Skip automatic export'
        )
        
        # RESET command - reset module to clean state
        reset_parser = subparsers.add_parser(
            'reset',
            help='Reset module to clean state (backup + unexport + restore)'
        )
        reset_parser.add_argument(
            'module_number',
            help='Module number to reset (01, 02, 03, etc.)'
        )
        reset_parser.add_argument(
            '--soft',
            action='store_true',
            help='Soft reset: backup + restore (keep exports)'
        )
        reset_parser.add_argument(
            '--hard',
            action='store_true',
            help='Hard reset: backup + unexport + restore [DEFAULT]'
        )
        reset_parser.add_argument(
            '--from-git',
            action='store_true',
            help='Restore from git HEAD [DEFAULT]'
        )
        reset_parser.add_argument(
            '--restore-backup',
            metavar='TIMESTAMP',
            help='Restore from specific backup'
        )
        reset_parser.add_argument(
            '--list-backups',
            action='store_true',
            help='List available backups'
        )
        reset_parser.add_argument(
            '--no-backup',
            action='store_true',
            help='Skip backup (dangerous)'
        )
        reset_parser.add_argument(
            '--force',
            action='store_true',
            help='Skip confirmation'
        )

        # STATUS command - show progress
        status_parser = subparsers.add_parser(
            'status',
            help='Show module completion status and progress'
        )
    
    def get_module_mapping(self) -> Dict[str, str]:
        """Get mapping from numbers to module names."""
        return {
            "01": "01_tensor",
            "02": "02_activations",
            "03": "03_layers",
            "04": "04_losses",
            "05": "05_autograd",
            "06": "06_optimizers",
            "07": "07_training",
            "08": "08_spatial",
            "09": "09_dataloader",
            "10": "10_tokenization",
            "11": "11_embeddings",
            "12": "12_attention",
            "13": "13_transformers",
            "14": "14_profiling",
            "15": "15_acceleration",
            "16": "16_quantization",
            "17": "17_compression",
            "18": "18_caching",
            "19": "19_benchmarking",
            "20": "20_capstone",
            "21": "21_mlops"
        }
    
    def normalize_module_number(self, module_input: str) -> str:
        """Normalize module input to 2-digit format."""
        if module_input.isdigit():
            return f"{int(module_input):02d}"
        return module_input
    
    def start_module(self, module_number: str) -> int:
        """Start working on a module (first time)."""
        module_mapping = self.get_module_mapping()
        normalized = self.normalize_module_number(module_number)
        
        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            self.console.print("💡 Available modules: 01-21")
            return 1
        
        module_name = module_mapping[normalized]
        
        # Check if already started
        if self.is_module_started(normalized):
            self.console.print(f"[yellow]⚠️  Module {normalized} already started[/yellow]")
            self.console.print(f"💡 Did you mean: [bold cyan]tito module resume {normalized}[/bold cyan]")
            return 1
        
        # Mark as started
        self.mark_module_started(normalized)
        
        self.console.print(f"🚀 Starting Module {normalized}: {module_name}")
        self.console.print("💡 Work in Jupyter, save your changes, then run:")
        self.console.print(f"   [bold cyan]tito module complete {normalized}[/bold cyan]")
        
        return self._open_jupyter(module_name)
    
    def resume_module(self, module_number: Optional[str] = None) -> int:
        """Resume working on a module (continue previous work)."""
        module_mapping = self.get_module_mapping()
        
        # If no module specified, resume last worked
        if not module_number:
            last_worked = self.get_last_worked_module()
            if not last_worked:
                self.console.print("[yellow]⚠️  No module to resume[/yellow]")
                self.console.print("💡 Start with: [bold cyan]tito module start 01[/bold cyan]")
                return 1
            module_number = last_worked
        
        normalized = self.normalize_module_number(module_number)
        
        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            self.console.print("💡 Available modules: 01-21")
            return 1
        
        module_name = module_mapping[normalized]
        
        # Check if module was started
        if not self.is_module_started(normalized):
            self.console.print(f"[yellow]⚠️  Module {normalized} not started yet[/yellow]")
            self.console.print(f"💡 Start with: [bold cyan]tito module start {normalized}[/bold cyan]")
            return 1
        
        # Update last worked
        self.update_last_worked(normalized)
        
        self.console.print(f"🔄 Resuming Module {normalized}: {module_name}")
        self.console.print("💡 Continue your work, then run:")
        self.console.print(f"   [bold cyan]tito module complete {normalized}[/bold cyan]")
        
        return self._open_jupyter(module_name)
    
    def _open_jupyter(self, module_name: str) -> int:
        """Open Jupyter Lab for a module."""
        # Use the existing view command
        fake_args = Namespace()
        fake_args.module = module_name
        fake_args.force = False
        
        view_command = ViewCommand(self.config)
        return view_command.run(fake_args)
    
    def complete_module(self, module_number: Optional[str] = None, skip_tests: bool = False, skip_export: bool = False) -> int:
        """Complete a module with testing and export."""
        module_mapping = self.get_module_mapping()
        
        # If no module specified, complete current/last worked
        if not module_number:
            last_worked = self.get_last_worked_module()
            if not last_worked:
                self.console.print("[yellow]⚠️  No module to complete[/yellow]")
                self.console.print("💡 Start with: [bold cyan]tito module start 01[/bold cyan]")
                return 1
            module_number = last_worked
        
        normalized = self.normalize_module_number(module_number)
        
        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            return 1
        
        module_name = module_mapping[normalized]
        
        self.console.print(Panel(
            f"🎯 Completing Module {normalized}: {module_name}",
            title="Module Completion Workflow",
            border_style="bright_green"
        ))
        
        success = True
        
        # Step 1: Run integration tests
        if not skip_tests:
            self.console.print("🧪 Running integration tests...")
            test_result = self.run_module_tests(module_name)
            if test_result != 0:
                self.console.print(f"[red]❌ Tests failed for {module_name}[/red]")
                self.console.print("💡 Fix the issues and try again")
                return 1
            self.console.print("✅ All tests passed!")
        
        # Step 2: Export to package
        if not skip_export:
            self.console.print("📦 Exporting to TinyTorch package...")
            export_result = self.export_module(module_name)
            if export_result != 0:
                self.console.print(f"[red]❌ Export failed for {module_name}[/red]")
                success = False
            else:
                self.console.print("✅ Module exported successfully!")
        
        # Step 3: Update progress tracking
        self.update_progress(normalized, module_name)
        
        # Step 4: Check for milestone unlocks
        if success:
            self._check_milestone_unlocks(module_name)
        
        # Step 5: Show next steps
        self.show_next_steps(normalized)
        
        return 0 if success else 1
    
    def run_module_tests(self, module_name: str) -> int:
        """Run tests for a specific module."""
        try:
            # Run the module's inline tests from source file in src/
            project_root = Path.cwd()
            src_dir = project_root / "src" / module_name
            dev_file = src_dir / f"{module_name}.py"
            
            if not dev_file.exists():
                self.console.print(f"[yellow]⚠️  No source file found: {dev_file}[/yellow]")
                return 0
            
            # Execute the Python file to run inline tests
            result = subprocess.run([
                sys.executable, str(dev_file.absolute())
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                return 0
            else:
                self.console.print(f"[red]Test output:[/red]\n{result.stdout}")
                if result.stderr:
                    self.console.print(f"[red]Errors:[/red]\n{result.stderr}")
                return 1
                
        except Exception as e:
            self.console.print(f"[red]Error running tests: {e}[/red]")
            return 1
    
    def export_module(self, module_name: str) -> int:
        """Export module to the TinyTorch package."""
        try:
            # Use the new source command for exporting
            from .src import SrcCommand
            
            fake_args = Namespace()
            fake_args.src_command = 'export'  # Subcommand
            fake_args.modules = [module_name]     # List of modules to export
            fake_args.test_checkpoint = False
            
            src_command = SrcCommand(self.config)
            return src_command.run(fake_args)
            
        except Exception as e:
            self.console.print(f"[red]Error exporting module: {e}[/red]")
            return 1
    
    def get_progress_data(self) -> dict:
        """Get current progress data."""
        progress_file = self.config.project_root / "progress.json"
        
        try:
            import json
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            'started_modules': [],
            'completed_modules': [],
            'last_worked': None,
            'last_completed': None,
            'last_updated': None
        }
    
    def save_progress_data(self, progress: dict) -> None:
        """Save progress data."""
        progress_file = self.config.project_root / "progress.json"
        
        try:
            import json
            from datetime import datetime
            progress['last_updated'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not save progress: {e}[/yellow]")
    
    def is_module_started(self, module_number: str) -> bool:
        """Check if a module has been started."""
        progress = self.get_progress_data()
        return module_number in progress.get('started_modules', [])
    
    def is_module_completed(self, module_number: str) -> bool:
        """Check if a module has been completed."""
        progress = self.get_progress_data()
        return module_number in progress.get('completed_modules', [])
    
    def mark_module_started(self, module_number: str) -> None:
        """Mark a module as started."""
        progress = self.get_progress_data()
        
        if 'started_modules' not in progress:
            progress['started_modules'] = []
        
        if module_number not in progress['started_modules']:
            progress['started_modules'].append(module_number)
        
        progress['last_worked'] = module_number
        self.save_progress_data(progress)
    
    def update_last_worked(self, module_number: str) -> None:
        """Update the last worked module."""
        progress = self.get_progress_data()
        progress['last_worked'] = module_number
        self.save_progress_data(progress)
    
    def get_last_worked_module(self) -> Optional[str]:
        """Get the last worked module."""
        progress = self.get_progress_data()
        return progress.get('last_worked')
    
    def update_progress(self, module_number: str, module_name: str) -> None:
        """Update user progress tracking."""
        progress = self.get_progress_data()
        
        # Update completed modules
        if 'completed_modules' not in progress:
            progress['completed_modules'] = []
        
        if module_number not in progress['completed_modules']:
            progress['completed_modules'].append(module_number)
        
        progress['last_completed'] = module_number
        self.save_progress_data(progress)
        
        self.console.print(f"📈 Progress updated: {len(progress['completed_modules'])} modules completed")
    
    def show_next_steps(self, completed_module: str) -> None:
        """Show next steps after completing a module."""
        module_mapping = self.get_module_mapping()
        completed_num = int(completed_module)
        next_num = f"{completed_num + 1:02d}"
        
        if next_num in module_mapping:
            next_module = module_mapping[next_num]
            self.console.print(Panel(
                f"🎉 Module {completed_module} completed!\n\n"
                f"Next steps:\n"
                f"  [bold cyan]tito module {next_num}[/bold cyan] - Start {next_module}\n"
                f"  [dim]tito module status[/dim] - View overall progress",
                title="What's Next?",
                border_style="green"
            ))
        else:
            self.console.print(Panel(
                f"🎉 Module {completed_module} completed!\n\n"
                "🏆 Congratulations! You've completed all available modules!\n"
                "🚀 You're now ready to build production ML systems!",
                title="All Modules Complete!",
                border_style="gold1"
            ))
    
    def show_status(self) -> int:
        """Show module completion status."""
        module_mapping = self.get_module_mapping()
        progress = self.get_progress_data()
        
        started = progress.get('started_modules', [])
        completed = progress.get('completed_modules', [])
        last_worked = progress.get('last_worked')
        
        self.console.print(Panel(
            "📊 Module Status & Progress",
            title="Your Learning Journey",
            border_style="bright_blue"
        ))
        
        for num, name in module_mapping.items():
            if num in completed:
                status = "✅"
                state = "completed"
            elif num in started:
                status = "🚀" if num == last_worked else "💻"
                state = "in progress" if num == last_worked else "started"
            else:
                status = "⏳"
                state = "not started"
            
            marker = " ← current" if num == last_worked else ""
            self.console.print(f"  {status} Module {num}: {name} ({state}){marker}")
        
        # Summary
        self.console.print(f"\n📈 Progress: {len(completed)}/{len(module_mapping)} completed, {len(started)} started")
        
        # Next steps
        if last_worked:
            if last_worked not in completed:
                self.console.print(f"💡 Continue: [bold cyan]tito module resume {last_worked}[/bold cyan]")
                self.console.print(f"💡 Or complete: [bold cyan]tito module complete {last_worked}[/bold cyan]")
            else:
                next_num = f"{int(last_worked) + 1:02d}"
                if next_num in module_mapping:
                    self.console.print(f"💡 Next: [bold cyan]tito module start {next_num}[/bold cyan]")
        else:
            self.console.print("💡 Start with: [bold cyan]tito module start 01[/bold cyan]")
        
        return 0
    
    def run(self, args: Namespace) -> int:
        """Execute the module workflow command."""
        # Handle subcommands
        if hasattr(args, 'module_command') and args.module_command:
            if args.module_command == 'start':
                return self.start_module(args.module_number)
            elif args.module_command == 'resume':
                return self.resume_module(getattr(args, 'module_number', None))
            elif args.module_command == 'complete':
                return self.complete_module(
                    getattr(args, 'module_number', None),
                    getattr(args, 'skip_tests', False),
                    getattr(args, 'skip_export', False)
                )
            elif args.module_command == 'reset':
                # Delegate to ModuleResetCommand
                reset_command = ModuleResetCommand(self.config)
                return reset_command.run(args)
            elif args.module_command == 'status':
                return self.show_status()
        
        # Show help if no valid command
        self.console.print(Panel(
            "[bold cyan]Module Lifecycle Commands[/bold cyan]\n\n"
            "[bold]Core Workflow:[/bold]\n"
            "  [bold green]tito module start 01[/bold green]     - Start working on Module 01 (first time)\n"
            "  [bold green]tito module resume 01[/bold green]    - Resume working on Module 01 (continue)\n"
            "  [bold green]tito module complete 01[/bold green]  - Complete Module 01 (test + export)\n"
            "  [bold yellow]tito module reset 01[/bold yellow]    - Reset Module 01 to clean state (with backup)\n\n"
            "[bold]Smart Defaults:[/bold]\n"
            "  [bold]tito module resume[/bold]        - Resume last worked module\n"
            "  [bold]tito module complete[/bold]      - Complete current module\n"
            "  [bold]tito module status[/bold]        - Show progress with states\n\n"
            "[bold]Natural Learning Flow:[/bold]\n"
            "  1. [dim]tito module start 01[/dim]     → Begin tensors (first time)\n"
            "  2. [dim]Work in Jupyter, save[/dim]    → Ctrl+S to save progress\n"
            "  3. [dim]tito module complete 01[/dim]  → Test, export, track progress\n"
            "  4. [dim]tito module start 02[/dim]     → Begin activations\n"
            "  5. [dim]tito module resume 02[/dim]    → Continue activations later\n\n"
            "[bold]Module States:[/bold]\n"
            "  ⏳ Not started  🚀 In progress  ✅ Completed\n\n"
            "[bold]Reset Options:[/bold]\n"
            "  [dim]tito module reset 01 --list-backups[/dim]   - View available backups\n"
            "  [dim]tito module reset 01 --soft[/dim]           - Keep package exports\n"
            "  [dim]tito module reset 01 --restore-backup[/dim] - Restore from backup",
            title="Module Development Workflow",
            border_style="bright_cyan"
        ))
        
        return 0
    
    def _check_milestone_unlocks(self, module_name: str) -> None:
        """Check if completing this module unlocks any milestones."""
        try:
            # Import milestone tracker
            import sys
            from pathlib import Path as PathLib
            milestone_tracker_path = PathLib(__file__).parent.parent.parent / "tests" / "milestones"
            if str(milestone_tracker_path) not in sys.path:
                sys.path.insert(0, str(milestone_tracker_path))
            
            from milestone_tracker import check_module_export
            
            # Let milestone tracker handle everything
            check_module_export(module_name, console=self.console)
        
        except ImportError:
            # Milestone tracker not available, skip silently
            pass
        except Exception as e:
            # Don't fail the workflow if milestone checking fails
            self.console.print(f"[dim]Note: Could not check milestone unlocks: {e}[/dim]")
