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
        # Add subcommands first
        subparsers = parser.add_subparsers(
            dest='module_command',
            help='Module operations'
        )
        
        # Complete command - the key workflow
        complete_parser = subparsers.add_parser(
            'complete',
            help='Complete module: run tests, export if passing, update progress'
        )
        complete_parser.add_argument(
            'module_number',
            help='Module number to complete (01, 02, 03, etc.)'
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
        
        # Status command
        status_parser = subparsers.add_parser(
            'status',
            help='Show module completion status'
        )
        
        # Advanced commands (less commonly used)
        test_parser = subparsers.add_parser(
            'test',
            help='Run tests for specific module'
        )
        test_parser.add_argument('module_number', help='Module to test')
        
        export_parser = subparsers.add_parser(
            'export',
            help='Export module to package'
        )
        export_parser.add_argument('module_number', help='Module to export')
    
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
    
    def open_module(self, module_number: str) -> int:
        """Open a module in Jupyter Lab."""
        module_mapping = self.get_module_mapping()
        normalized = self.normalize_module_number(module_number)
        
        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            self.console.print("💡 Available modules: 01-21")
            return 1
        
        module_name = module_mapping[normalized]
        
        self.console.print(f"🚀 Opening Module {normalized}: {module_name}")
        self.console.print("💡 Work in Jupyter, save your changes, then run:")
        self.console.print(f"   [bold cyan]tito module complete {normalized}[/bold cyan]")
        
        # Use the existing view command
        fake_args = Namespace()
        fake_args.module = module_name
        fake_args.force = False
        
        view_command = ViewCommand(self.config)
        return view_command.run(fake_args)
    
    def complete_module(self, module_number: str, skip_tests: bool = False, skip_export: bool = False) -> int:
        """Complete a module with testing and export."""
        module_mapping = self.get_module_mapping()
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
        
        # Step 4: Show next steps
        self.show_next_steps(normalized)
        
        return 0 if success else 1
    
    def run_module_tests(self, module_name: str) -> int:
        """Run tests for a specific module."""
        try:
            # Run the module's inline tests
            module_dir = self.config.modules_dir / module_name
            dev_file = module_dir / f"{module_name.split('_')[1]}_dev.py"
            
            if not dev_file.exists():
                self.console.print(f"[yellow]⚠️  No dev file found: {dev_file}[/yellow]")
                return 0
            
            # Execute the Python file to run inline tests
            result = subprocess.run([
                sys.executable, str(dev_file)
            ], capture_output=True, text=True, cwd=module_dir)
            
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
            # Use the existing export command
            fake_args = Namespace()
            fake_args.module = module_name
            fake_args.force = False
            
            export_command = ExportCommand(self.config)
            return export_command.run(fake_args)
            
        except Exception as e:
            self.console.print(f"[red]Error exporting module: {e}[/red]")
            return 1
    
    def update_progress(self, module_number: str, module_name: str) -> None:
        """Update user progress tracking."""
        progress_file = self.config.project_root / "progress.json"
        
        try:
            import json
            from datetime import datetime
            
            # Load existing progress
            progress = {}
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
            
            # Update progress
            if 'completed_modules' not in progress:
                progress['completed_modules'] = []
            
            if module_number not in progress['completed_modules']:
                progress['completed_modules'].append(module_number)
            
            progress['last_completed'] = module_number
            progress['last_updated'] = datetime.now().isoformat()
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            self.console.print(f"📈 Progress updated: {len(progress['completed_modules'])} modules completed")
            
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not update progress: {e}[/yellow]")
    
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
        progress_file = self.config.project_root / "progress.json"
        module_mapping = self.get_module_mapping()
        
        try:
            progress = {}
            if progress_file.exists():
                import json
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
            
            completed = progress.get('completed_modules', [])
            
            self.console.print(Panel(
                "📊 Module Completion Status",
                title="Your Progress",
                border_style="bright_blue"
            ))
            
            for num, name in module_mapping.items():
                status = "✅" if num in completed else "⏳"
                self.console.print(f"  {status} Module {num}: {name}")
            
            self.console.print(f"\n📈 Progress: {len(completed)}/{len(module_mapping)} modules completed")
            
            if completed:
                last = progress.get('last_completed', completed[-1])
                next_num = f"{int(last) + 1:02d}"
                if next_num in module_mapping:
                    self.console.print(f"💡 Next: [bold cyan]tito module {next_num}[/bold cyan]")
            else:
                self.console.print("💡 Start with: [bold cyan]tito module 01[/bold cyan]")
            
            return 0
            
        except Exception as e:
            self.console.print(f"[red]Error reading progress: {e}[/red]")
            return 1
    
    def run(self, args: Namespace) -> int:
        """Execute the module workflow command."""
        # Handle subcommands
        if hasattr(args, 'module_command') and args.module_command:
            if args.module_command == 'complete':
                return self.complete_module(
                    args.module_number,
                    getattr(args, 'skip_tests', False),
                    getattr(args, 'skip_export', False)
                )
            elif args.module_command == 'status':
                return self.show_status()
            elif args.module_command == 'test':
                module_mapping = self.get_module_mapping()
                normalized = self.normalize_module_number(args.module_number)
                if normalized in module_mapping:
                    return self.run_module_tests(module_mapping[normalized])
                else:
                    self.console.print(f"[red]❌ Module {normalized} not found[/red]")
                    return 1
            elif args.module_command == 'export':
                module_mapping = self.get_module_mapping()
                normalized = self.normalize_module_number(args.module_number)
                if normalized in module_mapping:
                    return self.export_module(module_mapping[normalized])
                else:
                    self.console.print(f"[red]❌ Module {normalized} not found[/red]")
                    return 1
        
        # Show help if no valid command
        self.console.print(Panel(
            "[bold cyan]Module Workflow Commands[/bold cyan]\n\n"
            "[bold]Main Workflow:[/bold]\n"
            "  [bold green]tito module 01[/bold green]           - Open Module 01 in Jupyter Lab\n"
            "  [bold green]tito module complete 01[/bold green]  - Complete Module 01 (test + export)\n\n"
            "[bold]Status & Management:[/bold]\n"
            "  [bold]tito module status[/bold]        - Show completion progress\n"
            "  [bold]tito module test 01[/bold]       - Run tests for Module 01\n"
            "  [bold]tito module export 01[/bold]     - Export Module 01 to package\n\n"
            "[bold]Natural Workflow:[/bold]\n"
            "  1. [dim]tito module 01[/dim]           → Open and work in Jupyter\n"
            "  2. [dim]Save your work[/dim]           → Ctrl+S in Jupyter\n"
            "  3. [dim]tito module complete 01[/dim]  → Test, export, track progress\n"
            "  4. [dim]tito module 02[/dim]           → Continue to next module",
            title="Module Development Workflow",
            border_style="bright_cyan"
        ))
        
        return 0
