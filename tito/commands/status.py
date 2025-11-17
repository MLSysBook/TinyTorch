"""
Status command for TinyTorch CLI: checks status of all modules in modules/ directory.

Supports both basic status checking and comprehensive system analysis.
"""

import subprocess
import sys
import yaml
import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Union, Dict, Any, Optional

from .base import BaseCommand
from ..core.status_analyzer import TinyTorchStatusAnalyzer

class StatusCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "status"

    @property
    def description(self) -> str:
        return "Check status of all modules"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--progress", action="store_true", help="Show user progress (modules + milestones) - DEFAULT")
        parser.add_argument("--files", action="store_true", help="Show file structure and module status")
        parser.add_argument("--details", action="store_true", help="Show detailed file structure")
        parser.add_argument("--metadata", action="store_true", help="Show module metadata information")
        parser.add_argument("--test-status", action="store_true", help="Include test execution status (slower)")
        parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive system health dashboard (environment + compliance + testing)")

    def _get_export_target(self, module_path: Path) -> str:
        """
        Read the actual export target from the dev file's #| default_exp directive.
        Same logic as the export command.
        """
        # Extract short name from module directory name for dev file
        module_name = module_path.name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        dev_file = module_path / f"{short_name}.py"
        if not dev_file.exists():
            return "not_found"
        
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for #| default_exp directive
                match = re.search(r'#\|\s*default_exp\s+([^\n\r]+)', content)
                if match:
                    return match.group(1).strip()
                return "no_export"
        except Exception:
            return "read_error"

    def _count_test_functions(self, dev_file: Path) -> int:
        """Count the number of test functions in a dev file."""
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count lines that start with "def test_"
                lines = content.split('\n')
                test_functions = [line for line in lines if line.strip().startswith('def test_')]
                return len(test_functions)
        except Exception:
            return 0

    def _count_export_functions(self, dev_file: Path) -> int:
        """Count the number of exported functions/classes in a dev file."""
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count lines that have #| export directive
                lines = content.split('\n')
                export_lines = [line for line in lines if line.strip().startswith('#| export')]
                return len(export_lines)
        except Exception:
            return 0

    def run(self, args: Namespace) -> int:
        console = self.console

        # Handle comprehensive analysis mode
        if args.comprehensive:
            return self._run_comprehensive_analysis()

        # Handle progress view (default if no flags, or --progress)
        if not args.files and not args.details and not args.metadata and not args.test_status:
            return self._run_progress_view()

        if args.progress:
            return self._run_progress_view()

        # Standard file status check mode
        return self._run_standard_status(args)
    
    def _run_progress_view(self) -> int:
        """Show unified user progress view (modules + milestones)."""
        console = self.console
        import json
        from datetime import datetime

        # Load progress data
        progress_file = Path(".tito") / "progress.json"
        milestones_file = Path(".tito") / "milestones.json"

        # Load module progress
        if progress_file.exists():
            progress_data = json.loads(progress_file.read_text())
            completed_modules = progress_data.get("completed_modules", [])
            completion_dates = progress_data.get("completion_dates", {})
        else:
            completed_modules = []
            completion_dates = {}

        # Load milestone achievements
        if milestones_file.exists():
            milestones_data = json.loads(milestones_file.read_text())
            completed_milestones = milestones_data.get("completed_milestones", [])
            milestone_dates = milestones_data.get("completion_dates", {})
        else:
            completed_milestones = []
            milestone_dates = {}

        # Calculate progress percentages
        total_modules = 20
        total_milestones = 6
        modules_percent = int((len(completed_modules) / total_modules) * 100)
        milestones_percent = int((len(completed_milestones) / total_milestones) * 100)

        # Create summary panel
        summary_text = Text()
        summary_text.append(f"📦 Modules Completed: ", style="bold")
        summary_text.append(f"{len(completed_modules)}/{total_modules} ({modules_percent}%)\n", style="cyan")
        summary_text.append(f"🏆 Milestones Achieved: ", style="bold")
        summary_text.append(f"{len(completed_milestones)}/{total_milestones} ({milestones_percent}%)\n\n", style="magenta")

        # Last activity
        all_dates = list(completion_dates.values()) + list(milestone_dates.values())
        if all_dates:
            latest_date = max(all_dates)
            summary_text.append("📍 Last Activity: ", style="bold")
            summary_text.append(f"{latest_date}\n", style="dim")

        console.print(Panel(
            summary_text,
            title="📊 TinyTorch Progress",
            border_style="bright_cyan"
        ))

        # Module Progress Table
        if completed_modules:
            console.print("\n[bold]Module Progress:[/bold]")
            for i in range(1, total_modules + 1):
                mod_num = i
                if mod_num in completed_modules:
                    module_name = self._get_module_name(mod_num)
                    console.print(f"  [green]✅ {mod_num:02d} {module_name}[/green]")
                elif i <= len(completed_modules) + 3:  # Show next few modules
                    module_name = self._get_module_name(mod_num)
                    console.print(f"  [dim]🔒 {mod_num:02d} {module_name}[/dim]")

        # Milestone Achievements
        if completed_milestones or (completed_modules and len(completed_modules) >= 1):
            console.print("\n[bold]Milestone Achievements:[/bold]")
            milestone_names = {
                "01": "Perceptron (1957)",
                "02": "Backpropagation (1986)",
                "03": "MLP Revival (1986)",
                "04": "CNN Revolution (1998)",
                "05": "Transformer Era (2017)",
                "06": "MLPerf (2018)"
            }
            for mid in ["01", "02", "03", "04", "05", "06"]:
                if mid in completed_milestones:
                    console.print(f"  [magenta]✅ {mid} - {milestone_names[mid]}[/magenta]")
                else:
                    # Check if ready
                    prereqs_met = self._check_milestone_prereqs(mid, completed_modules)
                    if prereqs_met:
                        console.print(f"  [yellow]🎯 {mid} - {milestone_names[mid]} [Ready!][/yellow]")
                    else:
                        console.print(f"  [dim]🔒 {mid} - {milestone_names[mid]}[/dim]")

        console.print()
        return 0

    def _get_module_name(self, module_num: int) -> str:
        """Get module name from number."""
        module_names = {
            1: "Tensor", 2: "Activations", 3: "Layers", 4: "Losses",
            5: "Autograd", 6: "Optimizers", 7: "Training", 8: "DataLoader",
            9: "Convolutions", 10: "Normalization", 11: "Tokenization",
            12: "Embeddings", 13: "Attention", 14: "Transformers",
            15: "Profiling", 16: "Quantization", 17: "Compression",
            18: "Memoization", 19: "Benchmarking", 20: "Capstone"
        }
        return module_names.get(module_num, "Unknown")

    def _check_milestone_prereqs(self, milestone_id: str, completed_modules: list) -> bool:
        """Check if milestone prerequisites are met."""
        prereqs = {
            "01": [1],
            "02": [1, 2, 3, 4, 5],
            "03": [1, 2, 3, 4, 5, 6, 7],
            "04": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "05": [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
            "06": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 19]
        }
        required = prereqs.get(milestone_id, [])
        return all(mod in completed_modules for mod in required)

    def _run_comprehensive_analysis(self) -> int:
        """Run comprehensive system health dashboard."""
        console = self.console
        start_time = time.time()
        
        console.print("🚀 Starting TinyTorch Comprehensive Status Check...", style="bold green")
        
        # Initialize analyzer
        analyzer = TinyTorchStatusAnalyzer()
        
        # Run full analysis
        result = analyzer.run_full_analysis()
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report(console)
        
        # Summary
        total_time = time.time() - start_time
        console.print(f"\n⏱️ Comprehensive analysis completed in {total_time:.1f}s", style="dim")
        
        # Return appropriate exit code
        if result['summary']['environment_healthy'] and result['summary']['working_modules'] >= result['summary']['total_modules'] * 0.8:
            return 0  # Success
        else:
            return 1  # Issues found
    
    def _run_standard_status(self, args: Namespace) -> int:
        """Run standard status check mode."""
        console = self.console
        
        # Scan modules directory
        modules_dir = Path("modules")
        if not modules_dir.exists():
            console.print(Panel("[red]❌ modules/ directory not found[/red]", 
                              title="Error", border_style="red"))
            return 1
        
        # Find all module directories (exclude special directories)
        exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
        module_dirs = [d for d in modules_dir.iterdir() 
                      if d.is_dir() and d.name not in exclude_dirs]
        
        if not module_dirs:
            console.print(Panel("[yellow]⚠️  No modules found in modules/ directory[/yellow]", 
                              title="Warning", border_style="yellow"))
            return 0
        
        console.print(Panel(f"📋 Found {len(module_dirs)} modules in modules directory", 
                          title="Module Status Check", border_style="bright_cyan"))
        
        # Create status table
        status_table = Table(title="Module Status Overview", show_header=True, header_style="bold blue")
        status_table.add_column("Module", style="bold cyan", width=17)
        status_table.add_column("Status", width=12, justify="center")
        status_table.add_column("Dev File", width=12, justify="center")
        status_table.add_column("Inline Tests", width=12, justify="center")
        status_table.add_column("External Tests", width=12, justify="center")
        status_table.add_column("README", width=12, justify="center")
        
        if args.metadata:
            status_table.add_column("Export Target", width=20, justify="center")
            status_table.add_column("Prerequisites", width=15, justify="center")
        
        # Check each module
        modules_status = []
        for module_dir in sorted(module_dirs):
            module_name = module_dir.name
            status = self._check_module_status(module_dir, args.test_status)
            modules_status.append((module_name, status))
            
            # Add to table
            row = [
                module_name,
                self._format_status(status['overall_status']),
                self._format_file_status(status['dev_file'], status.get('export_count', 0)),
                self._format_inline_tests(status['inline_test_count']),
                self._format_external_tests(status['external_tests'], status.get('external_test_status')),
                "✅" if status['readme'] else "❌"
            ]
            
            # Add metadata columns if requested
            if args.metadata:
                metadata = status.get('metadata', {})
                export_target = status.get('export_target', 'unknown')
                row.append(export_target if export_target not in ['not_found', 'no_export', 'read_error'] else export_target)
                
                # Show prerequisites from dependencies
                deps = metadata.get('dependencies', {})
                prereqs = deps.get('prerequisites', [])
                row.append(', '.join(prereqs) if prereqs else 'none')
            
            status_table.add_row(*row)
        
        console.print(status_table)
        
        # Summary with better logic
        total_modules = len(modules_status)
        
        # A module is "working" if it has a dev file with implementations
        working_modules = sum(1 for _, status in modules_status 
                            if status['dev_file'] and status.get('export_count', 0) > 0)
        
        # A module is "complete" if it has everything
        complete_modules = sum(1 for _, status in modules_status 
                             if status['dev_file'] and status['external_tests'] and status['readme'] and status.get('export_count', 0) > 0)
        
        console.print(f"\n📊 Summary:")
        console.print(f"   🏗️  Working modules: {working_modules}/{total_modules} (have implementations)")
        console.print(f"   ✅ Complete modules: {complete_modules}/{total_modules} (have implementations, tests, docs)")
        
        # Helpful commands
        console.print(f"\n💡 Quick commands:")
        console.print(f"   [bold cyan]tito status --comprehensive[/bold cyan]      # Full system health dashboard")
        console.print(f"   [bold cyan]tito module test --all[/bold cyan]           # Test all modules")
        console.print(f"   [bold cyan]tito module test MODULE_NAME[/bold cyan]     # Test specific module")
        console.print(f"   [bold cyan]pytest modules/*/  -k test_[/bold cyan]  # Run pytest on inline tests")
        console.print(f"   [bold cyan]pytest tests/test_*.py[/bold cyan]           # Run external tests")
        
        # Detailed view
        if args.details:
            console.print("\n" + "="*60)
            console.print("📁 Detailed Module Structure")
            console.print("="*60)
            
            for module_name, status in modules_status:
                self._print_module_details(module_name, status)
        
        # Metadata view
        if args.metadata:
            console.print("\n" + "="*60)
            console.print("📊 Module Metadata")
            console.print("="*60)
            
            for module_name, status in modules_status:
                if status.get('metadata'):
                    self._print_module_metadata(module_name, status['metadata'])
        
        return 0
    
    def _check_module_status(self, module_dir: Path, check_tests: bool = False) -> dict:
        """Check the status of a single module."""
        module_name = module_dir.name
        
        # Check for required files
        # Extract short name from module directory name for dev file
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        dev_file = module_dir / f"{short_name}.py"
        readme_file = module_dir / "README.md"
        metadata_file = module_dir / "module.yaml"
        
        # Check for tests in main tests directory
        # Extract short name from module directory name (e.g., "01_tensor" -> "tensor")
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        
        main_test_file = Path("tests") / f"test_{short_name}.py"
        
        status = {
            'dev_file': dev_file.exists(),
            'readme': readme_file.exists(),
            'metadata_file': metadata_file.exists(),
            'external_tests': main_test_file.exists(),
            'inline_test_count': 0,
            'export_count': 0,
            'export_target': 'not_found',
            'external_test_status': None,
            'overall_status': 'unknown',
            'metadata': None
        }
        
        # Count inline tests and exports if dev file exists
        if dev_file.exists():
            status['inline_test_count'] = self._count_test_functions(dev_file)
            status['export_count'] = self._count_export_functions(dev_file)
            status['export_target'] = self._get_export_target(module_dir)
        
        # Run external tests if requested (slower)
        if check_tests and main_test_file.exists():
            status['external_test_status'] = self._check_external_tests(main_test_file)
        
        # Determine overall status
        status['overall_status'] = self._determine_overall_status(status)
        
        # Load metadata if available
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                    status['metadata'] = metadata
            except Exception as e:
                status['metadata'] = {'error': str(e)}
        
        return status
    
    def _determine_overall_status(self, status: dict) -> str:
        """Determine overall module status based on files and implementation."""
        # If no dev file, module is not started
        if not status['dev_file']:
            return 'not_started'
        
        # If dev file exists but no implementations, module is empty
        if status.get('export_count', 0) == 0:
            return 'empty'
        
        # If has implementations but no tests, module is in progress
        if status.get('inline_test_count', 0) == 0 and not status.get('external_tests', False):
            return 'no_tests'
        
        # If has implementations and tests, module is working
        if status.get('export_count', 0) > 0 and (status.get('inline_test_count', 0) > 0 or status.get('external_tests', False)):
            return 'working'
        
        return 'unknown'
    
    def _check_external_tests(self, test_file: Path) -> str:
        """Check if external tests pass (used only when --test-status is specified)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return 'passing'
            else:
                return 'failing'
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 'error'
    
    def _format_status(self, status: str) -> str:
        """Format overall module status with appropriate emoji and color."""
        status_map = {
            'working': '✅',        # Has implementations and tests
            'no_tests': '🚧',       # Has implementations but no tests
            'empty': '📝',          # Has dev file but no implementations
            'not_started': '❌',    # No dev file
            'unknown': '❓'
        }
        return status_map.get(status, '❓')
    
    def _format_file_status(self, exists: bool, export_count: int) -> str:
        """Format dev file status showing if it has implementations."""
        if not exists:
            return "❌"
        if export_count == 0:
            return "📝"  # File exists but empty
        return f"✅({export_count})"  # File exists with implementations
    
    def _format_inline_tests(self, test_count: int) -> str:
        """Format inline test count."""
        if test_count == 0:
            return "❌"
        return f"✅({test_count})"
    
    def _format_external_tests(self, exists: bool, test_status: Optional[str] = None) -> str:
        """Format external test status."""
        if not exists:
            return "❌"
        if test_status == 'passing':
            return "✅"
        elif test_status == 'failing':
            return "🔴"
        elif test_status == 'error':
            return "⚠️"
        else:
            return "✅"  # Exists but not tested
    
    def _print_module_details(self, module_name: str, status: dict) -> None:
        """Print detailed information about a module."""
        console = self.console
        
        # Module header
        console.print(f"\n📦 {module_name.upper()}", style="bold cyan")
        console.print("-" * 40)
        
        # File structure
        files_table = Table(show_header=False, box=None, padding=(0, 2))
        files_table.add_column("File", style="dim")
        files_table.add_column("Status")
        
        dev_status = "✅ Found" if status['dev_file'] else "❌ Missing"
        if status['dev_file']:
            dev_status += f" ({status.get('export_count', 0)} exports, {status.get('inline_test_count', 0)} inline tests)"
        
        files_table.add_row(f"{module_name}.py", dev_status)
        files_table.add_row("tests/test_*.py", "✅ Found" if status['external_tests'] else "❌ Missing")
        files_table.add_row("README.md", "✅ Found" if status['readme'] else "❌ Missing")
        
        console.print(files_table)
        
        # Pytest commands
        if status['dev_file'] or status['external_tests']:
            console.print("\n[dim]💡 Test commands:[/dim]")
            if status['dev_file']:
                console.print(f"[dim]   pytest modules/{module_name}/{module_name}.py -k test_[/dim]")
            if status['external_tests']:
                short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
                console.print(f"[dim]   pytest tests/test_{short_name}.py -v[/dim]")
    
    def _print_module_metadata(self, module_name: str, metadata: dict) -> None:
        """Print detailed metadata information about a module."""
        console = self.console
        
        # Module header
        title = metadata.get('title', module_name.title())
        console.print(f"\n📦 {title}", style="bold cyan")
        console.print("-" * (len(title) + 4))
        
        # Basic info
        if metadata.get('description'):
            console.print(f"📝 {metadata['description']}")
        
        # Export info (read from dev file - source of truth)
        module_path = Path(f"modules/{module_name}")
        export_target = self._get_export_target(module_path)
        if export_target not in ['not_found', 'no_export', 'read_error']:
            console.print(f"📦 Exports to: {export_target}")
        
        # Dependencies
        if metadata.get('dependencies'):
            deps = metadata['dependencies']
            console.print("\n🔗 Dependencies:")
            if deps.get('prerequisites'):
                console.print(f"  Prerequisites: {', '.join(deps['prerequisites'])}")
            if deps.get('enables'):
                console.print(f"  Enables: {', '.join(deps['enables'])}")
        
        # Components
        if metadata.get('components'):
            console.print("\n🧩 Components:")
            for component in metadata['components']:
                console.print(f"  • {component}")
        
        # Files
        if metadata.get('files'):
            files = metadata['files']
            console.print("\n📁 Files:")
            if files.get('dev_file'):
                console.print(f"  • Dev: {files['dev_file']}")
            if files.get('test_file'):
                console.print(f"  • Test: {files['test_file']}")
            if files.get('readme'):
                console.print(f"  • README: {files['readme']}") 