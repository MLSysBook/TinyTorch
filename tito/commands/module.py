"""
Module command group for TinyTorch CLI: development workflow and module management.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.align import Align
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
import sys
import importlib.util
import json
import time
import subprocess
from datetime import datetime
from typing import Optional

from .base import BaseCommand
from .status import StatusCommand
from .test import TestCommand
from .notebooks import NotebooksCommand
from .clean import CleanCommand
from .export import ExportCommand
from .view import ViewCommand
from .checkpoint import CheckpointSystem
from .milestone import MilestoneSystem
from .leaderboard import LeaderboardCommand
from ..core.console import print_ascii_logo
from pathlib import Path

# Example mapping - shows what TinyTorch can do after each module
EXAMPLES = {
    "01_setup": None,  # Just environment setup
    "02_tensor": None,  # Foundation only
    "03_activations": None,  # Building blocks only
    "04_layers": None,  # Components only
    "05_dense": "xornet",  # 🔥 XORnet - Neural network fundamentals
    "06_spatial": None,  # CNN components (no working example yet)
    "07_attention": None,  # Attention building blocks
    "08_dataloader": None,  # Data loading components
    "09_autograd": None,  # XORnet already shows autograd
    "10_optimizers": None,  # Optimization components
    "11_training": "cifar10",  # 🎯 CIFAR-10 - Real computer vision
    "12_compression": None,  # Advanced optimization
    "13_kernels": None,  # Performance optimization
    "14_benchmarking": None,  # Performance analysis
    "15_mlops": None,  # Production deployment concepts
    "16_tinygpt": None  # Complete GPT implementation (Module 16)
}

class ModuleCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "module"

    @property
    def description(self) -> str:
        return "Module development and management commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='module_command',
            help='Module subcommands',
            metavar='SUBCOMMAND'
        )
        
        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='Check status of all modules'
        )
        status_cmd = StatusCommand(self.config)
        status_cmd.add_arguments(status_parser)
        
        # Test subcommand
        test_parser = subparsers.add_parser(
            'test',
            help='Run module tests'
        )
        test_cmd = TestCommand(self.config)
        test_cmd.add_arguments(test_parser)
        
        # Notebooks subcommand
        notebooks_parser = subparsers.add_parser(
            'notebooks',
            help='Build notebooks from Python files'
        )
        notebooks_cmd = NotebooksCommand(self.config)
        notebooks_cmd.add_arguments(notebooks_parser)
        
        # Clean subcommand
        clean_parser = subparsers.add_parser(
            'clean',
            help='Clean up module directories (notebooks, cache, etc.)'
        )
        clean_cmd = CleanCommand(self.config)
        clean_cmd.add_arguments(clean_parser)
        
        # Export subcommand
        export_parser = subparsers.add_parser(
            'export',
            help='Export module code to Python package'
        )
        export_cmd = ExportCommand(self.config)
        export_cmd.add_arguments(export_parser)
        
        # View subcommand
        view_parser = subparsers.add_parser(
            'view',
            help='Generate notebooks and open Jupyter Lab'
        )
        view_cmd = ViewCommand(self.config)
        view_cmd.add_arguments(view_parser)
        
        # Complete subcommand (new integration testing workflow)
        complete_parser = subparsers.add_parser(
            'complete',
            help='Complete module with automatic export and checkpoint testing'
        )
        complete_parser.add_argument(
            'module_name',
            help='Module name to complete (e.g., 02_tensor, tensor)'
        )
        complete_parser.add_argument(
            '--skip-test',
            action='store_true',
            help='Skip checkpoint test after export'
        )

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if not hasattr(args, 'module_command') or not args.module_command:
            console.print(Panel(
                "[bold cyan]Module Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]status[/bold]     - Check status of all modules\n"
                "  • [bold]test[/bold]       - Run module tests\n"
                "  • [bold]notebooks[/bold]  - Build notebooks from Python files\n"
                "  • [bold]clean[/bold]      - Clean up module directories\n"
                "  • [bold]export[/bold]     - Export module code to Python package\n"
                "  • [bold]view[/bold]       - Generate notebooks and open Jupyter Lab\n"
                "  • [bold]complete[/bold]   - Complete module with export, testing, and capability showcase\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito module status --metadata[/dim]\n"
                "[dim]  tito module test --all[/dim]\n"
                "[dim]  tito module test tensor[/dim]\n"
                "[dim]  tito module export --all[/dim]\n"
                "[dim]  tito module export 01_tensor[/dim]\n"
                "[dim]  tito module export 01_tensor 02_activations[/dim]\n"
                "[dim]  tito module clean --all[/dim]\n"
                "[dim]  tito module clean tensor[/dim]\n"
                "[dim]  tito module view 02_tensor[/dim]\n"
                "[dim]  tito module view --force[/dim]\n"
                "[dim]  tito module complete 02_tensor[/dim]\n"
                "[dim]  tito module complete tensor --skip-test[/dim]",
                title="Module Command Group",
                border_style="bright_cyan"
            ))
            return 0
        
        # Execute the appropriate subcommand
        if args.module_command == 'status':
            cmd = StatusCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'test':
            cmd = TestCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'notebooks':
            cmd = NotebooksCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'clean':
            cmd = CleanCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'export':
            cmd = ExportCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'view':
            cmd = ViewCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'complete':
            return self._handle_complete_command(args)
        else:
            console.print(Panel(
                f"[red]Unknown module subcommand: {args.module_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1 

    def _handle_complete_command(self, args: Namespace) -> int:
        """Handle the module complete command with full workflow."""
        console = self.console
        module_name = args.module_name
        
        # Normalize module name (handle both 02_tensor and tensor formats)
        normalized_name = self._normalize_module_name(module_name)
        if not normalized_name:
            console.print(Panel(
                f"[red]❌ Module '{module_name}' not found[/red]\n\n"
                f"[yellow]Available modules:[/yellow]\n"
                f"{self._get_available_modules_text()}",
                title="Module Not Found",
                border_style="red"
            ))
            return 1
        
        # Show workflow start
        console.print(Panel(
            f"[bold cyan]🚀 Module Completion Workflow[/bold cyan]\n\n"
            f"[bold]Module:[/bold] {normalized_name}\n"
            f"[bold]Steps:[/bold]\n"
            f"  1. Export module to package\n"
            f"  2. Run Package Manager integration test\n"
            f"  3. Run capability checkpoint test\n"
            f"  4. Show progress and next steps\n\n"
            f"[dim]Two-tier validation: Integration → Capability[/dim]",
            title="Module Complete",
            border_style="bright_cyan"
        ))
        
        # Step 1: Export the module
        console.print(f"\n[bold]Step 1: Exporting {normalized_name}...[/bold]")
        export_result = self._run_export(normalized_name)
        if export_result != 0:
            console.print(Panel(
                f"[red]❌ Export failed for {normalized_name}[/red]\n\n"
                f"[yellow]Cannot proceed with checkpoint testing until export succeeds.[/yellow]\n"
                f"[dim]Check the module implementation and try again.[/dim]",
                title="Export Failed",
                border_style="red"
            ))
            return export_result
        
        # Step 2: Run Package Manager integration test
        console.print(f"\n[bold]Step 2: Running Package Manager integration test...[/bold]")
        integration_result = self._run_integration_test(normalized_name)
        
        if not integration_result["success"]:
            console.print(Panel(
                f"[red]❌ Integration test failed for {normalized_name}[/red]\n\n"
                f"[yellow]Module exported but integration issues detected:[/yellow]\n"
                f"{integration_result.get('error', 'Unknown integration error')}\n\n"
                f"[cyan]This means the module may not work properly with the package.[/cyan]",
                title="Integration Test Failed",
                border_style="yellow"
            ))
            return 1
        
        # Show integration success
        console.print(f"[green]✅ Module {normalized_name} integrated into package successfully![/green]")
        
        # Step 3: Run checkpoint test (unless skipped)
        if not args.skip_test:
            console.print(f"\n[bold]Step 3: Testing capabilities...[/bold]")
            checkpoint_result = self._run_checkpoint_for_module(normalized_name)
            
            # Step 4: Check for milestone unlock (NEW!)
            console.print(f"\n[bold]Step 4: Checking for milestone unlocks...[/bold]")
            milestone_result = self._check_milestone_unlock(normalized_name)
            
            self._show_completion_results(checkpoint_result, normalized_name, integration_result, milestone_result)
        else:
            console.print(f"\n[bold yellow]Step 3: Checkpoint test skipped[/bold yellow]")
            console.print(f"[dim]Module integrated successfully. Run checkpoint test manually if needed.[/dim]")
        
        return 0

    def _normalize_module_name(self, module_name: str) -> str:
        """Normalize module name to full format (e.g., tensor -> 02_tensor)."""
        # If already in full format, validate it exists
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            module_path = Path(f"modules/{module_name}")
            if module_path.exists():
                return module_name
            return ""
        
        # Try to find the module by short name
        source_dir = Path("modules")
        if source_dir.exists():
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name.endswith(f"_{module_name}"):
                    return module_dir.name
        
        return ""

    def _get_available_modules_text(self) -> str:
        """Get formatted text listing available modules."""
        source_dir = Path("modules")
        modules = []
        
        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache', 'utils'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    modules.append(module_dir.name)
        
        if modules:
            return "\n".join(f"  • {module}" for module in sorted(modules))
        return "  No modules found"

    def _run_export(self, module_name: str) -> int:
        """Run export command for the module."""
        # Create a mock args object for export
        export_args = Namespace()
        export_args.module = module_name
        export_args.all = False
        export_args.from_release = False
        export_args.test_checkpoint = False  # We'll handle testing separately
        
        export_cmd = ExportCommand(self.config)
        return export_cmd.execute(export_args)

    def _run_checkpoint_for_module(self, module_name: str) -> dict:
        """Run checkpoint test for a module."""
        # Use the same mapping as ExportCommand
        from .export import ExportCommand
        module_to_checkpoint = ExportCommand.MODULE_TO_CHECKPOINT
        
        if module_name not in module_to_checkpoint:
            return {"skipped": True, "reason": f"No checkpoint mapping for module {module_name}"}
        
        checkpoint_id = module_to_checkpoint[module_name]
        checkpoint_system = CheckpointSystem(self.config)
        
        console = self.console
        checkpoint = checkpoint_system.CHECKPOINTS[checkpoint_id]
        console.print(f"[bold]Checkpoint {checkpoint_id}: {checkpoint['name']}[/bold]")
        console.print(f"[dim]Testing: {checkpoint['capability']}[/dim]")
        
        with console.status(f"[bold green]Running checkpoint {checkpoint_id} test...", spinner="dots"):
            result = checkpoint_system.run_checkpoint_test(checkpoint_id)
        
        return result

    def _run_integration_test(self, module_name: str) -> dict:
        """Run Package Manager integration test for a module."""
        try:
            # Import the Package Manager integration system
            integration_module_path = Path("tests/integration/package_manager_integration.py")
            
            if not integration_module_path.exists():
                return {
                    "success": False,
                    "error": "Package Manager integration system not found"
                }
            
            # Load the integration module
            spec = importlib.util.spec_from_file_location(
                "package_manager_integration", 
                integration_module_path
            )
            integration_module = importlib.util.module_from_spec(spec)
            sys.modules["package_manager_integration"] = integration_module
            spec.loader.exec_module(integration_module)
            
            # Run the integration test
            manager = integration_module.PackageManagerIntegration()
            result = manager.run_module_integration_test(module_name)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run integration test: {e}",
                "module_name": module_name
            }
    
    def _get_checkpoint_id_for_module(self, module_name: str) -> Optional[str]:
        """Get the checkpoint ID that corresponds to a module."""
        from .export import ExportCommand
        return ExportCommand.MODULE_TO_CHECKPOINT.get(module_name)

    def _check_milestone_unlock(self, module_name: str) -> dict:
        """Check if completing this module unlocks a milestone."""
        milestone_system = MilestoneSystem(self.config)
        return milestone_system.check_milestone_unlock(module_name)

    def _show_completion_results(self, result: dict, module_name: str, integration_result: dict = None, milestone_result: dict = None) -> None:
        """Show results of module completion workflow."""
        console = self.console
        
        if result.get("skipped"):
            console.print(f"\n[dim]No checkpoint test available for {module_name}[/dim]")
            console.print(f"[green]✅ Module {module_name} exported successfully![/green]")
            # Still record completion even if skipped
            self._record_module_completion(module_name)
            
            # Update leaderboard profile even for skipped checkpoints
            checkpoint_unlocked = self._get_checkpoint_id_for_module(module_name)
            leaderboard_cmd = LeaderboardCommand(self.config)
            leaderboard_cmd.update_profile_on_module_completion(module_name, checkpoint_unlocked)
            return
        
        if result["success"]:
            # Record successful completion first
            self._record_module_completion(module_name)
            
            # Update leaderboard profile with module completion
            checkpoint_unlocked = self._get_checkpoint_id_for_module(module_name)
            leaderboard_cmd = LeaderboardCommand(self.config)
            leaderboard_cmd.update_profile_on_module_completion(module_name, checkpoint_unlocked)
            
            # Check for EPIC MILESTONE UNLOCK first!
            if milestone_result and milestone_result.get("milestone_unlocked"):
                self._show_epic_milestone_celebration(milestone_result)
            else:
                # Show regular capability celebration
                self._show_capability_unlock_celebration(module_name, result)
            
            # Check for capability showcase
            self._check_and_run_capability_showcase(module_name)
            
            # Celebration and progress feedback
            checkpoint_name = result.get("checkpoint_name", "Unknown")
            capability = result.get("capability", "")
            
            # Show completion status (enhanced for milestones)
            if milestone_result and milestone_result.get("milestone_unlocked"):
                milestone_data = milestone_result["milestone_data"]
                console.print(Panel(
                    f"[bold green]🎉 EPIC MILESTONE UNLOCKED![/bold green]\n\n"
                    f"[green]✅ Package Integration: Module exported and integrated[/green]\n"
                    f"[green]✅ Capability Test: {checkpoint_name} checkpoint achieved![/green]\n"
                    f"[yellow]🎯 MILESTONE {milestone_result['milestone_id']}: {milestone_data['title']}![/yellow]\n\n"
                    f"[bold magenta]🚀 New Capability: {milestone_data['capability']}[/bold magenta]\n"
                    f"[cyan]Real-world impact: {milestone_data['real_world_impact']}[/cyan]\n\n"
                    f"[bold cyan]🔓 You've unlocked a major ML engineering capability![/bold cyan]",
                    title=f"🌟 MILESTONE {milestone_result['milestone_id']} ACHIEVED",
                    border_style="bright_magenta"
                ))
            else:
                # Standard completion message
                console.print(Panel(
                    f"[bold green]🎉 Module Complete![/bold green]\n\n"
                    f"[green]✅ Package Integration: Module exported and integrated[/green]\n"
                    f"[green]✅ Capability Test: {checkpoint_name} checkpoint achieved![/green]\n"
                    f"[green]🚀 Capability unlocked: {capability}[/green]\n\n"
                    f"[bold cyan]🔄 Two-Tier Validation Success[/bold cyan]\n"
                    f"Module {module_name} passed both integration and capability tests!\n"
                    f"Your module is fully functional in the TinyTorch ecosystem.",
                    title=f"🏆 {module_name} Achievement",
                    border_style="green"
                ))
            
            # Show progress and next steps (enhanced for milestones)
            self._enhanced_show_progress_and_next_steps(module_name, milestone_result)
        else:
            console.print(Panel(
                f"[bold yellow]⚠️  Integration Complete, Capability Test Failed[/bold yellow]\n\n"
                f"[green]✅ Package Integration: Module exported and integrated[/green]\n"
                f"[yellow]❌ Capability Test: {result.get('checkpoint_name', 'Checkpoint')} test failed[/yellow]\n\n"
                f"[bold]This usually indicates:[/bold]\n"
                f"• Basic integration works, but some advanced functionality is missing\n"
                f"• Implementation needs refinement for full capability\n"
                f"• Module requirements partially met\n\n"
                f"[cyan]💡 Next steps:[/cyan]\n"
                f"• Review module implementation for missing features\n"
                f"• Test individual components\n"
                f"• Try: tito module complete {module_name}",
                title="Capability Test Failed",
                border_style="yellow"
            ))

    def _show_progress_and_next_steps(self, completed_module: str) -> None:
        """Show overall progress and suggest next steps."""
        console = self.console
        
        # Show checkpoint status
        console.print(f"\n[bold cyan]📊 Progress Update[/bold cyan]")
        
        # Get current checkpoint status
        checkpoint_system = CheckpointSystem(self.config)
        progress_data = checkpoint_system.get_overall_progress()
        overall_percent = progress_data["overall_progress"]
        total_complete = progress_data["total_complete"]
        total_checkpoints = progress_data["total_checkpoints"]
        
        console.print(f"[bold]Overall Progress:[/bold] {overall_percent:.0f}% ({total_complete}/{total_checkpoints} checkpoints)")
        
        # Suggest next module
        if completed_module.startswith(tuple(f"{i:02d}_" for i in range(100))):
            try:
                module_num = int(completed_module[:2])
                next_num = module_num + 1
                
                # Next module suggestions
                next_modules = {
                    1: ("02_tensor", "Tensor operations - the foundation of ML"),
                    2: ("03_activations", "Activation functions - adding intelligence"),
                    3: ("04_layers", "Neural layers - building blocks"),
                    4: ("05_dense", "Dense networks - complete architectures"),
                    5: ("06_spatial", "Spatial processing - convolutional operations"),
                    6: ("07_attention", "Attention mechanisms - sequence understanding"),
                    7: ("08_dataloader", "Data loading - efficient training"),
                    8: ("09_autograd", "Automatic differentiation - gradient computation"),
                    9: ("10_optimizers", "Optimization algorithms - sophisticated learning"),
                    10: ("11_training", "Training loops - end-to-end learning"),
                    11: ("12_compression", "Model compression - efficient deployment"),
                    12: ("13_kernels", "High-performance kernels - optimized computation"),
                    13: ("14_benchmarking", "Performance analysis - bottleneck identification"),
                    14: ("15_mlops", "MLOps - production deployment"),
                    15: ("16_tinygpt", "TinyGPT - Language models and transformers"),
                }
                
                if next_num in next_modules:
                    next_module, next_desc = next_modules[next_num]
                    console.print(f"\n[bold cyan]🎯 Continue Your Journey[/bold cyan]")
                    console.print(f"[bold]Next Module:[/bold] {next_module}")
                    console.print(f"[dim]{next_desc}[/dim]")
                    console.print(f"\n[green]Ready to continue?[/green]")
                    console.print(f"[dim]  tito module view {next_module}[/dim]")
                    console.print(f"[dim]  tito module complete {next_module}[/dim]")
                elif next_num > 16:
                    console.print(f"\n[bold green]🏆 Congratulations![/bold green]")
                    console.print(f"[green]You've completed all TinyTorch modules![/green]")
                    console.print(f"[dim]Run 'tito checkpoint status' to see your complete progress[/dim]")
            except (ValueError, IndexError):
                pass
        
        # General next steps
        console.print(f"\n[bold]Track Your Progress:[/bold]")
        console.print(f"[dim]  tito checkpoint status       - View detailed progress[/dim]")
        console.print(f"[dim]  tito checkpoint timeline     - Visual progress timeline[/dim]")
    
    def _show_gamified_intro(self, module_name: str) -> None:
        """Show animated gamified introduction for module completion."""
        console = self.console
        
        # Module introduction with capability context
        capability_info = self._get_module_capability_info(module_name)
        
        console.print(Panel(
            f"[bold cyan]🚀 Starting Module Completion Quest[/bold cyan]\n\n"
            f"[bold]Module:[/bold] {module_name}\n"
            f"[bold]Capability to Unlock:[/bold] {capability_info['title']}\n"
            f"[dim]{capability_info['description']}[/dim]\n\n"
            f"[bold yellow]Quest Steps:[/bold yellow]\n"
            f"  1. 📦 Export module to TinyTorch package\n"
            f"  2. 🔧 Run integration validation\n"
            f"  3. ⚡ Test capability unlock\n"
            f"  4. 🎉 Celebrate achievement!\n\n"
            f"[bold green]Ready to unlock your next ML superpower?[/bold green]",
            title=f"🎮 Module Quest: {module_name}",
            border_style="bright_magenta"
        ))
        
        # Brief pause for dramatic effect
        time.sleep(1)
    
    def _run_export_with_animation(self, module_name: str) -> int:
        """Run export with Rich progress animation."""
        console = self.console
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Exporting to TinyTorch package..."),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("export", total=100)
            
            # Simulate export stages with progress updates
            for i, stage in enumerate([
                "Reading module source...",
                "Processing NBDev directives...", 
                "Generating package code...",
                "Validating exports...",
                "Updating package structure..."
            ]):
                progress.update(task, description=f"[bold blue]{stage}", completed=i*20)
                time.sleep(0.3)  # Brief pause for visual effect
            
            # Run actual export
            result = self._run_export(module_name)
            progress.update(task, completed=100)
            
            if result == 0:
                progress.update(task, description="[bold green]✅ Export completed successfully!")
            else:
                progress.update(task, description="[bold red]❌ Export failed")
            
            time.sleep(0.5)  # Show final state
            
        return result
    
    def _run_integration_with_animation(self, module_name: str) -> dict:
        """Run integration test with Rich progress animation."""
        console = self.console
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Running integration tests..."),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("integration", total=100)
            
            # Simulate integration test stages
            for i, stage in enumerate([
                "Loading package manager...",
                "Validating module imports...",
                "Testing integration points...",
                "Checking dependencies...",
                "Finalizing validation..."
            ]):
                progress.update(task, description=f"[bold yellow]{stage}", completed=i*20)
                time.sleep(0.2)
            
            # Run actual integration test
            result = self._run_integration_test(module_name)
            progress.update(task, completed=100)
            
            if result["success"]:
                progress.update(task, description="[bold green]✅ Integration test passed!")
            else:
                progress.update(task, description="[bold red]❌ Integration test failed")
            
            time.sleep(0.5)
        
        return result
    
    def _run_capability_test_with_animation(self, module_name: str) -> dict:
        """Run capability test with Rich progress animation."""
        console = self.console
        
        # Get capability info for this module
        capability_info = self._get_module_capability_info(module_name)
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold magenta]Testing capability: {capability_info['title']}..."),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("capability", total=100)
            
            # Simulate capability test stages
            for i, stage in enumerate([
                "Preparing capability test...",
                "Loading checkpoint system...",
                "Executing capability validation...",
                "Analyzing results...",
                "Finalizing capability check..."
            ]):
                progress.update(task, description=f"[bold magenta]{stage}", completed=i*20)
                time.sleep(0.3)
            
            # Run actual checkpoint test
            result = self._run_checkpoint_for_module(module_name)
            progress.update(task, completed=100)
            
            if result["success"]:
                progress.update(task, description="[bold green]✅ Capability unlocked!")
            else:
                progress.update(task, description="[bold red]❌ Capability test failed")
            
            time.sleep(0.5)
        
        return result
    
    def _show_capability_unlock_celebration(self, module_name: str, checkpoint_result: dict) -> None:
        """Show exciting capability unlock celebration with ASCII art."""
        console = self.console
        capability_info = self._get_module_capability_info(module_name)
        
        # Special celebration for TinyGPT (North Star achievement)
        if module_name == "16_tinygpt":
            self._show_north_star_celebration()
            return
        
        # Get celebration level based on module
        celebration_level = self._get_celebration_level(module_name)
        
        # Animated capability unlock
        time.sleep(0.5)
        
        if celebration_level == "major":  # Training, Regularization, etc.
            ascii_art = self._get_major_celebration_art()
            border_style = "bright_magenta"
            title_color = "bold magenta"
        elif celebration_level == "milestone":  # Networks, Attention, etc.
            ascii_art = self._get_milestone_celebration_art()
            border_style = "bright_yellow"
            title_color = "bold yellow"
        else:  # Standard celebration
            ascii_art = self._get_standard_celebration_art()
            border_style = "bright_green"
            title_color = "bold green"
        
        # Show animated unlock sequence
        console.print("\n" * 2)
        console.print(Align.center(Text("⚡ CAPABILITY UNLOCKED! ⚡", style="bold blink magenta")))
        console.print("\n")
        
        # Main celebration panel
        console.print(Panel(
            f"{ascii_art}\n\n"
            f"[{title_color}]🎉 {capability_info['title']} UNLOCKED! 🎉[/{title_color}]\n\n"
            f"[bold white]{capability_info['description']}[/bold white]\n\n"
            f"[green]✅ Capability Test:[/green] {checkpoint_result.get('checkpoint_name', 'Completed')}\n"
            f"[cyan]🚀 Achievement:[/cyan] {checkpoint_result.get('capability', 'ML Systems Engineering')}\n\n"
            f"[bold yellow]You are becoming an ML Systems Engineer![/bold yellow]",
            title=f"🏆 {module_name} MASTERED",
            border_style=border_style,
            box=box.ROUNDED
        ))
        
        # Brief pause for celebration
        time.sleep(1.5)
    
    def _show_epic_milestone_celebration(self, milestone_result: dict) -> None:
        """Show epic celebration for milestone unlock achievement."""
        console = self.console
        milestone_data = milestone_result["milestone_data"]
        milestone_id = milestone_result["milestone_id"]
        
        # Clear screen effect
        console.print("\n" * 2)
        
        # Epic milestone unlock sequence
        console.print(Align.center(Text("⚡ EPIC MILESTONE UNLOCKED! ⚡", style="bold blink bright_magenta")))
        console.print("\n")
        
        # Show the TinyTorch logo for special milestones
        if milestone_id in ["3", "5"]:  # Training and Language Generation
            print_ascii_logo()
        
        # Create milestone-specific celebration art
        celebration_art = self._get_milestone_celebration_art(milestone_id)
        
        # Epic celebration panel
        console.print(Panel(
            f"{celebration_art}\n\n"
            f"[bold bright_yellow]🎉 MILESTONE {milestone_id} ACHIEVED! 🎉[/bold bright_yellow]\n\n"
            f"[bold white]{milestone_data['emoji']} {milestone_data['title']} {milestone_data['emoji']}[/bold white]\n\n"
            f"[bold cyan]✨ NEW CAPABILITY UNLOCKED: ✨[/bold cyan]\n"
            f"[bold magenta]{milestone_data['capability']}[/bold magenta]\n\n"
            f"[green]🌟 Real-World Impact:[/green]\n"
            f"[yellow]{milestone_data['real_world_impact']}[/yellow]\n\n"
            f"[bold cyan]🚀 You're becoming an ML Systems Engineer! 🚀[/bold cyan]",
            title=f"🏆 MILESTONE {milestone_id}: {milestone_data['name'].upper()}",
            border_style="bright_magenta",
            box=box.ROUNDED
        ))
        
        # Extended celebration pause
        time.sleep(3)
        
        # Milestone-specific encouragement
        encouragement = self._get_milestone_encouragement(milestone_id)
        console.print(Panel(
            encouragement,
            title="🌟 Your Journey Continues",
            border_style="bright_cyan"
        ))
        
        time.sleep(2)
    
    def _get_milestone_celebration_art(self, milestone_id: str) -> str:
        """Get ASCII art for milestone celebrations."""
        if milestone_id == "1":
            return """
                    🧠 NEURAL NETWORKS UNLOCKED! 🧠
                    ╔══════════════════════════════╗
                    ║     ⚡ BASIC INFERENCE ⚡     ║
                    ║    Your first AI breakthrough   ║
                    ╚══════════════════════════════╝
            """
        elif milestone_id == "2":
            return """
                    👁️ COMPUTER VISION UNLOCKED! 👁️
                    ╔══════════════════════════════╗
                    ║      🖼️ MACHINES CAN SEE 🖼️     ║
                    ║   Image recognition mastery     ║
                    ╚══════════════════════════════╝
            """
        elif milestone_id == "3":
            return """
                🎓 FULL TRAINING PIPELINE UNLOCKED! 🎓
                ╔════════════════════════════════════╗
                ║     🏭 PRODUCTION TRAINING 🏭      ║
                ║   End-to-end ML system mastery    ║
                ║      🚀 Industry-ready skills      ║
                ╚════════════════════════════════════╝
            """
        elif milestone_id == "4":
            return """
                ⚡ ADVANCED VISION SYSTEMS UNLOCKED! ⚡
                ╔═══════════════════════════════════════╗
                ║      💎 PRODUCTION VISION 💎          ║
                ║    High-performance optimization      ║
                ║       🏢 Tech company level           ║
                ╚═══════════════════════════════════════╝
            """
        elif milestone_id == "5":
            return """
            🤖 LANGUAGE GENERATION UNLOCKED! 🤖
            ╔═══════════════════════════════════════════╗
            ║           👑 BUILD THE FUTURE 👑          ║
            ║      Transformer architecture mastery     ║
            ║         🌟 ChatGPT-level systems         ║
            ║           🚀 AI PIONEER STATUS 🚀         ║
            ╚═══════════════════════════════════════════╝
            """
        else:
            return """
                    ✨ MILESTONE ACHIEVED ✨
                       🏆 CAPABILITY 🏆
                         🌟 UNLOCKED 🌟
            """
    
    def _get_milestone_encouragement(self, milestone_id: str) -> str:
        """Get milestone-specific encouragement message."""
        encouragements = {
            "1": "[bold green]Amazing start![/bold green] You've built your first working neural networks.\n[cyan]You can now solve classification problems and understand gradient flow.\n[yellow]Next up: Teaching machines to see with computer vision![/yellow]",
            
            "2": "[bold green]Incredible breakthrough![/bold green] Machines can now see through your code.\n[cyan]You've mastered convolutional networks and image processing.\n[yellow]Next up: Building complete training pipelines![/yellow]",
            
            "3": "[bold green]Outstanding achievement![/bold green] You can now train production ML models.\n[cyan]You've built end-to-end systems that rivals industry standards.\n[yellow]Next up: Optimizing for real-world performance![/yellow]",
            
            "4": "[bold green]Exceptional mastery![/bold green] You build production-ready vision systems.\n[cyan]Your skills now match those of tech company engineers.\n[yellow]Next up: The ultimate challenge - language generation![/yellow]",
            
            "5": "[bold green]🎉 LEGENDARY STATUS ACHIEVED! 🎉[/bold green]\n[cyan]You can build the future of AI - transformer language models!\n[magenta]You are now a true ML Systems Engineer![/magenta]\n[yellow]Share your achievement and inspire others![/yellow]"
        }
        
        return encouragements.get(milestone_id, "[green]Great job on achieving this milestone![/green]")
    
    def _check_and_run_capability_showcase(self, module_name: str) -> None:
        """Check if example exists and prompt user to run it."""
        example_dir = EXAMPLES.get(module_name)
        if not example_dir:
            return
        
        example_path = Path("examples") / example_dir
        if not example_path.exists():
            return
        
        # Look for the main example file
        main_files = ["train.py", "demo.py", "main.py", "run.py"]
        example_file = None
        for filename in main_files:
            if (example_path / filename).exists():
                example_file = example_path / filename
                break
        
        if not example_file:
            return
        
        # Prompt user to run example
        if self._prompt_for_showcase(module_name):
            self._run_capability_showcase(module_name, example_file)
    
    def _prompt_for_showcase(self, module_name: str) -> bool:
        """Prompt user to run showcase with countdown."""
        console = self.console
        
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]🎯 Want to see your {module_name} capability in action?[/bold green]\n\n"
            f"[yellow]We have a live demonstration ready to show what you've built![/yellow]\n\n"
            f"[cyan]This showcase will demonstrate your newly unlocked capability\n"
            f"with real examples and visualizations.[/cyan]\n\n"
            f"[dim]Auto-running in 5 seconds...\n"
            f"Press 'n' + Enter to skip, or just Enter to run now[/dim]",
            title="🚀 Capability Showcase Available",
            border_style="bright_green"
        ))
        
        # Simple countdown with input check
        try:
            import select
            import sys
            
            # Countdown with periodic input checking
            for i in range(5, 0, -1):
                console.print(f"[dim]Starting showcase in {i}... (press 'n' + Enter to skip)[/dim]")
                
                # Check for input on Unix-like systems
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([sys.stdin], [], [], 1)
                    if ready:
                        user_input = sys.stdin.readline().strip().lower()
                        if user_input == 'n' or user_input == 'no':
                            console.print("[dim]Showcase skipped.[/dim]")
                            return False
                        else:
                            console.print("[green]Running showcase![/green]")
                            return True
                else:
                    # Windows fallback - just wait
                    time.sleep(1)
            
            console.print("[green]Auto-running showcase![/green]")
            return True
            
        except Exception:
            # Fallback: simple prompt without countdown
            console.print("[yellow]Run capability showcase? (Y/n):[/yellow]")
            try:
                user_input = input().strip().lower()
                if user_input == 'n' or user_input == 'no':
                    console.print("[dim]Showcase skipped.[/dim]")
                    return False
            except:
                pass
            
            console.print("[green]Running showcase![/green]")
            return True
    
    def _run_capability_showcase(self, module_name: str, achievement_file: Path) -> None:
        """Run the achievement demonstration for a module."""
        console = self.console
        showcase_path = achievement_file  # Already a full Path object
        
        console.print("\n[bold cyan]🚀 Launching Capability Showcase...[/bold cyan]")
        console.print(f"[yellow]See what you've built in action![/yellow]\n")
        
        console.print(Panel(
            f"[bold white]Running: {showcase_file}[/bold white]\n\n"
            f"[cyan]This demonstration shows your {module_name} capability\n"
            f"working with real data and examples.[/cyan]\n\n"
            f"[dim]The showcase will run in your terminal below...[/dim]",
            title=f"🎬 {module_name} Capability Demo",
            border_style="bright_cyan"
        ))
        
        try:
            # Run the showcase
            result = subprocess.run(
                [sys.executable, str(showcase_path)], 
                capture_output=False,  # Let output show in terminal
                text=True
            )
            
            if result.returncode == 0:
                console.print("\n" + "="*60)
                console.print(Panel(
                    f"[bold green]✅ Showcase completed successfully![/bold green]\n\n"
                    f"[yellow]You've now seen your {module_name} capability in action!\n"
                    f"This is what you've accomplished through your implementation.[/yellow]\n\n"
                    f"[cyan]💡 Try exploring the code in: capabilities/{showcase_file}[/cyan]",
                    title="🎉 Demo Complete",
                    border_style="green"
                ))
            else:
                console.print(f"\n[yellow]⚠️  Showcase completed with status code: {result.returncode}[/yellow]")
                
        except Exception as e:
            console.print(f"\n[red]❌ Error running showcase: {e}[/red]")
            console.print(f"[dim]You can manually run: python capabilities/{showcase_file}[/dim]")
    
    def _show_north_star_celebration(self) -> None:
        """Show epic North Star celebration for TinyGPT completion."""
        console = self.console
        
        # Clear screen effect
        console.print("\n" * 3)
        
        # Show the beautiful TinyTorch logo for ultimate celebration
        print_ascii_logo()
        
        # Animated stars
        stars = "✨ ⭐ 🌟 ✨ ⭐ 🌟 ✨ ⭐ 🌟 ✨"
        console.print(Align.center(Text(stars, style="bold bright_yellow blink")))
        console.print("\n")
        
        # Epic ASCII art
        north_star_art = """
                    🌟 NORTH STAR ACHIEVED! 🌟
                           ⭐ TinyGPT ⭐
                    
                     🏆 🎓 YOU ARE AN ML ENGINEER! 🎓 🏆
                    
                        ╔══════════════════════╗
                        ║   FROM SCRATCH TO    ║
                        ║   LANGUAGE MODEL     ║  
                        ║                      ║
                        ║    🧠 → 🤖 → 🚀      ║
                        ╚══════════════════════╝
        """
        
        console.print(Panel(
            north_star_art + "\n\n"
            "[bold bright_yellow]🎉 CONGRATULATIONS! 🎉[/bold bright_yellow]\n\n"
            "[bold white]You have mastered the complete ML systems engineering journey![/bold white]\n"
            "[bold white]From tensors to transformers - all built from scratch![/bold white]\n\n"
            "[bold cyan]🔓 All Capabilities Unlocked:[/bold cyan]\n"
            "  • Foundation & Intelligence\n"
            "  • Networks & Spatial Processing\n"
            "  • Attention & Differentiation\n"
            "  • Training & Optimization\n"
            "  • Deployment & Production\n"
            "  • Language Models & Transformers\n\n"
            "[bold magenta]You ARE an ML Systems Engineer! 🚀[/bold magenta]",
            title="🌟 NORTH STAR: ML SYSTEMS MASTERY 🌟",
            border_style="bright_yellow",
            box=box.ROUNDED
        ))
        
        # Final animated message
        time.sleep(2)
        console.print(Align.center(Text("⭐ Welcome to the ranks of ML Systems Engineers! ⭐", style="bold bright_cyan blink")))
        console.print("\n" * 2)
    
    def _get_module_capability_info(self, module_name: str) -> dict:
        """Get capability information for a module."""
        capabilities = {
            "01_setup": {
                "title": "Development Environment",
                "description": "Master the tools and setup for ML systems engineering"
            },
            "02_tensor": {
                "title": "Foundation Intelligence", 
                "description": "Create and manipulate the building blocks of machine learning"
            },
            "03_activations": {
                "title": "Neural Intelligence",
                "description": "Add nonlinearity - the key to neural network intelligence"
            },
            "04_layers": {
                "title": "Network Components",
                "description": "Build the fundamental building blocks of neural networks"
            },
            "05_dense": {
                "title": "Forward Inference", 
                "description": "Build complete multi-layer neural networks for inference"
            },
            "06_spatial": {
                "title": "Spatial Learning",
                "description": "Process images and spatial data with convolutional operations"
            },
            "07_attention": {
                "title": "Sequence Understanding",
                "description": "Build attention mechanisms for sequence and language understanding"
            },
            "08_dataloader": {
                "title": "Data Engineering",
                "description": "Efficiently load and process training data at scale"
            },
            "09_autograd": {
                "title": "Automatic Differentiation",
                "description": "Automatically compute gradients for neural network learning"
            },
            "10_optimizers": {
                "title": "Advanced Optimization",
                "description": "Optimize neural networks with sophisticated algorithms"
            },
            "11_training": {
                "title": "Neural Network Training",
                "description": "Build complete training loops for end-to-end learning"
            },
            "12_compression": {
                "title": "Robust Vision Models",
                "description": "Prevent overfitting and build robust, deployable models"
            },
            "13_kernels": {
                "title": "High-Performance Computing",
                "description": "Implement optimized computational kernels for ML acceleration"
            },
            "14_benchmarking": {
                "title": "Performance Engineering",
                "description": "Analyze performance and identify bottlenecks in ML systems"
            },
            "15_mlops": {
                "title": "Production Deployment",
                "description": "Deploy and monitor ML systems in production environments"
            },
            "16_tinygpt": {
                "title": "NORTH STAR: GPT FROM SCRATCH",
                "description": "Build complete transformer language models from first principles"
            }
        }
        
        return capabilities.get(module_name, {
            "title": "ML Systems Capability",
            "description": "Advance your ML systems engineering skills"
        })
    
    def _get_celebration_level(self, module_name: str) -> str:
        """Determine celebration level for module completion."""
        major_milestones = ["05_dense", "11_training", "12_compression", "16_tinygpt"]
        milestones = ["04_layers", "07_attention", "09_autograd", "15_mlops"]
        
        if module_name in major_milestones:
            return "major"
        elif module_name in milestones:
            return "milestone" 
        else:
            return "standard"
    
    def _get_standard_celebration_art(self) -> str:
        """Get ASCII art for standard celebrations."""
        return """
                         🎉
                     ⭐ SUCCESS ⭐
                         🚀
        """
    
    def _get_milestone_celebration_art(self) -> str:
        """Get ASCII art for milestone celebrations."""
        return """
                    ✨ MILESTONE ACHIEVED ✨
                       🏆 CAPABILITY 🏆
                         🌟 UNLOCKED 🌟
                            🚀
        """
    
    def _get_major_celebration_art(self) -> str:
        """Get ASCII art for major celebrations."""
        return """
    ╔═══════════════════════════════╗
    ║  🔥 TinyTorch Major Unlock 🔥 ║
    ╚═══════════════════════════════╝
    
            ⚡ MAJOR BREAKTHROUGH ⚡
               🏅 CRITICAL SKILL 🏅
                 🌟 MASTERED 🌟
                    🚀 → 🎯
        """
    
    def _is_module_completed(self, module_name: str) -> bool:
        """Check if module has been completed before."""
        progress_data = self._get_module_progress_data()
        return module_name in progress_data["completed_modules"]
    
    def _record_module_completion(self, module_name: str) -> None:
        """Record module completion in progress tracking."""
        progress_data = self._get_module_progress_data()
        
        if module_name not in progress_data["completed_modules"]:
            progress_data["completed_modules"].append(module_name)
            progress_data["completion_dates"][module_name] = datetime.now().isoformat()
        
        self._save_module_progress_data(progress_data)
    
    def _get_module_progress_data(self) -> dict:
        """Get or create module progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "progress.json"
        
        # Create directory if it doesn't exist
        progress_dir.mkdir(exist_ok=True)
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Return default structure
        return {
            "completed_modules": [],
            "completion_dates": {},
            "achievements": [],
            "total_capabilities_unlocked": 0
        }
    
    def _save_module_progress_data(self, progress_data: dict) -> None:
        """Save module progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "progress.json"
        
        progress_dir.mkdir(exist_ok=True)
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except IOError:
            pass  # Fail silently if we can't save
    
    def _enhanced_show_progress_and_next_steps(self, completed_module: str, milestone_result: dict = None) -> None:
        """Show enhanced progress visualization and suggest next steps."""
        console = self.console
        
        # Get progress data
        progress_data = self._get_module_progress_data()
        checkpoint_system = CheckpointSystem(self.config)
        checkpoint_progress = checkpoint_system.get_overall_progress()
        
        # Show animated progress update
        console.print(f"\n[bold cyan]📊 Progress Update[/bold cyan]")
        
        # Module completion progress bar
        total_modules = 16  # Updated count (01 through 16)
        completed_modules = len(progress_data["completed_modules"])
        module_progress_percent = (completed_modules / total_modules) * 100
        
        # Create visual progress bar
        progress_bar_width = 30
        filled = int((completed_modules / total_modules) * progress_bar_width)
        bar = "█" * filled + "░" * (progress_bar_width - filled)
        
        console.print(Panel(
            f"[bold green]Module Progress:[/bold green] [{bar}] {module_progress_percent:.0f}%\n"
            f"[bold]Modules Completed:[/bold] {completed_modules}/{total_modules}\n\n"
            f"[bold green]Checkpoint Progress:[/bold green] {checkpoint_progress['overall_progress']:.0f}%\n"
            f"[bold]Capabilities Unlocked:[/bold] {checkpoint_progress['total_complete']}/{checkpoint_progress['total_checkpoints']}",
            title="🚀 Your ML Systems Engineering Journey",
            border_style="bright_green"
        ))
        
        # Milestone celebrations
        self._check_milestone_achievements(completed_modules, total_modules)
        
        # Suggest next module with enhanced presentation
        self._suggest_next_module(completed_module)
        
        # Show achievement summary
        self._show_achievement_summary(progress_data)
        
        # General next steps with enhanced formatting
        console.print(Panel(
            f"[bold cyan]🎯 Continue Your Journey[/bold cyan]\n\n"
            f"[green]Track Progress:[/green]\n"
            f"  • [dim]tito checkpoint status --detailed[/dim]\n"
            f"  • [dim]tito checkpoint timeline[/dim]\n\n"
            f"[yellow]Quick Actions:[/yellow]\n"
            f"  • [dim]tito module view [module_name][/dim]\n"
            f"  • [dim]tito module complete [module_name][/dim]\n\n"
            f"[cyan]Show Capabilities:[/cyan]\n"
            f"  • [dim]tito checkpoint status[/dim]",
            title="Next Steps",
            border_style="bright_blue",
            box=box.ROUNDED
        ))
    
    def _check_milestone_achievements(self, completed_modules: int, total_modules: int) -> None:
        """Check and celebrate milestone achievements."""
        console = self.console
        
        milestones = {
            4: "🎯 Getting Started! 25% Complete",
            8: "🚀 Making Progress! 50% Complete", 
            12: "⚡ Almost There! 75% Complete",
            16: "🏆 FULL MASTERY! 100% Complete"
        }
        
        for milestone, message in milestones.items():
            if completed_modules == milestone:
                console.print(Panel(
                    f"[bold bright_yellow]🎊 MILESTONE REACHED! 🎊[/bold bright_yellow]\n\n"
                    f"[bold white]{message}[/bold white]\n\n"
                    f"[green]Keep going - you're becoming an ML Systems Engineer![/green]",
                    title="🏅 Achievement Unlocked",
                    border_style="bright_yellow"
                ))
                break
    
    def _suggest_next_module(self, completed_module: str) -> None:
        """Suggest next module with enhanced presentation."""
        console = self.console
        
        if completed_module.startswith(tuple(f"{i:02d}_" for i in range(100))):
            try:
                module_num = int(completed_module[:2])
                next_num = module_num + 1
                
                next_modules = {
                    1: ("02_tensor", "Tensor operations - the foundation of ML", "🧮"),
                    2: ("03_activations", "Activation functions - adding intelligence", "🧠"),
                    3: ("04_layers", "Neural layers - building blocks", "🔗"),
                    4: ("05_dense", "Dense networks - complete architectures", "🏗️"),
                    5: ("06_spatial", "Spatial processing - convolutional operations", "🖼️"),
                    6: ("07_attention", "Attention mechanisms - sequence understanding", "👁️"),
                    7: ("08_dataloader", "Data loading - efficient training", "📊"),
                    8: ("09_autograd", "Automatic differentiation - gradient computation", "🔄"),
                    9: ("10_optimizers", "Optimization algorithms - sophisticated learning", "⚡"),
                    10: ("11_training", "Training loops - end-to-end learning", "🎓"),
                    11: ("12_compression", "Model compression - efficient deployment", "📦"),
                    12: ("13_kernels", "High-performance kernels - optimized computation", "🚀"),
                    13: ("14_benchmarking", "Performance analysis - bottleneck identification", "📈"),
                    14: ("15_mlops", "MLOps - production deployment", "🌐"),
                    15: ("16_tinygpt", "TinyGPT - Language models and transformers", "🤖"),
                }
                
                if next_num in next_modules:
                    next_module, next_desc, emoji = next_modules[next_num]
                    console.print(Panel(
                        f"[bold cyan]{emoji} Next Adventure Awaits![/bold cyan]\n\n"
                        f"[bold yellow]Up Next:[/bold yellow] {next_module}\n"
                        f"[dim]{next_desc}[/dim]\n\n"
                        f"[bold green]Ready to continue your journey?[/bold green]\n\n"
                        f"[cyan]Quick Start:[/cyan]\n"
                        f"  • [dim]tito module view {next_module}[/dim]\n"
                        f"  • [dim]tito module complete {next_module}[/dim]",
                        title="🎯 Continue Your Quest",
                        border_style="bright_cyan"
                    ))
                elif next_num > 16:
                    console.print(Panel(
                        f"[bold green]🏆 QUEST COMPLETE! 🏆[/bold green]\n\n"
                        f"[green]You've mastered all TinyTorch modules![/green]\n"
                        f"[bold white]You are now an ML Systems Engineer![/bold white]\n\n"
                        f"[cyan]Share your achievement:[/cyan]\n"
                        f"[dim]  tito checkpoint status[/dim]\n"
                        f"[dim]  tito checkpoint timeline[/dim]",
                        title="🌟 FULL MASTERY ACHIEVED",
                        border_style="bright_green"
                    ))
            except (ValueError, IndexError):
                pass
    
    def _show_achievement_summary(self, progress_data: dict) -> None:
        """Show summary of recent achievements."""
        console = self.console
        completed_count = len(progress_data["completed_modules"])
        
        if completed_count > 0:
            recent_modules = progress_data["completed_modules"][-3:]  # Last 3 completed
            
            console.print(Panel(
                f"[bold yellow]🏅 Recent Achievements[/bold yellow]\n\n" +
                "\n".join(f"  ✅ {module}" for module in recent_modules) +
                f"\n\n[bold]Total Modules Mastered:[/bold] {completed_count}/16",
                title="Your Progress",
                border_style="yellow"
            ))
    
    def _show_capability_test_failure(self, module_name: str, checkpoint_result: dict, integration_result: dict) -> None:
        """Show helpful feedback when capability test fails but integration passes."""
        console = self.console
        
        console.print(Panel(
            f"[bold yellow]⚠️ Partial Success[/bold yellow]\n\n"
            f"[green]✅ Package Integration:[/green] Module exported and integrated successfully\n"
            f"[yellow]❌ Capability Test:[/yellow] {checkpoint_result.get('checkpoint_name', 'Checkpoint')} validation failed\n\n"
            f"[bold cyan]What this means:[/bold cyan]\n"
            f"• Your module integrates with the TinyTorch package\n"
            f"• Some advanced functionality may be missing\n"
            f"• Implementation needs refinement for full capability unlock\n\n"
            f"[bold green]💡 Next steps:[/bold green]\n"
            f"• Review the module implementation\n"
            f"• Test individual components manually\n"
            f"• Try: [dim]tito module complete {module_name}[/dim] again\n"
            f"• Debug: [dim]tito checkpoint test[/dim] for detailed feedback",
            title="Capability Unlock Pending",
            border_style="yellow"
        ))