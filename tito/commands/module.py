"""
Module command group for TinyTorch CLI: development workflow and module management.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
import sys
import importlib.util

from .base import BaseCommand
from .status import StatusCommand
from .test import TestCommand
from .notebooks import NotebooksCommand
from .clean import CleanCommand
from .export import ExportCommand
from .view import ViewCommand
from .checkpoint import CheckpointSystem
from pathlib import Path

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
                "  • [bold]complete[/bold]   - Complete module with export and checkpoint testing\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito module status --metadata[/dim]\n"
                "[dim]  tito module test --all[/dim]\n"
                "[dim]  tito module test tensor[/dim]\n"
                "[dim]  tito module export --all[/dim]\n"
                "[dim]  tito module export tensor[/dim]\n"
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
            self._show_completion_results(checkpoint_result, normalized_name, integration_result)
        else:
            console.print(f"\n[bold yellow]Step 3: Checkpoint test skipped[/bold yellow]")
            console.print(f"[dim]Module integrated successfully. Run checkpoint test manually if needed.[/dim]")
        
        return 0

    def _normalize_module_name(self, module_name: str) -> str:
        """Normalize module name to full format (e.g., tensor -> 02_tensor)."""
        # If already in full format, validate it exists
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            module_path = Path(f"modules/source/{module_name}")
            if module_path.exists():
                return module_name
            return ""
        
        # Try to find the module by short name
        source_dir = Path("modules/source")
        if source_dir.exists():
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name.endswith(f"_{module_name}"):
                    return module_dir.name
        
        return ""

    def _get_available_modules_text(self) -> str:
        """Get formatted text listing available modules."""
        source_dir = Path("modules/source")
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
        module_to_checkpoint = {
            "01_setup": "00",          # Setup → Environment checkpoint  
            "02_tensor": "01",         # Tensor → Foundation checkpoint
            "03_activations": "02",    # Activations → Intelligence checkpoint
            "04_layers": "03",         # Layers → Components checkpoint
            "05_dense": "04",          # Dense → Networks checkpoint
            "06_spatial": "05",        # Spatial → Learning checkpoint
            "07_attention": "06",      # Attention → Attention checkpoint
            "08_dataloader": "07",     # Dataloader → Stability checkpoint (data prep)
            "09_autograd": "08",       # Autograd → Differentiation checkpoint
            "10_optimizers": "09",     # Optimizers → Optimization checkpoint
            "11_training": "10",       # Training → Training checkpoint
            "12_compression": "11",    # Compression → Regularization checkpoint
            "13_kernels": "12",        # Kernels → Kernels checkpoint
            "14_benchmarking": "13",   # Benchmarking → Benchmarking checkpoint
            "15_mlops": "14",          # MLOps → Deployment checkpoint
            "16_capstone": "15",       # Capstone → Capstone checkpoint
        }
        
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

    def _show_completion_results(self, result: dict, module_name: str, integration_result: dict = None) -> None:
        """Show results of module completion workflow."""
        console = self.console
        
        if result.get("skipped"):
            console.print(f"\n[dim]No checkpoint test available for {module_name}[/dim]")
            console.print(f"[green]✅ Module {module_name} exported successfully![/green]")
            return
        
        if result["success"]:
            # Celebration and progress feedback
            checkpoint_name = result.get("checkpoint_name", "Unknown")
            capability = result.get("capability", "")
            
            # Show both integration and capability success
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
            
            # Show progress and next steps
            self._show_progress_and_next_steps(module_name)
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
                    15: ("16_capstone", "Capstone project - complete ML systems"),
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