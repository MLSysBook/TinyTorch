"""
Checkpoint tracking and visualization command for TinyTorch CLI.

Provides capability-based progress tracking through the ML systems engineering journey:
Foundation → Architecture → Training → Inference → Serving
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns

from .base import BaseCommand
from ..core.config import CLIConfig
from ..core.console import get_console, print_error, print_success


class CheckpointSystem:
    """Core checkpoint tracking system."""
    
    # Define the checkpoint structure
    CHECKPOINTS = {
        "foundation": {
            "name": "Foundation",
            "description": "Core ML primitives and environment setup",
            "modules": ["01_setup", "02_tensor", "03_activations"],
            "capability": "Can build mathematical operations and ML primitives"
        },
        "architecture": {
            "name": "Neural Architecture",
            "description": "Building complete neural network architectures", 
            "modules": ["04_layers", "05_dense", "06_spatial", "07_attention"],
            "capability": "Can design and construct any neural network architecture"
        },
        "training": {
            "name": "Training",
            "description": "Complete model training pipeline",
            "modules": ["08_dataloader", "09_autograd", "10_optimizers", "11_training"],
            "capability": "Can train neural networks on real datasets"
        },
        "inference": {
            "name": "Inference Deployment",
            "description": "Optimized model deployment and serving",
            "modules": ["12_compression", "13_kernels", "14_benchmarking", "15_mlops"],
            "capability": "Can deploy optimized models for production inference"
        },
        "serving": {
            "name": "Serving",
            "description": "Complete ML system integration",
            "modules": ["16_capstone"],
            "capability": "Have built a complete, production-ready ML framework"
        }
    }
    
    def __init__(self, config: CLIConfig):
        """Initialize checkpoint system."""
        self.config = config
        self.console = get_console()
        self.modules_dir = config.project_root / "modules" / "source"
    
    def get_module_status(self, module_name: str) -> Dict[str, bool]:
        """Get the completion status of a module."""
        module_dir = self.modules_dir / module_name
        
        # Check if module directory exists
        if not module_dir.exists():
            return {"exists": False, "has_dev": False, "has_tests": False}
        
        # Check for dev file
        dev_files = list(module_dir.glob("*_dev.py"))
        has_dev = len(dev_files) > 0
        
        # Check for test files or test indicators
        test_files = list(module_dir.glob("test_*.py")) + list(module_dir.glob("*_test.py"))
        has_tests = len(test_files) > 0
        
        return {
            "exists": True,
            "has_dev": has_dev,
            "has_tests": has_tests,
            "complete": has_dev  # For now, consider complete if dev file exists
        }
    
    def get_checkpoint_progress(self, checkpoint_key: str) -> Dict:
        """Get progress information for a checkpoint."""
        checkpoint = self.CHECKPOINTS[checkpoint_key]
        modules_status = []
        completed_count = 0
        
        for module in checkpoint["modules"]:
            status = self.get_module_status(module)
            modules_status.append({
                "name": module,
                "status": status,
                "complete": status.get("complete", False)
            })
            if status.get("complete", False):
                completed_count += 1
        
        total_modules = len(checkpoint["modules"])
        progress_percent = (completed_count / total_modules) * 100 if total_modules > 0 else 0
        
        return {
            "checkpoint": checkpoint,
            "modules": modules_status,
            "completed": completed_count,
            "total": total_modules,
            "progress": progress_percent,
            "is_complete": completed_count == total_modules,
            "is_current": progress_percent > 0 and progress_percent < 100
        }
    
    def get_overall_progress(self) -> Dict:
        """Get overall progress across all checkpoints."""
        checkpoints_progress = {}
        current_checkpoint = None
        total_modules_complete = 0
        total_modules = 0
        
        for key in self.CHECKPOINTS.keys():
            progress = self.get_checkpoint_progress(key)
            checkpoints_progress[key] = progress
            total_modules_complete += progress["completed"]
            total_modules += progress["total"]
            
            # Determine current checkpoint (first incomplete one with progress)
            if current_checkpoint is None and progress["is_current"]:
                current_checkpoint = key
            elif current_checkpoint is None and progress["progress"] == 0:
                current_checkpoint = key
                break
        
        # Calculate overall percentage
        overall_percent = (total_modules_complete / total_modules * 100) if total_modules > 0 else 0
        
        return {
            "checkpoints": checkpoints_progress,
            "current": current_checkpoint,
            "overall_progress": overall_percent,
            "total_modules_complete": total_modules_complete,
            "total_modules": total_modules
        }


class CheckpointCommand(BaseCommand):
    """Checkpoint tracking and visualization command."""
    
    name = "checkpoint"
    description = "Track and visualize ML systems engineering progress through checkpoints"
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add checkpoint-specific arguments."""
        subparsers = parser.add_subparsers(
            dest='checkpoint_command',
            help='Checkpoint operations',
            metavar='COMMAND'
        )
        
        # Status command
        status_parser = subparsers.add_parser(
            'status',
            help='Show current checkpoint progress'
        )
        status_parser.add_argument(
            '--detailed', '-d',
            action='store_true',
            help='Show detailed module-level progress'
        )
        
        # Timeline command
        timeline_parser = subparsers.add_parser(
            'timeline',
            help='Show visual progress timeline'
        )
        timeline_parser.add_argument(
            '--horizontal',
            action='store_true',
            help='Show horizontal timeline (default: vertical)'
        )
        
        # Test command
        test_parser = subparsers.add_parser(
            'test',
            help='Test checkpoint capabilities'
        )
        test_parser.add_argument(
            'checkpoint_name',
            nargs='?',
            help='Specific checkpoint to test (current checkpoint if not specified)'
        )
        
        # Unlock command
        unlock_parser = subparsers.add_parser(
            'unlock',
            help='Attempt to unlock next checkpoint'
        )
    
    def run(self, args: argparse.Namespace) -> int:
        """Execute checkpoint command."""
        checkpoint_system = CheckpointSystem(self.config)
        
        if not args.checkpoint_command:
            return self._show_help(args)
        
        if args.checkpoint_command == 'status':
            return self._show_status(checkpoint_system, args)
        elif args.checkpoint_command == 'timeline':
            return self._show_timeline(checkpoint_system, args)
        elif args.checkpoint_command == 'test':
            return self._test_checkpoint(checkpoint_system, args)
        elif args.checkpoint_command == 'unlock':
            return self._unlock_checkpoint(checkpoint_system, args)
        else:
            print_error(f"Unknown checkpoint command: {args.checkpoint_command}")
            return 1
    
    def _show_help(self, args: argparse.Namespace) -> int:
        """Show checkpoint command help."""
        console = get_console()
        console.print(Panel(
            "[bold cyan]TinyTorch Checkpoint System[/bold cyan]\n\n"
            "[bold]Track your progress through ML systems engineering:[/bold]\n"
            "  🎯 Foundation      → Core ML primitives and setup\n"
            "  🎯 Architecture    → Neural network building\n"
            "  🎯 Training        → Model training pipeline\n"
            "  🎯 Inference       → Deployment and optimization\n"
            "  🎯 Serving         → Complete system integration\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  [green]status[/green]     - Show current progress and capabilities\n"
            "  [green]timeline[/green]   - Visual progress timeline\n"
            "  [green]test[/green]       - Test checkpoint capabilities\n"
            "  [green]unlock[/green]     - Attempt to unlock next checkpoint\n\n"
            "[bold]Examples:[/bold]\n"
            "  [dim]tito checkpoint status --detailed[/dim]\n"
            "  [dim]tito checkpoint timeline --horizontal[/dim]\n"
            "  [dim]tito checkpoint test foundation[/dim]",
            title="Checkpoint System",
            border_style="bright_blue"
        ))
        return 0
    
    def _show_status(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Show checkpoint status."""
        console = get_console()
        progress_data = checkpoint_system.get_overall_progress()
        
        # Header
        console.print(Panel(
            "[bold cyan]🚀 TinyTorch Framework Capabilities[/bold cyan]",
            border_style="bright_blue"
        ))
        
        # Overall progress
        overall_percent = progress_data["overall_progress"]
        console.print(f"\n[bold]Overall Progress:[/bold] {overall_percent:.0f}% ({progress_data['total_modules_complete']}/{progress_data['total_modules']} modules)")
        
        # Current status summary
        current = progress_data["current"]
        if current:
            current_progress = progress_data["checkpoints"][current]
            current_name = current_progress["checkpoint"]["name"]
            current_percent = current_progress["progress"]
            
            console.print(f"[bold]Current Checkpoint:[/bold] {current_name}")
            console.print(f"[bold]Checkpoint Progress:[/bold] {current_percent:.0f}% complete")
            
            if current_progress["is_complete"]:
                console.print(f"[bold green]✅ {current_name} checkpoint achieved![/bold green]")
                console.print(f"[dim]Capability unlocked: {current_progress['checkpoint']['capability']}[/dim]")
            else:
                next_modules = [m for m in current_progress["modules"] if not m["complete"]]
                if next_modules:
                    console.print(f"[bold]Next Module:[/bold] {next_modules[0]['name']}")
        
        console.print()
        
        # Checkpoint progress
        for key, checkpoint_data in progress_data["checkpoints"].items():
            checkpoint = checkpoint_data["checkpoint"]
            progress = checkpoint_data["progress"]
            
            # Checkpoint header
            if checkpoint_data["is_complete"]:
                status_icon = "✅"
                status_color = "green"
            elif checkpoint_data["is_current"]:
                status_icon = "🔄"
                status_color = "yellow"
            else:
                status_icon = "⏳"
                status_color = "dim"
            
            console.print(f"[bold]{status_icon} {checkpoint['name']}[/bold] [{status_color}]{progress:.0f}%[/{status_color}]")
            
            if args.detailed:
                # Show module-level progress
                for module_info in checkpoint_data["modules"]:
                    module_status = "✅" if module_info["complete"] else "⏳"
                    module_name = module_info["name"].replace("_", " ").title()
                    console.print(f"   {module_status} {module_name}")
            else:
                # Show ticker-style progress
                tickers = ""
                for module_info in checkpoint_data["modules"]:
                    tickers += "✅ " if module_info["complete"] else "⏳ "
                console.print(f"   {tickers.strip()}")
            
            console.print(f"   [dim]{checkpoint['capability']}[/dim]\n")
        
        return 0
    
    def _show_timeline(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Show visual timeline with Rich progress bar."""
        console = get_console()
        progress_data = checkpoint_system.get_overall_progress()
        
        console.print("\n[bold cyan]🚀 TinyTorch Framework Progress Timeline[/bold cyan]\n")
        
        if args.horizontal:
            # Enhanced horizontal timeline with progress line
            # First, show the overall progress bar
            overall_percent = progress_data["overall_progress"]
            total_modules = progress_data["total_modules"]
            complete_modules = progress_data["total_modules_complete"]
            
            # Create a visual progress bar
            filled = int(overall_percent / 2)  # 50 characters total width
            bar = "█" * filled + "░" * (50 - filled)
            console.print(f"[bold]Overall:[/bold] [{bar}] {overall_percent:.0f}%")
            console.print(f"[dim]{complete_modules}/{total_modules} modules complete[/dim]\n")
            
            # Show checkpoint progression with connecting lines
            checkpoints_list = list(progress_data["checkpoints"].items())
            
            # Build the checkpoint line
            checkpoint_line = ""
            progress_line = ""
            
            for i, (key, checkpoint_data) in enumerate(checkpoints_list):
                checkpoint = checkpoint_data["checkpoint"]
                
                # Checkpoint status
                if checkpoint_data["is_complete"]:
                    checkpoint_marker = f"[green]●[/green]"
                    checkpoint_name = f"[green]{checkpoint['name']}[/green]"
                elif checkpoint_data["is_current"]:
                    checkpoint_marker = f"[yellow]◉[/yellow]"
                    checkpoint_name = f"[yellow]{checkpoint['name']}[/yellow]"
                else:
                    checkpoint_marker = f"[dim]○[/dim]"
                    checkpoint_name = f"[dim]{checkpoint['name']}[/dim]"
                
                # Add checkpoint
                checkpoint_line += checkpoint_marker
                
                # Add connecting line (except for last checkpoint)
                if i < len(checkpoints_list) - 1:
                    if checkpoint_data["is_complete"]:
                        checkpoint_line += "[green]━━━━[/green]"
                    elif checkpoint_data["is_current"]:
                        # Partial line based on progress
                        progress_chars = int(checkpoint_data["progress"] / 25)  # 4 chars max
                        checkpoint_line += "[yellow]" + "━" * progress_chars + "[/yellow]"
                        checkpoint_line += "[dim]" + "┅" * (4 - progress_chars) + "[/dim]"
                    else:
                        checkpoint_line += "[dim]┅┅┅┅[/dim]"
            
            console.print(checkpoint_line)
            
            # Show checkpoint names below
            names_line = ""
            for i, (key, checkpoint_data) in enumerate(checkpoints_list):
                checkpoint = checkpoint_data["checkpoint"]
                
                if checkpoint_data["is_complete"]:
                    name = f"[green]{checkpoint['name'][:8]:^8}[/green]"
                elif checkpoint_data["is_current"]:
                    name = f"[yellow]{checkpoint['name'][:8]:^8}[/yellow]"
                else:
                    name = f"[dim]{checkpoint['name'][:8]:^8}[/dim]"
                
                names_line += name + "  "
            
            console.print(names_line)
            
            # Show progress percentages
            progress_line = ""
            for key, checkpoint_data in checkpoints_list:
                progress = checkpoint_data["progress"]
                if checkpoint_data["is_complete"]:
                    progress_text = f"[green]{progress:^6.0f}%[/green]"
                elif checkpoint_data["is_current"]:
                    progress_text = f"[yellow]{progress:^6.0f}%[/yellow]"
                else:
                    progress_text = f"[dim]{progress:^6.0f}%[/dim]"
                progress_line += progress_text + "    "
            
            console.print(progress_line)
            
        else:
            # Vertical timeline (tree structure)
            tree = Tree("ML Systems Engineering Journey")
            
            for key, checkpoint_data in progress_data["checkpoints"].items():
                checkpoint = checkpoint_data["checkpoint"]
                
                if checkpoint_data["is_complete"]:
                    checkpoint_text = f"[green]✅ {checkpoint['name']}[/green]"
                elif checkpoint_data["is_current"]:
                    checkpoint_text = f"[yellow]🔄 {checkpoint['name']} ({checkpoint_data['progress']:.0f}%)[/yellow]"
                else:
                    checkpoint_text = f"[dim]⏳ {checkpoint['name']}[/dim]"
                
                checkpoint_node = tree.add(checkpoint_text)
                
                # Add modules as sub-nodes
                for module_info in checkpoint_data["modules"]:
                    module_name = module_info["name"].replace("_", " ").title()
                    if module_info["complete"]:
                        checkpoint_node.add(f"[green]✅ {module_name}[/green]")
                    else:
                        checkpoint_node.add(f"[dim]⏳ {module_name}[/dim]")
            
            console.print(tree)
        
        console.print()
        return 0
    
    def _test_checkpoint(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Test checkpoint capabilities."""
        console = get_console()
        
        # For now, just show what would be tested
        checkpoint_name = args.checkpoint_name
        if not checkpoint_name:
            progress_data = checkpoint_system.get_overall_progress()
            checkpoint_name = progress_data["current"]
        
        if checkpoint_name not in checkpoint_system.CHECKPOINTS:
            print_error(f"Unknown checkpoint: {checkpoint_name}")
            return 1
        
        checkpoint = checkpoint_system.CHECKPOINTS[checkpoint_name]
        console.print(f"\n[bold]Testing {checkpoint['name']} Capabilities[/bold]\n")
        console.print(f"[dim]Would test: {checkpoint['capability']}[/dim]")
        console.print(f"[dim]Modules involved: {', '.join(checkpoint['modules'])}[/dim]")
        console.print("\n[yellow]Checkpoint testing not yet implemented[/yellow]")
        
        return 0
    
    def _unlock_checkpoint(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Attempt to unlock next checkpoint."""
        console = get_console()
        
        # For now, just show what would be unlocked
        progress_data = checkpoint_system.get_overall_progress()
        current = progress_data["current"]
        
        if not current:
            console.print("[green]All checkpoints completed! 🎉[/green]")
            return 0
        
        current_progress = progress_data["checkpoints"][current]
        
        if current_progress["is_complete"]:
            console.print(f"[green]✅ {current_progress['checkpoint']['name']} checkpoint already complete![/green]")
            
            # Find next checkpoint
            checkpoint_keys = list(checkpoint_system.CHECKPOINTS.keys())
            current_index = checkpoint_keys.index(current)
            if current_index < len(checkpoint_keys) - 1:
                next_key = checkpoint_keys[current_index + 1]
                next_checkpoint = checkpoint_system.CHECKPOINTS[next_key]
                console.print(f"[bold]Next checkpoint:[/bold] {next_checkpoint['name']}")
            else:
                console.print("[bold]🎉 All checkpoints completed![/bold]")
        else:
            incomplete_modules = [m for m in current_progress["modules"] if not m["complete"]]
            console.print(f"[yellow]Complete these modules to unlock {current_progress['checkpoint']['name']}:[/yellow]")
            for module in incomplete_modules:
                console.print(f"  ⏳ {module['name']}")
        
        return 0