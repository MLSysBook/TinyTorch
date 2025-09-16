"""
Milestone tracking and visualization command for TinyTorch CLI.

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

from .base import BaseCommand
from ..core.config import CLIConfig
from ..core.console import get_console, print_error, print_success


class MilestoneSystem:
    """Core milestone tracking system."""
    
    # Define the milestone structure
    MILESTONES = {
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
        """Initialize milestone system."""
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
    
    def get_milestone_progress(self, milestone_key: str) -> Dict:
        """Get progress information for a milestone."""
        milestone = self.MILESTONES[milestone_key]
        modules_status = []
        completed_count = 0
        
        for module in milestone["modules"]:
            status = self.get_module_status(module)
            modules_status.append({
                "name": module,
                "status": status,
                "complete": status.get("complete", False)
            })
            if status.get("complete", False):
                completed_count += 1
        
        total_modules = len(milestone["modules"])
        progress_percent = (completed_count / total_modules) * 100 if total_modules > 0 else 0
        
        return {
            "milestone": milestone,
            "modules": modules_status,
            "completed": completed_count,
            "total": total_modules,
            "progress": progress_percent,
            "is_complete": completed_count == total_modules,
            "is_current": progress_percent > 0 and progress_percent < 100
        }
    
    def get_overall_progress(self) -> Dict:
        """Get overall progress across all milestones."""
        milestones_progress = {}
        current_milestone = None
        
        for key in self.MILESTONES.keys():
            progress = self.get_milestone_progress(key)
            milestones_progress[key] = progress
            
            # Determine current milestone (first incomplete one with progress)
            if current_milestone is None and progress["is_current"]:
                current_milestone = key
            elif current_milestone is None and progress["progress"] == 0:
                current_milestone = key
                break
        
        return {
            "milestones": milestones_progress,
            "current": current_milestone
        }


class MilestoneCommand(BaseCommand):
    """Milestone tracking and visualization command."""
    
    name = "milestone"
    description = "Track and visualize ML systems engineering progress"
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add milestone-specific arguments."""
        subparsers = parser.add_subparsers(
            dest='milestone_command',
            help='Milestone operations',
            metavar='COMMAND'
        )
        
        # Status command
        status_parser = subparsers.add_parser(
            'status',
            help='Show current milestone progress'
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
            help='Test milestone capabilities'
        )
        test_parser.add_argument(
            'milestone_name',
            nargs='?',
            help='Specific milestone to test (current milestone if not specified)'
        )
        
        # Unlock command
        unlock_parser = subparsers.add_parser(
            'unlock',
            help='Attempt to unlock next milestone'
        )
    
    def run(self, args: argparse.Namespace) -> int:
        """Execute milestone command."""
        milestone_system = MilestoneSystem(self.config)
        
        if not args.milestone_command:
            return self._show_help(args)
        
        if args.milestone_command == 'status':
            return self._show_status(milestone_system, args)
        elif args.milestone_command == 'timeline':
            return self._show_timeline(milestone_system, args)
        elif args.milestone_command == 'test':
            return self._test_milestone(milestone_system, args)
        elif args.milestone_command == 'unlock':
            return self._unlock_milestone(milestone_system, args)
        else:
            print_error(f"Unknown milestone command: {args.milestone_command}")
            return 1
    
    def _show_help(self, args: argparse.Namespace) -> int:
        """Show milestone command help."""
        console = get_console()
        console.print(Panel(
            "[bold cyan]TinyTorch Milestone System[/bold cyan]\n\n"
            "[bold]Track your progress through ML systems engineering:[/bold]\n"
            "  🎯 Foundation      → Core ML primitives and setup\n"
            "  🎯 Architecture    → Neural network building\n"
            "  🎯 Training        → Model training pipeline\n"
            "  🎯 Inference       → Deployment and optimization\n"
            "  🎯 Serving         → Complete system integration\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  [green]status[/green]     - Show current progress and capabilities\n"
            "  [green]timeline[/green]   - Visual progress timeline\n"
            "  [green]test[/green]       - Test milestone capabilities\n"
            "  [green]unlock[/green]     - Attempt to unlock next milestone\n\n"
            "[bold]Examples:[/bold]\n"
            "  [dim]tito milestone status --detailed[/dim]\n"
            "  [dim]tito milestone timeline --horizontal[/dim]\n"
            "  [dim]tito milestone test foundation[/dim]",
            title="Milestone System",
            border_style="bright_blue"
        ))
        return 0
    
    def _show_status(self, milestone_system: MilestoneSystem, args: argparse.Namespace) -> int:
        """Show milestone status."""
        console = get_console()
        progress_data = milestone_system.get_overall_progress()
        
        # Header
        console.print(Panel(
            "[bold cyan]🚀 TinyTorch Framework Capabilities[/bold cyan]",
            border_style="bright_blue"
        ))
        
        # Current status summary
        current = progress_data["current"]
        if current:
            current_progress = progress_data["milestones"][current]
            current_name = current_progress["milestone"]["name"]
            current_percent = current_progress["progress"]
            
            console.print(f"\n[bold]Current Focus:[/bold] {current_name}")
            console.print(f"[bold]Progress:[/bold] {current_percent:.0f}% complete")
            
            if current_progress["is_complete"]:
                console.print(f"[bold green]✅ {current_name} milestone achieved![/bold green]")
                console.print(f"[dim]Capability unlocked: {current_progress['milestone']['capability']}[/dim]")
            else:
                next_modules = [m for m in current_progress["modules"] if not m["complete"]]
                if next_modules:
                    console.print(f"[bold]Next:[/bold] {next_modules[0]['name']}")
        
        console.print()
        
        # Milestone progress
        for key, milestone_data in progress_data["milestones"].items():
            milestone = milestone_data["milestone"]
            progress = milestone_data["progress"]
            
            # Milestone header
            if milestone_data["is_complete"]:
                status_icon = "✅"
                status_color = "green"
            elif milestone_data["is_current"]:
                status_icon = "🔄"
                status_color = "yellow"
            else:
                status_icon = "⏳"
                status_color = "dim"
            
            console.print(f"[bold]{status_icon} {milestone['name']}[/bold] [{status_color}]{progress:.0f}%[/{status_color}]")
            
            if args.detailed:
                # Show module-level progress
                for module_info in milestone_data["modules"]:
                    module_status = "✅" if module_info["complete"] else "⏳"
                    module_name = module_info["name"].replace("_", " ").title()
                    console.print(f"   {module_status} {module_name}")
            else:
                # Show ticker-style progress
                tickers = ""
                for module_info in milestone_data["modules"]:
                    tickers += "✅ " if module_info["complete"] else "⏳ "
                console.print(f"   {tickers.strip()}")
            
            console.print(f"   [dim]{milestone['capability']}[/dim]\n")
        
        return 0
    
    def _show_timeline(self, milestone_system: MilestoneSystem, args: argparse.Namespace) -> int:
        """Show visual timeline."""
        console = get_console()
        progress_data = milestone_system.get_overall_progress()
        
        if args.horizontal:
            # Horizontal timeline
            console.print("\n[bold cyan]🚀 TinyTorch Framework Progress Timeline[/bold cyan]\n")
            
            timeline = ""
            for i, (key, milestone_data) in enumerate(progress_data["milestones"].items()):
                milestone = milestone_data["milestone"]
                
                if milestone_data["is_complete"]:
                    timeline += f"[green]✅ {milestone['name']}[/green]"
                elif milestone_data["is_current"]:
                    timeline += f"[yellow]🔄 {milestone['name']}[/yellow]"
                else:
                    timeline += f"[dim]⏳ {milestone['name']}[/dim]"
                
                if i < len(progress_data["milestones"]) - 1:
                    timeline += " ━━━━━→ "
            
            console.print(timeline)
            
        else:
            # Vertical timeline (tree structure)
            console.print("\n[bold cyan]🚀 TinyTorch Framework Capabilities[/bold cyan]\n")
            
            tree = Tree("ML Systems Engineering Journey")
            
            for key, milestone_data in progress_data["milestones"].items():
                milestone = milestone_data["milestone"]
                
                if milestone_data["is_complete"]:
                    milestone_text = f"[green]✅ {milestone['name']}[/green]"
                elif milestone_data["is_current"]:
                    milestone_text = f"[yellow]🔄 {milestone['name']} ({milestone_data['progress']:.0f}%)[/yellow]"
                else:
                    milestone_text = f"[dim]⏳ {milestone['name']}[/dim]"
                
                milestone_node = tree.add(milestone_text)
                
                # Add modules as sub-nodes
                for module_info in milestone_data["modules"]:
                    module_name = module_info["name"].replace("_", " ").title()
                    if module_info["complete"]:
                        milestone_node.add(f"[green]✅ {module_name}[/green]")
                    else:
                        milestone_node.add(f"[dim]⏳ {module_name}[/dim]")
            
            console.print(tree)
        
        console.print()
        return 0
    
    def _test_milestone(self, milestone_system: MilestoneSystem, args: argparse.Namespace) -> int:
        """Test milestone capabilities."""
        console = get_console()
        
        # For now, just show what would be tested
        milestone_name = args.milestone_name
        if not milestone_name:
            progress_data = milestone_system.get_overall_progress()
            milestone_name = progress_data["current"]
        
        if milestone_name not in milestone_system.MILESTONES:
            print_error(f"Unknown milestone: {milestone_name}")
            return 1
        
        milestone = milestone_system.MILESTONES[milestone_name]
        console.print(f"\n[bold]Testing {milestone['name']} Capabilities[/bold]\n")
        console.print(f"[dim]Would test: {milestone['capability']}[/dim]")
        console.print(f"[dim]Modules involved: {', '.join(milestone['modules'])}[/dim]")
        console.print("\n[yellow]Milestone testing not yet implemented[/yellow]")
        
        return 0
    
    def _unlock_milestone(self, milestone_system: MilestoneSystem, args: argparse.Namespace) -> int:
        """Attempt to unlock next milestone."""
        console = get_console()
        
        # For now, just show what would be unlocked
        progress_data = milestone_system.get_overall_progress()
        current = progress_data["current"]
        
        if not current:
            console.print("[green]All milestones completed! 🎉[/green]")
            return 0
        
        current_progress = progress_data["milestones"][current]
        
        if current_progress["is_complete"]:
            console.print(f"[green]✅ {current_progress['milestone']['name']} milestone already complete![/green]")
            
            # Find next milestone
            milestone_keys = list(milestone_system.MILESTONES.keys())
            current_index = milestone_keys.index(current)
            if current_index < len(milestone_keys) - 1:
                next_key = milestone_keys[current_index + 1]
                next_milestone = milestone_system.MILESTONES[next_key]
                console.print(f"[bold]Next milestone:[/bold] {next_milestone['name']}")
            else:
                console.print("[bold]🎉 All milestones completed![/bold]")
        else:
            incomplete_modules = [m for m in current_progress["modules"] if not m["complete"]]
            console.print(f"[yellow]Complete these modules to unlock {current_progress['milestone']['name']}:[/yellow]")
            for module in incomplete_modules:
                console.print(f"  ⏳ {module['name']}")
        
        return 0