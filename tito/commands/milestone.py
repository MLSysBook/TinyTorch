"""
Milestone command group for TinyTorch CLI: capability-based learning progression.

The milestone system transforms module completion into meaningful capability achievements.
Instead of just finishing modules, students unlock epic milestones that represent 
real-world ML engineering skills.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.align import Align
from rich.text import Text
from rich.layout import Layout
from rich.tree import Tree
from rich.columns import Columns
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

from .base import BaseCommand
from .checkpoint import CheckpointSystem
from ..core.console import print_ascii_logo
from ..core.console import get_console


class MilestoneSystem:
    """Core milestone tracking and management system."""
    
    def __init__(self, config):
        self.config = config
        self.console = get_console()
        
        # Define the 5 Epic Milestones
        self.MILESTONES = {
            "1": {
                "id": "1",
                "name": "Basic Inference",
                "title": "Neural Networks Work!",
                "trigger_module": "05_dense",
                "required_checkpoints": ["00", "01", "02", "03", "04"],
                "victory_condition": "Build multi-layer networks that solve XOR problem",
                "capability": "I can create working neural networks!",
                "real_world_impact": "Foundation of any AI system",
                "emoji": "🥉",
                "celebration_level": "milestone",
                "demo_file": "01_basic_inference_demo.py"
            },
            "2": {
                "id": "2", 
                "name": "Computer Vision",
                "title": "Machines Can See!",
                "trigger_module": "06_spatial",
                "required_checkpoints": ["05"],
                "victory_condition": "Achieve 95%+ accuracy on MNIST with CNN",
                "capability": "I can teach machines to see!",
                "real_world_impact": "Image recognition like autonomous vehicles",
                "emoji": "🥈",
                "celebration_level": "major",
                "demo_file": "02_computer_vision_demo.py"
            },
            "3": {
                "id": "3",
                "name": "Full Training", 
                "title": "Production Training!",
                "trigger_module": "11_training",
                "required_checkpoints": ["06", "07", "08", "09", "10"],
                "victory_condition": "Train CNN from scratch to convergence on CIFAR-10",
                "capability": "I can train production ML models!",
                "real_world_impact": "Train models like those used in industry",
                "emoji": "🏆",
                "celebration_level": "major",
                "demo_file": "03_full_training_demo.py"
            },
            "4": {
                "id": "4",
                "name": "Advanced Vision",
                "title": "Production Vision!",
                "trigger_module": "13_kernels", 
                "required_checkpoints": ["11", "12"],
                "victory_condition": "Achieve 75%+ accuracy on CIFAR-10 with optimized kernels",
                "capability": "I build production-ready vision systems!",
                "real_world_impact": "Computer vision systems used in tech companies",
                "emoji": "💎",
                "celebration_level": "epic",
                "demo_file": "04_advanced_vision_demo.py"
            },
            "5": {
                "id": "5",
                "name": "Language Generation",
                "title": "Build the Future!",
                "trigger_module": "16_tinygpt",
                "required_checkpoints": ["13", "14", "15"],
                "victory_condition": "Generate coherent text and Python code with TinyGPT",
                "capability": "I can build the future of AI!",
                "real_world_impact": "Build systems like ChatGPT and GitHub Copilot",
                "emoji": "👑",
                "celebration_level": "north_star",
                "demo_file": "05_language_generation_demo.py"
            }
        }
    
    def get_milestone_status(self) -> dict:
        """Get current milestone progress status."""
        checkpoint_system = CheckpointSystem(self.config)
        milestone_data = self._get_milestone_progress_data()
        
        status = {
            "milestones": {},
            "overall_progress": 0,
            "total_unlocked": 0,
            "next_milestone": None
        }
        
        total_milestones = len(self.MILESTONES)
        unlocked_count = 0
        
        for milestone_id, milestone in self.MILESTONES.items():
            # Check if all required checkpoints are complete
            required_complete = all(
                checkpoint_system.get_checkpoint_status(cp_id).get("is_complete", False)
                for cp_id in milestone["required_checkpoints"]
            )
            
            # Check if milestone is unlocked
            is_unlocked = milestone_id in milestone_data.get("unlocked_milestones", [])
            
            # Check if trigger module is completed
            trigger_complete = self._is_module_completed(milestone["trigger_module"])
            
            milestone_status = {
                "id": milestone_id,
                "name": milestone["name"], 
                "title": milestone["title"],
                "emoji": milestone["emoji"],
                "trigger_module": milestone["trigger_module"],
                "required_checkpoints": milestone["required_checkpoints"],
                "victory_condition": milestone["victory_condition"],
                "capability": milestone["capability"],
                "real_world_impact": milestone["real_world_impact"],
                "required_complete": required_complete,
                "trigger_complete": trigger_complete,
                "is_unlocked": is_unlocked,
                "can_unlock": required_complete and trigger_complete and not is_unlocked,
                "unlock_date": milestone_data.get("unlock_dates", {}).get(milestone_id)
            }
            
            status["milestones"][milestone_id] = milestone_status
            
            if is_unlocked:
                unlocked_count += 1
            elif milestone_status["can_unlock"] and not status["next_milestone"]:
                status["next_milestone"] = milestone_id
        
        status["total_unlocked"] = unlocked_count
        status["overall_progress"] = (unlocked_count / total_milestones) * 100
        
        return status
    
    def check_milestone_unlock(self, completed_module: str) -> dict:
        """Check if completing a module unlocks a milestone."""
        result = {
            "milestone_unlocked": False,
            "milestone_id": None,
            "milestone_data": None,
            "celebration_needed": False
        }
        
        # Find milestone triggered by this module
        for milestone_id, milestone in self.MILESTONES.items():
            if milestone["trigger_module"] == completed_module:
                status = self.get_milestone_status()
                milestone_status = status["milestones"][milestone_id]
                
                if milestone_status["can_unlock"]:
                    # Unlock the milestone!
                    self._unlock_milestone(milestone_id)
                    result.update({
                        "milestone_unlocked": True,
                        "milestone_id": milestone_id,
                        "milestone_data": milestone,
                        "celebration_needed": True
                    })
                break
        
        return result
    
    def run_milestone_test(self, milestone_id: str) -> dict:
        """Run tests to validate milestone achievement."""
        if milestone_id not in self.MILESTONES:
            return {"success": False, "error": f"Milestone {milestone_id} not found"}
        
        milestone = self.MILESTONES[milestone_id]
        
        # Check all required checkpoints
        checkpoint_system = CheckpointSystem(self.config)
        failed_checkpoints = []
        
        for cp_id in milestone["required_checkpoints"]:
            if not checkpoint_system.get_checkpoint_status(cp_id).get("is_complete", False):
                failed_checkpoints.append(cp_id)
        
        if failed_checkpoints:
            return {
                "success": False,
                "error": f"Required checkpoints not completed: {', '.join(failed_checkpoints)}",
                "milestone_name": milestone["name"]
            }
        
        # Check trigger module completion
        if not self._is_module_completed(milestone["trigger_module"]):
            return {
                "success": False,
                "error": f"Trigger module {milestone['trigger_module']} not completed",
                "milestone_name": milestone["name"]
            }
        
        # All tests passed
        return {
            "success": True,
            "milestone_id": milestone_id,
            "milestone_name": milestone["name"],
            "title": milestone["title"],
            "capability": milestone["capability"],
            "victory_condition": milestone["victory_condition"]
        }
    
    def _unlock_milestone(self, milestone_id: str) -> None:
        """Record milestone unlock in progress tracking."""
        milestone_data = self._get_milestone_progress_data()
        
        if milestone_id not in milestone_data["unlocked_milestones"]:
            milestone_data["unlocked_milestones"].append(milestone_id)
            milestone_data["unlock_dates"][milestone_id] = datetime.now().isoformat()
            milestone_data["total_unlocked"] = len(milestone_data["unlocked_milestones"])
        
        self._save_milestone_progress_data(milestone_data)
    
    def _is_module_completed(self, module_name: str) -> bool:
        """Check if a module has been completed."""
        # Check module progress file
        progress_file = Path(".tito") / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    return module_name in progress_data.get("completed_modules", [])
            except (json.JSONDecodeError, IOError):
                pass
        return False
    
    def _get_milestone_progress_data(self) -> dict:
        """Get or create milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"
        
        progress_dir.mkdir(exist_ok=True)
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "unlocked_milestones": [],
            "unlock_dates": {},
            "total_unlocked": 0,
            "achievements": []
        }
    
    def _save_milestone_progress_data(self, milestone_data: dict) -> None:
        """Save milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"
        
        progress_dir.mkdir(exist_ok=True)
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(milestone_data, f, indent=2)
        except IOError:
            pass


class MilestoneCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "milestone"

    @property
    def description(self) -> str:
        return "Milestone achievement and capability unlock commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='milestone_command',
            help='Milestone subcommands',
            metavar='SUBCOMMAND'
        )
        
        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='View milestone progress and achievements'
        )
        status_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed milestone information'
        )
        
        # Timeline subcommand
        timeline_parser = subparsers.add_parser(
            'timeline',
            help='View milestone timeline and progression'
        )
        timeline_parser.add_argument(
            '--horizontal',
            action='store_true',
            help='Show horizontal progress bar instead of tree'
        )
        
        # Test subcommand
        test_parser = subparsers.add_parser(
            'test',
            help='Test milestone achievement requirements'
        )
        test_parser.add_argument(
            'milestone_id',
            nargs='?',
            help='Milestone ID to test (1-5), or test next available'
        )
        
        # Demo subcommand
        demo_parser = subparsers.add_parser(
            'demo',
            help='Run milestone capability demonstration'
        )
        demo_parser.add_argument(
            'milestone_id',
            help='Milestone ID to demonstrate (1-5)'
        )

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if not hasattr(args, 'milestone_command') or not args.milestone_command:
            console.print(Panel(
                "[bold cyan]Milestone Commands[/bold cyan]\n\n"
                "Transform module completion into epic capability achievements!\n\n"
                "Available subcommands:\n"
                "  • [bold]status[/bold]     - View milestone progress and achievements\n"
                "  • [bold]timeline[/bold]   - View milestone timeline and progression\n"
                "  • [bold]test[/bold]       - Test milestone achievement requirements\n"
                "  • [bold]demo[/bold]       - Run milestone capability demonstration\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito milestone status --detailed[/dim]\n"
                "[dim]  tito milestone timeline --horizontal[/dim]\n"
                "[dim]  tito milestone test 2[/dim]\n"
                "[dim]  tito milestone demo 1[/dim]",
                title="🎮 Milestone System",
                border_style="bright_cyan"
            ))
            return 0
        
        # Execute the appropriate subcommand
        if args.milestone_command == 'status':
            return self._handle_status_command(args)
        elif args.milestone_command == 'timeline':
            return self._handle_timeline_command(args)
        elif args.milestone_command == 'test':
            return self._handle_test_command(args)
        elif args.milestone_command == 'demo':
            return self._handle_demo_command(args)
        else:
            console.print(Panel(
                f"[red]Unknown milestone subcommand: {args.milestone_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1

    def _handle_status_command(self, args: Namespace) -> int:
        """Handle milestone status command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()
        
        # Show header with overall progress
        console.print(Panel(
            f"[bold cyan]🎮 TinyTorch Milestone Progress[/bold cyan]\n\n"
            f"[bold]Capabilities Unlocked:[/bold] {status['total_unlocked']}/5 milestones\n"
            f"[bold]Overall Progress:[/bold] {status['overall_progress']:.0f}%\n\n"
            f"[dim]Transform from student to ML Systems Engineer![/dim]",
            title="🚀 Your Epic Journey",
            border_style="bright_blue"
        ))
        
        # Show milestone status
        for milestone_id in ["1", "2", "3", "4", "5"]:
            milestone = status["milestones"][milestone_id]
            self._show_milestone_status(milestone, args.detailed)
        
        # Show next steps
        if status["next_milestone"]:
            next_milestone = status["milestones"][status["next_milestone"]]
            console.print(Panel(
                f"[bold cyan]🎯 Next Achievement[/bold cyan]\n\n"
                f"[bold yellow]{next_milestone['emoji']} {next_milestone['title']}[/bold yellow]\n"
                f"[dim]{next_milestone['victory_condition']}[/dim]\n\n"
                f"[green]Ready to unlock![/green] Complete: {next_milestone['trigger_module']}\n"
                f"[dim]tito module complete {next_milestone['trigger_module']}[/dim]",
                title="Next Milestone",
                border_style="bright_green"
            ))
        elif status["total_unlocked"] == 5:
            console.print(Panel(
                f"[bold green]🏆 QUEST COMPLETE! 🏆[/bold green]\n\n"
                f"[green]You've unlocked all 5 epic milestones![/green]\n"
                f"[bold white]You are now an ML Systems Engineer![/bold white]\n\n"
                f"[cyan]Share your achievement and inspire others![/cyan]",
                title="🌟 FULL MASTERY ACHIEVED",
                border_style="bright_green"
            ))
        
        return 0
    
    def _show_milestone_status(self, milestone: dict, detailed: bool = False) -> None:
        """Show status for a single milestone."""
        console = self.console
        
        # Status indicator
        if milestone["is_unlocked"]:
            status_icon = "🔓"
            status_color = "green"
            status_text = "UNLOCKED"
        elif milestone["can_unlock"]:
            status_icon = "⚡"
            status_color = "yellow"
            status_text = "READY TO UNLOCK"
        elif milestone["required_complete"] and not milestone["trigger_complete"]:
            status_icon = "🔒"
            status_color = "cyan"
            status_text = f"COMPLETE: {milestone['trigger_module']}"
        else:
            status_icon = "🔒"
            status_color = "dim"
            status_text = "LOCKED"
        
        # Basic display
        milestone_content = (
            f"[{status_color}]{status_icon} {milestone['emoji']} {milestone['title']}[/{status_color}]\n"
            f"[dim]{milestone['victory_condition']}[/dim]"
        )
        
        # Add detailed information if requested
        if detailed:
            req_status = "✅" if milestone["required_complete"] else "❌"
            trigger_status = "✅" if milestone["trigger_complete"] else "❌"
            
            milestone_content += (
                f"\n\n[bold]Requirements:[/bold]\n"
                f"  {req_status} Checkpoints: {', '.join(milestone['required_checkpoints'])}\n"
                f"  {trigger_status} Module: {milestone['trigger_module']}\n"
                f"[bold]Capability:[/bold] {milestone['capability']}\n"
                f"[bold]Impact:[/bold] {milestone['real_world_impact']}"
            )
            
            if milestone["is_unlocked"] and milestone.get("unlock_date"):
                unlock_date = datetime.fromisoformat(milestone["unlock_date"]).strftime("%Y-%m-%d")
                milestone_content += f"\n[dim]Unlocked: {unlock_date}[/dim]"
        
        console.print(Panel(
            milestone_content,
            title=f"Milestone {milestone['id']}",
            border_style=status_color
        ))

    def _handle_timeline_command(self, args: Namespace) -> int:
        """Handle milestone timeline command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()
        
        if args.horizontal:
            self._show_horizontal_timeline(status)
        else:
            self._show_tree_timeline(status)
        
        return 0
    
    def _show_horizontal_timeline(self, status: dict) -> None:
        """Show horizontal progress bar timeline."""
        console = self.console
        
        console.print(Panel(
            f"[bold cyan]🎮 Milestone Timeline[/bold cyan]\n\n"
            f"[bold]Progress:[/bold] {status['total_unlocked']}/5 milestones unlocked",
            title="Your Epic Journey",
            border_style="bright_blue"
        ))
        
        # Create progress bar
        progress_width = 50
        unlocked_width = int((status["total_unlocked"] / 5) * progress_width)
        
        # Create milestone markers
        timeline = []
        for i in range(5):
            milestone_id = str(i + 1)
            milestone = status["milestones"][milestone_id]
            
            if milestone["is_unlocked"]:
                marker = f"[green]{milestone['emoji']}[/green]"
            elif milestone["can_unlock"]:
                marker = f"[yellow blink]{milestone['emoji']}[/yellow blink]"
            else:
                marker = f"[dim]{milestone['emoji']}[/dim]"
            
            timeline.append(marker)
        
        # Show timeline
        console.print(f"\n{'  '.join(timeline)}")
        
        # Progress bar
        filled = "█" * unlocked_width
        empty = "░" * (progress_width - unlocked_width)
        console.print(f"\n[green]{filled}[/green][dim]{empty}[/dim]")
        console.print(f"[dim]{status['overall_progress']:.0f}% complete[/dim]\n")
    
    def _show_tree_timeline(self, status: dict) -> None:
        """Show tree-style milestone timeline."""
        console = self.console
        
        console.print(Panel(
            f"[bold cyan]🎮 Milestone Progression Tree[/bold cyan]\n\n"
            f"[bold]Your journey from student to ML Systems Engineer[/bold]",
            title="Epic Timeline",
            border_style="bright_blue"
        ))
        
        # Create tree structure
        tree = Tree("🚀 [bold]TinyTorch Mastery Journey[/bold]")
        
        for milestone_id in ["1", "2", "3", "4", "5"]:
            milestone = status["milestones"][milestone_id]
            
            if milestone["is_unlocked"]:
                node_style = "green"
                icon = "✅"
            elif milestone["can_unlock"]:
                node_style = "yellow"
                icon = "⚡"
            else:
                node_style = "dim"
                icon = "🔒"
            
            branch = tree.add(
                f"[{node_style}]{icon} {milestone['emoji']} {milestone['title']}[/{node_style}]"
            )
            
            # Add capability description
            branch.add(f"[dim]{milestone['capability']}[/dim]")
            
            # Add trigger module info
            if milestone["trigger_complete"]:
                branch.add(f"[green]✅ {milestone['trigger_module']} completed[/green]")
            else:
                branch.add(f"[dim]🎯 Complete: {milestone['trigger_module']}[/dim]")
        
        console.print(tree)
        console.print()

    def _handle_test_command(self, args: Namespace) -> int:
        """Handle milestone test command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        
        # Determine which milestone to test
        if args.milestone_id:
            milestone_id = args.milestone_id
        else:
            # Test next available milestone
            status = milestone_system.get_milestone_status()
            if status["next_milestone"]:
                milestone_id = status["next_milestone"]
            else:
                console.print(Panel(
                    "[yellow]No milestone available to test.[/yellow]\n\n"
                    "Either all milestones are unlocked or none are ready.\n"
                    "Use [dim]tito milestone status[/dim] to see your progress.",
                    title="No Test Available",
                    border_style="yellow"
                ))
                return 0
        
        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1
        
        milestone = milestone_system.MILESTONES[milestone_id]
        
        console.print(Panel(
            f"[bold cyan]🧪 Testing Milestone {milestone_id}[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[dim]{milestone['victory_condition']}[/dim]",
            title="Milestone Test",
            border_style="bright_cyan"
        ))
        
        # Run the test with progress animation
        with console.status(f"[bold green]Testing milestone requirements...", spinner="dots"):
            result = milestone_system.run_milestone_test(milestone_id)
        
        # Show results
        if result["success"]:
            console.print(Panel(
                f"[bold green]✅ Milestone Test Passed![/bold green]\n\n"
                f"[green]All requirements met for {result['milestone_name']}[/green]\n"
                f"[cyan]Capability: {result['capability']}[/cyan]\n\n"
                f"[bold yellow]Complete the trigger module to unlock:[/bold yellow]\n"
                f"[dim]tito module complete {milestone['trigger_module']}[/dim]",
                title="🎉 Ready to Unlock!",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠️ Milestone Requirements Not Met[/bold yellow]\n\n"
                f"[yellow]Milestone: {result.get('milestone_name', 'Unknown')}[/yellow]\n"
                f"[red]Issue: {result.get('error', 'Unknown error')}[/red]\n\n"
                f"[cyan]Complete the required modules and try again.[/cyan]",
                title="Requirements Missing",
                border_style="yellow"
            ))
        
        return 0

    def _handle_demo_command(self, args: Namespace) -> int:
        """Handle milestone demo command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        milestone_id = args.milestone_id
        
        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1
        
        milestone = milestone_system.MILESTONES[milestone_id]
        status = milestone_system.get_milestone_status()
        milestone_status = status["milestones"][milestone_id]
        
        # Check if milestone is unlocked
        if not milestone_status["is_unlocked"]:
            console.print(Panel(
                f"[yellow]Milestone {milestone_id} not yet unlocked.[/yellow]\n\n"
                f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
                f"[dim]{milestone['victory_condition']}[/dim]\n\n"
                f"[cyan]Complete the requirements first:[/cyan]\n"
                f"[dim]tito milestone test {milestone_id}[/dim]",
                title="Milestone Locked",
                border_style="yellow"
            ))
            return 0
        
        # Check if demo file exists
        demo_path = Path("capabilities") / milestone["demo_file"]
        if not demo_path.exists():
            console.print(Panel(
                f"[yellow]Demo not available for Milestone {milestone_id}[/yellow]\n\n"
                f"Demo file not found: {milestone['demo_file']}\n"
                f"[dim]This demo may be coming in a future update.[/dim]",
                title="Demo Unavailable",
                border_style="yellow"
            ))
            return 0
        
        # Run the demo
        console.print(Panel(
            f"[bold cyan]🎬 Launching Milestone {milestone_id} Demo[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[yellow]Watch your capability in action![/yellow]\n\n"
            f"[cyan]Demonstrating: {milestone['capability']}[/cyan]\n"
            f"[dim]Running: {milestone['demo_file']}[/dim]",
            title="Capability Demo",
            border_style="bright_cyan"
        ))
        
        try:
            result = subprocess.run(
                [sys.executable, str(demo_path)],
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                console.print(Panel(
                    f"[bold green]✅ Demo completed successfully![/bold green]\n\n"
                    f"[yellow]You've seen your {milestone['title']} capability in action![/yellow]\n"
                    f"[cyan]Real-world impact: {milestone['real_world_impact']}[/cyan]",
                    title="🎉 Demo Complete",
                    border_style="green"
                ))
            else:
                console.print(f"[yellow]⚠️ Demo completed with status: {result.returncode}[/yellow]")
        
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error running demo: {e}[/red]\n\n"
                f"[dim]You can manually run: python capabilities/{milestone['demo_file']}[/dim]",
                title="Demo Error",
                border_style="red"
            ))
            return 1
        
        return 0