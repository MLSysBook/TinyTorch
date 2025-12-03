"""
Milestones command for TinyTorch CLI: track progress through ML history.
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import BaseCommand


class MilestonesCommand(BaseCommand):
    """Track and run milestone verification tests."""
    
    @property
    def name(self) -> str:
        return "milestones"
    
    @property
    def description(self) -> str:
        return "Track progress through ML history milestones"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="milestone_action", help="Milestone actions")
        
        # Progress command
        progress_parser = subparsers.add_parser("progress", help="Show milestone progress")
        
        # List command
        list_parser = subparsers.add_parser("list", help="List unlocked milestone tests")
        
        # Run command
        run_parser = subparsers.add_parser("run", help="Run a milestone verification test")
        run_parser.add_argument("milestone", help="Milestone ID (perceptron, xor, mlp_digits, cnn, transformer)")
    
    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Import milestone tracker
        milestone_tracker_path = Path(__file__).parent.parent.parent / "tests" / "milestones"
        if str(milestone_tracker_path) not in sys.path:
            sys.path.insert(0, str(milestone_tracker_path))
        
        try:
            from milestone_tracker import MilestoneTracker, MILESTONES
        except ImportError:
            console.print("[red]❌ Milestone tracker not available[/red]")
            return 1
        
        tracker = MilestoneTracker()
        
        if not hasattr(args, 'milestone_action') or args.milestone_action is None:
            # Default: show progress
            tracker.show_progress()
            return 0
        
        if args.milestone_action == "progress":
            tracker.show_progress()
            return 0
        
        elif args.milestone_action == "list":
            tracker.list_unlocked_tests()
            return 0
        
        elif args.milestone_action == "run":
            milestone_id = args.milestone
            
            if milestone_id not in MILESTONES:
                console.print(f"[red]❌ Unknown milestone: {milestone_id}[/red]")
                console.print(f"[yellow]Available: {', '.join(MILESTONES.keys())}[/yellow]")
                return 1
            
            if not tracker.can_run_milestone(milestone_id):
                milestone = MILESTONES[milestone_id]
                console.print(f"[yellow]🔒 Milestone locked: {milestone['name']}[/yellow]")
                console.print(f"\n[bold]Complete these modules first:[/bold]")
                for req in milestone["requires"]:
                    status = "✅" if req in tracker.progress["completed_modules"] else "❌"
                    console.print(f"  {status} {req}")
                return 1
            
            # Run the test
            import subprocess
            milestone = MILESTONES[milestone_id]
            test_name = milestone["test"]
            
            console.print(f"[bold cyan]🧪 Running {milestone['name']}[/bold cyan]")
            console.print(f"[dim]{milestone['description']}[/dim]\n")
            
            # Run pytest
            test_file = Path(__file__).parent.parent.parent / "tests" / "milestones" / "test_learning_verification.py"
            
            result = subprocess.run([
                "pytest",
                f"{test_file}::{test_name}",
                "-v",
                "--tb=short"
            ], cwd=Path.cwd())
            
            if result.returncode == 0:
                tracker.mark_milestone_complete(milestone_id)
                
                # Show what's next
                console.print()
                self._show_next_milestone(tracker, milestone_id)
            else:
                console.print()
                console.print("[yellow]💡 The test didn't pass. Check your implementation and try again.[/yellow]")
            
            return result.returncode
        
        else:
            console.print(f"[red]❌ Unknown action: {args.milestone_action}[/red]")
            return 1
    
    def _show_next_milestone(self, tracker, completed_id):
        """Show the next milestone after completing one."""
        from rich.panel import Panel
        
        milestone_order = ["perceptron", "xor", "mlp_digits", "cnn", "transformer"]
        
        try:
            current_index = milestone_order.index(completed_id)
            if current_index < len(milestone_order) - 1:
                next_id = milestone_order[current_index + 1]
                
                if next_id in tracker.progress.get("unlocked_milestones", []):
                    from milestone_tracker import MILESTONES
                    next_milestone = MILESTONES[next_id]
                    
                    self.console.print(Panel(
                        f"[bold cyan]🎯 Next Milestone Available![/bold cyan]\n\n"
                        f"[bold]{next_milestone['name']}[/bold]\n"
                        f"{next_milestone['description']}\n\n"
                        f"[bold]Run it now:[/bold]\n"
                        f"[yellow]tito milestones run {next_id}[/yellow]",
                        title="Continue Your Journey",
                        border_style="cyan"
                    ))
                else:
                    self.console.print(Panel(
                        f"[bold yellow]🔒 Next milestone locked[/bold yellow]\n\n"
                        f"Complete more modules to unlock the next milestone.\n\n"
                        f"[dim]Check progress:[/dim]\n"
                        f"[dim]tito milestones progress[/dim]",
                        title="Keep Building",
                        border_style="yellow"
                    ))
            else:
                self.console.print(Panel(
                    f"[bold green]🏆 ALL MILESTONES COMPLETED![/bold green]\n\n"
                    f"You've verified 60+ years of neural network history!\n"
                    f"Your TinyTorch implementation is complete and working. 🎓",
                    title="Congratulations!",
                    border_style="gold1"
                ))
        except (ValueError, IndexError):
            pass

