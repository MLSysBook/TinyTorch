"""
TinyTorch Community Leaderboard Command

Inclusive community showcase where everyone belongs, regardless of performance level.
Celebrates the journey, highlights improvements, and builds community through shared learning.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import uuid

from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.prompt import Prompt, Confirm
from rich.console import Group
from rich.align import Align

from .base import BaseCommand
from ..core.exceptions import TinyTorchCLIError


class LeaderboardCommand(BaseCommand):
    """Community leaderboard - Everyone welcome, celebrate the journey!"""
    
    @property
    def name(self) -> str:
        return "leaderboard"
    
    @property
    def description(self) -> str:
        return "Community showcase - Join, share progress, celebrate achievements together"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add leaderboard subcommands."""
        subparsers = parser.add_subparsers(
            dest='leaderboard_command',
            help='Leaderboard operations',
            metavar='COMMAND'
        )
        
        # Join/Register command (join is primary, register is alias)
        join_parser = subparsers.add_parser(
            'join',
            help='Join the TinyTorch community (inclusive, welcoming)',
            aliases=['register']
        )
        join_parser.add_argument(
            '--username',
            help='Your display name (defaults to system username)'
        )
        join_parser.add_argument(
            '--institution',
            help='Institution/organization (optional)'
        )
        join_parser.add_argument(
            '--country',
            help='Country (optional, for global community view)'
        )
        join_parser.add_argument(
            '--update',
            action='store_true',
            help='Update existing registration'
        )
        
        # Submit command
        submit_parser = subparsers.add_parser(
            'submit',
            help='Submit your results (baseline or improvements welcome!)'
        )
        submit_parser.add_argument(
            '--task',
            default='cifar10',
            choices=['cifar10', 'mnist', 'tinygpt'],
            help='Task to submit results for (default: cifar10)'
        )
        submit_parser.add_argument(
            '--accuracy',
            type=float,
            help='Accuracy achieved (any level welcome!)'
        )
        submit_parser.add_argument(
            '--model',
            help='Model description (e.g., "CNN-3-layer", "Custom-Architecture")'
        )
        submit_parser.add_argument(
            '--notes',
            help='Optional notes about your approach, learnings, challenges'
        )
        submit_parser.add_argument(
            '--checkpoint',
            help='Which TinyTorch checkpoint you completed (e.g., "05", "10", "15")'
        )
        
        # View command
        view_parser = subparsers.add_parser(
            'view',
            help='See the community progress (everyone included!)'
        )
        view_parser.add_argument(
            '--task',
            default='cifar10',
            choices=['cifar10', 'mnist', 'tinygpt'],
            help='Task leaderboard to view (default: cifar10)'
        )
        view_parser.add_argument(
            '--distribution',
            action='store_true',
            help='Show performance distribution graph'
        )
        view_parser.add_argument(
            '--recent',
            action='store_true',
            help='Focus on recent achievements and improvements'
        )
        view_parser.add_argument(
            '--all',
            action='store_true',
            help='Show complete community (not just top performers)'
        )
        
        # Profile command
        profile_parser = subparsers.add_parser(
            'profile',
            help='Your personal achievement journey'
        )
        profile_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed progress across all tasks'
        )
        
        # Status command (quick personal stats)
        status_parser = subparsers.add_parser(
            'status',
            help='Quick personal stats and next steps'
        )
        
        # Help command
        help_parser = subparsers.add_parser(
            'help',
            help='Explain the community leaderboard system'
        )
    
    def run(self, args: Namespace) -> int:
        """Execute leaderboard command."""
        command = getattr(args, 'leaderboard_command', None)
        
        if not command:
            self._show_leaderboard_overview()
            return 0
        
        if command in ['join', 'register']:
            return self._register_user(args)
        elif command == 'submit':
            return self._submit_results(args)
        elif command == 'view':
            return self._view_leaderboard(args)
        elif command == 'profile':
            return self._show_profile(args)
        elif command == 'status':
            return self._show_status(args)
        elif command == 'help':
            return self._show_help()
        else:
            raise TinyTorchCLIError(f"Unknown leaderboard command: {command}")
    
    def _show_leaderboard_overview(self) -> None:
        """Show leaderboard overview and welcome message."""
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🌟 TinyTorch Community Leaderboard 🌟[/bold bright_green]"),
                "",
                "[bold]Everyone Welcome![/bold] This is your inclusive community showcase where:",
                "• [green]Every achievement matters[/green] - 10% accuracy gets the same celebration as 90%",
                "• [blue]Progress is the goal[/blue] - We celebrate improvements and learning journeys",
                "• [yellow]Community first[/yellow] - Help each other, share insights, grow together",
                "• [magenta]No minimum required[/magenta] - Join with any level of progress",
                "",
                "[bold]Available Commands:[/bold]",
                "  [green]join[/green]      - Join our welcoming community (free, inclusive)",
                "  [green]submit[/green]    - Share your progress (any level welcome!)",
                "  [green]view[/green]      - See everyone's journey together",
                "  [green]profile[/green]   - Your personal achievement story",
                "  [green]status[/green]    - Quick stats and encouragement",
                "  [green]help[/green]      - Learn about the community system",
                "",
                "[dim]💡 Tip: Start with 'tito leaderboard join' to join the community![/dim]",
            ),
            title="Community Leaderboard",
            border_style="bright_green",
            padding=(1, 2)
        ))
    
    def _register_user(self, args: Namespace) -> int:
        """Register user for the leaderboard with welcoming experience."""
        # Get user data directory
        user_data_dir = self._get_user_data_dir()
        profile_file = user_data_dir / "profile.json"
        
        # Check existing registration
        existing_profile = None
        if profile_file.exists() and not args.update:
            with open(profile_file, 'r') as f:
                existing_profile = json.load(f)
            
            self.console.print(Panel(
                f"[green]✅ You're already registered![/green]\n\n"
                f"[bold]Username:[/bold] {existing_profile.get('username', 'Unknown')}\n"
                f"[bold]Institution:[/bold] {existing_profile.get('institution', 'Not specified')}\n"
                f"[bold]Country:[/bold] {existing_profile.get('country', 'Not specified')}\n"
                f"[bold]Joined:[/bold] {existing_profile.get('joined_date', 'Unknown')}\n\n"
                f"[dim]Use --update to modify your registration[/dim]",
                title="🎉 Welcome Back!",
                border_style="green"
            ))
            return 0
        
        # Welcome new user
        if not existing_profile:
            self.console.print(Panel(
                Group(
                    Align.center("[bold bright_green]🎉 Welcome to the TinyTorch Community! 🎉[/bold bright_green]"),
                    "",
                    "You're joining a welcoming community of ML systems learners where:",
                    "• [green]Every skill level is celebrated[/green]",
                    "• [blue]Progress matters more than perfection[/blue]",
                    "• [yellow]Learning together makes us all stronger[/yellow]",
                    "",
                    "[bold]Let's get you registered![/bold]",
                ),
                border_style="bright_green",
                padding=(1, 2)
            ))
        
        # Gather registration information
        username = args.username
        if not username:
            default_username = os.getenv('USER', 'tinytorch_learner')
            username = Prompt.ask(
                "[bold]Choose your display name[/bold]",
                default=default_username
            )
        
        institution = args.institution
        if not institution:
            institution = Prompt.ask(
                "[bold]Institution/Organization[/bold] (optional, or press Enter to skip)",
                default=""
            )
        
        country = args.country
        if not country:
            country = Prompt.ask(
                "[bold]Country[/bold] (optional, helps show global community)",
                default=""
            )
        
        # Create profile
        profile = {
            "user_id": str(uuid.uuid4()),
            "username": username,
            "institution": institution or None,
            "country": country or None,
            "joined_date": datetime.now().isoformat(),
            "updated_date": datetime.now().isoformat(),
            "submissions": [],
            "achievements": [],
            "checkpoints_completed": []
        }
        
        # Save profile
        user_data_dir.mkdir(parents=True, exist_ok=True)
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        # Celebration message
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🌟 Registration Complete! 🌟[/bold bright_green]"),
                "",
                f"[bold]Welcome, {username}![/bold]",
                "",
                "You're now part of our inclusive community! Here's what you can do:",
                "• Submit any results with [green]tito leaderboard submit[/green]",
                "• View the community with [blue]tito leaderboard view[/blue]",
                "• Track your progress with [yellow]tito leaderboard profile[/yellow]",
                "",
                "[bold bright_blue]🚀 Next Steps:[/bold bright_blue]",
                "1. Train any model on CIFAR-10 (even 10% accuracy counts!)",
                "2. Submit with: [dim]tito leaderboard submit --accuracy 15.2[/dim]",
                "3. Celebrate your achievement with the community!",
                "",
                "[dim]💝 Remember: Every step forward is worth celebrating![/dim]",
            ),
            title="🎊 You're In!",
            border_style="bright_green",
            padding=(1, 2)
        ))
        
        return 0
    
    def _submit_results(self, args: Namespace) -> int:
        """Submit results with encouraging experience."""
        # Check registration
        profile = self._load_user_profile()
        if not profile:
            self.console.print(Panel(
                "[yellow]Please join our community first![/yellow]\n\n"
                "Run: [bold]tito leaderboard join[/bold]",
                title="📝 Community Membership Required",
                border_style="yellow"
            ))
            return 1
        
        # Get submission details
        task = args.task
        accuracy = args.accuracy
        model = args.model
        notes = args.notes
        checkpoint = args.checkpoint
        
        # Interactive prompts if not provided
        if accuracy is None:
            accuracy = float(Prompt.ask(
                f"[bold]Accuracy achieved on {task.upper()}[/bold] (any level welcome!)",
                default="0.0"
            ))
        
        if not model:
            model = Prompt.ask(
                "[bold]Model description[/bold] (e.g., 'CNN-3-layer', 'My-Custom-Net')",
                default="Custom Model"
            )
        
        if not checkpoint:
            checkpoint = Prompt.ask(
                "[bold]TinyTorch checkpoint completed[/bold] (e.g., '05', '10', '15')",
                default=""
            )
        
        if not notes:
            notes = Prompt.ask(
                "[bold]Notes about your approach/learnings[/bold] (optional)",
                default=""
            )
        
        # Create submission
        submission = {
            "submission_id": str(uuid.uuid4()),
            "task": task,
            "accuracy": accuracy,
            "model": model,
            "notes": notes or None,
            "checkpoint": checkpoint or None,
            "submitted_date": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Add to profile
        profile["submissions"].append(submission)
        profile["updated_date"] = datetime.now().isoformat()
        
        # Update achievements based on accuracy
        achievement = self._calculate_achievement_level(accuracy, task)
        if achievement:
            profile["achievements"].append({
                "type": "accuracy_milestone",
                "level": achievement,
                "task": task,
                "accuracy": accuracy,
                "earned_date": datetime.now().isoformat()
            })
        
        # Save updated profile
        self._save_user_profile(profile)
        
        # Celebration based on performance level
        celebration_message = self._get_celebration_message(accuracy, task, model)
        
        self.console.print(Panel(
            Group(
                Align.center(celebration_message["title"]),
                "",
                celebration_message["message"],
                "",
                f"[bold]Your Submission:[/bold]",
                f"• Task: {task.upper()}",
                f"• Accuracy: {accuracy:.1f}%",
                f"• Model: {model}",
                f"• Checkpoint: {checkpoint or 'Not specified'}",
                "",
                celebration_message["encouragement"],
                "",
                "[bold bright_blue]🔍 Next:[/bold bright_blue]",
                "• View community: [dim]tito leaderboard view[/dim]",
                "• See your profile: [dim]tito leaderboard profile[/dim]",
                "• Try improving: [dim]tito leaderboard submit[/dim]",
            ),
            title=celebration_message["panel_title"],
            border_style=celebration_message["border_style"],
            padding=(1, 2)
        ))
        
        return 0
    
    def _view_leaderboard(self, args: Namespace) -> int:
        """View community leaderboard with inclusive display."""
        task = args.task
        
        # Load community data (mock for now - would connect to real backend)
        community_data = self._load_community_data(task)
        
        if args.distribution:
            self._show_performance_distribution(community_data, task)
        elif args.recent:
            self._show_recent_achievements(community_data, task)
        else:
            self._show_community_leaderboard(community_data, task, show_all=args.all)
        
        return 0
    
    def _show_profile(self, args: Namespace) -> int:
        """Show user's personal achievement journey."""
        profile = self._load_user_profile()
        if not profile:
            self.console.print(Panel(
                "[yellow]Please join our community first to see your profile![/yellow]\n\n"
                "Run: [bold]tito leaderboard join[/bold]",
                title="📝 Community Membership Required",
                border_style="yellow"
            ))
            return 1
        
        # Create profile display
        self._display_user_profile(profile, detailed=args.detailed)
        return 0
    
    def _show_status(self, args: Namespace) -> int:
        """Show quick personal stats and encouragement."""
        profile = self._load_user_profile()
        if not profile:
            self.console.print(Panel(
                "[yellow]Please join our community first![/yellow]\n\n"
                "Run: [bold]tito leaderboard join[/bold]",
                title="📝 Community Membership Required",
                border_style="yellow"
            ))
            return 1
        
        # Calculate quick stats
        submissions = profile.get("submissions", [])
        best_cifar10 = max([s["accuracy"] for s in submissions if s["task"] == "cifar10"], default=0)
        total_submissions = len(submissions)
        
        # Encouraging status message
        if total_submissions == 0:
            status_message = "[bold bright_blue]🚀 Ready for your first submission![/bold bright_blue]"
            next_step = "Train any model and submit with: [green]tito leaderboard submit[/green]"
        else:
            status_message = f"[bold bright_green]🌟 {total_submissions} submission{'s' if total_submissions != 1 else ''} made![/bold bright_green]"
            if best_cifar10 > 0:
                next_step = f"Best CIFAR-10: {best_cifar10:.1f}% - Keep improving! 🚀"
            else:
                next_step = "Try CIFAR-10 next: [green]tito leaderboard submit --task cifar10[/green]"
        
        self.console.print(Panel(
            Group(
                Align.center(status_message),
                "",
                f"[bold]{profile['username']}[/bold]'s Quick Status:",
                f"• Total submissions: {total_submissions}",
                f"• Best CIFAR-10 accuracy: {best_cifar10:.1f}%",
                f"• Community member since: {profile.get('joined_date', 'Unknown')[:10]}",
                "",
                "[bold]Next Step:[/bold]",
                next_step,
            ),
            title="⚡ Quick Status",
            border_style="bright_blue",
            padding=(1, 2)
        ))
        
        return 0
    
    def _show_help(self) -> int:
        """Show detailed explanation of the leaderboard system."""
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🎓 TinyTorch Community Leaderboard Guide 🎓[/bold bright_green]"),
                "",
                "[bold bright_blue]🌟 What is this?[/bold bright_blue]",
                "The TinyTorch Community Leaderboard is an [bold]inclusive showcase[/bold] where ML learners",
                "share their journey, celebrate achievements, and support each other's growth.",
                "",
                "[bold bright_blue]🚀 What gets submitted?[/bold bright_blue]",
                "• [green]Model checkpoints[/green] - Your trained TinyTorch models (not PyTorch!)",
                "• [green]Accuracy results[/green] - Performance on tasks like CIFAR-10, MNIST",
                "• [green]Learning progress[/green] - Which TinyTorch checkpoints you've completed",
                "• [green]Model descriptions[/green] - Your architecture choices and approaches",
                "",
                "[bold bright_blue]🔍 How does verification work?[/bold bright_blue]",
                "• Models must be built with [yellow]TinyTorch[/yellow] (the framework you're learning)",
                "• We verify checkpoints contain your custom implementations",
                "• Community review ensures submissions are genuine learning artifacts",
                "• [dim]Focus is on learning, not gaming the system[/dim]",
                "",
                "[bold bright_blue]🌈 Community guidelines:[/bold bright_blue]",
                "• [green]Celebrate all levels[/green] - 10% accuracy gets same respect as 90%",
                "• [blue]Share knowledge[/blue] - Help others learn from your approaches",
                "• [yellow]Be encouraging[/yellow] - Everyone started as a beginner",
                "• [magenta]Learn together[/magenta] - Community success > individual ranking",
                "",
                "[bold bright_blue]🎯 Getting started:[/bold bright_blue]",
                "1. [dim]tito leaderboard join[/dim] - Join our welcoming community",
                "2. Train any model using your TinyTorch implementations",
                "3. [dim]tito leaderboard submit --accuracy 25.3[/dim] - Share your results",
                "4. [dim]tito leaderboard view[/dim] - See the community progress",
                "",
                "[bold bright_green]💝 Remember: This is about learning, growing, and supporting each other![/bold bright_green]",
            ),
            title="🤗 Community Leaderboard System",
            border_style="bright_green",
            padding=(1, 2)
        ))
        return 0
    
    def _get_user_data_dir(self) -> Path:
        """Get user data directory for leaderboard."""
        data_dir = Path.home() / ".tinytorch" / "leaderboard"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def _load_user_profile(self) -> Optional[Dict[str, Any]]:
        """Load user profile if it exists."""
        profile_file = self._get_user_data_dir() / "profile.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_user_profile(self, profile: Dict[str, Any]) -> None:
        """Save user profile."""
        profile_file = self._get_user_data_dir() / "profile.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def _calculate_achievement_level(self, accuracy: float, task: str) -> Optional[str]:
        """Calculate achievement level based on accuracy."""
        if task == "cifar10":
            if accuracy >= 75:
                return "expert"
            elif accuracy >= 50:
                return "advanced"
            elif accuracy >= 25:
                return "intermediate"
            elif accuracy >= 10:
                return "beginner"
        return None
    
    def _get_celebration_message(self, accuracy: float, task: str, model: str) -> Dict[str, str]:
        """Get appropriate celebration message based on performance."""
        if accuracy >= 75:
            return {
                "title": "[bold bright_green]🏆 OUTSTANDING ACHIEVEMENT! 🏆[/bold bright_green]",
                "message": f"[green]WOW! {accuracy:.1f}% on {task.upper()} is exceptional![/green]",
                "encouragement": "[bold]You've mastered this challenge! Consider helping others in the community. 🌟[/bold]",
                "panel_title": "🚀 Elite Performance",
                "border_style": "bright_green"
            }
        elif accuracy >= 50:
            return {
                "title": "[bold bright_blue]🎯 STRONG PERFORMANCE! 🎯[/bold bright_blue]",
                "message": f"[blue]Great work! {accuracy:.1f}% on {task.upper()} shows solid progress![/blue]",
                "encouragement": "[bold]You're doing really well! Can you push toward 75%? 💪[/bold]",
                "panel_title": "📈 Solid Progress",
                "border_style": "bright_blue"
            }
        elif accuracy >= 25:
            return {
                "title": "[bold bright_yellow]🌱 GOOD PROGRESS! 🌱[/bold bright_yellow]",
                "message": f"[yellow]Nice! {accuracy:.1f}% on {task.upper()} shows you're learning![/yellow]",
                "encouragement": "[bold]You're on the right track! Keep experimenting and improving! 🚀[/bold]",
                "panel_title": "🌟 Learning Journey",
                "border_style": "bright_yellow"
            }
        elif accuracy >= 10:
            return {
                "title": "[bold bright_magenta]🎉 FIRST STEPS! 🎉[/bold bright_magenta]",
                "message": f"[magenta]Fantastic! {accuracy:.1f}% on {task.upper()} - you've started![/magenta]",
                "encouragement": "[bold]Every expert was once a beginner! Keep going! 🌈[/bold]",
                "panel_title": "🌸 Getting Started",
                "border_style": "bright_magenta"
            }
        else:
            return {
                "title": "[bold bright_cyan]🌟 BRAVE ATTEMPT! 🌟[/bold bright_cyan]",
                "message": f"[cyan]Thank you for sharing {accuracy:.1f}% on {task.upper()}![/cyan]",
                "encouragement": "[bold]Every submission helps you learn! Try different approaches! 💡[/bold]",
                "panel_title": "💝 Courage Counts",
                "border_style": "bright_cyan"
            }
    
    def _load_community_data(self, task: str) -> List[Dict[str, Any]]:
        """Load community data (mock implementation)."""
        # Mock community data for demonstration - shows diverse community with all skill levels
        if task == "cifar10":
            return [
                {"username": "alex_chen", "accuracy": 78.2, "model": "ResNet-Custom", "country": "USA", "recent": True, "checkpoint": "15"},
                {"username": "neural_ninja", "accuracy": 72.1, "model": "CNN-5-layer", "country": "Canada", "recent": False, "checkpoint": "12"},
                {"username": "sara_codes", "accuracy": 65.8, "model": "AttentionCNN", "country": "Sweden", "recent": True, "checkpoint": "11"},
                {"username": "ml_explorer", "accuracy": 58.4, "model": "DeepCNN", "country": "Brazil", "recent": False, "checkpoint": "10"},
                {"username": "code_learner", "accuracy": 52.3, "model": "ModernCNN", "country": "South Korea", "recent": True, "checkpoint": "09"},
                {"username": "ml_student", "accuracy": 45.3, "model": "Basic-CNN", "country": "UK", "recent": True, "checkpoint": "08"},
                {"username": "data_dreamer", "accuracy": 38.9, "model": "Experimental", "country": "Netherlands", "recent": False, "checkpoint": "07"},
                {"username": "curious_coder", "accuracy": 32.7, "model": "First-CNN", "country": "Germany", "recent": True, "checkpoint": "06"},
                {"username": "ai_enthusiast", "accuracy": 27.1, "model": "SimpleNet", "country": "Japan", "recent": False, "checkpoint": "05"},
                {"username": "future_engineer", "accuracy": 22.8, "model": "LearnNet", "country": "Mexico", "recent": True, "checkpoint": "04"},
                {"username": "tinytorch_fan", "accuracy": 18.2, "model": "TryHard-Net", "country": "Australia", "recent": False, "checkpoint": "03"},
                {"username": "beginner_ml", "accuracy": 14.9, "model": "FirstModel", "country": "India", "recent": True, "checkpoint": "02"},
                {"username": "brave_starter", "accuracy": 11.3, "model": "BasicTorch", "country": "Nigeria", "recent": True, "checkpoint": "02"},
                {"username": "learning_path", "accuracy": 8.7, "model": "StartNet", "country": "Philippines", "recent": False, "checkpoint": "01"},
                {"username": "new_to_ml", "accuracy": 6.2, "model": "FirstTry", "country": "Egypt", "recent": True, "checkpoint": "01"},
            ]
        elif task == "mnist":
            return [
                {"username": "digit_master", "accuracy": 94.2, "model": "Deep-MNIST", "country": "USA", "recent": True, "checkpoint": "08"},
                {"username": "neural_ninja", "accuracy": 89.1, "model": "CNN-MNIST", "country": "Canada", "recent": False, "checkpoint": "07"},
                {"username": "ml_student", "accuracy": 76.3, "model": "Simple-CNN", "country": "UK", "recent": True, "checkpoint": "05"},
                {"username": "beginner_ml", "accuracy": 52.9, "model": "Basic-Net", "country": "India", "recent": True, "checkpoint": "03"},
            ]
        else:  # tinygpt or other tasks
            return [
                {"username": "language_lover", "accuracy": 45.2, "model": "TinyGPT-v1", "country": "USA", "recent": True, "checkpoint": "16"},
                {"username": "transformer_fan", "accuracy": 32.1, "model": "MiniTransformer", "country": "UK", "recent": False, "checkpoint": "14"},
                {"username": "nlp_explorer", "accuracy": 18.7, "model": "BasicGPT", "country": "Germany", "recent": True, "checkpoint": "13"},
            ]
    
    def _show_community_leaderboard(self, data: List[Dict[str, Any]], task: str, show_all: bool = False) -> None:
        """Show inclusive community leaderboard."""
        # Sort by accuracy but show everyone
        sorted_data = sorted(data, key=lambda x: x["accuracy"], reverse=True)
        
        if not show_all:
            # Show top performers + some middle + recent submissions
            display_data = sorted_data[:3] + sorted_data[-3:] if len(sorted_data) > 6 else sorted_data
        else:
            display_data = sorted_data
        
        # Create inclusive leaderboard table
        table = Table(title=f"🌟 {task.upper()} Community Leaderboard 🌟")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Username", style="bold")
        table.add_column("Accuracy", style="green", justify="right")
        table.add_column("Model", style="blue")
        table.add_column("Checkpoint", style="magenta", justify="center")
        table.add_column("Country", style="cyan")
        table.add_column("Status", style="yellow")
        
        for i, entry in enumerate(display_data, 1):
            rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            status = "🔥 Recent" if entry.get("recent") else "⭐"
            
            table.add_row(
                rank,
                entry["username"],
                f"{entry['accuracy']:.1f}%",
                entry["model"],
                entry.get("checkpoint", "—"),
                entry.get("country", "Global"),
                status
            )
        
        self.console.print(table)
        
        # Encouraging footer
        self.console.print(Panel(
            Group(
                "[bold bright_green]🎉 Everyone's journey matters![/bold bright_green]",
                "",
                "• [green]Top performers[/green]: Inspiring the community with excellence",
                "• [blue]All achievers[/blue]: Every percentage point represents real learning",
                "• [yellow]Recent submissions[/yellow]: Fresh progress and new insights",
                "",
                "[dim]💡 See your progress: tito leaderboard profile[/dim]",
                "[dim]🚀 Submit improvements: tito leaderboard submit[/dim]",
            ),
            title="Community Insights",
            border_style="bright_blue",
            padding=(0, 1)
        ))
    
    def _show_performance_distribution(self, data: List[Dict[str, Any]], task: str) -> None:
        """Show performance distribution to normalize achievements."""
        accuracies = [entry["accuracy"] for entry in data]
        
        # Create distribution buckets
        buckets = {
            "🏆 Expert (75%+)": len([a for a in accuracies if a >= 75]),
            "🎯 Advanced (50-75%)": len([a for a in accuracies if 50 <= a < 75]),
            "🌱 Intermediate (25-50%)": len([a for a in accuracies if 25 <= a < 50]),
            "🌸 Beginner (10-25%)": len([a for a in accuracies if 10 <= a < 25]),
            "💝 Getting Started (<10%)": len([a for a in accuracies if a < 10]),
        }
        
        table = Table(title=f"📊 {task.upper()} Performance Distribution")
        table.add_column("Achievement Level", style="bold")
        table.add_column("Community Members", style="green", justify="right")
        table.add_column("Percentage", style="blue", justify="right")
        table.add_column("Visual", style="cyan")
        
        total = len(accuracies)
        for level, count in buckets.items():
            percentage = (count / total * 100) if total > 0 else 0
            visual = "█" * min(int(percentage / 5), 20)
            
            table.add_row(
                level,
                str(count),
                f"{percentage:.1f}%",
                visual
            )
        
        self.console.print(table)
        
        self.console.print(Panel(
            "[bold bright_blue]🌈 Every level has value![/bold bright_blue]\n\n"
            "This distribution shows our diverse, learning community where:\n"
            "• [green]Every achievement level contributes to collective knowledge[/green]\n"
            "• [blue]Different backgrounds bring different insights[/blue]\n"
            "• [yellow]Progress at any level deserves celebration[/yellow]\n\n"
            "[dim]Your journey is unique and valuable regardless of where you start! 💝[/dim]",
            title="🎊 Community Insights",
            border_style="bright_green"
        ))
    
    def _show_recent_achievements(self, data: List[Dict[str, Any]], task: str) -> None:
        """Show recent achievements to motivate continued participation."""
        recent_data = [entry for entry in data if entry.get("recent", False)]
        
        if not recent_data:
            self.console.print(Panel(
                "[yellow]No recent submissions![/yellow]\n\n"
                "Be the first to share recent progress:\n"
                "[bold]tito leaderboard submit[/bold]",
                title="🔥 Recent Activity",
                border_style="yellow"
            ))
            return
        
        # Sort recent by submission order (newest first)
        recent_data = sorted(recent_data, key=lambda x: x["accuracy"], reverse=True)
        
        table = Table(title=f"🔥 Recent {task.upper()} Achievements")
        table.add_column("Achievement", style="bold")
        table.add_column("Username", style="green")
        table.add_column("Accuracy", style="blue", justify="right")
        table.add_column("Model", style="cyan")
        table.add_column("Celebration", style="yellow")
        
        for entry in recent_data[:10]:  # Show top 10 recent
            # Determine achievement type
            accuracy = entry["accuracy"]
            if accuracy >= 75:
                achievement = "🏆 Expert Level"
                celebration = "Outstanding!"
            elif accuracy >= 50:
                achievement = "🎯 Strong Performance"
                celebration = "Excellent work!"
            elif accuracy >= 25:
                achievement = "🌱 Good Progress"
                celebration = "Keep going!"
            elif accuracy >= 10:
                achievement = "🌸 First Steps"
                celebration = "Great start!"
            else:
                achievement = "💝 Brave Attempt"
                celebration = "Learning!"
            
            table.add_row(
                achievement,
                entry["username"],
                f"{accuracy:.1f}%",
                entry["model"],
                celebration
            )
        
        self.console.print(table)
        
        self.console.print(Panel(
            "[bold bright_green]🎉 Celebrating recent community progress![/bold bright_green]\n\n"
            "Join the momentum:\n"
            "• [green]Share your latest results[/green]\n"
            "• [blue]Try a new approach[/blue]\n"
            "• [yellow]Learn from others' models[/yellow]\n\n"
            "[dim]Every submission adds to our collective learning! 🚀[/dim]",
            title="🌟 Community Energy",
            border_style="bright_blue"
        ))
    
    def _display_user_profile(self, profile: Dict[str, Any], detailed: bool = False) -> None:
        """Display user's personal achievement profile."""
        username = profile.get("username", "Unknown")
        submissions = profile.get("submissions", [])
        achievements = profile.get("achievements", [])
        
        # Calculate stats
        total_submissions = len(submissions)
        best_cifar10 = max([s["accuracy"] for s in submissions if s["task"] == "cifar10"], default=0)
        best_mnist = max([s["accuracy"] for s in submissions if s["task"] == "mnist"], default=0)
        
        # Create achievement summary
        if detailed:
            self._show_detailed_profile(profile)
        else:
            self._show_summary_profile(profile)
    
    def _show_summary_profile(self, profile: Dict[str, Any]) -> None:
        """Show summary profile view."""
        username = profile.get("username", "Unknown")
        submissions = profile.get("submissions", [])
        
        # Calculate stats
        total_submissions = len(submissions)
        best_cifar10 = max([s["accuracy"] for s in submissions if s["task"] == "cifar10"], default=0)
        tasks_tried = len(set(s["task"] for s in submissions))
        
        # Determine user level
        if best_cifar10 >= 75:
            level = "🏆 Expert"
            level_color = "bright_green"
        elif best_cifar10 >= 50:
            level = "🎯 Advanced"
            level_color = "bright_blue"
        elif best_cifar10 >= 25:
            level = "🌱 Intermediate"
            level_color = "bright_yellow"
        elif best_cifar10 >= 10:
            level = "🌸 Beginner"
            level_color = "bright_magenta"
        else:
            level = "💝 Getting Started"
            level_color = "bright_cyan"
        
        self.console.print(Panel(
            Group(
                Align.center(f"[bold {level_color}]{level}[/bold {level_color}]"),
                "",
                f"[bold]{username}[/bold]'s Achievement Journey",
                "",
                f"🎯 Total submissions: {total_submissions}",
                f"🏆 Best CIFAR-10: {best_cifar10:.1f}%",
                f"🌐 Tasks explored: {tasks_tried}",
                f"📅 Member since: {profile.get('joined_date', 'Unknown')[:10]}",
                "",
                "[bold bright_blue]🚀 What's Next?[/bold bright_blue]",
                self._get_next_steps_suggestion(best_cifar10, total_submissions),
            ),
            title=f"🌟 {username}'s Profile",
            border_style=level_color,
            padding=(1, 2)
        ))
    
    def _show_detailed_profile(self, profile: Dict[str, Any]) -> None:
        """Show detailed profile with all submissions."""
        username = profile.get("username", "Unknown")
        submissions = profile.get("submissions", [])
        
        # Summary first
        self._show_summary_profile(profile)
        
        if submissions:
            # Submissions table
            table = Table(title="📈 Your Submission History")
            table.add_column("Date", style="dim")
            table.add_column("Task", style="bold")
            table.add_column("Accuracy", style="green", justify="right")
            table.add_column("Model", style="blue")
            table.add_column("Checkpoint", style="yellow")
            
            for submission in sorted(submissions, key=lambda x: x["submitted_date"], reverse=True):
                date = submission["submitted_date"][:10]
                table.add_row(
                    date,
                    submission["task"].upper(),
                    f"{submission['accuracy']:.1f}%",
                    submission["model"],
                    submission.get("checkpoint", "—")
                )
            
            self.console.print(table)
    
    def _get_next_steps_suggestion(self, best_accuracy: float, total_submissions: int) -> str:
        """Get personalized next steps suggestion."""
        if total_submissions == 0:
            return "[green]Make your first submission![/green] Any accuracy level welcome."
        elif best_accuracy >= 75:
            return "[green]Help others in the community![/green] Share your insights and approaches."
        elif best_accuracy >= 50:
            return "[blue]Push toward expert level![/blue] Can you reach 75%?"
        elif best_accuracy >= 25:
            return "[yellow]Try advanced techniques![/yellow] Explore different architectures."
        elif best_accuracy >= 10:
            return "[magenta]Experiment and learn![/magenta] Each attempt teaches something new."
        else:
            return "[cyan]Keep experimenting![/cyan] Every expert started where you are now."