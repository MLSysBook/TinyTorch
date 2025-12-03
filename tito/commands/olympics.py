"""
TinyTorch Olympics Command

Special competition events with focused challenges, time-limited competitions,
and unique recognition opportunities beyond the regular community leaderboard.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.prompt import Prompt, Confirm
from rich.console import Group
from rich.align import Align

from .base import BaseCommand
from ..core.exceptions import TinyTorchCLIError


class OlympicsCommand(BaseCommand):
    """Special competition events - Focused challenges and recognition"""
    
    @property
    def name(self) -> str:
        return "olympics"
    
    @property
    def description(self) -> str:
        return "Special competition events with unique challenges and recognition"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add olympics subcommands."""
        subparsers = parser.add_subparsers(
            dest='olympics_command',
            help='Olympics operations',
            metavar='COMMAND'
        )
        
        # Events command
        events_parser = subparsers.add_parser(
            'events',
            help='View current and upcoming competition events'
        )
        events_parser.add_argument(
            '--upcoming',
            action='store_true',
            help='Show only upcoming events'
        )
        events_parser.add_argument(
            '--past',
            action='store_true',
            help='Show past competition results'
        )
        
        # Compete command
        compete_parser = subparsers.add_parser(
            'compete',
            help='Enter a specific competition event'
        )
        compete_parser.add_argument(
            '--event',
            required=True,
            help='Event ID to compete in'
        )
        compete_parser.add_argument(
            '--accuracy',
            type=float,
            help='Accuracy achieved for this competition'
        )
        compete_parser.add_argument(
            '--model',
            help='Model description and approach used'
        )
        compete_parser.add_argument(
            '--code-url',
            help='Optional: Link to your competition code/approach'
        )
        compete_parser.add_argument(
            '--notes',
            help='Competition-specific notes, innovations, learnings'
        )
        
        # Awards command
        awards_parser = subparsers.add_parser(
            'awards',
            help='View special recognition and achievement badges'
        )
        awards_parser.add_argument(
            '--personal',
            action='store_true',
            help='Show only your personal awards'
        )
        
        # History command
        history_parser = subparsers.add_parser(
            'history',
            help='View past competition events and memorable moments'
        )
        history_parser.add_argument(
            '--year',
            type=int,
            help='Filter by specific year'
        )
        history_parser.add_argument(
            '--event-type',
            choices=['speed', 'accuracy', 'innovation', 'efficiency', 'community'],
            help='Filter by event type'
        )
    
    def run(self, args: Namespace) -> int:
        """Execute olympics command."""
        command = getattr(args, 'olympics_command', None)
        
        if not command:
            self._show_olympics_overview()
            return 0
        
        if command == 'events':
            return self._show_events(args)
        elif command == 'compete':
            return self._compete_in_event(args)
        elif command == 'awards':
            return self._show_awards(args)
        elif command == 'history':
            return self._show_history(args)
        else:
            raise TinyTorchCLIError(f"Unknown olympics command: {command}")
    
    def _show_olympics_overview(self) -> None:
        """Show olympics overview and current special events."""
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_gold]🏅 TinyTorch Olympics 🏅[/bold bright_gold]"),
                "",
                "[bold]Special Competition Events![/bold] Beyond the regular community leaderboard:",
                "",
                "🎯 [bold bright_blue]Focused Challenges[/bold bright_blue]",
                "  • Time-limited competitions (24hr, 1week, 1month challenges)",
                "  • Specific constraints (memory-efficient, fastest training, novel architectures)",
                "  • Theme-based events (interpretability, fairness, efficiency)",
                "",
                "🏆 [bold bright_yellow]Special Recognition[/bold bright_yellow]",
                "  • Olympic medals and achievement badges",
                "  • Innovation awards for creative approaches",
                "  • Community impact recognition",
                "",
                "🌟 [bold bright_green]Current Active Events[/bold bright_green]",
                "  • Winter 2024 Speed Challenge (Training under 5 minutes)",
                "  • Memory Efficiency Olympics (Models under 1MB)",
                "  • Architecture Innovation Contest (Novel designs welcome)",
                "",
                "[bold]Available Commands:[/bold]",
                "  [green]events[/green]   - See current and upcoming competitions",
                "  [green]compete[/green]  - Enter a specific event",
                "  [green]awards[/green]   - View special recognition and badges",
                "  [green]history[/green]  - Past competitions and memorable moments",
                "",
                "[dim]💡 Note: Olympics are special events separate from daily community leaderboard[/dim]",
            ),
            title="🥇 Competition Central",
            border_style="bright_yellow",
            padding=(1, 2)
        ))
    
    def _show_events(self, args: Namespace) -> int:
        """Show current and upcoming competition events."""
        # Load events data (mock for now)
        events = self._load_olympics_events()
        
        if args.upcoming:
            events = [e for e in events if e["status"] == "upcoming"]
            title = "📅 Upcoming Competition Events"
        elif args.past:
            events = [e for e in events if e["status"] == "completed"]
            title = "🏛️ Past Competition Results"
        else:
            title = "🏅 All Competition Events"
        
        if not events:
            status_text = "upcoming" if args.upcoming else "past" if args.past else "available"
            self.console.print(Panel(
                f"[yellow]No {status_text} events at this time![/yellow]\n\n"
                "Check back soon for new competition opportunities!",
                title="📅 No Events",
                border_style="yellow"
            ))
            return 0
        
        # Create events table
        table = Table(title=title)
        table.add_column("Event", style="bold")
        table.add_column("Type", style="blue")
        table.add_column("Duration", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Prize/Recognition", style="bright_magenta")
        table.add_column("Participants", style="cyan", justify="right")
        
        for event in events:
            status_display = self._get_status_display(event["status"], event.get("end_date"))
            
            table.add_row(
                event["name"],
                event["type"],
                event["duration"],
                status_display,
                event["prize"],
                str(event.get("participants", 0))
            )
        
        self.console.print(table)
        
        # Show active event details
        active_events = [e for e in events if e["status"] == "active"]
        if active_events:
            self.console.print(Panel(
                Group(
                    "[bold bright_green]🔥 Active Competitions You Can Join Now![/bold bright_green]",
                    "",
                    *[f"• [bold]{event['name']}[/bold]: {event['description']}" for event in active_events[:3]],
                    "",
                    "[bold]Join a competition:[/bold]",
                    "[dim]tito olympics compete --event <event_id>[/dim]",
                ),
                title="⚡ Join Now",
                border_style="bright_green",
                padding=(0, 1)
            ))
        
        return 0
    
    def _compete_in_event(self, args: Namespace) -> int:
        """Enter a competition event."""
        # Check if user is registered for leaderboard
        if not self._is_user_registered():
            self.console.print(Panel(
                "[yellow]Please register for the community leaderboard first![/yellow]\n\n"
                "Olympics competitions require community membership:\n"
                "[bold]tito leaderboard register[/bold]",
                title="📝 Registration Required",
                border_style="yellow"
            ))
            return 1
        
        # Load event details
        event = self._get_event_details(args.event)
        if not event:
            self.console.print(Panel(
                f"[red]Event '{args.event}' not found![/red]\n\n"
                "See available events: [bold]tito olympics events[/bold]",
                title="❌ Event Not Found",
                border_style="red"
            ))
            return 1
        
        # Check if event is active
        if event["status"] != "active":
            self.console.print(Panel(
                f"[yellow]Event '{event['name']}' is not currently active![/yellow]\n\n"
                f"Status: {event['status']}\n"
                "See active events: [bold]tito olympics events[/bold]",
                title="⏰ Event Not Active",
                border_style="yellow"
            ))
            return 1
        
        # Show event details and confirm participation
        self._show_event_details(event)
        
        if not Confirm.ask("\n[bold]Compete in this event?[/bold]"):
            self.console.print("[dim]Maybe next time! 👋[/dim]")
            return 0
        
        # Gather competition submission
        submission = self._gather_competition_submission(event, args)
        
        # Validate submission meets event criteria
        validation_result = self._validate_submission(event, submission)
        if not validation_result["valid"]:
            self.console.print(Panel(
                f"[red]Submission doesn't meet event criteria![/red]\n\n"
                f"Issue: {validation_result['reason']}\n\n"
                "Please check event requirements and try again.",
                title="❌ Validation Failed",
                border_style="red"
            ))
            return 1
        
        # Save competition entry
        self._save_competition_entry(event, submission)
        
        # Show competition confirmation and standing
        self._show_competition_confirmation(event, submission)
        
        return 0
    
    def _show_awards(self, args: Namespace) -> int:
        """Show special recognition and achievement badges."""
        if args.personal:
            return self._show_personal_awards()
        else:
            return self._show_all_awards()
    
    def _show_personal_awards(self) -> int:
        """Show user's personal awards and badges."""
        if not self._is_user_registered():
            self.console.print(Panel(
                "[yellow]Please register first to see your awards![/yellow]\n\n"
                "Run: [bold]tito leaderboard register[/bold]",
                title="📝 Registration Required",
                border_style="yellow"
            ))
            return 1
        
        # Load user's Olympic achievements
        olympic_profile = self._load_user_olympic_profile()
        awards = olympic_profile.get("awards", [])
        competitions = olympic_profile.get("competitions", [])
        
        if not awards and not competitions:
            self.console.print(Panel(
                Group(
                    "[bold bright_blue]🌟 Your Olympic Journey Awaits![/bold bright_blue]",
                    "",
                    "You haven't participated in Olympics competitions yet.",
                    "",
                    "[bold]Start your journey:[/bold]",
                    "• Check active events: [green]tito olympics events[/green]",
                    "• Join a competition: [green]tito olympics compete --event <id>[/green]",
                    "• Earn your first Olympic badge! 🏅",
                    "",
                    "[dim]Every Olympic participant gets recognition for participation![/dim]",
                ),
                title="🏅 Your Olympic Profile",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            return 0
        
        # Show awards and achievements
        self._display_personal_olympic_achievements(olympic_profile)
        return 0
    
    def _show_all_awards(self) -> int:
        """Show community awards and notable achievements."""
        # Mock awards data
        notable_awards = self._load_notable_awards()
        
        # Recent awards table
        table = Table(title="🏆 Recent Olympic Achievements")
        table.add_column("Award", style="bold")
        table.add_column("Recipient", style="green")
        table.add_column("Event", style="blue")
        table.add_column("Achievement", style="yellow")
        table.add_column("Date", style="dim")
        
        for award in notable_awards[:10]:
            table.add_row(
                award["award_type"],
                award["recipient"],
                award["event"],
                award["description"],
                award["date"]
            )
        
        self.console.print(table)
        
        # Award categories explanation
        self.console.print(Panel(
            Group(
                "[bold bright_yellow]🏅 Olympic Award Categories[/bold bright_yellow]",
                "",
                "🥇 [bold]Performance Awards[/bold]",
                "  • Gold/Silver/Bronze medals for top competition results",
                "  • Speed records, accuracy achievements, efficiency milestones",
                "",
                "🌟 [bold]Innovation Awards[/bold]",
                "  • Novel Architecture Award for creative model designs",
                "  • Optimization Genius for breakthrough efficiency techniques",
                "  • Interpretability Champion for explainable AI contributions",
                "",
                "🤝 [bold]Community Awards[/bold]",
                "  • Mentor Badge for helping other competitors",
                "  • Knowledge Sharer for valuable insights and tutorials",
                "  • Sportsperson Award for exceptional community spirit",
                "",
                "🎯 [bold]Special Recognition[/bold]",
                "  • First Participation Badge (everyone gets this!)",
                "  • Consistency Award for regular competition participation",
                "  • Breakthrough Achievement for major personal improvements",
            ),
            title="🏆 Recognition System",
            border_style="bright_yellow",
            padding=(0, 1)
        ))
        
        return 0
    
    def _show_history(self, args: Namespace) -> int:
        """Show past competition events and memorable moments."""
        # Load historical data
        history = self._load_olympics_history()
        
        # Filter by year if specified
        if args.year:
            history = [h for h in history if h["year"] == args.year]
        
        # Filter by event type if specified
        if args.event_type:
            history = [h for h in history if h["type"] == args.event_type]
        
        if not history:
            filter_text = f" for {args.year}" if args.year else ""
            filter_text += f" ({args.event_type} events)" if args.event_type else ""
            
            self.console.print(Panel(
                f"[yellow]No competition history found{filter_text}![/yellow]\n\n"
                "The Olympics program is just getting started!",
                title="📚 No History",
                border_style="yellow"
            ))
            return 0
        
        # Create history table
        table = Table(title="📚 TinyTorch Olympics History")
        table.add_column("Event", style="bold")
        table.add_column("Date", style="dim")
        table.add_column("Type", style="blue")
        table.add_column("Winner", style="green")
        table.add_column("Achievement", style="yellow")
        table.add_column("Memorable Moment", style="cyan")
        
        for event in sorted(history, key=lambda x: x["date"], reverse=True):
            table.add_row(
                event["name"],
                event["date"],
                event["type"],
                event["winner"],
                event["winning_achievement"],
                event["memorable_moment"]
            )
        
        self.console.print(table)
        
        # Show legendary moments
        if not args.year and not args.event_type:
            self.console.print(Panel(
                Group(
                    "[bold bright_gold]🌟 Legendary Olympic Moments[/bold bright_gold]",
                    "",
                    "🏆 [bold]The Great Speed Challenge 2024[/bold]",
                    "   Winner achieved 75% CIFAR-10 accuracy in just 47 seconds!",
                    "",
                    "🧠 [bold]Architecture Innovation Contest[/bold]",
                    "   Revolutionary attention mechanism reduced parameters by 90%",
                    "",
                    "🤝 [bold]Community Spirit Award[/bold]",
                    "   Competitor shared winning code to help others improve",
                    "",
                    "[dim]Each Olympics creates new legends in the TinyTorch community! 💫[/dim]",
                ),
                title="🏛️ Hall of Fame",
                border_style="bright_gold",
                padding=(0, 1)
            ))
        
        return 0
    
    def _load_olympics_events(self) -> List[Dict[str, Any]]:
        """Load olympics events data (mock implementation)."""
        return [
            {
                "id": "winter2024_speed",
                "name": "Winter 2024 Speed Challenge",
                "type": "Speed",
                "status": "active",
                "duration": "24 hours",
                "description": "Train CIFAR-10 model to 70%+ accuracy in under 5 minutes",
                "prize": "🏆 Speed Medal + Recognition",
                "participants": 23,
                "start_date": "2024-01-15",
                "end_date": "2024-01-16",
                "criteria": {"min_accuracy": 70.0, "max_time_minutes": 5}
            },
            {
                "id": "memory2024_efficiency",
                "name": "Memory Efficiency Olympics",
                "type": "Efficiency",
                "status": "active", 
                "duration": "1 week",
                "description": "Best CIFAR-10 accuracy with model under 1MB",
                "prize": "🥇 Efficiency Champion",
                "participants": 15,
                "start_date": "2024-01-10",
                "end_date": "2024-01-17",
                "criteria": {"max_model_size_mb": 1.0}
            },
            {
                "id": "innovation2024_arch",
                "name": "Architecture Innovation Contest",
                "type": "Innovation",
                "status": "upcoming",
                "duration": "2 weeks",
                "description": "Novel architectures and creative approaches welcome",
                "prize": "🌟 Innovation Award",
                "participants": 0,
                "start_date": "2024-02-01",
                "end_date": "2024-02-14",
                "criteria": {"novelty_required": True}
            },
            {
                "id": "autumn2023_classic",
                "name": "Autumn 2023 Classic",
                "type": "Accuracy",
                "status": "completed",
                "duration": "1 month",
                "description": "Best overall CIFAR-10 accuracy challenge",
                "prize": "🥇 Gold Medal",
                "participants": 87,
                "start_date": "2023-10-01",
                "end_date": "2023-10-31",
                "winner": "neural_champion",
                "winning_score": 84.2
            }
        ]
    
    def _get_status_display(self, status: str, end_date: Optional[str] = None) -> str:
        """Get display-friendly status with timing information."""
        if status == "active":
            if end_date:
                # Calculate time remaining
                end = datetime.fromisoformat(end_date)
                now = datetime.now()
                if end > now:
                    remaining = end - now
                    if remaining.days > 0:
                        return f"🔥 Active ({remaining.days}d left)"
                    else:
                        hours = remaining.seconds // 3600
                        return f"🔥 Active ({hours}h left)"
            return "🔥 Active"
        elif status == "upcoming":
            return "📅 Upcoming"
        elif status == "completed":
            return "✅ Completed"
        else:
            return status.title()
    
    def _is_user_registered(self) -> bool:
        """Check if user is registered for community leaderboard."""
        from .leaderboard import LeaderboardCommand
        leaderboard_cmd = LeaderboardCommand(self.config)
        return leaderboard_cmd._load_user_profile() is not None
    
    def _get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific event."""
        events = self._load_olympics_events()
        return next((e for e in events if e["id"] == event_id), None)
    
    def _show_event_details(self, event: Dict[str, Any]) -> None:
        """Show detailed information about an event."""
        self.console.print(Panel(
            Group(
                f"[bold bright_blue]{event['name']}[/bold bright_blue]",
                "",
                f"[bold]Type:[/bold] {event['type']}",
                f"[bold]Duration:[/bold] {event['duration']}",
                f"[bold]Current Participants:[/bold] {event.get('participants', 0)}",
                "",
                f"[bold]Challenge:[/bold]",
                f"  {event['description']}",
                "",
                f"[bold]Recognition:[/bold]",
                f"  {event['prize']}",
                "",
                f"[bold]Requirements:[/bold]",
                *[f"  • {k.replace('_', ' ').title()}: {v}" for k, v in event.get('criteria', {}).items()],
            ),
            title=f"🏅 {event['type']} Competition",
            border_style="bright_blue",
            padding=(1, 2)
        ))
    
    def _gather_competition_submission(self, event: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
        """Gather submission details for competition."""
        submission = {
            "event_id": event["id"],
            "submitted_date": datetime.now().isoformat()
        }
        
        # Get accuracy
        if args.accuracy is not None:
            submission["accuracy"] = args.accuracy
        else:
            submission["accuracy"] = float(Prompt.ask(
                f"[bold]Accuracy achieved on {event.get('dataset', 'the task')}[/bold]",
                default="0.0"
            ))
        
        # Get model description
        if args.model:
            submission["model"] = args.model
        else:
            submission["model"] = Prompt.ask(
                "[bold]Model description[/bold] (architecture, approach, innovations)",
                default="Custom Model"
            )
        
        # Optional fields
        submission["code_url"] = args.code_url or Prompt.ask(
            "[bold]Code/approach URL[/bold] (optional)",
            default=""
        ) or None
        
        submission["notes"] = args.notes or Prompt.ask(
            "[bold]Competition notes[/bold] (innovations, challenges, learnings)",
            default=""
        ) or None
        
        # Event-specific metrics
        if "max_time_minutes" in event.get("criteria", {}):
            training_time = float(Prompt.ask(
                "[bold]Training time in minutes[/bold]",
                default="0.0"
            ))
            submission["training_time_minutes"] = training_time
        
        if "max_model_size_mb" in event.get("criteria", {}):
            model_size = float(Prompt.ask(
                "[bold]Model size in MB[/bold]",
                default="0.0"
            ))
            submission["model_size_mb"] = model_size
        
        return submission
    
    def _validate_submission(self, event: Dict[str, Any], submission: Dict[str, Any]) -> Dict[str, Any]:
        """Validate submission meets event criteria."""
        criteria = event.get("criteria", {})
        
        # Check minimum accuracy
        if "min_accuracy" in criteria:
            if submission["accuracy"] < criteria["min_accuracy"]:
                return {
                    "valid": False,
                    "reason": f"Accuracy {submission['accuracy']:.1f}% below required {criteria['min_accuracy']:.1f}%"
                }
        
        # Check maximum training time
        if "max_time_minutes" in criteria:
            if submission.get("training_time_minutes", 0) > criteria["max_time_minutes"]:
                return {
                    "valid": False,
                    "reason": f"Training time {submission['training_time_minutes']:.1f}min exceeds limit {criteria['max_time_minutes']:.1f}min"
                }
        
        # Check maximum model size
        if "max_model_size_mb" in criteria:
            if submission.get("model_size_mb", 0) > criteria["max_model_size_mb"]:
                return {
                    "valid": False,
                    "reason": f"Model size {submission['model_size_mb']:.1f}MB exceeds limit {criteria['max_model_size_mb']:.1f}MB"
                }
        
        return {"valid": True}
    
    def _save_competition_entry(self, event: Dict[str, Any], submission: Dict[str, Any]) -> None:
        """Save competition entry to user's Olympic profile."""
        olympic_profile = self._load_user_olympic_profile()
        
        if "competitions" not in olympic_profile:
            olympic_profile["competitions"] = []
        
        olympic_profile["competitions"].append(submission)
        
        # Add participation award if first competition
        if len(olympic_profile["competitions"]) == 1:
            award = {
                "type": "participation",
                "name": "First Olympic Participation",
                "description": "Welcomed to the Olympics community!",
                "event": event["name"],
                "earned_date": datetime.now().isoformat()
            }
            if "awards" not in olympic_profile:
                olympic_profile["awards"] = []
            olympic_profile["awards"].append(award)
        
        self._save_user_olympic_profile(olympic_profile)
    
    def _show_competition_confirmation(self, event: Dict[str, Any], submission: Dict[str, Any]) -> None:
        """Show confirmation and current standing."""
        # Determine performance level for this competition
        ranking_message = self._get_competition_ranking_message(event, submission)
        
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🎉 Competition Entry Submitted! 🎉[/bold bright_green]"),
                "",
                f"[bold]Event:[/bold] {event['name']}",
                f"[bold]Your Result:[/bold] {submission['accuracy']:.1f}% accuracy",
                f"[bold]Model:[/bold] {submission['model']}",
                "",
                ranking_message,
                "",
                "[bold bright_blue]🏅 Recognition Earned:[/bold bright_blue]",
                "• Olympic Participant Badge",
                "• Competition Experience Points",
                "• Community Recognition",
                "",
                "[bold]Next Steps:[/bold]",
                "• View your awards: [green]tito olympics awards --personal[/green]",
                "• See current standings: [green]tito olympics events[/green]",
                "• Join another event: [green]tito olympics events[/green]",
            ),
            title="🥇 Olympic Achievement",
            border_style="bright_green",
            padding=(1, 2)
        ))
    
    def _get_competition_ranking_message(self, event: Dict[str, Any], submission: Dict[str, Any]) -> str:
        """Get appropriate ranking/performance message for competition."""
        accuracy = submission["accuracy"]
        
        # Mock competition standings for encouragement
        if accuracy >= 80:
            return "[bright_green]🏆 Outstanding performance! You're in contention for top prizes![/bright_green]"
        elif accuracy >= 70:
            return "[bright_blue]🎯 Strong showing! You're competing well in this event![/bright_blue]"
        elif accuracy >= 60:
            return "[bright_yellow]🌟 Good effort! Every competition teaches valuable lessons![/bright_yellow]"
        else:
            return "[bright_magenta]💝 Thank you for participating! Competition experience is valuable![/bright_magenta]"
    
    def _load_user_olympic_profile(self) -> Dict[str, Any]:
        """Load user's Olympic competition profile."""
        data_dir = Path.home() / ".tinytorch" / "olympics"
        data_dir.mkdir(parents=True, exist_ok=True)
        profile_file = data_dir / "olympic_profile.json"
        
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        
        return {
            "competitions": [],
            "awards": [],
            "created_date": datetime.now().isoformat()
        }
    
    def _save_user_olympic_profile(self, profile: Dict[str, Any]) -> None:
        """Save user's Olympic competition profile."""
        data_dir = Path.home() / ".tinytorch" / "olympics"
        profile_file = data_dir / "olympic_profile.json"
        
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def _display_personal_olympic_achievements(self, olympic_profile: Dict[str, Any]) -> None:
        """Display user's personal Olympic achievements."""
        competitions = olympic_profile.get("competitions", [])
        awards = olympic_profile.get("awards", [])
        
        # Summary stats
        total_competitions = len(competitions)
        best_accuracy = max([c["accuracy"] for c in competitions], default=0)
        events_participated = len(set(c["event_id"] for c in competitions))
        
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_gold]🏅 Your Olympic Journey 🏅[/bold bright_gold]"),
                "",
                f"🎯 Competitions Entered: {total_competitions}",
                f"🏆 Best Performance: {best_accuracy:.1f}% accuracy",
                f"🌟 Events Participated: {events_participated}",
                f"🥇 Awards Earned: {len(awards)}",
            ),
            title="📊 Olympic Stats",
            border_style="bright_gold",
            padding=(1, 2)
        ))
        
        # Awards table
        if awards:
            awards_table = Table(title="🏆 Your Olympic Awards")
            awards_table.add_column("Award", style="bold")
            awards_table.add_column("Event", style="blue")
            awards_table.add_column("Description", style="green")
            awards_table.add_column("Date", style="dim")
            
            for award in sorted(awards, key=lambda x: x["earned_date"], reverse=True):
                awards_table.add_row(
                    award["name"],
                    award["event"],
                    award["description"],
                    award["earned_date"][:10]
                )
            
            self.console.print(awards_table)
        
        # Recent competitions
        if competitions:
            recent_comps = sorted(competitions, key=lambda x: x["submitted_date"], reverse=True)[:5]
            
            comps_table = Table(title="🎯 Recent Competition Entries")
            comps_table.add_column("Event", style="bold")
            comps_table.add_column("Accuracy", style="green", justify="right")
            comps_table.add_column("Model", style="blue")
            comps_table.add_column("Date", style="dim")
            
            for comp in recent_comps:
                comps_table.add_row(
                    comp["event_id"],
                    f"{comp['accuracy']:.1f}%",
                    comp["model"],
                    comp["submitted_date"][:10]
                )
            
            self.console.print(comps_table)
    
    def _load_notable_awards(self) -> List[Dict[str, Any]]:
        """Load notable community awards (mock implementation)."""
        return [
            {
                "award_type": "🥇 Gold Medal",
                "recipient": "speed_demon",
                "event": "Winter 2024 Speed Challenge",
                "description": "2.3 min training, 78.4% accuracy",
                "date": "2024-01-16"
            },
            {
                "award_type": "🌟 Innovation Award",
                "recipient": "arch_wizard",
                "event": "Memory Efficiency Olympics",
                "description": "Novel attention mechanism",
                "date": "2024-01-15"
            },
            {
                "award_type": "🤝 Community Spirit",
                "recipient": "helpful_mentor",
                "event": "Autumn 2023 Classic",
                "description": "Shared winning approach publicly",
                "date": "2023-11-01"
            },
            {
                "award_type": "🏆 Speed Record",
                "recipient": "lightning_fast",
                "event": "Winter 2024 Speed Challenge",
                "description": "47 second training record",
                "date": "2024-01-15"
            },
            {
                "award_type": "🎯 Accuracy Champion",
                "recipient": "precision_master",
                "event": "Architecture Innovation",
                "description": "86.7% CIFAR-10 accuracy",
                "date": "2024-01-10"
            }
        ]
    
    def _load_olympics_history(self) -> List[Dict[str, Any]]:
        """Load historical Olympics data (mock implementation)."""
        return [
            {
                "name": "Autumn 2023 Classic",
                "date": "2023-10-31",
                "year": 2023,
                "type": "accuracy",
                "winner": "neural_champion",
                "winning_achievement": "84.2% CIFAR-10 accuracy",
                "memorable_moment": "First 80%+ achievement in community"
            },
            {
                "name": "Summer 2023 Speed Trial",
                "date": "2023-07-15",
                "year": 2023,
                "type": "speed",
                "winner": "velocity_victor",
                "winning_achievement": "3.2 minute training",
                "memorable_moment": "Breakthrough GPU optimization technique"
            },
            {
                "name": "Spring 2023 Innovation Fair",
                "date": "2023-04-20",
                "year": 2023,
                "type": "innovation",
                "winner": "creative_genius",
                "winning_achievement": "Self-organizing architecture",
                "memorable_moment": "Inspired 12 follow-up research papers"
            }
        ]