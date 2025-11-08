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
from .checkpoint import CheckpointSystem


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
        
        # Progress command (module completion progress)
        progress_parser = subparsers.add_parser(
            'progress',
            help='Show learning progress and module completion'
        )
        progress_parser.add_argument(
            '--all',
            action='store_true',
            help='Show all module completion details'
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
        elif command == 'progress':
            return self._show_progress(args)
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
                "  [green]progress[/green]  - Track your module completion journey",
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
        """Register user for the leaderboard with comprehensive guided experience."""
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
        
        # Start the guided experience
        return self._guided_registration_experience(args, existing_profile)
    
    def _guided_registration_experience(self, args: Namespace, existing_profile: Optional[Dict[str, Any]]) -> int:
        """Comprehensive guided registration with progressive disclosure and Rich UI."""
        from rich.progress import Progress, BarColumn, TextColumn
        from rich.prompt import Prompt, Confirm
        import time
        import platform
        
        # Welcome message with community invitation
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🎉 Welcome to TinyTorch Community! 🎉[/bold bright_green]"),
                "",
                "Join thousands of ML systems learners from around the world who are:",
                "• [green]Building neural networks from scratch[/green]",
                "• [blue]Sharing progress and celebrating achievements[/blue]",
                "• [yellow]Learning together and supporting each other[/yellow]",
                "• [magenta]Making ML systems education accessible to everyone[/magenta]",
                "",
                "[bold]Let's get you connected! (Takes about 2 minutes)[/bold]",
                "",
                "[dim]This helps us build an inclusive global community map and match you with peers[/dim]",
            ),
            title="🌍 Join the Global ML Community",
            border_style="bright_green",
            padding=(1, 2)
        ))
        
        # Initialize profile data
        profile_data = {
            "user_id": str(uuid.uuid4()),
            "joined_date": datetime.now().isoformat(),
            "updated_date": datetime.now().isoformat(),
            "submissions": [],
            "achievements": [],
            "checkpoints_completed": [],
            "modules_completed": [],
            "progress_percentage": 0,
            "checkpoints_unlocked": [],
            "eligible_submissions": ["mnist"],  # MNIST available by default
            "next_module": "01_setup"
        }
        
        # Check if updating existing profile
        if existing_profile:
            profile_data.update(existing_profile)
            self.console.print("[dim]Updating your existing profile...[/dim]\n")
        
        # Step 1: Basic Identity (with progress tracking)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            registration_task = progress.add_task("Registration Progress", total=4)
            
            # Step 1: Basic Identity
            progress.update(registration_task, description="[cyan]Step 1/4: Basic Identity[/cyan]")
            self.console.print(Panel(
                Group(
                    "[bold bright_blue]🏷️ Step 1 of 4: Tell us about you[/bold bright_blue]",
                    "",
                    "This helps create your community identity and enables authentication for submissions.",
                ),
                title="Basic Identity",
                border_style="bright_blue",
                padding=(1, 1)
            ))
            
            # Display name
            if args.username:
                display_name = args.username
            else:
                default_name = existing_profile.get('username') if existing_profile else os.getenv('USER', 'tinytorch_learner')
                display_name = Prompt.ask(
                    "\n[bold]What should we call you?[/bold]",
                    default=default_name,
                    show_default=True
                )
            
            # GitHub username
            github_username = Prompt.ask(
                "[bold]GitHub username[/bold] (for submissions and PR authentication)",
                default=existing_profile.get('github_username', '') if existing_profile else ''
            )
            
            # Email (optional)
            email = Prompt.ask(
                "[bold]Email[/bold] (optional - for important community updates only, never shared)",
                default=existing_profile.get('email', '') if existing_profile else '',
                show_default=False
            )
            
            progress.advance(registration_task)
            
            # Step 2: Location for Community Map
            progress.update(registration_task, description="[cyan]Step 2/4: Location (Community Map)[/cyan]")
            self.console.print(Panel(
                Group(
                    "[bold bright_blue]🌍 Step 2 of 4: Help build our global community map[/bold bright_blue]",
                    "",
                    "We create beautiful visualizations showing our global ML learning community!",
                    "[dim]All location data is used only for community analytics and maps.[/dim]",
                ),
                title="Location & Community Map",
                border_style="bright_blue",
                padding=(1, 1)
            ))
            
            # Country (required for map)
            if args.country:
                country = args.country
            else:
                country = Prompt.ask(
                    "\n[bold]Country[/bold] (required - for global community map)",
                    default=existing_profile.get('country', '') if existing_profile else ''
                )
            
            # City/State (optional)
            city = Prompt.ask(
                "[bold]City, State/Province[/bold] (optional - for detailed regional view)",
                default=existing_profile.get('city', '') if existing_profile else '',
                show_default=False
            )
            
            # Auto-detect timezone for event scheduling
            try:
                import time
                timezone = time.tzname[0]
                self.console.print(f"[dim]Auto-detected timezone: {timezone}[/dim]")
            except:
                timezone = Prompt.ask(
                    "[bold]Timezone[/bold] (for event scheduling, e.g., 'EST', 'UTC', 'PST')",
                    default="UTC"
                )
            
            progress.advance(registration_task)
            
            # Step 3: Learning Context
            progress.update(registration_task, description="[cyan]Step 3/4: Learning Context[/cyan]")
            self.console.print(Panel(
                Group(
                    "[bold bright_blue]🎓 Step 3 of 4: Your learning journey[/bold bright_blue]",
                    "",
                    "This helps us understand our community demographics and create better content!",
                ),
                title="Learning Context",
                border_style="bright_blue",
                padding=(1, 1)
            ))
            
            # Institution/Company (optional)
            if args.institution:
                institution = args.institution
            else:
                institution = Prompt.ask(
                    "\n[bold]Institution/Company[/bold] (optional - helps show academic vs industry split)",
                    default=existing_profile.get('institution', '') if existing_profile else '',
                    show_default=False
                )
            
            # Role
            role_choices = ['Student', 'Professional', 'Educator', 'Hobbyist', 'Researcher']
            role = existing_profile.get('role', '') if existing_profile else ''
            while role not in role_choices:
                self.console.print(f"\n[bold]Role[/bold]: {', '.join(role_choices)}")
                role = Prompt.ask(
                    "Which best describes you",
                    choices=[r.lower() for r in role_choices],
                    default='student'
                ).title()
            
            # Experience level
            experience_choices = ['Beginner', 'Some ML', 'Experienced']
            experience = existing_profile.get('experience_level', '') if existing_profile else ''
            while experience not in experience_choices:
                self.console.print(f"\n[bold]Experience with ML[/bold]: {', '.join(experience_choices)}")
                experience = Prompt.ask(
                    "Your ML background",
                    choices=[e.lower().replace(' ', '_') for e in experience_choices],
                    default='beginner'
                ).replace('_', ' ').title()
            
            progress.advance(registration_task)
            
            # Step 4: Learning Goals & Community Preferences
            progress.update(registration_task, description="[cyan]Step 4/4: Goals & Community[/cyan]")
            self.console.print(Panel(
                Group(
                    "[bold bright_blue]🎯 Step 4 of 4: Goals and community preferences[/bold bright_blue]",
                    "",
                    "Help us connect you with the right learning opportunities and peers!",
                ),
                title="Goals & Community Preferences",
                border_style="bright_blue",
                padding=(1, 1)
            ))
            
            # Primary interest
            interest_choices = ['Computer Vision', 'NLP', 'Systems', 'General']
            interest = Prompt.ask(
                "\n[bold]Primary ML interest[/bold]",
                choices=[i.lower().replace(' ', '_') for i in interest_choices],
                default='general'
            ).replace('_', ' ').title()
            
            # Time commitment
            commitment_choices = ['Casual', 'Part-time', 'Intensive']
            commitment = Prompt.ask(
                "[bold]Time commitment[/bold]",
                choices=[c.lower() for c in commitment_choices],
                default='casual'
            ).title()
            
            # Learning goal
            goal_choices = ['Understanding', 'Career', 'Research', 'Fun']
            goal = Prompt.ask(
                "[bold]Primary goal[/bold]",
                choices=[g.lower() for g in goal_choices],
                default='understanding'
            ).title()
            
            # Community preferences
            study_partners = Confirm.ask("\n[bold]Open to finding study partners?[/bold]", default=True)
            help_others = Confirm.ask("[bold]Willing to help other learners?[/bold]", default=True)
            competitions = Confirm.ask("[bold]Interested in friendly competitions/challenges?[/bold]", default=True)
            
            progress.advance(registration_task)
            progress.update(registration_task, description="[green]Registration Complete![/green]")
        
        # Build complete profile
        profile_data.update({
            "username": display_name,
            "github_username": github_username or None,
            "email": email or None,
            "country": country,
            "city": city or None,
            "timezone": timezone,
            "institution": institution or None,
            "role": role,
            "experience_level": experience,
            "primary_interest": interest,
            "time_commitment": commitment,
            "learning_goal": goal,
            "community_preferences": {
                "study_partners": study_partners,
                "help_others": help_others,
                "competitions": competitions
            }
        })
        
        # Save profile with progress indication
        with self.console.status("[cyan]Saving your profile and connecting to community..."):
            user_data_dir = self._get_user_data_dir()
            user_data_dir.mkdir(parents=True, exist_ok=True)
            profile_file = user_data_dir / "profile.json"
            
            with open(profile_file, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            time.sleep(1)  # Brief pause for effect
        
        # Personalized celebration and next steps
        self._show_personalized_welcome(profile_data)
        
        return 0
    
    def _show_personalized_welcome(self, profile: Dict[str, Any]) -> None:
        """Show personalized welcome message based on user profile."""
        username = profile["username"]
        role = profile.get("role", "Learner")
        experience = profile.get("experience_level", "Beginner")
        interest = profile.get("primary_interest", "General")
        country = profile.get("country", "Global")
        
        # Personalize welcome based on profile
        if experience == "Beginner":
            experience_msg = "Perfect! TinyTorch is designed to take you from zero to ML systems expert."
            next_step = "Start with Module 01 (Setup) to configure your learning environment."
        elif experience == "Some ML":
            experience_msg = "Great! You'll love building ML systems from the ground up."
            next_step = "Jump to Module 02 (Tensors) to start building your own framework."
        else:  # Experienced
            experience_msg = "Excellent! You'll appreciate the systems engineering focus."
            next_step = "Explore our advanced modules or try the TinyGPT capstone project."
        
        if interest == "Computer Vision":
            interest_msg = "You'll love our CNN modules and CIFAR-10 training challenges!"
        elif interest == "NLP":
            interest_msg = "Check out our attention mechanisms and TinyGPT implementation!"
        elif interest == "Systems":
            interest_msg = "Perfect match! Every module emphasizes systems engineering principles."
        else:
            interest_msg = "Our comprehensive curriculum covers all aspects of ML systems!"
        
        # Community connections
        community_size = 1247  # Mock community size - would be real in production
        country_peers = 47 if country != "Global" else 0  # Mock country peers
        
        community_msg = f"You're now part of {community_size:,} learners worldwide"
        if country_peers > 0:
            community_msg += f", including {country_peers} from {country}"
        community_msg += "!"
        
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🌟 Welcome to the Community, {}! 🌟[/bold bright_green]".format(username)),
                "",
                f"[bold]🎓 {role} | {experience} | {interest}[/bold]",
                "",
                community_msg,
                "",
                "[bold bright_blue]✨ Personalized for You:[/bold bright_blue]",
                f"• [green]{experience_msg}[/green]",
                f"• [blue]{interest_msg}[/blue]",
                "",
                "[bold bright_blue]🚀 Your Next Steps:[/bold bright_blue]",
                f"1. [green]{next_step}[/green]",
                "2. [blue]Train your first model and share results[/blue]",
                "3. [yellow]Connect with peers: tito leaderboard view[/yellow]",
                "",
                "[bold bright_blue]🤝 Community Features Unlocked:[/bold bright_blue]",
                "• [green]Submit results: tito leaderboard submit[/green]",
                "• [blue]View global progress: tito leaderboard view[/blue]",
                "• [yellow]Track your journey: tito leaderboard profile[/yellow]",
                "• [magenta]Get encouragement: tito leaderboard status[/magenta]",
                "",
                "[dim]💝 Remember: Every step forward is celebrated here![/dim]",
            ),
            title=f"🎊 Welcome to TinyTorch, {username}!",
            border_style="bright_green",
            padding=(1, 2)
        ))
        
        # Show community preview
        self.console.print(Panel(
            Group(
                "[bold bright_blue]👥 Meet Your Community:[/bold bright_blue]",
                "",
                "Recent achievements from learners like you:",
                "• [green]Alex (Student, Canada)[/green]: 78% CIFAR-10 accuracy - 'Finally beat the 75% goal!'",
                "• [blue]Sara (Professional, Sweden)[/blue]: 65% CIFAR-10 - 'Attention mechanisms are amazing!'",
                "• [yellow]Maria (Beginner, Brazil)[/yellow]: 23% CIFAR-10 - 'First CNN working, so excited!'",
                "",
                "[dim]Join the conversation and share your progress![/dim]",
            ),
            title="🌍 Global Community Highlights",
            border_style="cyan",
            padding=(1, 1)
        ))
    
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
        
        # NEW: Check for prerequisite validation
        prerequisite_check = self._validate_submission_prerequisites(task, profile)
        if not prerequisite_check["valid"]:
            self.console.print(Panel(
                f"[yellow]⚠️  Submission accepted but prerequisites not fully met[/yellow]\n\n"
                f"[bold]Task:[/bold] {task.upper()}\n"
                f"[yellow]Missing prerequisites:[/yellow]\n" +
                "\n".join(f"  • {req}" for req in prerequisite_check["missing"]) +
                f"\n\n[cyan]Complete these modules to unlock full {task} capabilities![/cyan]",
                title="Prerequisites Check",
                border_style="yellow"
            ))
        
        # Save updated profile
        self._save_user_profile(profile)
        
        # NEW: Auto-update leaderboard progress
        self._update_leaderboard_progress_on_submission(profile, submission)
        
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
    
    def _show_progress(self, args: Namespace) -> int:
        """Show learning progress and module completion status."""
        profile = self._load_user_profile()
        if not profile:
            self.console.print(Panel(
                "[yellow]Please join our community first to see your progress![/yellow]\n\n"
                "Run: [bold]tito leaderboard join[/bold]",
                title="📝 Community Membership Required",
                border_style="yellow"
            ))
            return 1
        
        username = profile.get("username", "Unknown")
        modules_completed = profile.get("modules_completed", [])
        progress_percentage = profile.get("progress_percentage", 0)
        next_module = profile.get("next_module")
        eligible_submissions = profile.get("eligible_submissions", ["mnist"])
        checkpoints_unlocked = profile.get("checkpoints_unlocked", [])
        
        # Progress overview
        bar_width = 40
        filled = int((progress_percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        self.console.print(Panel(
            f"[bold bright_cyan]🚀 {username}'s Learning Journey[/bold bright_cyan]\n\n"
            f"[bold yellow]Module Progress:[/bold yellow]\n"
            f"[{bar}] {progress_percentage:.1f}%\n"
            f"[green]Completed:[/green] {len(modules_completed)}/16 modules\n"
            f"[blue]Checkpoints:[/blue] {len(checkpoints_unlocked)}/16 unlocked\n\n"
            f"[bold cyan]Next Adventure:[/bold cyan]\n" +
            (f"[yellow]📖 {next_module}[/yellow]\n" if next_module else 
             f"[green]🏆 All modules completed![/green]\n") +
            f"\n[bold magenta]Unlocked Capabilities:[/bold magenta]\n" +
            "• " + ", ".join(task.upper() for task in eligible_submissions),
            title="📊 Progress Dashboard",
            border_style="bright_cyan"
        ))
        
        # Detailed module breakdown if requested
        if args.all and modules_completed:
            # Load checkpoint system for detailed info
            checkpoint_system = CheckpointSystem(self.config)
            
            module_table = Table(title="📚 Detailed Module Completion History")
            module_table.add_column("Date", style="dim")
            module_table.add_column("Module", style="bold cyan")
            module_table.add_column("Checkpoint", style="yellow", justify="center")
            module_table.add_column("Capability Unlocked", style="green")
            
            for completion in sorted(modules_completed, key=lambda x: x["completed"]):
                date = completion["completed"]
                module = completion["module"]
                checkpoint = completion.get("checkpoint")
                
                # Get capability description
                capability = "Module Completed"
                if checkpoint is not None:
                    checkpoint_data = checkpoint_system.CHECKPOINTS.get(f"{checkpoint:02d}")
                    if checkpoint_data:
                        capability = checkpoint_data["name"]
                
                module_table.add_row(
                    date,
                    module,
                    f"#{checkpoint}" if checkpoint is not None else "—",
                    capability
                )
            
            self.console.print(module_table)
        
        # Quick action suggestions
        self.console.print(Panel(
            f"[bold cyan]🎯 Quick Actions[/bold cyan]\n\n" +
            (f"[green]Continue Learning:[/green]\n[dim]  tito module view {next_module}[/dim]\n\n" if next_module else "") +
            f"[yellow]Submit Results:[/yellow]\n[dim]  tito leaderboard submit --task mnist --accuracy XX.X[/dim]\n\n"
            f"[blue]View Community:[/blue]\n[dim]  tito leaderboard view[/dim]\n\n"
            f"[magenta]Track Progress:[/magenta]\n[dim]  tito checkpoint status[/dim]",
            title="🚀 Next Steps",
            border_style="bright_blue"
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
                {"username": "language_lover", "accuracy": 45.2, "model": "TinyGPT-v1", "country": "USA", "recent": True, "checkpoint": "15"},
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
        
        # Create inclusive leaderboard table with module progress
        table = Table(title=f"🌟 {task.upper()} Community Leaderboard 🌟")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Username", style="bold")
        table.add_column("Accuracy", style="green", justify="right")
        table.add_column("Model", style="blue")
        table.add_column("Progress", style="yellow", justify="center")
        table.add_column("Country", style="cyan")
        table.add_column("Status", style="magenta")
        
        for i, entry in enumerate(display_data, 1):
            rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            status = "🔥 Recent" if entry.get("recent") else "⭐"
            
            # Calculate progress display (mock data for now - would be real in production)
            checkpoint = entry.get("checkpoint", "—")
            if checkpoint != "—":
                progress_percent = min(int(checkpoint) * 6.25, 100)  # 16 checkpoints = 100%
                progress = f"{progress_percent:.0f}%"
            else:
                progress = "—"
            
            table.add_row(
                rank,
                entry["username"],
                f"{entry['accuracy']:.1f}%",
                entry["model"],
                progress,
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
        """Show summary profile view with enhanced module completion tracking."""
        username = profile.get("username", "Unknown")
        submissions = profile.get("submissions", [])
        modules_completed = profile.get("modules_completed", [])
        progress_percentage = profile.get("progress_percentage", 0)
        next_module = profile.get("next_module")
        eligible_submissions = profile.get("eligible_submissions", ["mnist"])
        
        # Calculate stats
        total_submissions = len(submissions)
        best_cifar10 = max([s["accuracy"] for s in submissions if s["task"] == "cifar10"], default=0)
        tasks_tried = len(set(s["task"] for s in submissions))
        total_modules_completed = len(modules_completed)
        
        # Determine user level based on modules + performance
        if total_modules_completed >= 12 or best_cifar10 >= 75:
            level = "🏆 Expert"
            level_color = "bright_green"
        elif total_modules_completed >= 8 or best_cifar10 >= 50:
            level = "🎯 Advanced"
            level_color = "bright_blue"
        elif total_modules_completed >= 4 or best_cifar10 >= 25:
            level = "🌱 Intermediate"
            level_color = "bright_yellow"
        elif total_modules_completed >= 1 or best_cifar10 >= 10:
            level = "🌸 Learning"
            level_color = "bright_magenta"
        else:
            level = "💝 Getting Started"
            level_color = "bright_cyan"
        
        # Progress bar visualization
        bar_width = 25
        filled = int((progress_percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        self.console.print(Panel(
            Group(
                Align.center(f"[bold {level_color}]{level}[/bold {level_color}]"),
                "",
                f"[bold]{username}[/bold]'s ML Systems Journey",
                "",
                f"[bold cyan]📚 Learning Progress:[/bold cyan]",
                f"[{bar}] {progress_percentage:.1f}% ({total_modules_completed}/16 modules)",
                "",
                f"[bold green]🎯 Submissions:[/bold green]",
                f"• Total submissions: {total_submissions}",
                f"• Best CIFAR-10: {best_cifar10:.1f}%",
                f"• Tasks explored: {tasks_tried}",
                "",
                f"[bold yellow]🔓 Unlocked Submissions:[/bold yellow]",
                "• " + ", ".join(task.upper() for task in eligible_submissions),
                "",
                f"[bold blue]📅 Member since:[/bold blue] {profile.get('joined_date', 'Unknown')[:10]}",
                "",
                "[bold bright_blue]🚀 What's Next?[/bold bright_blue]",
                self._get_enhanced_next_steps_suggestion(next_module, best_cifar10, total_submissions),
            ),
            title=f"🌟 {username}'s Profile",
            border_style=level_color,
            padding=(1, 2)
        ))
    
    def _show_detailed_profile(self, profile: Dict[str, Any]) -> None:
        """Show detailed profile with all submissions and module progress."""
        username = profile.get("username", "Unknown")
        submissions = profile.get("submissions", [])
        modules_completed = profile.get("modules_completed", [])
        
        # Summary first
        self._show_summary_profile(profile)
        
        # Module completion history
        if modules_completed:
            module_table = Table(title="📚 Module Completion Journey")
            module_table.add_column("Date", style="dim")
            module_table.add_column("Module", style="bold cyan")
            module_table.add_column("Checkpoint", style="yellow", justify="center")
            module_table.add_column("Capability", style="green")
            
            # Load checkpoint system for capability descriptions
            checkpoint_system = CheckpointSystem(self.config)
            
            for module_completion in sorted(modules_completed, key=lambda x: x["completed"], reverse=True):
                date = module_completion["completed"]
                module = module_completion["module"]
                checkpoint = module_completion.get("checkpoint")
                
                # Get capability description if checkpoint is known
                capability = "Module Completed"
                if checkpoint is not None:
                    checkpoint_data = checkpoint_system.CHECKPOINTS.get(f"{checkpoint:02d}")
                    if checkpoint_data:
                        capability = checkpoint_data["name"]
                
                module_table.add_row(
                    date,
                    module,
                    f"#{checkpoint}" if checkpoint is not None else "—",
                    capability
                )
            
            self.console.print(module_table)
        
        # Submissions table
        if submissions:
            submission_table = Table(title="🎯 Submission History")
            submission_table.add_column("Date", style="dim")
            submission_table.add_column("Task", style="bold")
            submission_table.add_column("Accuracy", style="green", justify="right")
            submission_table.add_column("Model", style="blue")
            submission_table.add_column("Notes", style="yellow")
            
            for submission in sorted(submissions, key=lambda x: x["submitted_date"], reverse=True):
                date = submission["submitted_date"][:10]
                notes = submission.get("notes", "")
                notes_display = (notes[:30] + "...") if len(notes) > 30 else notes
                
                submission_table.add_row(
                    date,
                    submission["task"].upper(),
                    f"{submission['accuracy']:.1f}%",
                    submission["model"],
                    notes_display or "—"
                )
            
            self.console.print(submission_table)
    
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
    
    def _get_enhanced_next_steps_suggestion(self, next_module: Optional[str], best_accuracy: float, total_submissions: int) -> str:
        """Get enhanced next steps suggestion combining modules and submissions."""
        suggestions = []
        
        # Module-based suggestions
        if next_module:
            suggestions.append(f"[green]Continue learning:[/green] {next_module}")
            suggestions.append(f"[dim]  tito module view {next_module}[/dim]")
        else:
            suggestions.append("[green]🏆 All modules complete![/green] You're an ML Systems Engineer!")
        
        # Submission-based suggestions
        if total_submissions == 0:
            suggestions.append("[yellow]Make your first submission:[/yellow] Any accuracy level welcome!")
            suggestions.append("[dim]  tito leaderboard submit --task mnist --accuracy XX.X[/dim]")
        elif best_accuracy >= 75:
            suggestions.append("[blue]Help others in the community:[/blue] Share your insights!")
        elif best_accuracy >= 50:
            suggestions.append("[yellow]Push toward expert level:[/yellow] Can you reach 75%?")
        else:
            suggestions.append("[yellow]Keep improving:[/yellow] Every experiment teaches something!")
        
        return "\n".join(suggestions)
    
    def update_profile_on_module_completion(self, module_name: str, checkpoint_unlocked: Optional[str] = None) -> None:
        """Update leaderboard profile when a module is completed via 'tito module complete'."""
        profile = self._load_user_profile()
        
        if not profile:
            # User hasn't joined leaderboard yet - show gentle invitation
            self.console.print(Panel(
                f"[bold green]✨ Great progress on {module_name}! ✨[/bold green]\n\n"
                f"[yellow]Join the TinyTorch community to track your journey:[/yellow]\n"
                f"  • Show your progress to the world\n"
                f"  • Get personalized next step suggestions\n"
                f"  • Connect with other ML learners\n"
                f"  • Celebrate achievements together\n\n"
                f"[bold cyan]Ready to join?[/bold cyan]\n"
                f"[dim]  tito leaderboard join[/dim]",
                title="🎆 Join the Community",
                border_style="bright_green"
            ))
            return
        
        # Update module completion tracking
        self._add_module_completion(profile, module_name, checkpoint_unlocked)
        
        # Recalculate progress metrics
        self._update_progress_metrics(profile)
        
        # Update submission eligibility
        self._update_submission_eligibility(profile)
        
        # Save updated profile
        self._save_user_profile(profile)
        
        # Show progress celebration
        self._show_module_completion_celebration(profile, module_name, checkpoint_unlocked)
    
    def _add_module_completion(self, profile: Dict[str, Any], module_name: str, checkpoint_unlocked: Optional[str]) -> None:
        """Add module completion to profile."""
        modules_completed = profile.get("modules_completed", [])
        
        # Check if already completed
        existing_completion = None
        for completion in modules_completed:
            if completion["module"] == module_name:
                existing_completion = completion
                break
        
        completion_data = {
            "module": module_name,
            "completed": datetime.now().isoformat()[:10],  # YYYY-MM-DD format
            "checkpoint": int(checkpoint_unlocked) if checkpoint_unlocked else None
        }
        
        if existing_completion:
            # Update existing completion
            existing_completion.update(completion_data)
        else:
            # Add new completion
            modules_completed.append(completion_data)
        
        profile["modules_completed"] = modules_completed
        
        # Update checkpoints unlocked
        if checkpoint_unlocked:
            checkpoints_unlocked = set(profile.get("checkpoints_unlocked", []))
            checkpoints_unlocked.add(int(checkpoint_unlocked))
            profile["checkpoints_unlocked"] = sorted(list(checkpoints_unlocked))
    
    def _update_progress_metrics(self, profile: Dict[str, Any]) -> None:
        """Update progress percentage and next module suggestion."""
        modules_completed = profile.get("modules_completed", [])
        total_modules = 16  # 01_setup through 16_tinygpt
        
        # Calculate progress percentage
        progress_percentage = (len(modules_completed) / total_modules) * 100
        profile["progress_percentage"] = round(progress_percentage, 1)
        
        # Determine next module
        completed_module_names = {comp["module"] for comp in modules_completed}
        
        for i in range(1, 17):
            module_name = f"{i:02d}_{self._get_module_short_name(i)}"
            if module_name not in completed_module_names:
                profile["next_module"] = module_name
                break
        else:
            profile["next_module"] = None  # All modules completed!
    
    def _get_module_short_name(self, module_num: int) -> str:
        """Get short name for module number."""
        module_names = {
            1: "setup", 2: "tensor", 3: "activations", 4: "layers", 5: "dense",
            6: "spatial", 7: "attention", 8: "dataloader", 9: "autograd", 10: "optimizers",
            11: "training", 12: "compression", 13: "kernels", 14: "benchmarking",
            15: "mlops", 16: "tinygpt"
        }
        return module_names.get(module_num, "unknown")
    
    def _update_submission_eligibility(self, profile: Dict[str, Any]) -> None:
        """Update which submissions the user is eligible for based on completed modules."""
        modules_completed = {comp["module"] for comp in profile.get("modules_completed", [])}
        eligible_submissions = ["mnist"]  # Always available
        
        # CIFAR-10 requires spatial modules (basic CNN pipeline)
        cifar10_prereqs = {"06_spatial", "08_dataloader", "09_autograd", "11_training"}
        if all(module in modules_completed for module in cifar10_prereqs):
            eligible_submissions.append("cifar10")
        
        # TinyGPT requires language/transformer modules
        tinygpt_prereqs = {"07_attention", "09_autograd", "11_training"}
        if all(module in modules_completed for module in tinygpt_prereqs):
            eligible_submissions.append("tinygpt")
        
        profile["eligible_submissions"] = eligible_submissions
    
    def _validate_submission_prerequisites(self, task: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if user has prerequisites for a submission task."""
        modules_completed = {comp["module"] for comp in profile.get("modules_completed", [])}
        
        if task == "mnist":
            # MNIST is always available (basic neural networks)
            required = {"05_dense"}  # Need basic dense networks
            missing = required - modules_completed
            return {"valid": len(missing) == 0, "missing": list(missing)}
        
        elif task == "cifar10":
            # CIFAR-10 requires CNN pipeline
            required = {"06_spatial", "08_dataloader", "09_autograd", "11_training"}
            missing = required - modules_completed
            return {"valid": len(missing) == 0, "missing": list(missing)}
        
        elif task == "tinygpt":
            # TinyGPT requires attention and training
            required = {"07_attention", "09_autograd", "11_training"}
            missing = required - modules_completed
            return {"valid": len(missing) == 0, "missing": list(missing)}
        
        # Unknown task - allow but warn
        return {"valid": True, "missing": []}
    
    def _update_leaderboard_progress_on_submission(self, profile: Dict[str, Any], submission: Dict[str, Any]) -> None:
        """Update leaderboard progress tracking when user submits results."""
        # This method is called when users submit results
        # It can provide helpful feedback about their progress
        
        task = submission["task"]
        accuracy = submission["accuracy"]
        
        # Check if this is a meaningful improvement
        previous_submissions = [s for s in profile.get("submissions", []) if s["task"] == task]
        if previous_submissions:
            best_previous = max(s["accuracy"] for s in previous_submissions[:-1])  # Exclude current
            if accuracy > best_previous:
                improvement = accuracy - best_previous
                self.console.print(Panel(
                    f"[bold green]🚀 Progress Update![/bold green]\n\n"
                    f"[yellow]Improvement on {task.upper()}:[/yellow] +{improvement:.1f}% accuracy!\n"
                    f"[cyan]Previous best:[/cyan] {best_previous:.1f}%\n"
                    f"[green]New best:[/green] {accuracy:.1f}%\n\n"
                    f"[bold]Keep pushing forward! 🎆[/bold]",
                    title="📈 Performance Boost",
                    border_style="bright_green"
                ))
    
    def _show_module_completion_celebration(self, profile: Dict[str, Any], module_name: str, checkpoint_unlocked: Optional[str]) -> None:
        """Show celebration for module completion with progress visualization."""
        console = self.console
        username = profile.get("username", "Learner")
        progress_percentage = profile.get("progress_percentage", 0)
        next_module = profile.get("next_module")
        
        # Progress bar visualization
        bar_width = 30
        filled = int((progress_percentage / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Celebrate the achievement
        console.print(Panel(
            f"[bold bright_green]🎉 Module Completion Recorded! 🎉[/bold bright_green]\n\n"
            f"[bold]{username}'s Progress Updated:[/bold]\n"
            f"[green]✓ Module:[/green] {module_name}\n" +
            (f"[green]✓ Checkpoint:[/green] {checkpoint_unlocked} unlocked\n" if checkpoint_unlocked else "") +
            f"\n[bold cyan]Overall Progress:[/bold cyan]\n"
            f"[{bar}] {progress_percentage:.1f}%\n\n" +
            (f"[bold yellow]🎯 Next Adventure:[/bold yellow] {next_module}\n" if next_module else 
             f"[bold green]🏆 All modules completed! You're an ML Systems Engineer![/bold green]\n") +
            f"\n[bold cyan]View your journey:[/bold cyan]\n"
            f"[dim]  tito leaderboard profile --detailed[/dim]\n"
            f"[dim]  tito leaderboard status[/dim]",
            title=f"🎆 {username}'s Achievement",
            border_style="bright_green"
        ))
        
        # Show submission eligibility update
        eligible_submissions = profile.get("eligible_submissions", [])
        if len(eligible_submissions) > 1:  # More than just MNIST
            console.print(Panel(
                f"[bold cyan]🔓 Unlocked Submissions:[/bold cyan]\n\n" +
                "\n".join(f"  • [green]{task.upper()}[/green]" for task in eligible_submissions) +
                f"\n\n[yellow]Ready to submit your results?[/yellow]\n"
                f"[dim]  tito leaderboard submit --task cifar10 --accuracy XX.X[/dim]",
                title="🏅 New Opportunities",
                border_style="bright_cyan"
            ))