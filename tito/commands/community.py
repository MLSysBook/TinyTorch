"""
Tiny🔥Torch Community Commands

Join, update, and manage your community profile for the global builder map.
"""

import json
import os
import webbrowser
import urllib.parse
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.console import Console

from .base import BaseCommand
from ..core.exceptions import TinyTorchCLIError


class CommunityCommand(BaseCommand):
    """Community commands - join, update, leave, and manage your profile."""
    
    @property
    def name(self) -> str:
        return "community"
    
    @property
    def description(self) -> str:
        return "Join the global community - connect with builders worldwide"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add community subcommands."""
        subparsers = parser.add_subparsers(
            dest='community_command',
            help='Community operations',
            metavar='COMMAND'
        )
        
        # Join command
        join_parser = subparsers.add_parser(
            'join',
            help='Join the TinyTorch community'
        )
        join_parser.add_argument(
            '--country',
            help='Your country (optional, auto-detected if possible)'
        )
        join_parser.add_argument(
            '--institution',
            help='Your institution/school (optional)'
        )
        join_parser.add_argument(
            '--course-type',
            choices=['university', 'bootcamp', 'self-paced', 'other'],
            help='Course type (optional)'
        )
        join_parser.add_argument(
            '--experience',
            choices=['beginner', 'intermediate', 'advanced', 'expert'],
            help='Experience level (optional)'
        )
        
        # Update command
        update_parser = subparsers.add_parser(
            'update',
            help='Update your community profile'
        )
        update_parser.add_argument(
            '--country',
            help='Update country'
        )
        update_parser.add_argument(
            '--institution',
            help='Update institution'
        )
        update_parser.add_argument(
            '--course-type',
            choices=['university', 'bootcamp', 'self-paced', 'other'],
            help='Update course type'
        )
        update_parser.add_argument(
            '--experience',
            choices=['beginner', 'intermediate', 'advanced', 'expert'],
            help='Update experience level'
        )
        
        # Leave command
        leave_parser = subparsers.add_parser(
            'leave',
            help='Leave the community (removes your profile)'
        )
        leave_parser.add_argument(
            '--force',
            action='store_true',
            help='Skip confirmation'
        )
        
        # Stats command
        stats_parser = subparsers.add_parser(
            'stats',
            help='View community statistics'
        )
        
        # Profile command
        profile_parser = subparsers.add_parser(
            'profile',
            help='View your community profile'
        )
    
    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: join, update, leave, stats, or profile[/yellow]")
            return 1
        
        if args.community_command == 'join':
            return self._join_community(args)
        elif args.community_command == 'update':
            return self._update_profile(args)
        elif args.community_command == 'leave':
            return self._leave_community(args)
        elif args.community_command == 'stats':
            return self._show_stats(args)
        elif args.community_command == 'profile':
            return self._show_profile(args)
        else:
            self.console.print(f"[red]Unknown community command: {args.community_command}[/red]")
            return 1
    
    def _join_community(self, args: Namespace) -> int:
        """Join the TinyTorch community - GitHub-first flow."""
        console = self.console

        # Check if already joined
        profile = self._get_profile()
        if profile:
            github_username = profile.get("github_username")
            profile_url = profile.get("profile_url", "https://tinytorch.ai/community")
            console.print(Panel(
                f"[yellow]⚠️  You're already in the community![/yellow]\n\n"
                f"GitHub: [cyan]@{github_username}[/cyan]\n"
                f"Profile: [cyan]{profile_url}[/cyan]\n\n"
                f"Update online: [cyan]{profile_url}[/cyan]\n"
                f"View profile: [cyan]tito community profile[/cyan]",
                title="Already Joined",
                border_style="yellow"
            ))
            return 0

        console.print(Panel(
            "[bold cyan]🌍 Join the TinyTorch Community[/bold cyan]\n\n"
            "Connect with ML systems builders worldwide!\n"
            "We'll ask 3 quick questions, then open your browser to complete your profile.",
            title="Welcome",
            border_style="cyan"
        ))

        console.print("\n[dim]Your data:[/dim]")
        console.print("  • Stored locally in [cyan].tinytorch/community.json[/cyan]")
        console.print("  • GitHub username for authentication")
        console.print("  • Basic info shared with community (country, institution)")
        console.print("  • Full profile completed on tinytorch.ai\n")

        # Question 1: GitHub username (REQUIRED)
        console.print("[bold]Question 1/3[/bold]")
        github_username = Prompt.ask(
            "[cyan]GitHub username[/cyan] (required for authentication)",
            default=""
        ).strip()

        if not github_username:
            console.print("[red]❌ GitHub username is required to join the community[/red]")
            console.print("[dim]Your GitHub username is used to:\n"
                        "  • Authenticate your profile\n  "
                        "  • Link to your projects\n"
                        "  • Connect with other builders[/dim]")
            return 1

        # Question 2: Country (optional, auto-detect)
        console.print("\n[bold]Question 2/3[/bold]")
        country = self._detect_country()
        if country:
            console.print(f"[dim]Auto-detected: {country}[/dim]")
        country = Prompt.ask(
            "[cyan]Country[/cyan] (for community map, optional)",
            default=country or "",
            show_default=False
        ).strip()

        # Question 3: Institution (optional)
        console.print("\n[bold]Question 3/3[/bold]")
        institution = Prompt.ask(
            "[cyan]Institution/University[/cyan] (optional)",
            default="",
            show_default=False
        ).strip()

        # Create local profile
        profile = {
            "github_username": github_username,
            "joined_at": datetime.now().isoformat(),
            "country": country or None,
            "institution": institution or None,
            "profile_url": f"https://tinytorch.ai/community/{github_username}",
            "last_synced": None
        }

        # Save profile locally
        self._save_profile(profile)

        # Build URL with pre-filled params
        base_url = "https://tinytorch.ai/community/join"
        params = {
            "github": github_username,
        }
        if country:
            params["country"] = country
        if institution:
            params["institution"] = institution

        signup_url = f"{base_url}?{urllib.parse.urlencode(params)}"

        # Show success and open browser
        console.print("\n")
        console.print(Panel(
            f"[bold green]✅ Local profile created![/bold green]\n\n"
            f"👤 GitHub: [cyan]@{github_username}[/cyan]\n"
            f"📍 Country: {country or '[dim]Not specified[/dim]'}\n"
            f"🏫 Institution: {institution or '[dim]Not specified[/dim]'}\n\n"
            f"[bold cyan]🌐 Opening browser to complete your profile...[/bold cyan]\n"
            f"[dim]URL: {signup_url}[/dim]\n\n"
            f"Complete your profile online to:\n"
            f"  • Authenticate with GitHub OAuth\n"
            f"  • Add bio, interests, and social links\n"
            f"  • Join the global community map\n"
            f"  • Connect with other builders",
            title="Almost There!",
            border_style="green"
        ))

        # Open browser
        try:
            webbrowser.open(signup_url)
            console.print("\n[green]✓[/green] Browser opened! Complete your profile there.")
        except Exception as e:
            console.print(f"\n[yellow]⚠️  Could not open browser automatically[/yellow]")
            console.print(f"[dim]Please visit: {signup_url}[/dim]")

        console.print(f"\n[dim]💡 View profile later: [cyan]tito community profile[/cyan][/dim]")

        return 0
    
    def _update_profile(self, args: Namespace) -> int:
        """Update community profile."""
        console = self.console
        
        # Get existing profile
        profile = self._get_profile()
        if not profile:
            console.print(Panel(
                "[yellow]⚠️  You're not in the community yet.[/yellow]\n\n"
                "Join first: [cyan]tito community join[/cyan]",
                title="Not Joined",
                border_style="yellow"
            ))
            return 1
        
        console.print(Panel(
            "[bold cyan]📝 Update Your Community Profile[/bold cyan]",
            title="Update Profile",
            border_style="cyan"
        ))
        
        # Update fields
        updated = False
        
        if args.country:
            profile["location"]["country"] = args.country
            updated = True
            console.print(f"[green]✅ Updated country: {args.country}[/green]")
        
        if args.institution:
            profile["institution"]["name"] = args.institution
            updated = True
            console.print(f"[green]✅ Updated institution: {args.institution}[/green]")
        
        if args.course_type:
            profile["context"]["course_type"] = args.course_type
            updated = True
            console.print(f"[green]✅ Updated course type: {args.course_type}[/green]")
        
        if args.experience:
            profile["context"]["experience_level"] = args.experience
            updated = True
            console.print(f"[green]✅ Updated experience level: {args.experience}[/green]")
        
        # If no args provided, do interactive update
        if not updated:
            console.print("\n[cyan]Interactive update (press Enter to keep current value):[/cyan]\n")
            
            # Country
            current_country = profile["location"].get("country", "")
            new_country = Prompt.ask(
                f"[cyan]Country[/cyan]",
                default=current_country or "",
                show_default=bool(current_country)
            )
            if new_country != current_country:
                profile["location"]["country"] = new_country or None
                updated = True
            
            # Institution
            current_institution = profile["institution"].get("name", "")
            new_institution = Prompt.ask(
                f"[cyan]Institution[/cyan]",
                default=current_institution or "",
                show_default=bool(current_institution)
            )
            if new_institution != current_institution:
                profile["institution"]["name"] = new_institution or None
                updated = True
        
        # Update progress if available
        self._update_progress(profile)
        
        # Save updated profile
        if updated:
            profile["updated_at"] = datetime.now().isoformat()
            self._save_profile(profile)
            console.print("\n[green]✅ Profile updated successfully![/green]")
        else:
            console.print("\n[yellow]No changes made.[/yellow]")
        
        return 0
    
    def _leave_community(self, args: Namespace) -> int:
        """Leave the community."""
        console = self.console
        
        # Get existing profile
        profile = self._get_profile()
        if not profile:
            console.print(Panel(
                "[yellow]⚠️  You're not in the community.[/yellow]",
                title="Not Joined",
                border_style="yellow"
            ))
        return 0
    
        # Confirm
        if not args.force:
            console.print(Panel(
                "[yellow]⚠️  Warning: This will remove your community profile[/yellow]\n\n"
                "This action cannot be undone.\n"
                "Your benchmark submissions will remain, but your profile will be removed.",
                title="Leave Community",
                border_style="yellow"
            ))
            
            confirm = Confirm.ask("\n[red]Are you sure you want to leave?[/red]", default=False)
            if not confirm:
                console.print("[cyan]Cancelled.[/cyan]")
                return 0
        
        # Remove profile
        profile_file = self._get_profile_file()
        if profile_file.exists():
            profile_file.unlink()
        
        # Stub: Notify website of leave
        self._notify_website_leave(profile.get("anonymous_id") if profile else None)
        
        console.print(Panel(
            "[green]✅ You've left the community.[/green]\n\n"
            "You can rejoin anytime with: [cyan]tito community join[/cyan]",
            title="Left Community",
            border_style="green"
        ))
        
        return 0
    
    def _show_stats(self, args: Namespace) -> int:
        """Show community statistics."""
        console = self.console
        
        # For now, show local stats
        # In production, this would fetch from a server
        profile = self._get_profile()
        
        console.print(Panel(
            "[bold cyan]🌍 TinyTorch Community Stats[/bold cyan]\n\n"
            "[dim]Note: Full community stats require server connection.[/dim]\n"
            "This shows your local information.",
            title="Community Stats",
            border_style="cyan"
        ))
        
        if profile:
            console.print(f"\n[cyan]Your Profile:[/cyan]")
            console.print(f"  • Country: {profile['location'].get('country', 'Not specified')}")
            console.print(f"  • Institution: {profile['institution'].get('name', 'Not specified')}")
            console.print(f"  • Course Type: {profile['context'].get('course_type', 'Not specified')}")
            console.print(f"  • Experience: {profile['context'].get('experience_level', 'Not specified')}")
            console.print(f"  • Cohort: {profile['context'].get('cohort', 'Not specified')}")
        else:
            console.print("\n[yellow]You're not in the community yet.[/yellow]")
            console.print("Join with: [cyan]tito community join[/cyan]")
        
        return 0
    
    def _show_profile(self, args: Namespace) -> int:
        """Show user's community profile."""
        console = self.console
        
        profile = self._get_profile()
        if not profile:
            console.print(Panel(
                "[yellow]⚠️  You're not in the community yet.[/yellow]\n\n"
                "Join with: [cyan]tito community join[/cyan]",
                title="Not Joined",
                border_style="yellow"
            ))
            return 1
        
        # Display profile
        profile_table = Table(title="Your Community Profile", show_header=False, box=None)
        profile_table.add_column("Field", style="cyan", width=20)
        profile_table.add_column("Value", style="green")
        
        profile_table.add_row("Anonymous ID", profile.get("anonymous_id", "N/A"))
        profile_table.add_row("Joined", self._format_date(profile.get("joined_at")))
        profile_table.add_row("Country", profile["location"].get("country", "Not specified"))
        profile_table.add_row("Institution", profile["institution"].get("name", "Not specified"))
        profile_table.add_row("Course Type", profile["context"].get("course_type", "Not specified"))
        profile_table.add_row("Experience", profile["context"].get("experience_level", "Not specified"))
        profile_table.add_row("Cohort", profile["context"].get("cohort", "Not specified"))
        
        progress = profile.get("progress", {})
        profile_table.add_row("", "")
        profile_table.add_row("[bold]Progress[/bold]", "")
        profile_table.add_row("Setup Verified", "✅" if progress.get("setup_verified") else "❌")
        profile_table.add_row("Milestones Passed", str(progress.get("milestones_passed", 0)))
        profile_table.add_row("Modules Completed", str(progress.get("modules_completed", 0)))
        capstone_score = progress.get("capstone_score")
        profile_table.add_row("Capstone Score", f"{capstone_score}/100" if capstone_score else "Not completed")
        
        console.print("\n")
        console.print(profile_table)
        
        return 0
    
    def _get_profile(self) -> Optional[Dict[str, Any]]:
        """Get user's community profile."""
        profile_file = self._get_profile_file()
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_profile(self, profile: Dict[str, Any]) -> None:
        """Save user's community profile."""
        profile_file = self._get_profile_file()
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        # Stub: Sync with website if configured
        self._sync_profile_to_website(profile)
    
    def _get_profile_file(self) -> Path:
        """Get path to profile file (project-local)."""
        return self.config.project_root / ".tinytorch" / "community" / "profile.json"
    
    def _get_config(self) -> Dict[str, Any]:
        """Get community configuration."""
        config_file = self.config.project_root / ".tinytorch" / "config.json"
        default_config = {
            "website": {
                "base_url": "https://tinytorch.ai",
                "community_map_url": "https://tinytorch.ai/community",
                "api_url": None,  # Set when API is available
                "enabled": False  # Set to True when website integration is ready
            },
            "local": {
                "enabled": True,  # Always use local storage
                "auto_sync": False  # Auto-sync to website when enabled
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
                    return default_config
            except Exception:
                pass
        
        # Create default config if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _sync_profile_to_website(self, profile: Dict[str, Any]) -> None:
        """Stub: Sync profile to website (local for now, website integration later)."""
        config = self._get_config()
        
        if not config.get("website", {}).get("enabled", False):
            # Website integration not enabled, just store locally
            return
        
        # Stub for future website API integration
        api_url = config.get("website", {}).get("api_url")
        if api_url:
            # TODO: Implement API call when website is ready
            # Example:
            # import requests
            # response = requests.post(f"{api_url}/api/community/profile", json=profile)
            # response.raise_for_status()
            pass
    
    def _detect_country(self) -> Optional[str]:
        """Try to detect country from system."""
        # Try timezone first
        try:
            import time
            tz = time.tzname[0] if time.daylight == 0 else time.tzname[1]
            # This is a simple heuristic - could be improved
            return None  # Don't auto-detect for privacy
        except Exception:
            return None
    
    def _determine_cohort(self) -> str:
        """Determine cohort based on current date."""
        now = datetime.now()
        month = now.month
        
        if month in [9, 10, 11, 12]:
            return f"Fall {now.year}"
        elif month in [1, 2, 3, 4, 5]:
            return f"Spring {now.year}"
        else:
            return f"Summer {now.year}"
    
    def _update_progress(self, profile: Dict[str, Any]) -> None:
        """Update progress information from local data."""
        # Check milestone progress
        milestone_file = Path(".tito") / "milestones.json"
        if milestone_file.exists():
            try:
                with open(milestone_file, 'r') as f:
                    milestones_data = json.load(f)
                    completed = milestones_data.get("completed_milestones", [])
                    profile["progress"]["milestones_passed"] = len(completed)
            except Exception:
                pass
        
        # Check module progress
        progress_file = Path(".tito") / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed = progress_data.get("completed_modules", [])
                    profile["progress"]["modules_completed"] = len(completed)
            except Exception:
                pass
        
        # Check capstone score
        benchmark_dir = Path(".tito") / "benchmarks"
        if benchmark_dir.exists():
            # Find latest capstone benchmark
            capstone_files = sorted(benchmark_dir.glob("capstone_*.json"), reverse=True)
            if capstone_files:
                try:
                    with open(capstone_files[0], 'r') as f:
                        capstone_data = json.load(f)
                        profile["progress"]["capstone_score"] = capstone_data.get("overall_score")
                except Exception:
                    pass
    
    def _format_date(self, date_str: Optional[str]) -> str:
        """Format ISO date string."""
        if not date_str:
            return "N/A"
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return date_str
    
    def _notify_website_join(self, profile: Dict[str, Any]) -> None:
        """Stub: Notify website when user joins (local for now, website integration later)."""
        config = self._get_config()
        
        if not config.get("website", {}).get("enabled", False):
            # Website integration not enabled
            return
        
        api_url = config.get("website", {}).get("api_url")
        if api_url:
            # TODO: Implement API call when website is ready
            # Example:
            # import requests
            # try:
            #     response = requests.post(
            #         f"{api_url}/api/community/join",
            #         json=profile,
            #         timeout=10,  # 10 second timeout
            #         headers={"Content-Type": "application/json"}
            #     )
            #     response.raise_for_status()
            # except requests.Timeout:
            #     self.console.print("[dim]Note: Website sync timed out. Your data is saved locally.[/dim]")
            # except requests.RequestException as e:
            #     # Log error but don't fail the command
            #     self.console.print(f"[dim]Note: Could not sync with website: {e}[/dim]")
            #     self.console.print("[dim]Your data is saved locally and can be synced later.[/dim]")
            pass
    
    def _notify_website_leave(self, anonymous_id: Optional[str]) -> None:
        """Stub: Notify website when user leaves (local for now, website integration later)."""
        config = self._get_config()
        
        if not config.get("website", {}).get("enabled", False):
            # Website integration not enabled
            return
        
        api_url = config.get("website", {}).get("api_url")
        if api_url and anonymous_id:
            # TODO: Implement API call when website is ready
            # Example:
            # import requests
            # try:
            #     response = requests.post(
            #         f"{api_url}/api/community/leave",
            #         json={"anonymous_id": anonymous_id},
            #         timeout=10,  # 10 second timeout
            #         headers={"Content-Type": "application/json"}
            #     )
            #     response.raise_for_status()
            # except requests.Timeout:
            #     self.console.print("[dim]Note: Website sync timed out. Profile removed locally.[/dim]")
            # except requests.RequestException as e:
            #     # Log error but don't fail the command
            #     self.console.print(f"[dim]Note: Could not sync with website: {e}[/dim]")
            #     self.console.print("[dim]Profile removed locally.[/dim]")
            pass
