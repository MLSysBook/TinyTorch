"""
Tiny🔥Torch Interactive Help System

Provides contextual, progressive guidance for new and experienced users.
"""

from argparse import ArgumentParser, Namespace
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

from .base import BaseCommand
from ..core.config import CLIConfig
from ..core.console import get_console
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm


class HelpCommand(BaseCommand):
    """Interactive help and onboarding system."""
    
    @property
    def name(self) -> str:
        return "help"
    
    @property
    def description(self) -> str:
        return "Interactive help system with guided onboarding"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add help command arguments."""
        parser.add_argument(
            'topic', 
            nargs='?', 
            help='Specific help topic (getting-started, commands, workflow, etc.)'
        )
        parser.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Launch interactive onboarding wizard'
        )
        parser.add_argument(
            '--quick', '-q',
            action='store_true',
            help='Show quick reference card'
        )
    
    def run(self, args: Namespace) -> int:
        """Execute help command."""
        console = get_console()
        
        # Interactive onboarding wizard
        if args.interactive:
            return self._interactive_onboarding()
        
        # Quick reference
        if args.quick:
            return self._show_quick_reference()
        
        # Topic-specific help
        if args.topic:
            return self._show_topic_help(args.topic)
        
        # Default: Show main help with user context
        return self._show_contextual_help()
    
    def _interactive_onboarding(self) -> int:
        """Launch interactive onboarding wizard."""
        console = get_console()
        
        # Welcome screen
        console.print(Panel.fit(
            "[bold blue]🚀 Welcome to Tiny🔥Torch![/bold blue]\n\n"
            "Let's get you started on your ML systems engineering journey.\n"
            "This quick wizard will help you understand what Tiny🔥Torch is\n"
            "and guide you to the right starting point.",
            title="Tiny🔥Torch Onboarding Wizard",
            border_style="blue"
        ))
        
        # User experience assessment
        experience = self._assess_user_experience()
        
        # Learning goal identification
        goals = self._identify_learning_goals()
        
        # Time commitment assessment
        time_commitment = self._assess_time_commitment()
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(experience, goals, time_commitment)
        
        # Show personalized path
        self._show_personalized_path(recommendations)
        
        # Offer to start immediately
        if Confirm.ask("\n[bold green]Ready to start your first steps?[/bold green]"):
            self._launch_first_steps(recommendations)
        
        return 0
    
    def _assess_user_experience(self) -> str:
        """Assess user's ML and programming experience."""
        console = get_console()
        
        console.print("\n[bold cyan]📋 Quick Experience Assessment[/bold cyan]")
        
        choices = [
            "New to ML and Python - need fundamentals",
            "Know Python, new to ML - want to learn systems",
            "Use PyTorch/TensorFlow - want to understand internals", 
            "ML Engineer - need to debug/optimize production systems",
            "Instructor - want to teach this course"
        ]
        
        console.print("\nWhat best describes your background?")
        for i, choice in enumerate(choices, 1):
            console.print(f"  {i}. {choice}")
        
        while True:
            try:
                selection = int(Prompt.ask("\nEnter your choice (1-5)"))
                if 1 <= selection <= 5:
                    return ['beginner', 'python_user', 'framework_user', 'ml_engineer', 'instructor'][selection-1]
                else:
                    console.print("[red]Please enter a number between 1-5[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")
    
    def _identify_learning_goals(self) -> List[str]:
        """Identify user's learning goals."""
        console = get_console()
        
        console.print("\n[bold cyan]🎯 Learning Goals[/bold cyan]")
        console.print("What do you want to achieve? (Select all that apply)")
        
        goals = [
            ("understand_internals", "Understand how PyTorch/TensorFlow work internally"),
            ("build_networks", "Build neural networks from scratch"),
            ("optimize_performance", "Learn to optimize ML system performance"),
            ("debug_production", "Debug production ML systems"),
            ("teach_course", "Teach ML systems to others"),
            ("career_transition", "Transition from software engineering to ML"),
            ("research_custom", "Implement custom operations for research")
        ]
        
        selected_goals = []
        for key, description in goals:
            if Confirm.ask(f"  • {description}?"):
                selected_goals.append(key)
        
        return selected_goals
    
    def _assess_time_commitment(self) -> str:
        """Assess available time commitment."""
        console = get_console()
        
        console.print("\n[bold cyan]⏰ Time Commitment[/bold cyan]")
        
        choices = [
            ("15_minutes", "15 minutes - just want a quick taste"),
            ("2_hours", "2 hours - explore a few modules"),
            ("weekend", "Weekend project - build something substantial"),
            ("semester", "8-12 weeks - complete learning journey"),
            ("teaching", "Teaching timeline - need instructor resources")
        ]
        
        console.print("How much time can you dedicate?")
        for i, (key, description) in enumerate(choices, 1):
            console.print(f"  {i}. {description}")
        
        while True:
            try:
                selection = int(Prompt.ask("\nEnter your choice (1-5)"))
                if 1 <= selection <= 5:
                    return choices[selection-1][0]
                else:
                    console.print("[red]Please enter a number between 1-5[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")
    
    def _generate_recommendations(self, experience: str, goals: List[str], time: str) -> Dict[str, Any]:
        """Generate personalized recommendations."""
        
        # Learning path mapping
        path_mapping = {
            'beginner': 'foundation_first',
            'python_user': 'guided_learning', 
            'framework_user': 'systems_focus',
            'ml_engineer': 'optimization_focus',
            'instructor': 'teaching_resources'
        }
        
        # Starting point mapping
        start_mapping = {
            '15_minutes': 'quick_demo',
            '2_hours': 'first_module',
            'weekend': 'milestone_project', 
            'semester': 'full_curriculum',
            'teaching': 'instructor_setup'
        }
        
        return {
            'learning_path': path_mapping.get(experience, 'guided_learning'),
            'starting_point': start_mapping.get(time, 'first_module'),
            'experience_level': experience,
            'goals': goals,
            'time_commitment': time
        }
    
    def _show_personalized_path(self, recommendations: Dict[str, Any]) -> None:
        """Show personalized learning path."""
        console = get_console()
        
        # Path descriptions
        paths = {
            'foundation_first': {
                'title': '🌱 Foundation First Path',
                'description': 'Build fundamentals step-by-step with extra explanations',
                'next_steps': ['Module 1: Setup & Environment', 'Python fundamentals review', 'Linear algebra primer']
            },
            'guided_learning': {
                'title': '🎯 Guided Learning Path', 
                'description': 'Structured progression through all major concepts',
                'next_steps': ['Module 1: Setup', 'Module 2: Tensors', 'Track progress with checkpoints']
            },
            'systems_focus': {
                'title': '⚡ Systems Focus Path',
                'description': 'Understand internals of frameworks you already use',
                'next_steps': ['Compare PyTorch vs your code', 'Profile memory usage', 'Optimization modules']
            },
            'optimization_focus': {
                'title': '🚀 Optimization Focus Path',
                'description': 'Performance debugging and production optimization',
                'next_steps': ['Profiling module', 'Benchmarking module', 'TinyMLPerf competition']
            },
            'teaching_resources': {
                'title': '🎓 Teaching Resources Path',
                'description': 'Instructor guides and classroom setup',
                'next_steps': ['Instructor guide', 'NBGrader setup', 'Student progress tracking']
            }
        }
        
        path_info = paths[recommendations['learning_path']]
        
        console.print(f"\n[bold green]✨ Your Personalized Learning Path[/bold green]")
        console.print(Panel(
            f"[bold]{path_info['title']}[/bold]\n\n"
            f"{path_info['description']}\n\n"
            f"[bold cyan]Your Next Steps:[/bold cyan]\n" +
            "\n".join(f"  • {step}" for step in path_info['next_steps']),
            border_style="green"
        ))
    
    def _launch_first_steps(self, recommendations: Dict[str, Any]) -> None:
        """Launch appropriate first steps based on recommendations."""
        console = get_console()
        
        starting_point = recommendations['starting_point']
        
        if starting_point == 'quick_demo':
            console.print("\n[bold blue]🚀 Launching Quick Demo...[/bold blue]")
            console.print("Running: [code]tito demo quick[/code]")
            os.system("tito demo quick")
            
        elif starting_point == 'first_module':
            console.print("\n[bold blue]🛠️ Setting up Module 1...[/bold blue]")
            console.print("Next commands:")
            console.print("  [code]cd modules/01_setup[/code]")
            console.print("  [code]jupyter lab setup.py[/code]")
            
        elif starting_point == 'milestone_project':
            console.print("\n[bold blue]🎯 Weekend Project Recommendations...[/bold blue]")
            console.print("Suggested goal: Build XOR solver (Modules 1-6)")
            console.print("Time estimate: 6-8 hours")
            
        elif starting_point == 'full_curriculum':
            console.print("\n[bold blue]📚 Full Curriculum Setup...[/bold blue]")
            console.print("Running checkpoint system initialization...")
            os.system("tito checkpoint status")
            
        elif starting_point == 'instructor_setup':
            console.print("\n[bold blue]🎓 Instructor Resources...[/bold blue]")
            console.print("Opening instructor guide...")
            console.print("Check: [code]book/usage-paths/classroom-use.html[/code]")
    
    def _show_quick_reference(self) -> int:
        """Show quick reference card."""
        console = get_console()
        
        # Essential commands table
        table = Table(title="🚀 TinyTorch Quick Reference", show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold", width=25)
        table.add_column("Description", width=40)
        table.add_column("Example", style="dim", width=30)
        
        essential_commands = [
            ("tito help --interactive", "Launch onboarding wizard", "First time users"),
            ("tito checkpoint status", "See your progress", "Track learning journey"),
            ("tito module complete 02", "Finish a module", "Export & test your code"),
            ("tito demo quick", "See framework in action", "5-minute demonstration"),
            ("tito leaderboard join", "Join community", "Connect with learners"),
            ("tito system doctor", "Check environment", "Troubleshoot issues")
        ]
        
        for cmd, desc, example in essential_commands:
            table.add_row(cmd, desc, example)
        
        console.print(table)
        
        # Common workflows
        console.print("\n[bold cyan]📋 Common Workflows:[/bold cyan]")
        workflows = [
            ("New User", "tito help -i → tito checkpoint status → cd modules/01_setup"),
            ("Continue Learning", "tito checkpoint status → work on next module → tito module complete XX"),
            ("Join Community", "tito leaderboard join → submit progress → see global rankings"),
            ("Get Help", "tito system doctor → check docs/FAQ → ask community")
        ]
        
        for workflow, commands in workflows:
            console.print(f"  [bold]{workflow}:[/bold] {commands}")
        
        return 0
    
    def _show_topic_help(self, topic: str) -> int:
        """Show help for specific topic."""
        console = get_console()
        
        topics = {
            'getting-started': self._help_getting_started,
            'commands': self._help_commands,
            'workflow': self._help_workflow,
            'modules': self._help_modules,
            'checkpoints': self._help_checkpoints,
            'community': self._help_community,
            'troubleshooting': self._help_troubleshooting
        }
        
        if topic in topics:
            topics[topic]()
            return 0
        else:
            console.print(f"[red]Unknown help topic: {topic}[/red]")
            console.print("Available topics: " + ", ".join(topics.keys()))
            return 1
    
    def _show_contextual_help(self) -> int:
        """Show contextual help based on user progress."""
        console = get_console()
        
        # Check user progress to provide contextual guidance
        progress = self._assess_user_progress()
        
        if progress['is_new_user']:
            self._show_new_user_help()
        elif progress['current_module']:
            self._show_in_progress_help(progress['current_module'])
        else:
            self._show_experienced_user_help()
        
        return 0
    
    def _assess_user_progress(self) -> Dict[str, Any]:
        """Assess user's current progress."""
        # Check for checkpoint files, completed modules, etc.
        # This would integrate with the checkpoint system
        
        # Simplified implementation for now
        checkpoints_dir = Path("tests/checkpoints")
        modules_dir = Path("modules")
        
        return {
            'is_new_user': not checkpoints_dir.exists(),
            'current_module': None,  # Would be determined by checkpoint status
            'completed_modules': [],  # Would be populated from checkpoint results
            'has_joined_community': False  # Would check leaderboard status
        }
    
    def _show_new_user_help(self) -> None:
        """Show help optimized for new users."""
        console = get_console()
        
        console.print(Panel.fit(
            "[bold blue]👋 Welcome to Tiny🔥Torch![/bold blue]\n\n"
            "You're about to build a complete ML framework from scratch.\n"
            "Here's how to get started:\n\n"
            "[bold cyan]Next Steps:[/bold cyan]\n"
            "1. [code]tito help --interactive[/code] - Personalized onboarding\n"
            "2. [code]tito system doctor[/code] - Check your environment\n"
            "3. [code]tito checkpoint status[/code] - See the learning journey\n\n"
            "[bold yellow]New to ML systems?[/bold yellow] Run the interactive wizard!",
            title="Getting Started",
            border_style="blue"
        ))
    
    def _help_getting_started(self) -> None:
        """Detailed getting started help."""
        console = get_console()
        
        console.print("[bold blue]🚀 Getting Started with Tiny🔥Torch[/bold blue]\n")
        
        # Installation steps
        install_panel = Panel(
            "[bold]1. Environment Setup[/bold]\n"
            "```bash\n"
            "git clone https://github.com/mlsysbook/Tiny🔥Torch.git\n"
            "cd Tiny🔥Torch\n"
            f"python -m venv {self.venv_path}\n"
            f"source {self.venv_path}/bin/activate  # Windows: .venv\\Scripts\\activate\n"
            "pip install -r requirements.txt\n"
            "pip install -e .\n"
            "```",
            title="Installation",
            border_style="green"
        )
        
        # First steps
        first_steps_panel = Panel(
            "[bold]2. First Steps[/bold]\n"
            "• [code]tito system doctor[/code] - Verify installation\n"
            "• [code]tito help --interactive[/code] - Personalized guidance\n"
            "• [code]tito checkpoint status[/code] - See learning path\n"
            "• [code]cd modules/01_setup[/code] - Start first module",
            title="First Steps",
            border_style="blue"
        )
        
        # Learning path
        learning_panel = Panel(
            "[bold]3. Learning Journey[/bold]\n"
            "📚 [bold]Modules 1-8:[/bold] Neural Network Foundations\n"
            "🔬 [bold]Modules 9-10:[/bold] Computer Vision (CNNs)\n"
            "🤖 [bold]Modules 11-14:[/bold] Language Models (Transformers)\n"
            "⚡ [bold]Modules 15-20:[/bold] System Optimization\n\n"
            "[dim]Each module: Build → Test → Export → Checkpoint[/dim]",
            title="Learning Path",
            border_style="yellow"
        )
        
        console.print(Columns([install_panel, first_steps_panel, learning_panel]))
    
    # Additional help methods would be implemented here...
    def _help_commands(self) -> None:
        """Show comprehensive command reference."""
        pass
    
    def _help_workflow(self) -> None:
        """Show common workflow patterns."""
        pass
    
    def _help_modules(self) -> None:
        """Show module system explanation."""
        pass
    
    def _help_checkpoints(self) -> None:
        """Show checkpoint system explanation.""" 
        pass
    
    def _help_community(self) -> None:
        """Show community features and leaderboard."""
        pass
    
    def _help_troubleshooting(self) -> None:
        """Show troubleshooting guide."""
        pass