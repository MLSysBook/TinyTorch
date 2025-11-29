"""
TinyTorch CLI Main Entry Point

A professional command-line interface with proper architecture:
- Clean separation of concerns
- Proper error handling
- Logging support
- Configuration management
- Extensible command system
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Type, Optional, List

# Set TINYTORCH_QUIET before any tinytorch imports to suppress autograd messages
os.environ['TINYTORCH_QUIET'] = '1'

from .core.config import CLIConfig
from .core.virtual_env_manager import get_venv_path
from .core.console import get_console, print_banner, print_error, print_ascii_logo
from .core.exceptions import TinyTorchCLIError
from rich.panel import Panel
from .commands.base import BaseCommand
from .commands.test import TestCommand
from .commands.export import ExportCommand
from .commands.src import SrcCommand
from .commands.system import SystemCommand
from .commands.module import ModuleWorkflowCommand
from .commands.package import PackageCommand
from .commands.nbgrader import NBGraderCommand
from .commands.book import BookCommand
from .commands.grade import GradeCommand
from .commands.demo import DemoCommand
from .commands.logo import LogoCommand
from .commands.milestone import MilestoneCommand
from .commands.leaderboard import LeaderboardCommand
from .commands.olympics import OlympicsCommand
from .commands.setup import SetupCommand
from .commands.benchmark import BenchmarkCommand
from .commands.community import CommunityCommand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tito-cli.log'),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

class TinyTorchCLI:
    """Main CLI application class."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.config = CLIConfig.from_project_root()
        self.console = get_console()
        # SINGLE SOURCE OF TRUTH: All valid commands registered here
        self.commands: Dict[str, Type[BaseCommand]] = {
            # Essential
            'setup': SetupCommand,
            # Workflow (student-facing)
            'system': SystemCommand,
            'module': ModuleWorkflowCommand,
            # Developer tools
            'src': SrcCommand,
            'package': PackageCommand,
            'nbgrader': NBGraderCommand,
            # Progress tracking
            'milestones': MilestoneCommand,
            # Community
            'leaderboard': LeaderboardCommand,
            'olympics': OlympicsCommand,
            'benchmark': BenchmarkCommand,
            'community': CommunityCommand,
            # Shortcuts
            'export': ExportCommand,
            'test': TestCommand,
            'book': BookCommand,
            'grade': GradeCommand,
            'demo': DemoCommand,
            'logo': LogoCommand,
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="tito",
            description="Tiny🔥Torch CLI - Build ML systems from scratch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Command Groups:
  system       System environment and configuration commands
  module       Module development workflow - start, complete, resume modules (students)
  source       Source file workflow - export src/ to modules/ and tinytorch/ (developers)
  package      Package management and nbdev integration commands
  nbgrader     Assignment management and auto-grading commands
  milestones   Track progress through ML history - epic achievements and capability unlocks
  leaderboard  Community showcase - share progress, connect with learners
  olympics     Competition events - friendly challenges and recognition

Convenience Shortcuts:
  export       Quick export (alias for: tito module export)
  test         Quick test (alias for: tito module test)
  book         Build Jupyter Book documentation
  grade        Simplified grading interface (wraps NBGrader)
  demo         Run capability demos (show what you've built!)

Getting Started:
  tito setup                    First-time environment setup
  tito module start 01          Start Module 01 (tensors, first time)
  tito module complete 01       Complete Module 01 (test + export + track)
  tito module resume 02         Resume working on Module 02
  tito module status            View your progress across all modules

Tracking Progress:
  tito milestones list          See all available milestones
  tito milestones status        View progress and achievements
  tito leaderboard join         Join the community
  tito leaderboard profile      View your achievement journey
            """
        )
        
        # Global options
        parser.add_argument(
            '--version',
            action='version',
            version='Tiny🔥Torch CLI 0.1.0'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--no-color',
            action='store_true',
            help='Disable colored output'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Add command parsers
        for command_name, command_class in self.commands.items():
            # Create temporary instance to get metadata
            temp_command = command_class(self.config)
            cmd_parser = subparsers.add_parser(
                command_name,
                help=temp_command.description
            )
            temp_command.add_arguments(cmd_parser)
        
        return parser
    
    def validate_environment(self) -> bool:
        """Validate the environment and show issues if any."""
        issues = self.config.validate(get_venv_path())
        
        if issues:
            print_error(
                "Environment validation failed:\n" + "\n".join(f"  • {issue}" for issue in issues),
                "Environment Issues"
            )
            self.console.print("\n[dim]Run 'tito doctor' for detailed diagnosis[/dim]")
            # Return True to allow command execution despite validation issues
            # This is temporary for development
            return True
        
        return True
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI application."""
        try:
            parser = self.create_parser()
            parsed_args = parser.parse_args(args)
            
            # Update config with global options
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                self.config.verbose = True
                logging.getLogger().setLevel(logging.DEBUG)
            
            if hasattr(parsed_args, 'no_color') and parsed_args.no_color:
                self.config.no_color = True
            
            # Show banner for interactive commands (except logo which has its own display)
            if parsed_args.command and not self.config.no_color and parsed_args.command != 'logo':
                print_banner()
            
            # Validate environment for most commands (skip for doctor)
            skip_validation = (
                parsed_args.command in [None, 'version', 'help'] or
                (parsed_args.command == 'system' and 
                 hasattr(parsed_args, 'system_command') and 
                 parsed_args.system_command == 'doctor')
            )
            if not skip_validation:
                if not self.validate_environment():
                    return 1
            
            # Handle no command
            if not parsed_args.command:
                # Show ASCII logo first
                print_ascii_logo()

                # Dynamically build help based on registered commands
                # Categorize commands by role
                essential = ['setup']
                student_workflow = ['module', 'milestones']
                community = ['leaderboard', 'olympics', 'community']
                developer = ['system', 'package', 'nbgrader', 'src']
                shortcuts = ['export', 'test', 'book', 'demo']
                other = ['benchmark', 'grade', 'logo']

                help_text = "[bold]Essential Commands:[/bold]\n"
                for cmd in essential:
                    if cmd in self.commands:
                        desc = self.commands[cmd](self.config).description
                        help_text += f"  [bold cyan]{cmd}[/bold cyan]        - {desc}\n"

                help_text += "\n[bold]Student Workflow:[/bold]\n"
                for cmd in student_workflow:
                    if cmd in self.commands:
                        desc = self.commands[cmd](self.config).description
                        help_text += f"  [bold green]{cmd}[/bold green]       - {desc}\n"

                help_text += "\n[bold]Community:[/bold]\n"
                for cmd in community:
                    if cmd in self.commands:
                        desc = self.commands[cmd](self.config).description
                        help_text += f"  [bold bright_blue]{cmd}[/bold bright_blue]  - {desc}\n"

                help_text += "\n[bold]Developer Tools:[/bold]\n"
                for cmd in developer:
                    if cmd in self.commands:
                        desc = self.commands[cmd](self.config).description
                        help_text += f"  [dim]{cmd}[/dim]       - {desc}\n"

                help_text += "\n[bold]Shortcuts:[/bold]\n"
                for cmd in shortcuts:
                    if cmd in self.commands:
                        desc = self.commands[cmd](self.config).description
                        help_text += f"  [bold yellow]{cmd}[/bold yellow]      - {desc}\n"

                help_text += "\n[bold]Quick Start:[/bold]\n"
                help_text += "  [dim]tito setup[/dim]                    - First-time setup (run once)\n"
                help_text += "  [dim]tito module start 01[/dim]          - Start Module 01 (tensors)\n"
                help_text += "  [dim]tito module complete 01[/dim]       - Complete it (test + export + track)\n"
                help_text += "  [dim]tito module status[/dim]            - View all progress\n"
                help_text += "\n[bold]Track Progress:[/bold]\n"
                help_text += "  [dim]tito milestones list[/dim]          - Available milestones\n"
                help_text += "  [dim]tito milestones status[/dim]        - Your progress\n"
                help_text += "  [dim]tito leaderboard profile[/dim]      - Community profile\n"
                help_text += "\n[bold]Get Help:[/bold]\n"
                help_text += "  [dim]tito <command>[/dim]                - Show command subcommands\n"
                help_text += "  [dim]tito --help[/dim]                   - Show full help"

                self.console.print(Panel(
                    help_text,
                    title="Welcome to Tiny🔥Torch!",
                    border_style="bright_green"
                ))
                return 0
            
            # Execute command
            if parsed_args.command in self.commands:
                command_class = self.commands[parsed_args.command]
                command = command_class(self.config)
                return command.execute(parsed_args)
            else:
                print_error(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation cancelled by user[/yellow]")
            return 130
        except TinyTorchCLIError as e:
            logger.error(f"CLI error: {e}")
            print_error(str(e))
            return 1
        except Exception as e:
            logger.exception("Unexpected error in CLI")
            print_error(f"Unexpected error: {e}")
            return 1

def main() -> int:
    """Main entry point for the CLI."""
    cli = TinyTorchCLI()
    return cli.run(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(main()) 