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

from .core.config import CLIConfig
from .core.console import get_console, print_banner, print_error, print_ascii_logo
from .core.exceptions import TinyTorchCLIError
from rich.panel import Panel
from .commands.base import BaseCommand
from .commands.notebooks import NotebooksCommand
from .commands.info import InfoCommand
from .commands.test import TestCommand
from .commands.doctor import DoctorCommand
from .commands.export import ExportCommand
from .commands.reset import ResetCommand
from .commands.jupyter import JupyterCommand
from .commands.nbdev import NbdevCommand
from .commands.status import StatusCommand
from .commands.system import SystemCommand
from .commands.module import ModuleCommand
from .commands.package import PackageCommand
from .commands.nbgrader import NBGraderCommand
from .commands.book import BookCommand
from .commands.checkpoint import CheckpointCommand
from .commands.grade import GradeCommand
from .commands.demo import DemoCommand
from .commands.logo import LogoCommand
from .commands.milestone import MilestoneCommand
from .commands.leaderboard import LeaderboardCommand
from .commands.olympics import OlympicsCommand

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
        self.commands: Dict[str, Type[BaseCommand]] = {
            # Hierarchical command groups only
            'system': SystemCommand,
            'module': ModuleCommand,
            'package': PackageCommand,
            'nbgrader': NBGraderCommand,
            'checkpoint': CheckpointCommand,
            'milestone': MilestoneCommand,
            'leaderboard': LeaderboardCommand,
            'olympics': OlympicsCommand,
            # Convenience commands
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
            description="TinyTorch CLI - Build ML systems from scratch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Command Groups:
  system       System environment and configuration commands
  module       Module development and management commands  
  package      Package management and nbdev integration commands
  nbgrader     Assignment management and auto-grading commands
  checkpoint   Track ML systems engineering progress through checkpoints
  milestone    Epic capability achievements and ML systems mastery
  leaderboard  Join the inclusive community, share progress, celebrate achievements
  olympics     Special competition events with focused challenges and recognition

Convenience Commands:
  export      Export modules to package (quick shortcut)
  test        Run tests (quick shortcut)
  book        Build and manage Jupyter Book
  grade       Simplified grading interface (wraps NBGrader)
  demo        Run AI capability demos (show what your framework can do!)

Examples:
  tito system info              Show system information
  tito module status --metadata Module status with metadata
  tito module view 01_setup     Start coding in Jupyter Lab
  tito export 01_tensor         Export specific module to package
  tito checkpoint timeline      Visual progress timeline
  tito leaderboard register     Join the inclusive community
  tito olympics events          See special competitions
  tito book build               Build the Jupyter Book locally
            """
        )
        
        # Global options
        parser.add_argument(
            '--version', 
            action='version',
            version='TinyTorch CLI 0.1.0'
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
        issues = self.config.validate()
        
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
                
                # Show enhanced help with command groups
                self.console.print(Panel(
                    "[bold]Command Groups:[/bold]\n"
                    "  [bold green]system[/bold green]       - System environment and configuration\n"
                    "  [bold green]module[/bold green]       - Module development and management\n"
                    "  [bold green]package[/bold green]      - Package management and nbdev integration\n"
                    "  [bold green]nbgrader[/bold green]     - Assignment management and auto-grading\n"
                    "  [bold green]checkpoint[/bold green]   - Track ML systems engineering progress\n"
                    "  [bold magenta]milestone[/bold magenta]    - Epic capability achievements and ML mastery\n"
                    "  [bold bright_blue]leaderboard[/bold bright_blue] - Join the inclusive community, share progress\n"
                    "  [bold bright_yellow]olympics[/bold bright_yellow]     - Special competition events and recognition\n\n"
                    "[bold]Convenience Commands:[/bold]\n"
                    "  [bold green]export[/bold green]      - Export modules to package\n"
                    "  [bold green]test[/bold green]        - Run tests\n"
                    "  [bold green]book[/bold green]        - Build and manage Jupyter Book\n"
                    "  [bold green]logo[/bold green]        - Learn about TinyTorch philosophy\n"
                    "[bold]Quick Start:[/bold]\n"
                    "  [dim]tito system info[/dim]              - Show system information\n"
                    "  [dim]tito module status --metadata[/dim] - Module status with metadata\n"
                    "  [dim]tito module view 01_setup[/dim]     - Start coding in Jupyter Lab\n"
                    "  [dim]tito checkpoint timeline[/dim]      - Visual progress timeline\n"
                    "  [dim]tito leaderboard register[/dim]     - Join the inclusive community\n"
                    "  [dim]tito olympics events[/dim]          - See special competitions\n"
                    "  [dim]tito milestone status[/dim]         - See your epic achievement progress\n"
                    "[bold]Get Help:[/bold]\n"
                    "  [dim]tito system[/dim]                   - Show system subcommands\n"
                    "  [dim]tito module[/dim]                   - Show module subcommands\n"
                    "  [dim]tito package[/dim]                  - Show package subcommands\n"
                    "  [dim]tito --help[/dim]                   - Show full help",
                    title="Welcome to TinyTorch!",
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
    return cli.run()

if __name__ == "__main__":
    sys.exit(main()) 