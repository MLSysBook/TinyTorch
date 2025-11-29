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
from .commands.grade import GradeCommand
from .commands.logo import LogoCommand
from .commands.milestone import MilestoneCommand
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
            'community': CommunityCommand,
            'benchmark': BenchmarkCommand,
            # Shortcuts
            'export': ExportCommand,
            'test': TestCommand,
            'grade': GradeCommand,
            'logo': LogoCommand,
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="tito",
            description="Tiny🔥Torch CLI - Build ML systems from scratch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Student Commands:
  module       Module workflow - start, work, complete modules
  milestones   Track progress - unlock capabilities as you build
  community    Join global community - connect with builders

Developer Commands:
  system       Environment and configuration
  src          Export src/ to modules/ and tinytorch/
  package      Package management (nbdev)
  nbgrader     Auto-grading tools

Quick Start:
  tito setup                    First-time setup
  tito module start 01          Start Module 01 (tensors)
  tito module complete 01       Test, export, and track progress
  tito module status            View your progress
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

                # Simple, focused welcome message
                help_text = "[bold cyan]Quick Start:[/bold cyan]\n"
                help_text += "  [green]tito setup[/green]                  First-time setup\n"
                help_text += "  [green]tito module start 01[/green]        Start Module 01 (tensors)\n"
                help_text += "  [green]tito module complete 01[/green]     Test, export, and track progress\n"
                help_text += "\n[bold cyan]Track Progress:[/bold cyan]\n"
                help_text += "  [yellow]tito module status[/yellow]        View module progress\n"
                help_text += "  [yellow]tito milestones status[/yellow]    View unlocked capabilities\n"
                help_text += "\n[bold cyan]Community:[/bold cyan]\n"
                help_text += "  [blue]tito community join[/blue]          Connect with builders worldwide\n"
                help_text += "\n[dim]More commands: tito --help[/dim]"

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