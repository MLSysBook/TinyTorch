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
from .core.virtual_env_manager import get_venv_path
from .core.console import get_console, print_banner, print_error, print_ascii_logo
from .core.exceptions import TinyTorchCLIError
from rich.panel import Panel
from .commands.base import BaseCommand
from .commands.info import InfoCommand
from .commands.test import TestCommand
from .commands.doctor import DoctorCommand
from .commands.export import ExportCommand
from .commands.reset import ResetCommand
from .commands.jupyter import JupyterCommand
from .commands.nbdev import NbdevCommand
from .commands.status import StatusCommand
from .commands.system import SystemCommand
from .commands.package import PackageCommand
from .commands.nbgrader import NBGraderCommand
from .commands.book import BookCommand
from .commands.checkpoint import CheckpointCommand
from .commands.grade import GradeCommand
from .commands.demo import DemoCommand
from .commands.logo import LogoCommand
from .commands.milestone import MilestoneCommand
from .commands.community import CommunityCommand
from .commands.olympics import OlympicsCommand
from .commands.submit import SubmitCommand
from .commands.setup import SetupCommand

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
            # Core workflow (what students actually use)
            'setup': SetupCommand,
            'export': ExportCommand,
            'test': TestCommand,
            'submit': SubmitCommand,
            # Community features
            'community': CommunityCommand,
            # System management
            'system': SystemCommand,
            'package': PackageCommand,
            # Assessment tools
            'nbgrader': NBGraderCommand,
            'grade': GradeCommand,
            # Progress tracking
            'checkpoint': CheckpointCommand,
            'milestone': MilestoneCommand,
            # Special events
            'olympics': OlympicsCommand,
            # Utilities
            'book': BookCommand,
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
Core Workflow (what you'll use daily):
  setup        First-time environment setup
  export       Export module code to tinytorch package
  test         Run module tests
  submit       Submit TinyMLPerf competition entry (Module 20)

Community Features:
  community    Join community, share progress, celebrate achievements

System Tools:
  system       System environment and configuration
  package      Package management and nbdev integration
  checkpoint   Track your learning progress
  milestone    Epic capability achievements

Assessment:
  nbgrader     Assignment management and auto-grading
  grade        Simplified grading interface

Special Events:
  olympics     Competition events and recognition

Utilities:
  book         Build and manage Jupyter Book
  demo         Run AI capability demos
  logo         Learn about TinyTorch philosophy

Daily Workflow:
  # 1. Edit code in your IDE
  cd modules/source/05_autograd/
  # Edit autograd_dev.py...
  
  # 2. Export and test
  tito export 05
  tito test 05
  
  # 3. Repeat for next module!

Competition Workflow (Module 20):
  # Edit Module 20, run cells → submission.json
  tito submit submission.json
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
                
                # Show enhanced help with simplified structure
                self.console.print(Panel(
                    "[bold bright_cyan]Core Workflow (Daily Use):[/bold bright_cyan]\n"
                    "  [bold green]setup[/bold green]        - First-time environment setup\n"
                    "  [bold green]export[/bold green]       - Export module code to package\n"
                    "  [bold green]test[/bold green]         - Run module tests\n"
                    "  [bold green]submit[/bold green]       - Submit competition entry (Module 20)\n\n"
                    "[bold bright_blue]Community:[/bold bright_blue]\n"
                    "  [bold bright_blue]community[/bold bright_blue]   - Join, share progress, celebrate together\n\n"
                    "[bold yellow]Progress & Achievements:[/bold yellow]\n"
                    "  [bold yellow]checkpoint[/bold yellow]  - Track your learning progress\n"
                    "  [bold yellow]milestone[/bold yellow]   - Epic capability achievements\n\n"
                    "[bold magenta]Tools & Utilities:[/bold magenta]\n"
                    "  [bold magenta]system[/bold magenta]      - System tools\n"
                    "  [bold magenta]package[/bold magenta]     - Package management\n"
                    "  [bold magenta]book[/bold magenta]        - Build documentation\n"
                    "  [bold magenta]demo[/bold magenta]        - Show capabilities\n\n"
                    "[bold bright_green]📚 Quick Start:[/bold bright_green]\n"
                    "  [dim]# 1. Setup (one time)[/dim]\n"
                    "  [cyan]tito setup[/cyan]\n\n"
                    "  [dim]# 2. Daily workflow (edit → export → test)[/dim]\n"
                    "  [dim]# Edit modules/source/05_autograd/autograd_dev.py[/dim]\n"
                    "  [cyan]tito export 05[/cyan]\n"
                    "  [cyan]tito test 05[/cyan]\n\n"
                    "  [dim]# 3. Competition (Module 20)[/dim]\n"
                    "  [dim]# Edit Module 20, run cells → submission.json[/dim]\n"
                    "  [cyan]tito submit submission.json[/cyan]\n\n"
                    "[bold]Get Help:[/bold]\n"
                    "  [dim]tito --help[/dim]     - Show all commands\n"
                    "  [dim]tito community[/dim]  - Community subcommands\n"
                    "  [dim]tito system[/dim]     - System subcommands",
                    title="🔥 Welcome to TinyTorch!",
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