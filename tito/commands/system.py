"""
System command group for TinyTorch CLI: environment, configuration, and system tools.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from .base import BaseCommand
from .info import InfoCommand
from .doctor import DoctorCommand
from .jupyter import JupyterCommand
from .protect import ProtectCommand

class SystemCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "System environment and configuration commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='system_command',
            help='System subcommands',
            metavar='SUBCOMMAND'
        )
        
        # Info subcommand
        info_parser = subparsers.add_parser(
            'info',
            help='Show system information and course navigation'
        )
        info_cmd = InfoCommand(self.config)
        info_cmd.add_arguments(info_parser)
        
        # Doctor subcommand
        doctor_parser = subparsers.add_parser(
            'doctor',
            help='Run environment diagnosis'
        )
        doctor_cmd = DoctorCommand(self.config)
        doctor_cmd.add_arguments(doctor_parser)
        
        # Jupyter subcommand
        jupyter_parser = subparsers.add_parser(
            'jupyter',
            help='Start Jupyter notebook server'
        )
        jupyter_cmd = JupyterCommand(self.config)
        jupyter_cmd.add_arguments(jupyter_parser)
        
        # Protect subcommand
        protect_parser = subparsers.add_parser(
            'protect',
            help='🛡️ Student protection system to prevent core file edits'
        )
        protect_cmd = ProtectCommand(self.config)
        protect_cmd.add_arguments(protect_parser)

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if not hasattr(args, 'system_command') or not args.system_command:
            console.print(Panel(
                "[bold cyan]System Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]info[/bold]    - Show system information and course navigation\n"
                "  • [bold]doctor[/bold]  - Run environment diagnosis\n"
                "  • [bold]jupyter[/bold] - Start Jupyter notebook server\n"
                "  • [bold]protect[/bold] - 🛡️ Student protection system management\n\n"
                "[dim]Example: tito system info[/dim]",
                title="System Command Group",
                border_style="bright_cyan"
            ))
            return 0
        
        # Execute the appropriate subcommand
        if args.system_command == 'info':
            cmd = InfoCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'doctor':
            cmd = DoctorCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'jupyter':
            cmd = JupyterCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'protect':
            cmd = ProtectCommand(self.config)
            return cmd.execute(args)
        else:
            console.print(Panel(
                f"[red]Unknown system subcommand: {args.system_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1 