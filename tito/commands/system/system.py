"""
System command group for TinyTorch CLI: environment, configuration, and system tools.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand
from .info import InfoCommand
from .health import HealthCommand
from ..check import CheckCommand
from ..version import VersionCommand
from ..clean_workspace import CleanWorkspaceCommand
from ..report import ReportCommand
from .jupyter import JupyterCommand
from ..protect import ProtectCommand

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
            help='Show system and environment information'
        )
        info_cmd = InfoCommand(self.config)
        info_cmd.add_arguments(info_parser)

        # Health subcommand (quick check)
        health_parser = subparsers.add_parser(
            'health',
            help='Quick environment health check'
        )
        health_cmd = HealthCommand(self.config)
        health_cmd.add_arguments(health_parser)

        # Check subcommand (comprehensive validation)
        check_parser = subparsers.add_parser(
            'check',
            help='Run comprehensive environment validation (60+ tests)'
        )
        check_cmd = CheckCommand(self.config)
        check_cmd.add_arguments(check_parser)

        # Version subcommand
        version_parser = subparsers.add_parser(
            'version',
            help='Show version information for TinyTorch and dependencies'
        )
        version_cmd = VersionCommand(self.config)
        version_cmd.add_arguments(version_parser)

        # Clean subcommand
        clean_parser = subparsers.add_parser(
            'clean',
            help='Clean up generated files, caches, and temporary files'
        )
        clean_cmd = CleanWorkspaceCommand(self.config)
        clean_cmd.add_arguments(clean_parser)

        # Report subcommand
        report_parser = subparsers.add_parser(
            'report',
            help='Generate comprehensive diagnostic report (JSON)'
        )
        report_cmd = ReportCommand(self.config)
        report_cmd.add_arguments(report_parser)

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
                "  • [bold]info[/bold]    - Show system/environment information\n"
                "  • [bold]health[/bold]  - Quick environment health check\n"
                "  • [bold]check[/bold]   - Comprehensive validation (60+ tests)\n"
                "  • [bold]version[/bold] - Show version information\n"
                "  • [bold]clean[/bold]   - Clean up workspace (caches, temp files)\n"
                "  • [bold]report[/bold]  - Generate diagnostic report (JSON)\n"
                "  • [bold]jupyter[/bold] - Start Jupyter notebook server\n"
                "  • [bold]protect[/bold] - 🛡️ Student protection system management\n\n"
                "[dim]Example: tito system health[/dim]",
                title="System Command Group",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.system_command == 'info':
            cmd = InfoCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'health':
            cmd = HealthCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'check':
            cmd = CheckCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'version':
            cmd = VersionCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'clean':
            cmd = CleanWorkspaceCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'report':
            cmd = ReportCommand(self.config)
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
