"""
Module command group for TinyTorch CLI: development workflow and module management.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from .base import BaseCommand
from .status import StatusCommand
from .test import TestCommand
from .notebooks import NotebooksCommand
from .clean import CleanCommand
from .export import ExportCommand

class ModuleCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "module"

    @property
    def description(self) -> str:
        return "Module development and management commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='module_command',
            help='Module subcommands',
            metavar='SUBCOMMAND'
        )
        
        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='Check status of all modules'
        )
        status_cmd = StatusCommand(self.config)
        status_cmd.add_arguments(status_parser)
        
        # Test subcommand
        test_parser = subparsers.add_parser(
            'test',
            help='Run module tests'
        )
        test_cmd = TestCommand(self.config)
        test_cmd.add_arguments(test_parser)
        
        # Notebooks subcommand
        notebooks_parser = subparsers.add_parser(
            'notebooks',
            help='Build notebooks from Python files'
        )
        notebooks_cmd = NotebooksCommand(self.config)
        notebooks_cmd.add_arguments(notebooks_parser)
        
        # Clean subcommand
        clean_parser = subparsers.add_parser(
            'clean',
            help='Clean up module directories (notebooks, cache, etc.)'
        )
        clean_cmd = CleanCommand(self.config)
        clean_cmd.add_arguments(clean_parser)
        
        # Export subcommand
        export_parser = subparsers.add_parser(
            'export',
            help='Export module code to Python package'
        )
        export_cmd = ExportCommand(self.config)
        export_cmd.add_arguments(export_parser)

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if not hasattr(args, 'module_command') or not args.module_command:
            console.print(Panel(
                "[bold cyan]Module Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]status[/bold]     - Check status of all modules\n"
                "  • [bold]test[/bold]       - Run module tests\n"
                "  • [bold]notebooks[/bold]  - Build notebooks from Python files\n"
                "  • [bold]clean[/bold]      - Clean up module directories\n"
                "  • [bold]export[/bold]     - Export module code to Python package\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito module status --metadata[/dim]\n"
                "[dim]  tito module test --all[/dim]\n"
                "[dim]  tito module test tensor[/dim]\n"
                "[dim]  tito module export --all[/dim]\n"
                "[dim]  tito module export tensor[/dim]\n"
                "[dim]  tito module clean --all[/dim]\n"
                "[dim]  tito module clean tensor[/dim]",
                title="Module Command Group",
                border_style="bright_cyan"
            ))
            return 0
        
        # Execute the appropriate subcommand
        if args.module_command == 'status':
            cmd = StatusCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'test':
            cmd = TestCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'notebooks':
            cmd = NotebooksCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'clean':
            cmd = CleanCommand(self.config)
            return cmd.execute(args)
        elif args.module_command == 'export':
            cmd = ExportCommand(self.config)
            return cmd.execute(args)
        else:
            console.print(Panel(
                f"[red]Unknown module subcommand: {args.module_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1 