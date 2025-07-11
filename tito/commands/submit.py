"""
Submit command for TinyTorch CLI: submits module for grading.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

class SubmitCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "submit"

    @property
    def description(self) -> str:
        return "Submit module"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--module", required=True, help="Module to submit")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        submit_text = Text()
        submit_text.append(f"📤 Submitting module: {args.module}\n\n", style="bold cyan")
        submit_text.append("🚧 Submission system not yet implemented.\n\n", style="yellow")
        submit_text.append("For now, make sure all tests pass with:\n", style="dim")
        submit_text.append(f"   python -m pytest modules/{args.module}/tests/test_{args.module}.py -v", style="bold white")
        
        console.print(Panel(submit_text, title="Module Submission", border_style="bright_yellow"))
        return 0 