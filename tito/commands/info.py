"""
Info command for TinyTorch CLI: shows system information and course navigation.
"""

from argparse import ArgumentParser, Namespace
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree

from .base import BaseCommand

class InfoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "info"

    @property
    def description(self) -> str:
        return "Show system information and course navigation"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--hello", action="store_true", help="Show hello message")
        parser.add_argument("--show-architecture", action="store_true", help="Show system architecture")

    def run(self, args: Namespace) -> int:
        console = self.console
        self.print_banner()
        console.print()
        
        # System Information Panel
        info_text = Text()
        info_text.append(f"Python: {sys.version.split()[0]}\n", style="cyan")
        info_text.append(f"Platform: {sys.platform}\n", style="cyan")
        info_text.append(f"Working Directory: {os.getcwd()}\n", style="cyan")
        
        # Virtual environment check
        venv_exists = self.venv_path.exists()
        in_venv = (
            os.environ.get('VIRTUAL_ENV') is not None or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            hasattr(sys, 'real_prefix')
        )
        
        if venv_exists and in_venv:
            venv_style = "green"
            venv_icon = "✅"
            venv_status = "Ready & Active"
        elif venv_exists:
            venv_style = "yellow"
            venv_icon = "✅"
            venv_status = "Ready (Not Active)"
        else:
            venv_style = "red"
            venv_icon = "❌"
            venv_status = "Not Found"
        
        info_text.append(f"Virtual Environment: {venv_icon} ", style=venv_style)
        info_text.append(venv_status, style=f"bold {venv_style}")
        
        console.print(Panel(info_text, title="📋 System Information", border_style="bright_blue"))
        console.print()
        
        # Course Navigation Panel
        nav_text = Text()
        nav_text.append("📖 Course Overview: ", style="dim")
        nav_text.append("README.md\n", style="cyan underline")
        nav_text.append("🎯 Detailed Guide: ", style="dim")
        nav_text.append("COURSE_GUIDE.md\n", style="cyan underline")
        nav_text.append("🚀 Start Here: ", style="dim")
        nav_text.append("modules/setup/README.md", style="cyan underline")
        
        console.print(Panel(nav_text, title="📋 Course Navigation", border_style="bright_green"))
        console.print()
        
        # Command Reference Panel
        cmd_text = Text()
        cmd_text.append("📊 Module Status: ", style="dim")
        cmd_text.append("tito module status\n", style="bold cyan")
        cmd_text.append("🧪 Run Tests: ", style="dim")
        cmd_text.append("tito module test --all\n", style="bold cyan")
        cmd_text.append("🔄 Export Code: ", style="dim")
        cmd_text.append("tito module export --all\n", style="bold cyan")
        cmd_text.append("🩺 Check Environment: ", style="dim")
        cmd_text.append("tito system doctor", style="bold cyan")
        
        console.print(Panel(cmd_text, title="📋 Quick Commands", border_style="bright_magenta"))
        
        # Optionally show hello message or architecture
        if args.hello and self.check_setup_status() == "✅ Implemented":
            try:
                from tinytorch.core.utils import hello_tinytorch
                hello_text = Text(hello_tinytorch(), style="bold red")
                console.print()
                console.print(Panel(hello_text, style="bright_red", padding=(1, 2)))
            except ImportError:
                pass
        
        if args.show_architecture:
            console.print()
            arch_tree = Tree("🏗️ TinyTorch System Architecture", style="bold blue")
            cli_branch = arch_tree.add("CLI Interface", style="cyan")
            cli_branch.add("tito/ - Command line tools", style="dim")
            training_branch = arch_tree.add("Training Orchestration", style="cyan")
            training_branch.add("trainer.py - Training loop management", style="dim")
            core_branch = arch_tree.add("Core Components", style="cyan")
            model_sub = core_branch.add("Model Definition", style="yellow")
            model_sub.add("modules.py - Neural network layers", style="dim")
            data_sub = core_branch.add("Data Pipeline", style="yellow")
            data_sub.add("dataloader.py - Efficient data loading", style="dim")
            opt_sub = core_branch.add("Optimization", style="yellow")
            opt_sub.add("optimizer.py - SGD, Adam, etc.", style="dim")
            autograd_branch = arch_tree.add("Automatic Differentiation Engine", style="cyan")
            autograd_branch.add("autograd.py - Gradient computation", style="dim")
            tensor_branch = arch_tree.add("Tensor Operations & Storage", style="cyan")
            tensor_branch.add("tensor.py - Core tensor implementation", style="dim")
            system_branch = arch_tree.add("System Tools", style="cyan")
            system_branch.add("profiler.py - Performance measurement", style="dim")
            system_branch.add("mlops.py - Production monitoring", style="dim")
            console.print(Panel(arch_tree, title="🏗️ System Architecture", border_style="bright_blue"))
        
        return 0

    def print_banner(self):
        banner_text = Text("Tiny🔥Torch: Build ML Systems from Scratch", style="bold red")
        self.console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

    def check_setup_status(self):
        try:
            from tinytorch.core.utils import hello_tinytorch
            return "✅ Implemented"
        except ImportError:
            return "❌ Not Implemented" 