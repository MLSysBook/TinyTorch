"""
Doctor command for TinyTorch CLI: runs comprehensive environment diagnosis.
"""

import sys
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand

class DoctorCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "doctor"

    @property
    def description(self) -> str:
        return "Run environment diagnosis"

    def add_arguments(self, parser: ArgumentParser) -> None:
        # Doctor command doesn't need additional arguments
        pass

    def run(self, args: Namespace) -> int:
        console = self.console
        
        console.print(Panel("🔬 TinyTorch Environment Diagnosis", 
                           title="System Doctor", border_style="bright_magenta"))
        console.print()
        
        # Environment checks table
        env_table = Table(title="Environment Check", show_header=True, header_style="bold blue")
        env_table.add_column("Component", style="cyan", width=20)
        env_table.add_column("Status", justify="left")
        env_table.add_column("Details", style="dim", width=30)
        
        # Python environment
        env_table.add_row("Python", "[green]✅ OK[/green]", f"{sys.version.split()[0]} ({sys.platform})")
        
        # Virtual environment - check if it exists and if we're using it
        venv_exists = self.venv_path.exists()
        in_venv = (
            # Method 1: Check VIRTUAL_ENV environment variable (most reliable for activation)
            os.environ.get('VIRTUAL_ENV') is not None or
            # Method 2: Check sys.prefix vs sys.base_prefix (works for running Python in venv)
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            # Method 3: Check for sys.real_prefix (older Python versions)
            hasattr(sys, 'real_prefix')
        )
        
        if venv_exists and in_venv:
            venv_status = "[green]✅ Ready & Active[/green]"
        elif venv_exists:
            venv_status = "[yellow]✅ Ready (Not Active)[/yellow]"
        else:
            venv_status = "[red]❌ Not Found[/red]"
        env_table.add_row("Virtual Environment", venv_status, f"{self.venv_path}")
        
        # Dependencies
        dependencies = [
            ('numpy', 'numpy'),
            ('matplotlib', 'matplotlib'),  
            ('pytest', 'pytest'),
            ('yaml', 'yaml'),  # PyYAML package imports as yaml
            ('black', 'black'),
            ('rich', 'rich')
        ]
        for display_name, import_name in dependencies:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                env_table.add_row(display_name.title(), "[green]✅ OK[/green]", f"v{version}")
            except ImportError:
                env_table.add_row(display_name.title(), "[red]❌ Missing[/red]", "Not installed")
        
        console.print(env_table)
        console.print()
        
        # Module structure table
        struct_table = Table(title="Module Structure", show_header=True, header_style="bold magenta")
        struct_table.add_column("Path", style="cyan", width=25)
        struct_table.add_column("Status", justify="left")
        struct_table.add_column("Type", style="dim", width=25)
        
        required_paths = [
            ('tinytorch/', 'Package directory'),
            ('tinytorch/core/', 'Core module directory'),
            ('modules/', 'Module directory'),
            ('bin/tito', 'CLI script'),
            ('requirements.txt', 'Dependencies file')
        ]
        
        for path, desc in required_paths:
            if Path(path).exists():
                struct_table.add_row(path, "[green]✅ Found[/green]", desc)
            else:
                struct_table.add_row(path, "[red]❌ Missing[/red]", desc)
        
        console.print(struct_table)
        console.print()
        
        # Module implementations
        console.print(Panel("📋 Implementation Status", 
                           title="Module Status", border_style="bright_blue"))
        
        # Import and run the info command to show module status
        from .info import InfoCommand
        info_cmd = InfoCommand(self.config)
        info_args = ArgumentParser()
        info_cmd.add_arguments(info_args)
        info_args = info_args.parse_args([])  # Empty args for info
        return info_cmd.run(info_args) 