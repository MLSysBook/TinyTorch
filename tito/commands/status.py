"""
Status command for TinyTorch CLI: checks status of all modules in modules/ directory.
"""

import subprocess
import sys
import yaml
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import BaseCommand

class StatusCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "status"

    @property
    def description(self) -> str:
        return "Check status of all modules"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--details", action="store_true", help="Show detailed file structure")
        parser.add_argument("--metadata", action="store_true", help="Show module metadata information")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Scan modules directory
        modules_dir = Path("modules")
        if not modules_dir.exists():
            console.print(Panel("[red]❌ modules/ directory not found[/red]", 
                              title="Error", border_style="red"))
            return 1
        
        # Find all module directories (exclude special directories)
        exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
        module_dirs = [d for d in modules_dir.iterdir() 
                      if d.is_dir() and d.name not in exclude_dirs]
        
        if not module_dirs:
            console.print(Panel("[yellow]⚠️  No modules found in modules/ directory[/yellow]", 
                              title="Warning", border_style="yellow"))
            return 0
        
        console.print(Panel(f"📋 Found {len(module_dirs)} modules in modules/ directory", 
                          title="Module Status Check", border_style="bright_cyan"))
        
        # Create status table
        status_table = Table(title="Module Status Overview", show_header=True, header_style="bold blue")
        status_table.add_column("Module", style="bold cyan", width=15)
        status_table.add_column("Status", width=12, justify="center")
        status_table.add_column("Dev File", width=12, justify="center")
        status_table.add_column("Tests", width=12, justify="center")
        status_table.add_column("README", width=12, justify="center")
        
        if args.metadata:
            status_table.add_column("Export", width=15, justify="center")
            status_table.add_column("Components", width=15, justify="center")
        
        # Check each module
        modules_status = []
        for module_dir in sorted(module_dirs):
            module_name = module_dir.name
            status = self._check_module_status(module_dir)
            modules_status.append((module_name, status))
            
            # Add to table
            row = [
                module_name,
                self._format_status(status.get('metadata', {}).get('status', 'unknown')),
                "✅" if status['dev_file'] else "❌",
                "✅" if status['tests'] else "❌", 
                "✅" if status['readme'] else "❌"
            ]
            
            # Add metadata columns if requested
            if args.metadata:
                metadata = status.get('metadata', {})
                row.append(metadata.get('exports_to', 'unknown'))
                components = metadata.get('components', [])
                row.append(f"{len(components)} items" if components else 'none')
            
            status_table.add_row(*row)
        
        console.print(status_table)
        
        # Summary
        total_modules = len(modules_status)
        complete_modules = sum(1 for _, status in modules_status 
                             if status['dev_file'] and status['tests'] and status['readme'])
        
        console.print(f"\n📊 Summary: {complete_modules}/{total_modules} modules complete")
        console.print(f"💡 To run tests: [bold cyan]tito test --all[/bold cyan]")
        
        # Detailed view
        if args.details:
            console.print("\n" + "="*60)
            console.print("📁 Detailed Module Structure")
            console.print("="*60)
            
            for module_name, status in modules_status:
                self._print_module_details(module_name, status)
        
        # Metadata view
        if args.metadata:
            console.print("\n" + "="*60)
            console.print("📊 Module Metadata")
            console.print("="*60)
            
            for module_name, status in modules_status:
                if status.get('metadata'):
                    self._print_module_metadata(module_name, status['metadata'])
        
        return 0
    
    def _check_module_status(self, module_dir: Path) -> dict:
        """Check the status of a single module."""
        module_name = module_dir.name
        
        # Check for required files
        dev_file = module_dir / f"{module_name}_dev.py"
        tests_dir = module_dir / "tests"
        test_file = tests_dir / f"test_{module_name}.py"
        readme_file = module_dir / "README.md"
        metadata_file = module_dir / "module.yaml"
        
        status = {
            'dev_file': dev_file.exists(),
            'tests': test_file.exists(),
            'readme': readme_file.exists(),
            'metadata_file': metadata_file.exists(),
        }
        
        # Load metadata if available
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                    status['metadata'] = metadata
            except Exception as e:
                status['metadata'] = {'error': str(e)}
        
        return status
    
    def _print_module_details(self, module_name: str, status: dict) -> None:
        """Print detailed information about a module."""
        console = self.console
        
        # Module header
        console.print(f"\n📦 {module_name.upper()}", style="bold cyan")
        console.print("-" * 40)
        
        # File structure
        files_table = Table(show_header=False, box=None, padding=(0, 2))
        files_table.add_column("File", style="dim")
        files_table.add_column("Status")
        
        files_table.add_row(f"{module_name}_dev.py", "✅ Found" if status['dev_file'] else "❌ Missing")
        files_table.add_row("tests/test_*.py", "✅ Found" if status['tests'] else "❌ Missing")
        files_table.add_row("README.md", "✅ Found" if status['readme'] else "❌ Missing")
        
        console.print(files_table)
    
    def _format_status(self, status: str) -> str:
        """Format module status with appropriate emoji and color."""
        status_map = {
            'complete': '✅',
            'in_progress': '🚧',
            'not_started': '❌',
            'deprecated': '⚠️',
            'unknown': '❓'
        }
        return status_map.get(status, '❓')
    
    def _print_module_metadata(self, module_name: str, metadata: dict) -> None:
        """Print detailed metadata information about a module."""
        console = self.console
        
        # Module header
        title = metadata.get('title', module_name.title())
        console.print(f"\n📦 {title}", style="bold cyan")
        console.print("-" * (len(title) + 4))
        
        # Basic info
        if metadata.get('description'):
            console.print(f"📝 {metadata['description']}")
        
        # Status and export info
        status_info = []
        if metadata.get('status'):
            status_info.append(f"Status: {self._format_status(metadata['status'])} {metadata['status']}")
        if metadata.get('exports_to'):
            status_info.append(f"Exports to: {metadata['exports_to']}")
        
        if status_info:
            console.print(" | ".join(status_info))
        
        # Dependencies
        if metadata.get('dependencies'):
            deps = metadata['dependencies']
            console.print("\n🔗 Dependencies:")
            if deps.get('prerequisites'):
                console.print(f"  Prerequisites: {', '.join(deps['prerequisites'])}")
            if deps.get('enables'):
                console.print(f"  Enables: {', '.join(deps['enables'])}")
        
        # Components
        if metadata.get('components'):
            console.print("\n🧩 Components:")
            for component in metadata['components']:
                console.print(f"  • {component}")
        
        # Files
        if metadata.get('files'):
            files = metadata['files']
            console.print("\n📁 Files:")
            if files.get('dev_file'):
                console.print(f"  • Dev: {files['dev_file']}")
            if files.get('test_file'):
                console.print(f"  • Test: {files['test_file']}")
            if files.get('readme'):
                console.print(f"  • README: {files['readme']}") 