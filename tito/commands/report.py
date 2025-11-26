"""
Report command for TinyTorch CLI: generate comprehensive diagnostic report.

This command generates a JSON report containing all environment information,
perfect for sharing with TAs, instructors, or when filing bug reports.
"""

import sys
import os
import platform
import shutil
import json
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand


class ReportCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "report"

    @property
    def description(self) -> str:
        return "Generate comprehensive diagnostic report (JSON)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '-o', '--output',
            type=str,
            default=None,
            help='Output file path (default: tinytorch-report-TIMESTAMP.json)'
        )
        parser.add_argument(
            '--stdout',
            action='store_true',
            help='Print JSON to stdout instead of file'
        )

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print()
        console.print(Panel(
            "📋 Generating TinyTorch Diagnostic Report\n\n"
            "[dim]This report contains all environment information\n"
            "needed for debugging and support.[/dim]",
            title="System Report",
            border_style="bright_yellow"
        ))
        console.print()

        # Collect all diagnostic information
        report = self._collect_report_data()

        # Determine output path
        if args.stdout:
            # Print to stdout
            print(json.dumps(report, indent=2))
            return 0
        else:
            # Write to file
            if args.output:
                output_path = Path(args.output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_path = Path.cwd() / f"tinytorch-report-{timestamp}.json"

            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)

                console.print(Panel(
                    f"[bold green]✅ Report Generated Successfully![/bold green]\n\n"
                    f"📄 File: [cyan]{output_path}[/cyan]\n"
                    f"📦 Size: {output_path.stat().st_size} bytes\n\n"
                    f"[dim]Share this file with your TA or instructor for support.[/dim]",
                    title="Report Complete",
                    border_style="green"
                ))

                return 0
            except Exception as e:
                console.print(Panel(
                    f"[red]❌ Failed to write report:[/red]\n\n{str(e)}",
                    title="Error",
                    border_style="red"
                ))
                return 1

    def _collect_report_data(self) -> dict:
        """Collect comprehensive diagnostic information."""
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "tinytorch_cli_version": self._get_tinytorch_version()
            },
            "system": self._collect_system_info(),
            "python": self._collect_python_info(),
            "environment": self._collect_environment_info(),
            "dependencies": self._collect_dependencies_info(),
            "tinytorch": self._collect_tinytorch_info(),
            "modules": self._collect_modules_info(),
            "git": self._collect_git_info(),
            "disk_memory": self._collect_disk_memory_info()
        }
        return report

    def _get_tinytorch_version(self) -> str:
        """Get TinyTorch version."""
        try:
            import tinytorch
            return getattr(tinytorch, '__version__', 'unknown')
        except ImportError:
            return "not_installed"

    def _collect_system_info(self) -> dict:
        """Collect system information."""
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "node": platform.node()
        }

    def _collect_python_info(self) -> dict:
        """Collect Python interpreter information."""
        return {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
                "serial": sys.version_info.serial
            },
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, 'base_prefix', sys.prefix),
            "path": sys.path
        }

    def _collect_environment_info(self) -> dict:
        """Collect environment variables and paths."""
        venv_exists = self.venv_path.exists()
        in_venv = (
            os.environ.get('VIRTUAL_ENV') is not None or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            hasattr(sys, 'real_prefix')
        )

        return {
            "working_directory": str(Path.cwd()),
            "virtual_environment": {
                "exists": venv_exists,
                "active": in_venv,
                "path": str(self.venv_path) if venv_exists else None,
                "VIRTUAL_ENV": os.environ.get('VIRTUAL_ENV'),
                "CONDA_DEFAULT_ENV": os.environ.get('CONDA_DEFAULT_ENV'),
                "CONDA_PREFIX": os.environ.get('CONDA_PREFIX')
            },
            "PATH": os.environ.get('PATH', '').split(os.pathsep),
            "PYTHONPATH": os.environ.get('PYTHONPATH', '').split(os.pathsep) if os.environ.get('PYTHONPATH') else []
        }

    def _collect_dependencies_info(self) -> dict:
        """Collect installed package information."""
        dependencies = {}

        # Core dependencies
        packages = [
            ('numpy', 'numpy'),
            ('pytest', 'pytest'),
            ('PyYAML', 'yaml'),
            ('rich', 'rich'),
            ('jupyterlab', 'jupyterlab'),
            ('jupytext', 'jupytext'),
            ('nbformat', 'nbformat'),
            ('nbgrader', 'nbgrader'),
            ('nbconvert', 'nbconvert'),
            ('jupyter', 'jupyter'),
            ('matplotlib', 'matplotlib'),
            ('psutil', 'psutil'),
            ('black', 'black'),
            ('isort', 'isort'),
            ('flake8', 'flake8')
        ]

        for display_name, import_name in packages:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                location = getattr(module, '__file__', 'unknown')
                dependencies[display_name] = {
                    "installed": True,
                    "version": version,
                    "location": location
                }
            except ImportError:
                dependencies[display_name] = {
                    "installed": False,
                    "version": None,
                    "location": None
                }

        return dependencies

    def _collect_tinytorch_info(self) -> dict:
        """Collect TinyTorch package information."""
        try:
            import tinytorch
            version = getattr(tinytorch, '__version__', 'unknown')
            location = Path(tinytorch.__file__).parent

            # Check if in development mode
            is_dev = (location / '../setup.py').exists() or (location / '../pyproject.toml').exists()

            return {
                "installed": True,
                "version": version,
                "location": str(location),
                "development_mode": is_dev,
                "package_structure": {
                    "has_init": (location / '__init__.py').exists(),
                    "has_core": (location / 'core').exists(),
                    "has_ops": (location / 'ops').exists()
                }
            }
        except ImportError:
            return {
                "installed": False,
                "version": None,
                "location": None,
                "development_mode": False,
                "package_structure": {}
            }

    def _collect_modules_info(self) -> dict:
        """Collect TinyTorch modules information."""
        modules_dir = Path.cwd() / "modules"

        if not modules_dir.exists():
            return {"exists": False, "modules": []}

        modules = []
        for module_path in sorted(modules_dir.iterdir()):
            if module_path.is_dir() and module_path.name.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                module_info = {
                    "name": module_path.name,
                    "path": str(module_path),
                    "has_notebook": any(module_path.glob("*.ipynb")),
                    "has_dev_py": any(module_path.glob("*_dev.py")),
                    "has_tests": (module_path / "tests").exists()
                }
                modules.append(module_info)

        return {
            "exists": True,
            "count": len(modules),
            "modules": modules
        }

    def _collect_git_info(self) -> dict:
        """Collect git repository information."""
        git_dir = Path.cwd() / ".git"

        if not git_dir.exists():
            return {"is_repo": False}

        try:
            import subprocess

            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get current commit
            commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get remote URL
            try:
                remote = subprocess.check_output(
                    ['git', 'remote', 'get-url', 'origin'],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
            except:
                remote = None

            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            has_changes = len(status) > 0

            return {
                "is_repo": True,
                "branch": branch,
                "commit": commit,
                "remote": remote,
                "has_uncommitted_changes": has_changes,
                "status": status if has_changes else None
            }
        except:
            return {"is_repo": True, "error": "Failed to get git info"}

    def _collect_disk_memory_info(self) -> dict:
        """Collect disk space and memory information."""
        info = {}

        # Disk space
        try:
            disk_usage = shutil.disk_usage(Path.cwd())
            info["disk"] = {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "percent_used": round((disk_usage.used / disk_usage.total) * 100, 1)
            }
        except Exception as e:
            info["disk"] = {"error": str(e)}

        # Memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory"] = {
                "total_bytes": mem.total,
                "available_bytes": mem.available,
                "used_bytes": mem.used,
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "percent_used": mem.percent
            }
        except ImportError:
            info["memory"] = {"error": "psutil not installed"}
        except Exception as e:
            info["memory"] = {"error": str(e)}

        return info
