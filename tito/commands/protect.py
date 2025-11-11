"""
🛡️ Protection command for TinyTorch CLI: Student protection system management.

Industry-standard approach to prevent students from accidentally breaking
critical Variable/Tensor compatibility fixes that enable CIFAR-10 training.
"""

import os
import stat
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import BaseCommand


class ProtectCommand(BaseCommand):
    """🛡️ Student Protection System for TinyTorch core files."""
    
    @property
    def name(self) -> str:
        return "protect"

    @property
    def description(self) -> str:
        return "🛡️ Student protection system to prevent accidental core file edits"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='protect_command',
            help='Protection subcommands',
            metavar='SUBCOMMAND'
        )
        
        # Enable protection
        enable_parser = subparsers.add_parser(
            'enable',
            help='🔒 Enable comprehensive student protection system'
        )
        enable_parser.add_argument(
            '--force',
            action='store_true',
            help='Force enable even if already protected'
        )
        
        # Disable protection (for development)
        disable_parser = subparsers.add_parser(
            'disable',
            help='🔓 Disable protection system (for development only)'
        )
        disable_parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm disabling protection'
        )
        
        # Check protection status
        status_parser = subparsers.add_parser(
            'status',
            help='🔍 Check current protection status'
        )
        
        # Validate core functionality
        validate_parser = subparsers.add_parser(
            'validate',
            help='✅ Validate core functionality works correctly'
        )
        validate_parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed validation output'
        )
        
        # Quick health check
        check_parser = subparsers.add_parser(
            'check',
            help='⚡ Quick health check of critical functionality'
        )

    def run(self, args: Namespace) -> int:
        """Execute the protection command."""
        console = Console()
        
        # Show header
        console.print(Panel.fit(
            "🛡️ [bold blue]TinyTorch Student Protection System[/bold blue]\n"
            "Prevents accidental edits to critical core functionality",
            border_style="blue"
        ))
        
        # Route to appropriate subcommand
        if args.protect_command == 'enable':
            return self._enable_protection(console, args)
        elif args.protect_command == 'disable':
            return self._disable_protection(console, args)
        elif args.protect_command == 'status':
            return self._show_protection_status(console)
        elif args.protect_command == 'validate':
            return self._validate_functionality(console, args)
        elif args.protect_command == 'check':
            return self._quick_health_check(console)
        else:
            console.print("[red]❌ No protection subcommand specified[/red]")
            console.print("Use: [yellow]tito system protect --help[/yellow]")
            return 1
    
    def _enable_protection(self, console: Console, args: Namespace) -> int:
        """🔒 Enable comprehensive protection system."""
        console.print("[blue]🔒 Enabling TinyTorch Student Protection System...[/blue]")
        console.print()
        
        protection_count = 0
        
        # 1. Set file permissions
        tinytorch_core = Path("tinytorch/core")
        if tinytorch_core.exists():
            console.print("[yellow]🔒 Setting core files to read-only...[/yellow]")
            for py_file in tinytorch_core.glob("*.py"):
                try:
                    # Make file read-only
                    py_file.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                    protection_count += 1
                except OSError as e:
                    console.print(f"[red]⚠️  Could not protect {py_file}: {e}[/red]")
            console.print(f"[green]✅ Protected {protection_count} core files[/green]")
        else:
            console.print("[yellow]⚠️  tinytorch/core/ not found - run export first[/yellow]")
        
        # 2. Create .gitattributes
        console.print("[yellow]📝 Setting up Git attributes...[/yellow]")
        gitattributes_content = """# 🛡️ TinyTorch Protection: Mark auto-generated files
# GitHub will show "Generated" label for these files
tinytorch/core/*.py linguist-generated=true
tinytorch/**/*.py linguist-generated=true

# Exclude from diff by default (reduces noise in pull requests)
tinytorch/core/*.py -diff
"""
        with open(".gitattributes", "w") as f:
            f.write(gitattributes_content)
        console.print("[green]✅ Git attributes configured[/green]")
        
        # 3. Create pre-commit hook
        console.print("[yellow]🚫 Installing Git pre-commit hook...[/yellow]")
        git_hooks_dir = Path(".git/hooks")
        if git_hooks_dir.exists():
            precommit_hook = git_hooks_dir / "pre-commit"
            hook_content = """#!/bin/bash
# 🛡️ TinyTorch Protection: Prevent committing auto-generated files

echo "🛡️ Checking for modifications to auto-generated files..."

# Check if any tinytorch/core files are staged
CORE_FILES_MODIFIED=$(git diff --cached --name-only | grep "^tinytorch/core/")

if [ ! -z "$CORE_FILES_MODIFIED" ]; then
    echo ""
    echo "🚨 ERROR: Attempting to commit auto-generated files!"
    echo "=========================================="
    echo ""
    echo "The following auto-generated files are staged:"
    echo "$CORE_FILES_MODIFIED"
    echo ""
    echo "🛡️ PROTECTION TRIGGERED: These files are auto-generated from modules/"
    echo ""
    echo "TO FIX:"
    echo "1. Unstage these files: git reset HEAD tinytorch/core/"
    echo "2. Make changes in modules/ instead"
    echo "3. Run: tito module complete <module_name>"
    echo "4. Commit the source changes, not the generated files"
    echo ""
    echo "⚠️  This protection prevents breaking CIFAR-10 training!"
    echo ""
    exit 1
fi

echo "✅ No auto-generated files being committed"
"""
            with open(precommit_hook, "w") as f:
                f.write(hook_content)
            precommit_hook.chmod(0o755)  # Make executable
            console.print("[green]✅ Git pre-commit hook installed[/green]")
        else:
            console.print("[yellow]⚠️  .git directory not found - skipping Git hooks[/yellow]")
        
        # 4. Create VSCode settings
        console.print("[yellow]⚙️  Setting up VSCode protection...[/yellow]")
        vscode_dir = Path(".vscode")
        vscode_dir.mkdir(exist_ok=True)
        
        python_default_interpreter = str(self.venv_path) + "/bin/python"
        vscode_settings = {
            "_comment_protection": "🛡️ TinyTorch Student Protection",
            "files.readonlyInclude": {
                "**/tinytorch/core/**/*.py": True
            },
            "files.readonlyFromPermissions": True,
            "files.decorations.colors": True,
            "files.decorations.badges": True,
            "explorer.decorations.colors": True,
            "explorer.decorations.badges": True,
            "python.defaultInterpreterPath": python_default_interpreter,
            "python.terminal.activateEnvironment": True
        }
        
        import json
        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(vscode_settings, f, indent=4)
        console.print("[green]✅ VSCode protection configured[/green]")
        
        console.print()
        console.print(Panel.fit(
            "[green]🎉 Protection System Activated![/green]\n\n"
            "🔒 Core files are read-only\n"
            "📝 GitHub will label files as 'Generated'\n"
            "🚫 Git prevents committing generated files\n"
            "⚙️  VSCode shows protection warnings\n\n"
            "[blue]Students are now protected from breaking CIFAR-10 training![/blue]",
            border_style="green"
        ))
        
        return 0
    
    def _disable_protection(self, console: Console, args: Namespace) -> int:
        """🔓 Disable protection system (for development)."""
        if not args.confirm:
            console.print("[red]❌ Protection disable requires --confirm flag[/red]")
            console.print("[yellow]This is to prevent accidental disabling[/yellow]")
            return 1
        
        console.print("[yellow]🔓 Disabling TinyTorch Protection System...[/yellow]")
        
        # Reset file permissions
        tinytorch_core = Path("tinytorch/core")
        if tinytorch_core.exists():
            for py_file in tinytorch_core.glob("*.py"):
                try:
                    py_file.chmod(0o644)  # Reset to normal permissions
                except OSError:
                    pass
        
        # Remove protection files
        protection_files = [".gitattributes", ".git/hooks/pre-commit", ".vscode/settings.json"]
        for file_path in protection_files:
            path = Path(file_path)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        
        console.print("[green]✅ Protection system disabled[/green]")
        console.print("[red]⚠️  Remember to re-enable before students use the system![/red]")
        
        return 0
    
    def _show_protection_status(self, console: Console) -> int:
        """🔍 Show current protection status."""
        console.print("[blue]🔍 TinyTorch Protection Status[/blue]")
        console.print()
        
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Protection Feature", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        # Check file permissions
        tinytorch_core = Path("tinytorch/core")
        if tinytorch_core.exists():
            readonly_count = 0
            total_files = 0
            for py_file in tinytorch_core.glob("*.py"):
                total_files += 1
                if not (py_file.stat().st_mode & stat.S_IWRITE):
                    readonly_count += 1
            
            if readonly_count == total_files and total_files > 0:
                table.add_row("🔒 File Permissions", "[green]✅ PROTECTED[/green]", f"{readonly_count}/{total_files} files read-only")
            elif readonly_count > 0:
                table.add_row("🔒 File Permissions", "[yellow]⚠️  PARTIAL[/yellow]", f"{readonly_count}/{total_files} files read-only")
            else:
                table.add_row("🔒 File Permissions", "[red]❌ UNPROTECTED[/red]", "Files are writable")
        else:
            table.add_row("🔒 File Permissions", "[yellow]⚠️  N/A[/yellow]", "tinytorch/core/ not found")
        
        # Check Git attributes
        gitattributes = Path(".gitattributes")
        if gitattributes.exists():
            table.add_row("📝 Git Attributes", "[green]✅ CONFIGURED[/green]", "Generated files marked")
        else:
            table.add_row("📝 Git Attributes", "[red]❌ MISSING[/red]", "No .gitattributes file")
        
        # Check pre-commit hook
        precommit_hook = Path(".git/hooks/pre-commit")
        if precommit_hook.exists():
            table.add_row("🚫 Git Pre-commit", "[green]✅ ACTIVE[/green]", "Prevents core file commits")
        else:
            table.add_row("🚫 Git Pre-commit", "[red]❌ MISSING[/red]", "No pre-commit protection")
        
        # Check VSCode settings
        vscode_settings = Path(".vscode/settings.json")
        if vscode_settings.exists():
            table.add_row("⚙️  VSCode Protection", "[green]✅ CONFIGURED[/green]", "Editor warnings enabled")
        else:
            table.add_row("⚙️  VSCode Protection", "[yellow]⚠️  MISSING[/yellow]", "No VSCode settings")
        
        console.print(table)
        console.print()
        
        # Overall status
        protection_features = [
            tinytorch_core.exists() and all(not (f.stat().st_mode & stat.S_IWRITE) for f in tinytorch_core.glob("*.py")),
            gitattributes.exists(),
            precommit_hook.exists()
        ]
        
        if all(protection_features):
            console.print("[green]🛡️  Overall Status: FULLY PROTECTED[/green]")
        elif any(protection_features):
            console.print("[yellow]🛡️  Overall Status: PARTIALLY PROTECTED[/yellow]")
            console.print("[yellow]💡 Run 'tito system protect enable' to complete protection[/yellow]")
        else:
            console.print("[red]🛡️  Overall Status: UNPROTECTED[/red]")
            console.print("[red]⚠️  Run 'tito system protect enable' to protect against student errors[/red]")
        
        return 0
    
    def _validate_functionality(self, console: Console, args: Namespace) -> int:
        """✅ Validate core functionality works correctly."""
        try:
            from tinytorch.core._validation import run_student_protection_checks
            console.print("[blue]🔍 Running comprehensive validation...[/blue]")
            console.print()
            
            try:
                run_student_protection_checks(verbose=args.verbose)
                console.print()
                console.print("[green]🎉 All validation checks passed![/green]")
                console.print("[green]✅ CIFAR-10 training should work correctly[/green]")
                return 0
            except Exception as e:
                console.print()
                console.print(f"[red]❌ Validation failed: {e}[/red]")
                console.print("[red]⚠️  CIFAR-10 training may not work properly[/red]")
                console.print("[yellow]💡 Check if core files have been accidentally modified[/yellow]")
                return 1
                
        except ImportError:
            console.print("[red]❌ Validation system not available[/red]")
            console.print("[yellow]💡 Run module export to generate validation system[/yellow]")
            return 1
    
    def _quick_health_check(self, console: Console) -> int:
        """⚡ Quick health check of critical functionality."""
        console.print("[blue]⚡ Quick Health Check[/blue]")
        console.print()
        
        checks = []
        
        # Check if core modules can be imported
        try:
            from tinytorch.core.tensor import Tensor
            checks.append(("Core Tensor", True, "Import successful"))
        except Exception as e:
            checks.append(("Core Tensor", False, str(e)))
        
        try:
            from tinytorch.core.autograd import Variable
            checks.append(("Core Autograd", True, "Import successful"))
        except Exception as e:
            checks.append(("Core Autograd", False, str(e)))
        
        try:
            from tinytorch.core.layers import matmul
            checks.append(("Core Layers", True, "Import successful"))
        except Exception as e:
            checks.append(("Core Layers", False, str(e)))
        
        # Quick Variable/Tensor compatibility test
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.autograd import Variable
            from tinytorch.core.layers import matmul
            
            a = Variable(Tensor([[1, 2]]), requires_grad=True)
            b = Variable(Tensor([[3], [4]]), requires_grad=True)
            result = matmul(a, b)
            
            if hasattr(result, 'requires_grad'):
                checks.append(("Variable Compatibility", True, "matmul works with Variables"))
            else:
                checks.append(("Variable Compatibility", False, "matmul doesn't return Variables"))
                
        except Exception as e:
            checks.append(("Variable Compatibility", False, str(e)))
        
        # Display results
        for check_name, passed, details in checks:
            status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
            console.print(f"{status} {check_name}: {details}")
        
        console.print()
        
        # Overall status
        all_passed = all(passed for _, passed, _ in checks)
        if all_passed:
            console.print("[green]🎉 All health checks passed![/green]")
            return 0
        else:
            console.print("[red]❌ Some health checks failed[/red]")
            console.print("[yellow]💡 Run 'tito system protect validate --verbose' for details[/yellow]")
            return 1