"""
Setup command for TinyTorch CLI: First-time environment setup and configuration.

This replaces the old 01_setup module with a proper CLI command that handles:
- Package installation and virtual environment setup
- Environment validation and compatibility checking  
- User profile creation for development tracking
- Workspace initialization for TinyTorch development
"""

import subprocess
import sys
import os
import platform
import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import BaseCommand
from ..core.console import get_console

class SetupCommand(BaseCommand):
    """First-time setup command for TinyTorch development environment."""
    
    @property
    def name(self) -> str:
        return "setup"
    
    @property
    def description(self) -> str:
        return "First-time setup: install packages, create profile, initialize workspace"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add setup command arguments."""
        parser.add_argument(
            '--skip-venv',
            action='store_true',
            help='Skip virtual environment creation'
        )
        parser.add_argument(
            '--skip-packages',
            action='store_true', 
            help='Skip package installation'
        )
        parser.add_argument(
            '--skip-profile',
            action='store_true',
            help='Skip user profile creation'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force setup even if already configured'
        )
    
    def check_existing_setup(self) -> bool:
        """Check if TinyTorch is already set up."""
        # Check for profile file
        profile_path = self.config.project_root / "profile.json"
        
        # Check for virtual environment
        venv_paths = [
            self.config.project_root / "venv",
            self.config.project_root / "tinytorch-env",
            Path.home() / ".tinytorch" / "venv"
        ]
        
        has_profile = profile_path.exists()
        has_venv = any(venv_path.exists() for venv_path in venv_paths)
        
        return has_profile and has_venv
    
    def install_packages(self) -> bool:
        """Install required packages for TinyTorch development."""
        self.console.print("📦 Installing TinyTorch dependencies...")
        
        # Essential packages for TinyTorch
        packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "jupytext>=1.13.0",
            "rich>=12.0.0",
            "pyyaml>=6.0",
            "psutil>=5.8.0"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            for package in packages:
                task = progress.add_task(f"Installing {package.split('>=')[0]}...", total=None)
                
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        progress.update(task, description=f"✅ {package.split('>=')[0]} installed")
                    else:
                        progress.update(task, description=f"❌ {package.split('>=')[0]} failed")
                        self.console.print(f"[red]Error installing {package}: {result.stderr}[/red]")
                        return False
                        
                except subprocess.TimeoutExpired:
                    progress.update(task, description=f"⏰ {package.split('>=')[0]} timed out")
                    self.console.print(f"[yellow]Warning: {package} installation timed out[/yellow]")
                except Exception as e:
                    progress.update(task, description=f"❌ {package.split('>=')[0]} error")
                    self.console.print(f"[red]Error installing {package}: {e}[/red]")
                    return False
        
        # Install TinyTorch in development mode
        try:
            self.console.print("🔧 Installing TinyTorch in development mode...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], cwd=self.config.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print("✅ TinyTorch installed in development mode")
                return True
            else:
                self.console.print(f"[red]Failed to install TinyTorch: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]Error installing TinyTorch: {e}[/red]")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment for TinyTorch development."""
        self.console.print("🐍 Setting up virtual environment...")
        
        venv_path = self.config.project_root / "tinytorch-env"
        
        if venv_path.exists():
            if not Confirm.ask(f"Virtual environment already exists at {venv_path}. Recreate?"):
                return True
            
            import shutil
            shutil.rmtree(venv_path)
        
        try:
            # Create virtual environment
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.console.print(f"[red]Failed to create virtual environment: {result.stderr}[/red]")
                return False
            
            self.console.print(f"✅ Virtual environment created at {venv_path}")
            
            # Create activation script
            self.create_activation_script(venv_path)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error creating virtual environment: {e}[/red]")
            return False
    
    def create_activation_script(self, venv_path: Path) -> None:
        """Create a convenient activation script."""
        script_path = self.config.project_root / "activate-tinytorch.sh"
        
        script_content = f"""#!/bin/bash
# TinyTorch Development Environment Activation Script

echo "🔥 Activating TinyTorch development environment..."

# Activate virtual environment
source {venv_path}/bin/activate

# Set environment variables
export TINYTORCH_DEV=1
export PYTHONPATH="${{PYTHONPATH}}:{self.config.project_root}"

echo "✅ TinyTorch environment activated!"
echo "💡 Try: tito 01 (to start with tensors)"
echo "🔄 To deactivate: deactivate"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        self.console.print(f"✅ Activation script created: {script_path}")
        self.console.print("💡 Use: source activate-tinytorch.sh")
    
    def create_user_profile(self) -> Dict[str, Any]:
        """Create user profile for development tracking."""
        self.console.print("👋 Creating your TinyTorch development profile...")
        
        profile_path = self.config.project_root / "profile.json"
        
        if profile_path.exists():
            if not Confirm.ask("Profile already exists. Update it?"):
                import json
                with open(profile_path, 'r') as f:
                    return json.load(f)
        
        # Collect user information
        name = Prompt.ask("Your name", default="TinyTorch Developer")
        email = Prompt.ask("Your email (optional)", default="dev@tinytorch.local")
        affiliation = Prompt.ask("Your affiliation (university, company, etc.)", default="Independent")
        
        # Create profile
        profile = {
            "name": name,
            "email": email,
            "affiliation": affiliation,
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created": datetime.datetime.now().isoformat(),
            "setup_version": "2.0",
            "modules_completed": [],
            "last_active": datetime.datetime.now().isoformat()
        }
        
        # Save profile
        import json
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        self.console.print(f"✅ Profile created for {profile['name']}")
        return profile
    
    def validate_environment(self) -> bool:
        """Validate the development environment setup."""
        self.console.print("🔍 Validating environment...")
        
        checks = [
            ("Python version", self.check_python_version),
            ("NumPy installation", self.check_numpy),
            ("Jupyter installation", self.check_jupyter),
            ("TinyTorch package", self.check_tinytorch_package)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.console.print(f"  ✅ {check_name}")
                else:
                    self.console.print(f"  ❌ {check_name}")
                    all_passed = False
            except Exception as e:
                self.console.print(f"  ❌ {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        return sys.version_info >= (3, 8)
    
    def check_numpy(self) -> bool:
        """Check if NumPy is installed and working."""
        try:
            import numpy as np
            # Test basic operation
            arr = np.array([1, 2, 3])
            return len(arr) == 3
        except ImportError:
            return False
    
    def check_jupyter(self) -> bool:
        """Check if Jupyter is installed."""
        try:
            import jupyter
            import jupyterlab
            return True
        except ImportError:
            return False
    
    def check_tinytorch_package(self) -> bool:
        """Check if TinyTorch package is installed."""
        try:
            import tinytorch
            return True
        except ImportError:
            return False
    
    def print_success_message(self, profile: Dict[str, Any]) -> None:
        """Print success message with next steps."""
        success_text = Text()
        success_text.append("🎉 TinyTorch setup completed successfully!\n\n", style="bold green")
        success_text.append(f"👋 Welcome, {profile['name']}!\n", style="bold")
        success_text.append(f"📧 Email: {profile['email']}\n", style="dim")
        success_text.append(f"🏢 Affiliation: {profile['affiliation']}\n", style="dim")
        success_text.append(f"💻 Platform: {profile['platform']}\n", style="dim")
        success_text.append(f"🐍 Python: {profile['python_version']}\n\n", style="dim")
        
        success_text.append("🚀 Ready to start building ML systems!\n\n", style="bold cyan")
        success_text.append("Next steps:\n", style="bold")
        success_text.append("  1. ", style="dim")
        success_text.append("tito 01", style="bold green")
        success_text.append(" - Start with tensors (the foundation)\n", style="dim")
        success_text.append("  2. ", style="dim")
        success_text.append("tito 02", style="bold green") 
        success_text.append(" - Add activation functions\n", style="dim")
        success_text.append("  3. ", style="dim")
        success_text.append("tito 03", style="bold green")
        success_text.append(" - Build neural network layers\n", style="dim")
        
        self.console.print(Panel(
            success_text,
            title="🔥 TinyTorch Setup Complete!",
            border_style="green"
        ))
    
    def run(self, args: Namespace) -> int:
        """Execute the setup command."""
        self.console.print(Panel(
            "🔥 TinyTorch First-Time Setup\n\n"
            "This will configure your development environment for building ML systems from scratch.",
            title="Welcome to TinyTorch!",
            border_style="bright_green"
        ))
        
        # Check if already set up
        if not args.force and self.check_existing_setup():
            if not Confirm.ask("TinyTorch appears to be already set up. Continue anyway?"):
                self.console.print("✅ Setup cancelled. You're ready to go!")
                self.console.print("💡 Try: tito 01")
                return 0
        
        try:
            # Step 1: Virtual environment (optional)
            if not args.skip_venv:
                if not self.create_virtual_environment():
                    self.console.print("[yellow]⚠️  Virtual environment setup failed, but continuing...[/yellow]")
            
            # Step 2: Install packages
            if not args.skip_packages:
                if not self.install_packages():
                    self.console.print("[red]❌ Package installation failed[/red]")
                    return 1
            
            # Step 3: Create user profile
            profile = {}
            if not args.skip_profile:
                profile = self.create_user_profile()
            
            # Step 4: Validate environment
            if not self.validate_environment():
                self.console.print("[yellow]⚠️  Some validation checks failed, but setup completed[/yellow]")
            
            # Success!
            if profile:  # Only print if profile was created
                self.print_success_message(profile)
            else:
                self.console.print("✅ Setup completed successfully!")
                self.console.print("💡 Try: tito 01")
            return 0
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return 130
        except Exception as e:
            self.console.print(f"[red]Setup failed: {e}[/red]")
            return 1
