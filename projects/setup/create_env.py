#!/usr/bin/env python3
"""
TinyTorch Environment Setup Script

This script automatically creates a virtual environment and installs
all required dependencies for the TinyTorch course.

Usage: python projects/setup/create_env.py
"""

import sys
import subprocess
import os
from pathlib import Path

def print_step(step, message):
    """Print a formatted step message."""
    print(f"\n🔥 Step {step}: {message}")
    print("-" * 50)

def run_command(cmd, check=True):
    """Run a command and handle errors gracefully."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Need Python 3.8+")
        print("Please install Python 3.8+ and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected (compatible)")
    return True

def create_virtual_environment():
    """Create the TinyTorch virtual environment."""
    env_path = Path("tinytorch-env")
    
    if env_path.exists():
        print(f"⚠️  Virtual environment already exists at {env_path}")
        response = input("Remove and recreate? [y/N]: ").lower().strip()
        if response == 'y':
            import shutil
            shutil.rmtree(env_path)
        else:
            print("Using existing environment...")
            return True
    
    # Create virtual environment
    result = run_command([sys.executable, "-m", "venv", "tinytorch-env"])
    if result is None:
        print("❌ Failed to create virtual environment")
        return False
    
    print("✅ Virtual environment created")
    return True

def get_venv_python():
    """Get the path to Python in the virtual environment."""
    if sys.platform == "win32":
        return Path("tinytorch-env/Scripts/python.exe")
    else:
        return Path("tinytorch-env/bin/python")

def install_dependencies():
    """Install required dependencies in the virtual environment."""
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print(f"❌ Virtual environment Python not found at {venv_python}")
        return False
    
    # Upgrade pip first
    print("Upgrading pip...")
    result = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    if result is None:
        return False
    
    # Install build tools first (required for Python 3.13+)
    print("Installing build tools...")
    result = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "setuptools", "wheel"])
    if result is None:
        return False
    
    # Try installing dependencies - first with requirements file
    print("Installing TinyTorch dependencies...")
    result = run_command([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])
    
    # If that fails, try installing core packages individually (fallback for Python 3.13)
    if result is None:
        print("⚠️  Requirements file failed, trying individual packages...")
        core_packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "PyYAML>=6.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0"
        ]
        
        for package in core_packages:
            print(f"Installing {package}...")
            result = run_command([str(venv_python), "-m", "pip", "install", package])
            if result is None:
                print(f"❌ Failed to install {package}")
                return False
    
    print("✅ Dependencies installed")
    return True

def verify_installation():
    """Verify that everything is installed correctly."""
    venv_python = get_venv_python()
    
    # Test core imports
    test_script = '''
import sys
try:
    import numpy
    import matplotlib
    import yaml
    import pytest
    print("✅ All core dependencies imported successfully")
    print(f"Python: {sys.version}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"Pytest: {pytest.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
    
    result = run_command([str(venv_python), "-c", test_script])
    return result is not None

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "=" * 60)
    print("🎉 Environment setup complete!")
    print("=" * 60)
    
    if sys.platform == "win32":
        activate_cmd = "tinytorch-env\\Scripts\\activate"
    else:
        activate_cmd = "source tinytorch-env/bin/activate"
    
    print(f"""
Next steps:

1. Activate your environment (do this every time you work):
   {activate_cmd}

2. Verify the setup:
   python3 projects/setup/check_setup.py

3. Start the first project:
   cd projects/setup/
   cat README.md

📝 Remember: Always activate your virtual environment before working!
""")

def main():
    """Run the complete environment setup."""
    print("🔥 TinyTorch Environment Setup 🔥")
    print("=" * 60)
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        return False
    
    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_virtual_environment():
        return False
    
    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        return False
    
    # Step 4: Verify installation
    print_step(4, "Verifying installation")
    if not verify_installation():
        return False
    
    # Print next steps
    print_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 