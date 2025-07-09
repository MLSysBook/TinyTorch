#!/usr/bin/env python3
"""
TinyTorch Setup Verification Script

This script performs comprehensive checks to ensure the student's 
environment is properly configured for the TinyTorch course.

Usage: python projects/setup/check_setup.py
"""

import sys
import os
import subprocess
import importlib.util

def print_header():
    """Print the verification header."""
    print("🔥 TinyTorch Setup Verification 🔥")
    print("=" * 50)
    print()

def check_python_version():
    """Verify Python version is 3.8+."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_virtual_environment():
    """Check if running in a virtual environment."""
    print("\n🐍 Checking virtual environment...")
    
    # Check if in virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✅ Virtual environment: Active")
        print(f"   Environment: {sys.prefix}")
        return True
    else:
        print("⚠️  Virtual environment: Not detected")
        print("   Recommendation: Use 'source tinytorch-env/bin/activate'")
        print("   (This is strongly recommended for consistency)")
        return True  # Don't fail, just warn

def check_dependencies():
    """Check that all required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'yaml': 'PyYAML',
        'pytest': 'pytest'
    }
    all_good = True
    
    for import_name, package_name in required_packages.items():
        try:
            if import_name == 'yaml':
                import yaml as module
            else:
                module = __import__(import_name)
            
            # Get version if available
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: missing")
            all_good = False
    
    if all_good:
        print("✅ Dependencies: All installed correctly")
    else:
        print("❌ Dependencies: Some packages missing")
        print("   Solution: Activate venv and run 'pip install -r requirements.txt'")
    
    return all_good

def check_cli_commands():
    """Test that tito CLI commands work."""
    print("\n🔧 Checking CLI commands...")
    
    try:
        # Test --version
        result = subprocess.run([sys.executable, 'bin/tito.py', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'Tiny🔥Torch' in result.stdout:
            print("✅ tito --version: Working")
        else:
            print("❌ tito --version: Failed")
            return False
        
        # Test info command
        result = subprocess.run([sys.executable, 'bin/tito.py', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'Implementation Status' in result.stdout:
            print("✅ tito info: Working")
        else:
            print("❌ tito info: Failed")
            return False
        
        print("✅ CLI commands: Working properly")
        return True
        
    except Exception as e:
        print(f"❌ CLI commands: Error - {e}")
        return False

def check_hello_implementation():
    """Check the student's hello_tinytorch implementation."""
    print("\n👋 Checking hello_tinytorch() implementation...")
    
    try:
        # Add project root to path
        project_root = os.path.join(os.path.dirname(__file__), '../..')
        sys.path.insert(0, project_root)
        
        from tinytorch.core.utils import hello_tinytorch
        
        # Test the function
        result = hello_tinytorch()
        
        if not isinstance(result, str):
            print(f"❌ hello_tinytorch(): Should return string, got {type(result)}")
            return False
        
        if len(result.strip()) == 0:
            print("❌ hello_tinytorch(): Should return non-empty string")
            return False
        
        if '🔥' not in result:
            print("❌ hello_tinytorch(): Should contain 🔥 emoji")
            return False
        
        print(f"✅ hello_tinytorch(): Implemented correctly")
        print(f"   Message: {result}")
        return True
        
    except ImportError as e:
        print(f"❌ hello_tinytorch(): Function not found - {e}")
        print("   Make sure you've added the function to tinytorch/core/utils.py")
        return False
    except Exception as e:
        print(f"❌ hello_tinytorch(): Error - {e}")
        return False

def run_test_suite():
    """Run the pytest test suite for setup."""
    print("\n🧪 Running test suite...")
    
    try:
        test_file = os.path.join(os.path.dirname(__file__), 'test_setup.py')
        result = subprocess.run([sys.executable, '-m', 'pytest', test_file, '-v'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Test suite: All tests passing")
            return True
        else:
            print("❌ Test suite: Some tests failing")
            print("   Test output:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test suite: Error running tests - {e}")
        return False

def print_summary(all_checks):
    """Print final summary."""
    print("\n" + "=" * 50)
    
    if all(all_checks):
        print("🎉 Setup complete! You're ready to build an ML system from scratch.")
        print("\nNext steps:")
        print("  cd ../tensor/")
        print("  cat README.md")
        print("\nYou can now submit this project:")
        print("  python bin/tito.py submit --project setup")
    else:
        print("❌ Setup incomplete. Please fix the issues above before continuing.")
        print("\nNeed help? Check the README.md or ask in office hours.")

def main():
    """Run all setup verification checks."""
    print_header()
    
    checks = [
        check_python_version(),
        check_virtual_environment(),
        check_dependencies(), 
        check_cli_commands(),
        check_hello_implementation(),
        run_test_suite()
    ]
    
    print_summary(checks)
    return all(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 