#!/usr/bin/env python3
"""
TinyTorch CLI (tito)

The main command-line interface for the TinyTorch ML system.
Students use this CLI for testing, training, and project management.

Usage: python bin/tito.py [command] [options]
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_banner():
    """Print the TinyTorch banner."""
    print("🔥 Tiny🔥Torch: Build ML Systems from Scratch 🔥")
    print("=" * 50)

def cmd_version(args):
    """Show TinyTorch version."""
    print("Tiny🔥Torch v0.1.0")
    print("Machine Learning Systems Course")

def cmd_info(args):
    """Show system information and status."""
    print_banner()
    print()
    
    # Python environment info
    print("📋 System Information")
    print("-" * 30)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Virtual environment check
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    venv_status = "✅ Active" if in_venv else "❌ Not Active"
    print(f"Virtual Environment: {venv_status}")
    
    print()
    
    # Course navigation
    print("📋 Course Navigation")
    print("-" * 30)
    print("📖 Course Overview: README.md")
    print("🎯 Detailed Guide: COURSE_GUIDE.md")
    print("🚀 Start Here: projects/setup/README.md")
    
    print()
    
    # Implementation status
    print("🚀 Implementation Status")
    print("-" * 30)
    
    # Check if hello function exists
    try:
        from tinytorch.core.utils import hello_tinytorch
        hello_status = "✅ Implemented"
        if args.hello:
            print(f"Hello Message: {hello_tinytorch()}")
    except ImportError:
        hello_status = "❌ Not Implemented"
    
    print(f"hello_tinytorch(): {hello_status}")
    
    # TODO: Add checks for other components as they're implemented
    print("tensor operations: ⏳ Coming in Project 1")
    print("autograd engine: ⏳ Coming in Project 4") 
    print("neural networks: ⏳ Coming in Project 2")
    print("training system: ⏳ Coming in Project 6")
    
    if args.show_architecture:
        print()
        print("🏗️  System Architecture")
        print("-" * 30)
        print("""
┌─────────────────────────────────────────────────────────────┐
│                    TinyTorch System                         │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface (bin/tito.py)                               │
├─────────────────────────────────────────────────────────────┤
│  Training Orchestration (trainer.py)                       │
├─────────────────────────────────────────────────────────────┤
│  Model Definition     │  Data Pipeline    │  Optimization   │
│  (modules.py)         │  (dataloader.py)  │  (optimizer.py) │
├─────────────────────────────────────────────────────────────┤
│  Automatic Differentiation Engine (autograd)               │
├─────────────────────────────────────────────────────────────┤
│  Tensor Operations & Storage (tensor.py)                   │
├─────────────────────────────────────────────────────────────┤
│  Profiling & MLOps (profiler.py, mlops.py)                 │
└─────────────────────────────────────────────────────────────┘
        """)

def cmd_test(args):
    """Run tests for a specific project."""
    print(f"🧪 Running tests for project: {args.project}")
    
    if args.project == "setup":
        # Run setup tests
        import subprocess
        test_file = "projects/setup/test_setup.py"
        result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                              capture_output=False)
        return result.returncode
    else:
        print(f"Tests for project '{args.project}' not yet implemented.")
        return 1

def cmd_submit(args):
    """Submit project for grading."""
    print(f"📤 Submitting project: {args.project}")
    print("🚧 Submission system not yet implemented.")
    print("For now, make sure all tests pass with:")
    print(f"   python -m pytest projects/{args.project}/test_*.py -v")

def cmd_status(args):
    """Check project status."""
    print(f"📊 Status for project: {args.project}")
    print("🚧 Status system not yet implemented.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tito", 
        description="TinyTorch CLI - Build ML systems from scratch"
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument("--hello", action="store_true", help="Show hello message")
    info_parser.add_argument("--show-architecture", action="store_true", help="Show system architecture")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run project tests")
    test_parser.add_argument("--project", required=True, help="Project to test")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit project")
    submit_parser.add_argument("--project", required=True, help="Project to submit")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check project status")
    status_parser.add_argument("--project", required=True, help="Project to check")
    
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        cmd_version(args)
        return 0
    
    # Handle commands
    if args.command == "info":
        cmd_info(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 