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
import numpy as np

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create Rich console instance
console = Console()

def print_banner():
    """Print the TinyTorch banner using Rich."""
    banner_text = Text("Tiny🔥Torch: Build ML Systems from Scratch", style="bold red")
    console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

def check_setup_status():
    """Check if setup project is complete."""
    try:
        from tinytorch.core.utils import hello_tinytorch
        return "✅ Implemented"
    except ImportError:
        return "❌ Not Implemented"

def check_tensor_status():
    """Check if tensor project is complete."""
    try:
        from tinytorch.core.tensor import Tensor
        # Test actual functionality, not just import
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2  # Should work if implemented
        return "✅ Implemented"
    except (ImportError, NotImplementedError):
        return "⏳ Not Started"

def check_mlp_status():
    """Check if MLP project is complete."""
    try:
        from tinytorch.core.modules import MLP
        # Test actual functionality
        mlp = MLP(input_size=10, hidden_size=5, output_size=2)
        from tinytorch.core.tensor import Tensor
        x = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        output = mlp(x)  # Should work if implemented
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_cnn_status():
    """Check if CNN project is complete."""
    try:
        from tinytorch.core.modules import Conv2d
        # Test actual functionality
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        from tinytorch.core.tensor import Tensor
        x = Tensor(np.random.randn(1, 3, 32, 32))
        output = conv(x)  # Should work if implemented
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_data_status():
    """Check if data project is complete."""
    try:
        from tinytorch.core.dataloader import DataLoader
        # Test actual functionality
        data = [(np.random.randn(3, 32, 32), 0) for _ in range(10)]
        loader = DataLoader(data, batch_size=2, shuffle=True)
        batch = next(iter(loader))  # Should work if implemented
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError, StopIteration):
        return "⏳ Not Started"

def check_training_status():
    """Check if training project is complete."""
    try:
        from tinytorch.core.optimizer import SGD
        from tinytorch.core.tensor import Tensor
        # Test actual functionality
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SGD([t], lr=0.01)
        t.backward()  # Should work if autograd is implemented
        optimizer.step()  # Should work if optimizer is implemented
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_profiling_status():
    """Check if profiling project is complete."""
    try:
        from tinytorch.core.profiler import Profiler
        # Test basic functionality
        profiler = Profiler()
        profiler.start("test")
        profiler.end("test")
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_compression_status():
    """Check if compression project is complete."""
    try:
        from tinytorch.core.compression import Pruner
        # Test basic functionality
        pruner = Pruner(sparsity=0.5)
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_kernels_status():
    """Check if kernels project is complete."""
    try:
        from tinytorch.core.kernels import optimized_matmul
        # Test basic functionality
        a = np.random.randn(3, 3)
        b = np.random.randn(3, 3)
        result = optimized_matmul(a, b)
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_benchmarking_status():
    """Check if benchmarking project is complete."""
    try:
        from tinytorch.core.benchmark import Benchmark
        # Test basic functionality
        benchmark = Benchmark()
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError):
        return "⏳ Not Started"

def check_mlops_status():
    """Check if MLOps project is complete."""
    try:
        from tinytorch.core.mlops import ModelMonitor
        from tinytorch.core.tensor import Tensor
        # Test that the methods are actually implemented, not just placeholders
        monitor = ModelMonitor(model=None, baseline_metrics={})
        # Try to use a method that should be implemented
        test_inputs = Tensor([1.0, 2.0, 3.0])
        test_predictions = Tensor([0.5, 0.8, 0.2]) 
        monitor.log_prediction(test_inputs, test_predictions)
        return "✅ Implemented"
    except (ImportError, NotImplementedError, AttributeError, TypeError):
        return "⏳ Not Started"

def validate_environment():
    """Validate environment setup before running commands."""
    issues = []
    
    # Check virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    if not in_venv:
        issues.append("Virtual environment not activated. Run: source .venv/bin/activate")
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check core dependencies
    try:
        import numpy
        import pytest
    except ImportError as e:
        issues.append(f"Missing dependency: {e.name}. Run: pip install -r requirements.txt")
    
    # Check TinyTorch package structure
    tinytorch_path = Path(__file__).parent.parent / "tinytorch"
    if not tinytorch_path.exists():
        issues.append("TinyTorch package not found. Check project structure.")
    
    if issues:
        issue_text = Text()
        issue_text.append("❌ Environment Issues Detected\n\n", style="bold red")
        for issue in issues:
            issue_text.append(f"  • {issue}\n", style="yellow")
        issue_text.append("\nRun 'python3 bin/tito.py doctor' for detailed diagnosis", style="dim")
        
        console.print(Panel(issue_text, title="Environment Problems", border_style="red"))
        return False
    
    return True

def cmd_version(args):
    """Show TinyTorch version."""
    version_text = Text()
    version_text.append("Tiny🔥Torch v0.1.0\n", style="bold red")
    version_text.append("Machine Learning Systems Course", style="cyan")
    console.print(Panel(version_text, title="Version Info", border_style="bright_blue"))

def cmd_info(args):
    """Show system information and status."""
    print_banner()
    console.print()
    
    # System Information Panel
    info_text = Text()
    info_text.append(f"Python: {sys.version.split()[0]}\n", style="cyan")
    info_text.append(f"Platform: {sys.platform}\n", style="cyan")
    info_text.append(f"Working Directory: {os.getcwd()}\n", style="cyan")
    
    # Virtual environment check - use same robust detection as doctor
    venv_path = Path(".venv")
    venv_exists = venv_path.exists()
    in_venv = (
        # Method 1: Check VIRTUAL_ENV environment variable (most reliable for activation)
        os.environ.get('VIRTUAL_ENV') is not None or
        # Method 2: Check sys.prefix vs sys.base_prefix (works for running Python in venv)
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        # Method 3: Check for sys.real_prefix (older Python versions)
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
    
    # Implementation status
    modules = [
        ("Setup", "hello_tinytorch function", check_setup_status),
        ("Tensor", "basic tensor operations", check_tensor_status),
        ("MLP", "multi-layer perceptron (manual)", check_mlp_status),
        ("CNN", "convolutional networks (basic)", check_cnn_status),
        ("Data", "data loading pipeline", check_data_status),
        ("Training", "autograd engine & optimization", check_training_status),
        ("Profiling", "performance profiling", check_profiling_status),
        ("Compression", "model compression", check_compression_status),
        ("Kernels", "custom compute kernels", check_kernels_status),
        ("Benchmarking", "performance benchmarking", check_benchmarking_status),
        ("MLOps", "production monitoring", check_mlops_status),
    ]
    
    # Module Status Table
    status_table = Table(title="🚀 Module Implementation Status", show_header=True, header_style="bold blue")
    status_table.add_column("ID", style="dim", width=3, justify="center")
    status_table.add_column("Project", style="bold cyan", width=12)
    status_table.add_column("Status", width=18, justify="center")
    status_table.add_column("Description", style="dim", width=40)
    
    for i, (name, desc, check_func) in enumerate(modules):
        status_text = check_func()
        if "✅" in status_text:
            status_style = "[green]✅ Implemented[/green]"
        elif "❌" in status_text:
            status_style = "[red]❌ Not Implemented[/red]"
        else:
            status_style = "[yellow]⏳ Not Started[/yellow]"
        
        status_table.add_row(str(i), name, status_style, desc)
    
    console.print(status_table)
    
    if args.hello and check_setup_status() == "✅ Implemented":
        try:
            from tinytorch.core.utils import hello_tinytorch
            hello_text = Text(hello_tinytorch(), style="bold red")
            console.print()
            console.print(Panel(hello_text, style="bright_red", padding=(1, 2)))
        except ImportError:
            pass
    
    if args.show_architecture:
        console.print()
        
        # Create architecture tree
        arch_tree = Tree("🏗️ TinyTorch System Architecture", style="bold blue")
        
        cli_branch = arch_tree.add("CLI Interface", style="cyan")
        cli_branch.add("bin/tito.py - Command line tools", style="dim")
        
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

def cmd_test(args):
    """Run tests for a specific module."""
    valid_modules = ["setup", "tensor", "mlp", "cnn", "data", "training", 
                     "profiling", "compression", "kernels", "benchmarking", "mlops"]
    
    if args.all:
        # Run all tests with progress bar
        import subprocess
        failed_modules = []
        
        # Count existing test files
        existing_tests = [p for p in valid_modules if Path(f"modules/{p}/test_{p}.py").exists()]
        
        console.print(Panel(f"🧪 Running tests for {len(existing_tests)} modules", 
                          title="Test Suite", border_style="bright_cyan"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running tests...", total=len(existing_tests))
            
            for module in existing_tests:
                progress.update(task, description=f"Testing {module}...")
                
                test_file = f"modules/{module}/test_{module}.py"
                result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                                      capture_output=True, text=True)
                
                if result.returncode != 0:
                    failed_modules.append(module)
                    console.print(f"[red]❌ {module} tests failed[/red]")
                else:
                    console.print(f"[green]✅ {module} tests passed[/green]")
                
                progress.advance(task)
        
        # Results summary
        if failed_modules:
            console.print(Panel(f"[red]❌ Failed modules: {', '.join(failed_modules)}[/red]", 
                              title="Test Results", border_style="red"))
            return 1
        else:
            console.print(Panel("[green]✅ All tests passed![/green]", 
                              title="Test Results", border_style="green"))
            return 0
    
    elif args.module in valid_modules:
        # Run specific module tests
        import subprocess
        test_file = f"modules/{args.module}/test_{args.module}.py"
        
        console.print(Panel(f"🧪 Running tests for module: [bold cyan]{args.module}[/bold cyan]", 
                          title="Single Module Test", border_style="bright_cyan"))
        
        if not Path(test_file).exists():
            console.print(Panel(f"[yellow]⏳ Test file not found: {test_file}\n"
                              f"Module '{args.module}' may not be implemented yet.[/yellow]", 
                              title="Test Not Found", border_style="yellow"))
            return 1
        
        console.print(f"[dim]Running: pytest {test_file} -v[/dim]")
        console.print()
        
        result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                              capture_output=False)
        
        # Show result summary
        if result.returncode == 0:
            console.print(Panel(f"[green]✅ All tests passed for {args.module}![/green]", 
                              title="Test Results", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ Some tests failed for {args.module}[/red]", 
                              title="Test Results", border_style="red"))
        
        return result.returncode
    else:
        console.print(Panel(f"[red]❌ Unknown module: {args.module}[/red]\n"
                          f"Valid modules: {', '.join(valid_modules)}", 
                          title="Invalid Module", border_style="red"))
        return 1

def cmd_submit(args):
    """Submit module for grading."""
    submit_text = Text()
    submit_text.append(f"📤 Submitting module: {args.module}\n\n", style="bold cyan")
    submit_text.append("🚧 Submission system not yet implemented.\n\n", style="yellow")
    submit_text.append("For now, make sure all tests pass with:\n", style="dim")
    submit_text.append(f"   python -m pytest modules/{args.module}/test_*.py -v", style="bold white")
    
    console.print(Panel(submit_text, title="Module Submission", border_style="bright_yellow"))

def cmd_status(args):
    """Check module status."""
    status_text = Text()
    status_text.append(f"📊 Status for module: {args.module}\n\n", style="bold cyan")
    status_text.append("🚧 Status system not yet implemented.", style="yellow")
    
    console.print(Panel(status_text, title="Module Status", border_style="bright_yellow"))




def cmd_doctor(args):
    """Run comprehensive environment diagnosis."""
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
    venv_path = Path(".venv")
    venv_exists = venv_path.exists()
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
    env_table.add_row("Virtual Environment", venv_status, ".venv")
    
    # Dependencies
    dependencies = ['numpy', 'matplotlib', 'pytest', 'yaml', 'black', 'rich']
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            env_table.add_row(dep.title(), "[green]✅ OK[/green]", f"v{version}")
        except ImportError:
            env_table.add_row(dep.title(), "[red]❌ Missing[/red]", "Not installed")
    
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
        ('bin/tito.py', 'CLI script'),
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
    cmd_info(argparse.Namespace(hello=False, show_architecture=False))

def cmd_jupyter(args):
    """Start Jupyter notebook server."""
    import subprocess
    
    console.print(Panel("📓 Jupyter Notebook Server", 
                       title="Interactive Development", border_style="bright_green"))
    
    # Determine which Jupyter to start
    if args.lab:
        cmd = ["jupyter", "lab", "--port", str(args.port)]
        console.print(f"🚀 Starting JupyterLab on port {args.port}...")
    else:
        cmd = ["jupyter", "notebook", "--port", str(args.port)]
        console.print(f"🚀 Starting Jupyter Notebook on port {args.port}...")
    
    console.print("💡 Open your browser to the URL shown above")
    console.print("📁 Navigate to your module's notebook directory")
    console.print("🔄 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n🛑 Jupyter server stopped")
    except FileNotFoundError:
        console.print(Panel("[red]❌ Jupyter not found. Install with: pip install jupyter[/red]", 
                          title="Error", border_style="red"))
        return 1
    
    return 0

def cmd_nbdev(args):
    """Run nbdev commands for notebook development."""
    import subprocess
    
    console.print(Panel("📓 nbdev Notebook Development", 
                       title="Notebook Tools", border_style="bright_cyan"))
    
    if args.build_lib:
        console.print("🔨 Building library from notebooks...")
        result = subprocess.run(["nbdev_build_lib"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(Panel("[green]✅ Library built successfully![/green]", 
                              title="Build Success", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ Build failed: {result.stderr}[/red]", 
                              title="Build Error", border_style="red"))
        return result.returncode
    
    elif args.build_docs:
        console.print("📚 Building documentation from notebooks...")
        result = subprocess.run(["nbdev_build_docs"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(Panel("[green]✅ Documentation built successfully![/green]", 
                              title="Docs Success", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ Docs build failed: {result.stderr}[/red]", 
                              title="Docs Error", border_style="red"))
        return result.returncode
    
    elif args.test:
        console.print("🧪 Running notebook tests...")
        result = subprocess.run(["nbdev_test"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(Panel("[green]✅ All notebook tests passed![/green]", 
                              title="Test Success", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ Some tests failed: {result.stderr}[/red]", 
                              title="Test Error", border_style="red"))
        return result.returncode
    
    elif args.clean:
        console.print("🧹 Cleaning notebook build artifacts...")
        result = subprocess.run(["nbdev_clean_nbs"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(Panel("[green]✅ Cleaned successfully![/green]", 
                              title="Clean Success", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ Clean failed: {result.stderr}[/red]", 
                              title="Clean Error", border_style="red"))
        return result.returncode
    
    else:
        help_text = Text()
        help_text.append("📓 nbdev Commands:\n\n", style="bold cyan")
        help_text.append("  tito nbdev --build-lib    - Build library from notebooks\n", style="white")
        help_text.append("  tito nbdev --build-docs   - Build documentation\n", style="white")
        help_text.append("  tito nbdev --test         - Run notebook tests\n", style="white")
        help_text.append("  tito nbdev --clean        - Clean build artifacts\n\n", style="white")
        help_text.append("💡 Development workflow:\n", style="bold yellow")
        help_text.append("  1. Work in modules/*/notebook/*_dev.ipynb\n", style="dim")
        help_text.append("  2. Test interactively\n", style="dim")
        help_text.append("  3. Run: tito nbdev --build-lib\n", style="dim")
        help_text.append("  4. Test compiled package\n", style="dim")
        
        console.print(Panel(help_text, title="nbdev Help", border_style="bright_cyan"))
        return 0

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
    test_parser = subparsers.add_parser("test", help="Run module tests")
    test_parser.add_argument("--module", help="Module to test")
    test_parser.add_argument("--all", action="store_true", help="Run all module tests")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit module")
    submit_parser.add_argument("--module", required=True, help="Module to submit")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check module status")
    status_parser.add_argument("--module", required=True, help="Module to check")
    
    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run environment diagnosis")
    
    # nbdev commands
    nbdev_parser = subparsers.add_parser("nbdev", help="nbdev notebook development commands")
    nbdev_parser.add_argument("--build-lib", action="store_true", help="Build library from notebooks")
    nbdev_parser.add_argument("--build-docs", action="store_true", help="Build documentation from notebooks")
    nbdev_parser.add_argument("--test", action="store_true", help="Run notebook tests")
    nbdev_parser.add_argument("--clean", action="store_true", help="Clean notebook build artifacts")
    
    # Jupyter command
    jupyter_parser = subparsers.add_parser("jupyter", help="Start Jupyter notebook server")
    jupyter_parser.add_argument("--notebook", action="store_true", help="Start classic notebook")
    jupyter_parser.add_argument("--lab", action="store_true", help="Start JupyterLab")
    jupyter_parser.add_argument("--port", type=int, default=8888, help="Port to run on (default: 8888)")
    
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        cmd_version(args)
        return 0
    
    # Environment validation (skip for doctor and info commands)
    if args.command not in ["doctor", "info"] and not validate_environment():
        return 1
    
    # Validate test command arguments
    if args.command == "test":
        if not args.all and not args.module:
            error_text = Text()
            error_text.append("❌ Error: Must specify either --module or --all\n\n", style="bold red")
            error_text.append("Usage: python bin/tito.py test --module <name> | --all", style="cyan")
            console.print(Panel(error_text, title="Invalid Arguments", border_style="red"))
            return 1
    
    # Handle commands
    if args.command == "info":
        cmd_info(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "nbdev":
        return cmd_nbdev(args)
    elif args.command == "jupyter":
        return cmd_jupyter(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 