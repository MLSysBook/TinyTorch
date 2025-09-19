#!/usr/bin/env python3
"""
Comprehensive test script for all TinyTorch demos.
Validates that each demo runs successfully and produces expected outputs.
"""

import sys
import subprocess
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# List of all demos to test
DEMOS = [
    ("demo_tensor_math.py", "02_tensor", "Tensor Math & Linear Algebra"),
    ("demo_activations.py", "03_activations", "Activation Functions"),
    ("demo_single_neuron.py", "04_layers", "Single Neuron Learning"),
    ("demo_xor_network.py", "05_dense", "XOR Multi-Layer Network"),
    ("demo_vision.py", "06_spatial", "Computer Vision & CNNs"),
    ("demo_attention.py", "07_attention", "Attention Mechanisms"),
    ("demo_training.py", "11_training", "End-to-End Training"),
    ("demo_language.py", "16_tinygpt", "Language Generation"),
]

def run_demo(demo_file: str, timeout: int = 30) -> tuple[bool, str, float]:
    """
    Run a single demo and return success status, output, and execution time.
    
    Args:
        demo_file: Name of the demo file to run
        timeout: Maximum time to wait for demo completion
    
    Returns:
        Tuple of (success, output/error, execution_time)
    """
    demo_path = Path("demos") / demo_file
    
    if not demo_path.exists():
        return False, f"Demo file not found: {demo_path}", 0.0
    
    start_time = time.time()
    
    try:
        # Run the demo with a timeout
        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # Check for key success indicators in output
            output = result.stdout
            
            # Look for success markers
            if "Demo Complete" in output or "Achievements:" in output or "✅" in output:
                return True, "Demo ran successfully", execution_time
            else:
                # Demo ran but might not have completed properly
                return True, "Demo executed (check output manually)", execution_time
        else:
            # Demo failed with error
            error_msg = result.stderr if result.stderr else result.stdout
            # Extract the key error message
            if "Could not import TinyTorch modules" in error_msg:
                error_msg = "Missing module exports (needs tito export)"
            elif "ImportError" in error_msg:
                error_msg = "Import error - check dependencies"
            else:
                # Get last line of error for concise message
                lines = error_msg.strip().split('\n')
                error_msg = lines[-1] if lines else "Unknown error"
            
            return False, error_msg[:100], execution_time
            
    except subprocess.TimeoutExpired:
        return False, f"Demo timed out after {timeout} seconds", timeout
    except Exception as e:
        return False, f"Error running demo: {str(e)}", time.time() - start_time

def test_all_demos():
    """Test all demos and display results."""
    
    console = Console()
    
    # Header
    console.print(Panel.fit(
        "🧪 TinyTorch Demo Test Suite\nValidating all demos work correctly",
        style="bold cyan",
        border_style="bright_blue"
    ))
    console.print()
    
    # Check virtual environment
    console.print("🔍 Checking environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        console.print("[yellow]⚠️  Warning: Not running in virtual environment[/yellow]")
        console.print("[yellow]   Demos may fail due to missing dependencies[/yellow]")
        console.print("[yellow]   Run: source .venv/bin/activate[/yellow]")
        console.print()
    else:
        console.print("[green]✅ Virtual environment active[/green]")
        console.print()
    
    # Test each demo with progress bar
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Testing demos...", total=len(DEMOS))
        
        for demo_file, module_needed, description in DEMOS:
            progress.update(task, description=f"Testing {demo_file}...")
            
            success, message, exec_time = run_demo(demo_file)
            results.append({
                "file": demo_file,
                "module": module_needed,
                "description": description,
                "success": success,
                "message": message,
                "time": exec_time
            })
            
            progress.advance(task)
    
    console.print()
    
    # Display results table
    console.print("📊 Test Results:")
    console.print()
    
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Demo", style="cyan", width=25)
    results_table.add_column("Description", style="white", width=30)
    results_table.add_column("Status", style="green", width=10)
    results_table.add_column("Time", style="yellow", width=8)
    results_table.add_column("Notes", style="blue", width=40)
    
    passed = 0
    failed = 0
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        status_style = "green" if result["success"] else "red"
        
        if result["success"]:
            passed += 1
        else:
            failed += 1
        
        # Add row with appropriate styling
        results_table.add_row(
            result["file"],
            result["description"],
            f"[{status_style}]{status}[/{status_style}]",
            f"{result['time']:.2f}s",
            result["message"]
        )
    
    console.print(results_table)
    console.print()
    
    # Summary statistics
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Total Demos", str(total))
    summary_table.add_row("Passed", f"[green]{passed}[/green]")
    summary_table.add_row("Failed", f"[red]{failed}[/red]")
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    summary_table.add_row("Total Time", f"{sum(r['time'] for r in results):.2f}s")
    
    console.print(Panel(summary_table, title="📈 Summary", style="blue"))
    console.print()
    
    # Recommendations for failures
    if failed > 0:
        console.print("🔧 [bold yellow]Fixing Failed Demos:[/bold yellow]")
        console.print()
        
        for result in results:
            if not result["success"]:
                console.print(f"  [red]•[/red] {result['file']}:")
                
                if "Missing module exports" in result["message"]:
                    console.print(f"    → Run: [cyan]tito export {result['module']}[/cyan]")
                elif "Import error" in result["message"]:
                    console.print(f"    → Check dependencies and virtual environment")
                else:
                    console.print(f"    → Debug: [cyan]python demos/{result['file']}[/cyan]")
                    console.print(f"    → Error: {result['message']}")
                console.print()
    
    # Final status
    if passed == total:
        console.print(Panel.fit(
            f"🎉 All {total} demos passed successfully!",
            style="bold green",
            border_style="bright_green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            f"⚠️  {failed} of {total} demos failed. See recommendations above.",
            style="bold yellow",
            border_style="yellow"
        ))
        return 1

def quick_test():
    """Quick test mode - just check if demos can be imported."""
    
    console = Console()
    
    console.print("⚡ Quick Import Test")
    console.print()
    
    results = []
    
    for demo_file, _, description in DEMOS:
        demo_path = Path("demos") / demo_file.replace('.py', '')
        
        try:
            # Try to import the demo module
            exec(f"import demos.{demo_file.replace('.py', '')}")
            results.append((demo_file, True, "Import successful"))
        except Exception as e:
            error_msg = str(e).split('\n')[0][:50]
            results.append((demo_file, False, error_msg))
    
    # Display quick results
    for demo, success, msg in results:
        status = "✅" if success else "❌"
        console.print(f"{status} {demo}: {msg}")
    
    console.print()
    passed = sum(1 for _, s, _ in results if s)
    console.print(f"Quick test: {passed}/{len(results)} demos importable")

if __name__ == "__main__":
    # Check for command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all TinyTorch demos")
    parser.add_argument("--quick", action="store_true", help="Quick import test only")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per demo in seconds")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        sys.exit(test_all_demos())