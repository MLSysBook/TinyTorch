#!/usr/bin/env python3
"""
Master Test Runner for All TinyTorch Modules
Runs tests for all modules and provides comprehensive reporting
"""

import sys
from pathlib import Path
import subprocess
from typing import Dict, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_module_tests() -> List[Path]:
    """Find all module test directories."""
    test_dirs = []
    tests_path = Path(__file__).parent
    
    for module_dir in sorted(tests_path.glob("module_*")):
        if module_dir.is_dir():
            run_script = module_dir / "run_all_tests.py"
            if run_script.exists():
                test_dirs.append(module_dir)
    
    return test_dirs


def run_module_test_suite(module_dir: Path) -> Dict:
    """Run tests for a single module."""
    module_name = module_dir.name
    run_script = module_dir / "run_all_tests.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(run_script)],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per module
        )
        
        # Parse output for pass/fail
        output = result.stdout + result.stderr
        passed = "All tests passed" in output
        
        return {
            'module': module_name,
            'status': 'PASSED' if passed else 'FAILED',
            'return_code': result.returncode,
            'output': output
        }
    except subprocess.TimeoutExpired:
        return {
            'module': module_name,
            'status': 'TIMEOUT',
            'return_code': -1,
            'output': 'Tests timed out after 30 seconds'
        }
    except Exception as e:
        return {
            'module': module_name,
            'status': 'ERROR',
            'return_code': -1,
            'output': str(e)
        }


def print_summary(results: List[Dict]):
    """Print summary of all test results."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    
    console = Console()
    
    # Header
    console.print("\n" + "="*70)
    console.print(Panel("[bold cyan]TinyTorch Complete Test Suite Results[/bold cyan]",
                       style="bold blue", expand=False))
    console.print("="*70 + "\n")
    
    # Results table
    table = Table(title="Module Test Summary", box=box.ROUNDED)
    table.add_column("Module", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    
    passed_count = 0
    failed_count = 0
    
    for result in results:
        status_display = {
            'PASSED': '[green]✅ PASSED[/green]',
            'FAILED': '[red]❌ FAILED[/red]',
            'TIMEOUT': '[yellow]⏱️ TIMEOUT[/yellow]',
            'ERROR': '[red]💥 ERROR[/red]'
        }.get(result['status'], '❓')
        
        # Extract test counts if available
        details = ""
        if "Total:" in result.get('output', ''):
            for line in result['output'].split('\n'):
                if "Total:" in line or "Passed:" in line or "Failed:" in line:
                    details = line.strip()
                    break
        
        table.add_row(
            result['module'],
            status_display,
            details
        )
        
        if result['status'] == 'PASSED':
            passed_count += 1
        else:
            failed_count += 1
    
    console.print(table)
    
    # Overall summary
    total = len(results)
    console.print(f"\n📊 Overall Summary:")
    console.print(f"  • Total Modules: {total}")
    console.print(f"  • ✅ Passed: {passed_count}")
    console.print(f"  • ❌ Failed: {failed_count}")
    
    # Final status
    if failed_count == 0:
        console.print("\n🎉 [green bold]All module tests passed![/green bold]")
        console.print("✨ TinyTorch is fully tested and ready!")
    else:
        console.print(f"\n⚠️  [red]{failed_count} module(s) have failing tests.[/red]")
        console.print("Please review and fix the failures above.")
    
    return failed_count == 0


def main():
    """Run tests for all modules."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all TinyTorch module tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Show detailed output from each module")
    parser.add_argument("--module", "-m",
                      help="Run tests for specific module only (e.g., module_05)")
    
    args = parser.parse_args()
    
    if args.module:
        # Run specific module
        module_dir = Path(__file__).parent / args.module
        if not module_dir.exists():
            print(f"❌ Module directory not found: {module_dir}")
            sys.exit(1)
        
        result = run_module_test_suite(module_dir)
        if args.verbose:
            print(result['output'])
        else:
            # Still show summary even without verbose
            print(f"\nModule {args.module}:")
            print(f"Status: {result['status']}")
            if 'output' in result:
                # Extract summary lines
                for line in result['output'].split('\n'):
                    if 'Summary:' in line or 'Total:' in line or 'Passed:' in line or 'Failed:' in line:
                        print(line)
        
        success = result['status'] == 'PASSED'
    else:
        # Run all modules
        module_dirs = find_module_tests()
        
        if not module_dirs:
            print("⚠️  No module test directories found!")
            print("Expected format: tests/module_XX/run_all_tests.py")
            sys.exit(1)
        
        results = []
        for module_dir in module_dirs:
            print(f"Running tests for {module_dir.name}...")
            result = run_module_test_suite(module_dir)
            results.append(result)
            
            if args.verbose:
                print(result['output'])
                print("-" * 60)
        
        success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()