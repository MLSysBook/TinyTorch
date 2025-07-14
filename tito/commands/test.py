"""
Enhanced Test command for TinyTorch CLI: runs both inline and external tests.
"""

import subprocess
import sys
import re
import importlib.util
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from .base import BaseCommand

class TestResult:
    """Container for test results."""
    def __init__(self, name: str, success: bool, output: str = "", error: str = ""):
        self.name = name
        self.success = success
        self.output = output
        self.error = error

class ModuleTestResult:
    """Container for module test results."""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.inline_tests: List[TestResult] = []
        self.external_tests: List[TestResult] = []
        self.compilation_success = True
        self.compilation_error = ""
        
    @property
    def all_tests_passed(self) -> bool:
        """Check if all tests passed."""
        if not self.compilation_success:
            return False
        
        all_inline_passed = all(test.success for test in self.inline_tests)
        all_external_passed = all(test.success for test in self.external_tests)
        
        return all_inline_passed and all_external_passed
    
    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.inline_tests) + len(self.external_tests)
    
    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        passed_inline = sum(1 for test in self.inline_tests if test.success)
        passed_external = sum(1 for test in self.external_tests if test.success)
        return passed_inline + passed_external

class TestCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run module tests (inline and external)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("module", nargs="?", help="Module to test (optional)")
        parser.add_argument("--all", action="store_true", help="Run all module tests")
        parser.add_argument("--inline-only", action="store_true", help="Run only inline tests")
        parser.add_argument("--external-only", action="store_true", help="Run only external tests")
        parser.add_argument("--detailed", action="store_true", help="Show detailed output for all tests")
        parser.add_argument("--summary", action="store_true", help="Show summary report only")

    def validate_args(self, args: Namespace) -> None:
        """Validate test command arguments."""
        if args.inline_only and args.external_only:
            raise ValueError("Cannot use --inline-only and --external-only together")

    def run(self, args: Namespace) -> int:
        console = self.console
        
        if args.all:
            return self._run_all_tests(args)
        elif args.module:
            return self._run_single_module_test(args)
        else:
            return self._show_available_modules()
    
    def _run_all_tests(self, args: Namespace) -> int:
        """Run tests for all modules."""
        console = self.console
        
        modules = self._discover_modules()
        if not modules:
            console.print(Panel("[yellow]⚠️  No modules found[/yellow]", 
                              title="No Modules", border_style="yellow"))
            return 0
        
        console.print(Panel(f"🧪 Running tests for {len(modules)} modules", 
                          title="Test Suite", border_style="bright_cyan"))
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running tests...", total=len(modules))
            
            for module_name in modules:
                progress.update(task, description=f"Testing {module_name}...")
                
                result = self._test_module(module_name, args)
                results.append(result)
                
                # Show immediate feedback
                if result.all_tests_passed:
                    console.print(f"[green]✅ {module_name} - All tests passed ({result.passed_tests}/{result.total_tests})[/green]")
                else:
                    console.print(f"[red]❌ {module_name} - Tests failed ({result.passed_tests}/{result.total_tests})[/red]")
                
                progress.advance(task)
        
        # Generate report
        if args.summary:
            self._generate_summary_report(results)
        elif args.detailed:
            self._generate_detailed_report(results)
        else:
            self._generate_default_report(results)
        
        # Return success if all modules passed
        failed_modules = [r for r in results if not r.all_tests_passed]
        return 0 if not failed_modules else 1
    
    def _run_single_module_test(self, args: Namespace) -> int:
        """Run tests for a single module with detailed output."""
        console = self.console
        
        module_name = args.module
        
        console.print(Panel(f"🧪 Running tests for module: [bold cyan]{module_name}[/bold cyan]", 
                          title="Single Module Test", border_style="bright_cyan"))
        
        result = self._test_module(module_name, args)
        
        # Always show detailed output for single module tests
        self._show_detailed_module_result(result)
        
        return 0 if result.all_tests_passed else 1
    
    def _test_module(self, module_name: str, args: Namespace) -> ModuleTestResult:
        """Test a single module comprehensively."""
        result = ModuleTestResult(module_name)
        
        # Test compilation first
        dev_file = self._get_dev_file_path(module_name)
        if not dev_file.exists():
            result.compilation_success = False
            result.compilation_error = f"Module file not found: {dev_file}"
            return result
        
        # Test Python compilation
        try:
            subprocess.run([sys.executable, "-m", "py_compile", str(dev_file)], 
                          check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            result.compilation_success = False
            result.compilation_error = f"Compilation error: {e.stderr}"
            return result
        
        # Run inline tests if requested
        if not args.external_only:
            inline_tests = self._run_inline_tests(dev_file)
            result.inline_tests = inline_tests
        
        # Run external tests if requested
        if not args.inline_only:
            external_tests = self._run_external_tests(module_name)
            result.external_tests = external_tests
        
        return result
    
    def _run_inline_tests(self, dev_file: Path) -> List[TestResult]:
        """Run inline tests using the module's standardized testing framework."""
        inline_tests = []
        
        # Instead of finding individual test functions, run the module as a script
        # This will trigger the if __name__ == "__main__" section with standardized testing
        try:
            result = subprocess.run(
                [sys.executable, str(dev_file)],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            output = result.stdout
            error = result.stderr
            
            # Check return code
            if result.returncode != 0:
                inline_tests.append(TestResult("script_execution", False, output, error))
                return inline_tests
            
            # Parse the output to determine success
            # Check if testing was successful based on output patterns
            if "🎉 All tests passed!" in output or "✅ All tests passed!" in output:
                inline_tests.append(TestResult("standardized_testing", True, output))
            elif "❌" in output or "FAILED" in output or error:
                inline_tests.append(TestResult("standardized_testing", False, output, error))
            elif "✅" in output and "Module Tests:" in output:
                # Handle the case where tests pass but don't have the final success message
                inline_tests.append(TestResult("standardized_testing", True, output))
            else:
                # If no clear success/failure indicator, consider it a failure
                inline_tests.append(TestResult("standardized_testing", False, output, 
                                             "No clear test result indicator found"))
                    
        except subprocess.TimeoutExpired:
            inline_tests.append(TestResult("timeout", False, "", "Test execution timed out"))
        except Exception as e:
            inline_tests.append(TestResult("subprocess_error", False, "", str(e)))
        
        return inline_tests
    
    def _run_external_tests(self, module_name: str) -> List[TestResult]:
        """Run external pytest tests for a module."""
        external_tests = []
        
        # Extract short name from module directory name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        
        test_file = Path("tests") / f"test_{short_name}.py"
        
        if not test_file.exists():
            return external_tests
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=300
            )
            
            # Parse pytest output to extract individual test results
            test_results = self._parse_pytest_output(result.stdout, result.stderr)
            
            # If parsing fails, create a single result for the whole file
            if not test_results:
                success = result.returncode == 0
                external_tests.append(TestResult(
                    f"external_tests_{short_name}",
                    success,
                    result.stdout,
                    result.stderr
                ))
            else:
                external_tests.extend(test_results)
                
        except subprocess.TimeoutExpired:
            external_tests.append(TestResult("external_tests_timeout", False, "", "Tests timed out after 5 minutes"))
        except Exception as e:
            external_tests.append(TestResult("external_tests_error", False, "", str(e)))
        
        return external_tests
    

    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract individual test results."""
        test_results = []
        
        # Simple parsing - look for test function results
        lines = stdout.split('\n')
        for line in lines:
            # Look for lines like "test_file.py::test_function PASSED"
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                parts = line.split('::')
                if len(parts) >= 2:
                    test_name = parts[1].split()[0]
                    success = 'PASSED' in line
                    test_results.append(TestResult(test_name, success, line))
        
        return test_results
    
    def _discover_modules(self) -> List[str]:
        """Discover available modules."""
        modules = []
        source_dir = Path("modules/source")
        
        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    # Check if dev file exists
                    dev_file = self._get_dev_file_path(module_dir.name)
                    if dev_file.exists():
                        modules.append(module_dir.name)
        
        return sorted(modules)
    
    def _get_dev_file_path(self, module_name: str) -> Path:
        """Get the path to a module's dev file."""
        # Extract short name from module directory name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        
        return Path("modules/source") / module_name / f"{short_name}_dev.py"
    
    def _generate_summary_report(self, results: List[ModuleTestResult]) -> None:
        """Generate a summary report for all modules."""
        console = self.console
        
        # Summary table
        table = Table(title="Test Summary Report", show_header=True, header_style="bold blue")
        table.add_column("Module", style="bold cyan", width=15)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Inline Tests", width=12, justify="center")
        table.add_column("External Tests", width=12, justify="center")
        table.add_column("Total", width=10, justify="center")
        
        total_modules = len(results)
        passed_modules = 0
        total_tests = 0
        total_passed = 0
        
        for result in results:
            status = "✅ PASS" if result.all_tests_passed else "❌ FAIL"
            if result.all_tests_passed:
                passed_modules += 1
            
            inline_status = f"{len([t for t in result.inline_tests if t.success])}/{len(result.inline_tests)}"
            external_status = f"{len([t for t in result.external_tests if t.success])}/{len(result.external_tests)}"
            
            total_tests += result.total_tests
            total_passed += result.passed_tests
            
            table.add_row(
                result.module_name,
                status,
                inline_status,
                external_status,
                f"{result.passed_tests}/{result.total_tests}"
            )
        
        console.print(table)
        
        # Overall summary
        console.print(f"\n📊 Overall Summary:")
        console.print(f"   Modules: {passed_modules}/{total_modules} passed")
        console.print(f"   Tests: {total_passed}/{total_tests} passed")
        console.print(f"   Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "   No tests found")
    
    def _generate_detailed_report(self, results: List[ModuleTestResult]) -> None:
        """Generate a detailed report for all modules."""
        console = self.console
        
        console.print(Panel("📋 Detailed Test Report", title="Test Results", border_style="bright_cyan"))
        
        for result in results:
            self._show_detailed_module_result(result)
    
    def _generate_default_report(self, results: List[ModuleTestResult]) -> None:
        """Generate the default report (between summary and detailed)."""
        console = self.console
        
        failed_modules = [r for r in results if not r.all_tests_passed]
        
        if failed_modules:
            console.print(Panel(f"[red]❌ {len(failed_modules)} modules failed[/red]", 
                              title="Failed Modules", border_style="red"))
            
            for result in failed_modules:
                console.print(f"\n[red]❌ {result.module_name}[/red]:")
                
                if not result.compilation_success:
                    console.print(f"   [red]Compilation Error: {result.compilation_error}[/red]")
                
                failed_inline = [t for t in result.inline_tests if not t.success]
                failed_external = [t for t in result.external_tests if not t.success]
                
                if failed_inline:
                    console.print(f"   [red]Failed inline tests: {', '.join(t.name for t in failed_inline)}[/red]")
                
                if failed_external:
                    console.print(f"   [red]Failed external tests: {', '.join(t.name for t in failed_external)}[/red]")
        else:
            console.print(Panel("[green]✅ All modules passed![/green]", 
                              title="Test Results", border_style="green"))
    
    def _show_detailed_module_result(self, result: ModuleTestResult) -> None:
        """Show detailed results for a single module."""
        console = self.console
        
        status_color = "green" if result.all_tests_passed else "red"
        status_icon = "✅" if result.all_tests_passed else "❌"
        
        console.print(f"\n[{status_color}]{status_icon} {result.module_name}[/{status_color}]")
        
        if not result.compilation_success:
            console.print(f"   [red]❌ Compilation failed: {result.compilation_error}[/red]")
            return
        
        # Show inline test results
        if result.inline_tests:
            console.print("   📝 Inline Tests:")
            for test in result.inline_tests:
                icon = "✅" if test.success else "❌"
                color = "green" if test.success else "red"
                console.print(f"      [{color}]{icon} {test.name}[/{color}]")
                if not test.success and test.error:
                    console.print(f"         Error: {test.error}")
        
        # Show external test results
        if result.external_tests:
            console.print("   🧪 External Tests:")
            for test in result.external_tests:
                icon = "✅" if test.success else "❌"
                color = "green" if test.success else "red"
                console.print(f"      [{color}]{icon} {test.name}[/{color}]")
                if not test.success and test.error:
                    console.print(f"         Error: {test.error}")
        
        # Summary for this module
        console.print(f"   📊 Summary: {result.passed_tests}/{result.total_tests} tests passed")
    
    def _show_available_modules(self) -> int:
        """Show available modules when no arguments are provided."""
        console = self.console
        
        modules = self._discover_modules()
        
        if modules:
            console.print(Panel(f"[red]❌ Please specify a module to test[/red]\n\n"
                              f"Available modules: {', '.join(modules)}\n\n"
                              f"[dim]Examples:[/dim]\n"
                              f"[dim]  tito module test tensor       - Test specific module[/dim]\n"
                              f"[dim]  tito module test --all        - Test all modules[/dim]\n"
                              f"[dim]  tito module test --all --summary - Summary report[/dim]", 
                              title="Module Required", border_style="red"))
        else:
            console.print(Panel("[red]❌ No modules found in modules/source directory[/red]", 
                              title="Error", border_style="red"))
        
        return 1 