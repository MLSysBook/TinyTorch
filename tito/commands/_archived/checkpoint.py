"""
Checkpoint tracking and visualization command for TinyTorch CLI.

Provides capability-based progress tracking through the ML systems engineering journey:
Foundation → Architecture → Training → Inference → Serving
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich.status import Status

from .base import BaseCommand
from ..core.config import CLIConfig
from ..core.console import get_console, print_error, print_success


class CheckpointSystem:
    """Core checkpoint tracking system."""
    
    # Define the 20-checkpoint structure for complete ML systems engineering journey
    CHECKPOINTS = {
        "00": {
            "name": "Environment",
            "description": "Development environment setup and configuration",
            "test_file": "checkpoint_00_environment.py",
            "capability": "Can I configure my TinyTorch development environment?"
        },
        "01": {
            "name": "Foundation",
            "description": "Basic tensor operations and ML building blocks",
            "test_file": "checkpoint_01_foundation.py",
            "capability": "Can I create and manipulate the building blocks of ML?"
        },
        "02": {
            "name": "Intelligence",
            "description": "Nonlinear activation functions",
            "test_file": "checkpoint_02_intelligence.py",
            "capability": "Can I add nonlinearity - the key to neural network intelligence?"
        },
        "03": {
            "name": "Components",
            "description": "Fundamental neural network building blocks",
            "test_file": "checkpoint_03_components.py",
            "capability": "Can I build the fundamental building blocks of neural networks?"
        },
        "04": {
            "name": "Networks",
            "description": "Complete multi-layer neural networks",
            "test_file": "checkpoint_04_networks.py",
            "capability": "Can I build complete multi-layer neural networks?"
        },
        "05": {
            "name": "Learning",
            "description": "Spatial data processing with convolutional operations",
            "test_file": "checkpoint_05_learning.py",
            "capability": "Can I process spatial data like images with convolutional operations?"
        },
        "06": {
            "name": "Attention",
            "description": "Attention mechanisms for sequence understanding",
            "test_file": "checkpoint_06_attention.py",
            "capability": "Can I build attention mechanisms for sequence understanding?"
        },
        "07": {
            "name": "Stability",
            "description": "Training stabilization with normalization",
            "test_file": "checkpoint_07_stability.py",
            "capability": "Can I stabilize training with normalization techniques?"
        },
        "08": {
            "name": "Differentiation",
            "description": "Automatic gradient computation for learning",
            "test_file": "checkpoint_08_differentiation.py",
            "capability": "Can I automatically compute gradients for learning?"
        },
        "09": {
            "name": "Optimization",
            "description": "Sophisticated optimization algorithms",
            "test_file": "checkpoint_09_optimization.py",
            "capability": "Can I optimize neural networks with sophisticated algorithms?"
        },
        "10": {
            "name": "Training",
            "description": "Complete training loops for end-to-end learning",
            "test_file": "checkpoint_10_training.py",
            "capability": "Can I build complete training loops for end-to-end learning?"
        },
        "11": {
            "name": "Regularization",
            "description": "Overfitting prevention and robust model building",
            "test_file": "checkpoint_11_regularization.py",
            "capability": "Can I prevent overfitting and build robust models?"
        },
        "12": {
            "name": "Kernels",
            "description": "High-performance computational kernels",
            "test_file": "checkpoint_12_kernels.py",
            "capability": "Can I implement high-performance computational kernels?"
        },
        "13": {
            "name": "Benchmarking",
            "description": "Performance analysis and bottleneck identification",
            "test_file": "checkpoint_13_benchmarking.py",
            "capability": "Can I analyze performance and identify bottlenecks in ML systems?"
        },
        "14": {
            "name": "Deployment",
            "description": "Production deployment and monitoring",
            "test_file": "checkpoint_14_deployment.py",
            "capability": "Can I deploy and monitor ML systems in production?"
        },
        "15": {
            "name": "Acceleration",
            "description": "Algorithmic optimization and acceleration techniques",
            "test_file": "checkpoint_15_acceleration.py",
            "capability": "Can I accelerate computations through algorithmic optimization?"
        },
        "16": {
            "name": "Quantization",
            "description": "Trading precision for speed with INT8 quantization",
            "test_file": "checkpoint_16_quantization.py",
            "capability": "Can I trade precision for speed with INT8 quantization?"
        },
        "17": {
            "name": "Compression",
            "description": "Neural network pruning for edge deployment",
            "test_file": "checkpoint_17_compression.py",
            "capability": "Can I remove 70% of parameters while maintaining accuracy?"
        },
        "18": {
            "name": "Caching",
            "description": "KV caching for transformer inference optimization",
            "test_file": "checkpoint_18_caching.py",
            "capability": "Can I transform O(N²) to O(N) complexity with intelligent caching?"
        },
        "19": {
            "name": "Competition",
            "description": "TinyMLPerf competition system for optimization mastery",
            "test_file": "checkpoint_19_competition.py",
            "capability": "Can I build competition-grade benchmarking infrastructure?"
        },
        "20": {
            "name": "TinyGPT Capstone",
            "description": "Complete language model demonstrating ML systems mastery",
            "test_file": "checkpoint_20_capstone.py",
            "capability": "Can I build a complete language model that generates coherent text from scratch?"
        }
    }
    
    def __init__(self, config: CLIConfig):
        """Initialize checkpoint system."""
        self.config = config
        self.console = get_console()
        self.modules_dir = config.project_root / "modules" / "source"
        self.checkpoints_dir = config.project_root / "tests" / "checkpoints"
    
    def get_checkpoint_test_status(self, checkpoint_id: str) -> Dict[str, bool]:
        """Get the status of a checkpoint test file."""
        if checkpoint_id not in self.CHECKPOINTS:
            return {"exists": False, "tested": False, "passed": False}
        
        test_file = self.CHECKPOINTS[checkpoint_id]["test_file"]
        test_path = self.checkpoints_dir / test_file
        
        return {
            "exists": test_path.exists(),
            "tested": False,  # Will be set when we run tests
            "passed": False   # Will be set based on test results
        }
    
    def get_checkpoint_status(self, checkpoint_id: str) -> Dict:
        """Get status information for a checkpoint."""
        checkpoint = self.CHECKPOINTS[checkpoint_id]
        test_status = self.get_checkpoint_test_status(checkpoint_id)
        
        return {
            "checkpoint": checkpoint,
            "test_status": test_status,
            "is_available": test_status["exists"],
            "is_complete": test_status.get("passed", False),
            "checkpoint_id": checkpoint_id
        }
    
    def get_overall_progress(self) -> Dict:
        """Get overall progress across all checkpoints."""
        checkpoints_status = {}
        current_checkpoint = None
        total_complete = 0
        total_checkpoints = len(self.CHECKPOINTS)
        
        for checkpoint_id in self.CHECKPOINTS.keys():
            status = self.get_checkpoint_status(checkpoint_id)
            checkpoints_status[checkpoint_id] = status
            
            if status["is_complete"]:
                total_complete += 1
            elif current_checkpoint is None and status["is_available"]:
                # First available but incomplete checkpoint is current
                current_checkpoint = checkpoint_id
        
        # If all are complete, set current to last checkpoint
        if current_checkpoint is None and total_complete == total_checkpoints:
            current_checkpoint = list(self.CHECKPOINTS.keys())[-1]
        # If none are complete, start with first
        elif current_checkpoint is None:
            current_checkpoint = "00"
        
        # Calculate overall percentage
        overall_percent = (total_complete / total_checkpoints * 100) if total_checkpoints > 0 else 0
        
        return {
            "checkpoints": checkpoints_status,
            "current": current_checkpoint,
            "overall_progress": overall_percent,
            "total_complete": total_complete,
            "total_checkpoints": total_checkpoints
        }
    
    def run_checkpoint_test(self, checkpoint_id: str) -> Dict:
        """Run a specific checkpoint test and return results."""
        if checkpoint_id not in self.CHECKPOINTS:
            return {"success": False, "error": f"Unknown checkpoint: {checkpoint_id}"}
        
        checkpoint = self.CHECKPOINTS[checkpoint_id]
        test_file = checkpoint["test_file"]
        test_path = self.checkpoints_dir / test_file
        
        if not test_path.exists():
            return {"success": False, "error": f"Test file not found: {test_file}"}
        
        try:
            # Run the test using subprocess to capture output
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                timeout=30  # 30 second timeout
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "checkpoint_name": checkpoint["name"],
                "capability": checkpoint["capability"]
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test timed out after 30 seconds"}
        except Exception as e:
            return {"success": False, "error": f"Test execution failed: {str(e)}"}


class CheckpointCommand(BaseCommand):
    """Checkpoint tracking and visualization command."""
    
    name = "checkpoint"
    description = "Track and visualize ML systems engineering progress through checkpoints"
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add checkpoint-specific arguments."""
        subparsers = parser.add_subparsers(
            dest='checkpoint_command',
            help='Checkpoint operations',
            metavar='COMMAND'
        )
        
        # Status command
        status_parser = subparsers.add_parser(
            'status',
            help='Show current checkpoint progress'
        )
        status_parser.add_argument(
            '--detailed', '-d',
            action='store_true',
            help='Show detailed module-level progress'
        )
        
        # Timeline command
        timeline_parser = subparsers.add_parser(
            'timeline',
            help='Show visual progress timeline'
        )
        timeline_parser.add_argument(
            '--horizontal',
            action='store_true',
            help='Show horizontal timeline (default: vertical)'
        )
        
        # Test command
        test_parser = subparsers.add_parser(
            'test',
            help='Test checkpoint capabilities'
        )
        test_parser.add_argument(
            'checkpoint_id',
            nargs='?',
            help='Checkpoint ID to test (00-20, current checkpoint if not specified)'
        )
        
        # Run command (new)
        run_parser = subparsers.add_parser(
            'run',
            help='Run specific checkpoint tests with progress tracking'
        )
        run_parser.add_argument(
            'checkpoint_id',
            help='Checkpoint ID to run (00-20)'
        )
        run_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed test output'
        )
        
        # Unlock command
        unlock_parser = subparsers.add_parser(
            'unlock',
            help='Attempt to unlock next checkpoint'
        )
    
    def run(self, args: argparse.Namespace) -> int:
        """Execute checkpoint command."""
        checkpoint_system = CheckpointSystem(self.config)
        
        if not args.checkpoint_command:
            return self._show_help(args)
        
        if args.checkpoint_command == 'status':
            return self._show_status(checkpoint_system, args)
        elif args.checkpoint_command == 'timeline':
            return self._show_timeline(checkpoint_system, args)
        elif args.checkpoint_command == 'test':
            return self._test_checkpoint(checkpoint_system, args)
        elif args.checkpoint_command == 'run':
            return self._run_checkpoint(checkpoint_system, args)
        elif args.checkpoint_command == 'unlock':
            return self._unlock_checkpoint(checkpoint_system, args)
        else:
            print_error(f"Unknown checkpoint command: {args.checkpoint_command}")
            return 1
    
    def _show_help(self, args: argparse.Namespace) -> int:
        """Show checkpoint command help."""
        console = get_console()
        console.print(Panel(
            "[bold cyan]TinyTorch Checkpoint System[/bold cyan]\n\n"
            "[bold]Track your progress through 20 capability checkpoints:[/bold]\n"
            "  00-04: Foundation  → Environment, tensors, networks\n"
            "  05-09: Architecture → Spatial, attention, autograd, optimization\n"
            "  10-14: Systems     → Training, kernels, benchmarking, deployment\n"
            "  15-19: Optimization → Acceleration, quantization, compression, caching, competition\n"
            "  20: Capstone       → Complete TinyGPT language model\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  [green]status[/green]     - Show current progress and capabilities\n"
            "  [green]timeline[/green]   - Visual progress timeline\n"
            "  [green]test[/green]       - Test checkpoint capabilities\n"
            "  [green]run[/green]        - Run specific checkpoint with progress\n"
            "  [green]unlock[/green]     - Attempt to unlock next checkpoint\n\n"
            "[bold]Examples:[/bold]\n"
            "  [dim]tito checkpoint status --detailed[/dim]\n"
            "  [dim]tito checkpoint timeline --horizontal[/dim]\n"
            "  [dim]tito checkpoint test 16[/dim]\n"
            "  [dim]tito checkpoint run 20 --verbose[/dim]",
            title="Checkpoint System (20 Checkpoints)",
            border_style="bright_blue"
        ))
        return 0
    
    def _show_status(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Show checkpoint status."""
        console = get_console()
        progress_data = checkpoint_system.get_overall_progress()
        
        # Header
        console.print(Panel(
            "[bold cyan]🚀 TinyTorch Framework Capabilities[/bold cyan]",
            border_style="bright_blue"
        ))
        
        # Overall progress
        overall_percent = progress_data["overall_progress"]
        console.print(f"\n[bold]Overall Progress:[/bold] {overall_percent:.0f}% ({progress_data['total_complete']}/{progress_data['total_checkpoints']} checkpoints)")
        
        # Current status summary
        current = progress_data["current"]
        if current:
            current_status = progress_data["checkpoints"][current]
            current_name = current_status["checkpoint"]["name"]
            
            console.print(f"[bold]Current Checkpoint:[/bold] {current:0>2} - {current_name}")
            
            if current_status["is_complete"]:
                console.print(f"[bold green]✅ {current_name} checkpoint achieved![/bold green]")
                console.print(f"[dim]Capability unlocked: {current_status['checkpoint']['capability']}[/dim]")
            else:
                console.print(f"[bold yellow]🎯 Ready to test {current_name} capabilities[/bold yellow]")
                console.print(f"[dim]Goal: {current_status['checkpoint']['capability']}[/dim]")
        
        console.print()
        
        # Checkpoint progress  
        for checkpoint_id, checkpoint_data in progress_data["checkpoints"].items():
            checkpoint = checkpoint_data["checkpoint"]
            
            # Checkpoint header
            if checkpoint_data["is_complete"]:
                status_icon = "✅"
                status_color = "green"
            elif checkpoint_id == current:
                status_icon = "🎯"
                status_color = "yellow"
            else:
                status_icon = "⏳"
                status_color = "dim"
            
            console.print(f"[bold]{status_icon} {checkpoint_id:0>2}: {checkpoint['name']}[/bold] [{status_color}]{'COMPLETE' if checkpoint_data['is_complete'] else 'PENDING'}[/{status_color}]")
            
            if args.detailed:
                # Show test file and availability
                test_status = checkpoint_data["test_status"]
                test_available = "✅" if test_status["exists"] else "❌"
                console.print(f"   {test_available} Test: {checkpoint['test_file']}")
            
            console.print(f"   [dim]{checkpoint['capability']}[/dim]\n")
        
        return 0
    
    def _show_timeline(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Show visual timeline with Rich progress bar."""
        console = get_console()
        progress_data = checkpoint_system.get_overall_progress()
        
        console.print("\n[bold cyan]🚀 TinyTorch Framework Progress Timeline[/bold cyan]\n")
        
        if args.horizontal:
            # Enhanced horizontal timeline with progress line
            overall_percent = progress_data["overall_progress"]
            total_checkpoints = progress_data["total_checkpoints"]
            complete_checkpoints = progress_data["total_complete"]
            
            # Create a visual progress bar
            filled = int(overall_percent / 2)  # 50 characters total width
            bar = "█" * filled + "░" * (50 - filled)
            console.print(f"[bold]Overall:[/bold] [{bar}] {overall_percent:.0f}%")
            console.print(f"[dim]{complete_checkpoints}/{total_checkpoints} checkpoints complete[/dim]\n")
            
            # Show checkpoint progression - group in rows of 8
            checkpoints_list = list(progress_data["checkpoints"].items())
            
            for row_start in range(0, len(checkpoints_list), 8):
                row_checkpoints = checkpoints_list[row_start:row_start + 8]
                
                # Build the checkpoint line for this row
                checkpoint_line = ""
                names_line = ""
                
                for i, (checkpoint_id, checkpoint_data) in enumerate(row_checkpoints):
                    checkpoint = checkpoint_data["checkpoint"]
                    
                    # Checkpoint status
                    if checkpoint_data["is_complete"]:
                        checkpoint_marker = f"[green]●[/green]"
                        name_color = "green"
                    elif checkpoint_id == progress_data["current"]:
                        checkpoint_marker = f"[yellow]◉[/yellow]"
                        name_color = "yellow"
                    else:
                        checkpoint_marker = f"[dim]○[/dim]"
                        name_color = "dim"
                    
                    # Add checkpoint with ID
                    checkpoint_line += f"{checkpoint_marker}{checkpoint_id}"
                    names_line += f"[{name_color}]{checkpoint['name'][:9]:^9}[/{name_color}]"
                    
                    # Add spacing (except for last in row)
                    if i < len(row_checkpoints) - 1:
                        if checkpoint_data["is_complete"]:
                            checkpoint_line += "[green]━━[/green]"
                        else:
                            checkpoint_line += "[dim]━━[/dim]"
                        names_line += "  "
                
                console.print(checkpoint_line)
                console.print(names_line)
                console.print()  # Empty line between rows
            
        else:
            # Vertical timeline (tree structure)
            tree = Tree("ML Systems Engineering Journey (20 Checkpoints)")
            
            for checkpoint_id, checkpoint_data in progress_data["checkpoints"].items():
                checkpoint = checkpoint_data["checkpoint"]
                
                if checkpoint_data["is_complete"]:
                    checkpoint_text = f"[green]✅ {checkpoint_id}: {checkpoint['name']}[/green]"
                elif checkpoint_id == progress_data["current"]:
                    checkpoint_text = f"[yellow]🎯 {checkpoint_id}: {checkpoint['name']} (CURRENT)[/yellow]"
                else:
                    checkpoint_text = f"[dim]⏳ {checkpoint_id}: {checkpoint['name']}[/dim]"
                
                checkpoint_node = tree.add(checkpoint_text)
                checkpoint_node.add(f"[dim]{checkpoint['capability']}[/dim]")
            
            console.print(tree)
        
        console.print()
        return 0
    
    def _test_checkpoint(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Test checkpoint capabilities."""
        console = get_console()
        
        # Determine which checkpoint to test
        checkpoint_id = args.checkpoint_id
        if not checkpoint_id:
            progress_data = checkpoint_system.get_overall_progress()
            checkpoint_id = progress_data["current"]
        
        # Validate checkpoint ID
        if checkpoint_id not in checkpoint_system.CHECKPOINTS:
            print_error(f"Unknown checkpoint: {checkpoint_id}")
            console.print(f"[dim]Available checkpoints: {', '.join(checkpoint_system.CHECKPOINTS.keys())}[/dim]")
            return 1
        
        checkpoint = checkpoint_system.CHECKPOINTS[checkpoint_id]
        
        # Show what we're testing
        console.print(f"\n[bold cyan]Testing Checkpoint {checkpoint_id}: {checkpoint['name']}[/bold cyan]")
        console.print(f"[bold]Capability Question:[/bold] {checkpoint['capability']}\n")
        
        # Run the test
        with console.status(f"[bold green]Running checkpoint {checkpoint_id} test...", spinner="dots") as status:
            result = checkpoint_system.run_checkpoint_test(checkpoint_id)
        
        # Display results
        if result["success"]:
            console.print(f"[bold green]✅ Checkpoint {checkpoint_id} PASSED![/bold green]")
            console.print(f"[green]Capability achieved: {checkpoint['capability']}[/green]\n")
            
            # Show brief output
            if result.get("stdout") and "🎉" in result["stdout"]:
                # Extract the completion message
                lines = result["stdout"].split('\n')
                for line in lines:
                    if "🎉" in line or "📝" in line or "🎯" in line:
                        console.print(f"[dim]{line}[/dim]")
            
            print_success(f"Checkpoint {checkpoint_id} test completed successfully!")
            return 0
        else:
            console.print(f"[bold red]❌ Checkpoint {checkpoint_id} FAILED[/bold red]\n")
            
            # Show error details
            if "error" in result:
                console.print(f"[red]Error: {result['error']}[/red]")
            elif result.get("stderr"):
                console.print(f"[red]Error output:[/red]")
                console.print(f"[dim]{result['stderr']}[/dim]")
            elif result.get("stdout"):
                console.print(f"[yellow]Test output:[/yellow]")
                console.print(f"[dim]{result['stdout']}[/dim]")
            
            print_error(f"Checkpoint {checkpoint_id} test failed")
            return 1
    
    def _run_checkpoint(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Run specific checkpoint test with detailed progress tracking."""
        console = get_console()
        checkpoint_id = args.checkpoint_id
        
        # Validate checkpoint ID
        if checkpoint_id not in checkpoint_system.CHECKPOINTS:
            print_error(f"Unknown checkpoint: {checkpoint_id}")
            console.print(f"[dim]Available checkpoints: {', '.join(checkpoint_system.CHECKPOINTS.keys())}[/dim]")
            return 1
        
        checkpoint = checkpoint_system.CHECKPOINTS[checkpoint_id]
        
        # Show detailed information
        console.print(Panel(
            f"[bold cyan]Checkpoint {checkpoint_id}: {checkpoint['name']}[/bold cyan]\n\n"
            f"[bold]Capability Question:[/bold]\n{checkpoint['capability']}\n\n"
            f"[bold]Test File:[/bold] {checkpoint['test_file']}\n"
            f"[bold]Description:[/bold] {checkpoint['description']}",
            title=f"Running Checkpoint {checkpoint_id}",
            border_style="bright_blue"
        ))
        
        # Check if test file exists
        test_path = checkpoint_system.checkpoints_dir / checkpoint["test_file"]
        if not test_path.exists():
            print_error(f"Test file not found: {checkpoint['test_file']}")
            return 1
        
        console.print(f"\n[bold]Executing test...[/bold]")
        
        # Run the test with status feedback
        with console.status(f"[bold green]Running checkpoint {checkpoint_id} test...", spinner="dots"):
            result = checkpoint_system.run_checkpoint_test(checkpoint_id)
        
        console.print()
        
        # Display detailed results
        if result["success"]:
            console.print(Panel(
                f"[bold green]✅ SUCCESS![/bold green]\n\n"
                f"[green]Checkpoint {checkpoint_id} completed successfully![/green]\n"
                f"[green]Capability achieved: {checkpoint['capability']}[/green]",
                title="Test Results",
                border_style="green"
            ))
            
            # Show test output if verbose or if it contains key markers
            if args.verbose or (result.get("stdout") and any(marker in result["stdout"] for marker in ["🎉", "✅", "📝", "🎯"])):
                console.print(f"\n[bold]Test Output:[/bold]")
                if result.get("stdout"):
                    console.print(result["stdout"])
            
            return 0
        else:
            console.print(Panel(
                f"[bold red]❌ FAILED[/bold red]\n\n"
                f"[red]Checkpoint {checkpoint_id} test failed[/red]\n"
                f"[yellow]This indicates the required capabilities are not yet implemented.[/yellow]",
                title="Test Results",
                border_style="red"
            ))
            
            # Show error details
            if "error" in result:
                console.print(f"\n[bold red]Error:[/bold red] {result['error']}")
            
            if args.verbose or "error" in result:
                if result.get("stdout"):
                    console.print(f"\n[bold]Standard Output:[/bold]")
                    console.print(result["stdout"])
                if result.get("stderr"):
                    console.print(f"\n[bold]Error Output:[/bold]")
                    console.print(result["stderr"])
            
            return 1
    
    def _unlock_checkpoint(self, checkpoint_system: CheckpointSystem, args: argparse.Namespace) -> int:
        """Attempt to unlock next checkpoint."""
        console = get_console()
        progress_data = checkpoint_system.get_overall_progress()
        current = progress_data["current"]
        
        if not current:
            console.print("[green]All checkpoints completed! 🎉[/green]")
            return 0
        
        current_status = progress_data["checkpoints"][current]
        
        if current_status["is_complete"]:
            console.print(f"[green]✅ Checkpoint {current} ({current_status['checkpoint']['name']}) already complete![/green]")
            
            # Find next checkpoint
            checkpoint_ids = list(checkpoint_system.CHECKPOINTS.keys())
            try:
                current_index = checkpoint_ids.index(current)
                if current_index < len(checkpoint_ids) - 1:
                    next_id = checkpoint_ids[current_index + 1]
                    next_checkpoint = checkpoint_system.CHECKPOINTS[next_id]
                    console.print(f"[bold]Next checkpoint:[/bold] {next_id} - {next_checkpoint['name']}")
                    console.print(f"[dim]Goal: {next_checkpoint['capability']}[/dim]")
                else:
                    console.print("[bold]🎉 All checkpoints completed![/bold]")
            except ValueError:
                console.print("[yellow]Cannot determine next checkpoint[/yellow]")
        else:
            console.print(f"[yellow]Test checkpoint {current} to unlock your next capability:[/yellow]")
            console.print(f"[bold]Goal:[/bold] {current_status['checkpoint']['capability']}")
            console.print(f"[dim]Run: tito checkpoint run {current}[/dim]")
        
        return 0