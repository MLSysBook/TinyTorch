#!/usr/bin/env python3
"""
TinyTorch CLI - Main entry point for training and evaluation.

This command-line interface provides access to all TinyTorch functionality:
- Training models on various datasets
- Evaluating trained models  
- Profiling and benchmarking
- System information and diagnostics
- Configuration management
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path to find tinytorch package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinytorch.core import __version__


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="tinytorch",
        description="Tiny🔥Torch ML System - Build neural networks from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/tito.py train --config tinytorch/configs/default.yaml
  python bin/tito.py eval --model checkpoints/best_model.pth --dataset mnist
  python bin/tito.py benchmark --ops matmul,conv2d
  python bin/tito.py info --show-architecture
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"Tiny🔥Torch {__version__}"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration file"
    )
    train_parser.add_argument(
        "--resume", 
        type=str,
        help="Path to checkpoint to resume from"
    )
    train_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="logs/runs",
        help="Directory for output logs and checkpoints"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    eval_parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["mnist", "cifar10"],
        help="Dataset to evaluate on"
    )
    eval_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size for evaluation"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark operations")
    benchmark_parser.add_argument(
        "--ops", 
        type=str,
        help="Comma-separated list of operations to benchmark"
    )
    benchmark_parser.add_argument(
        "--sizes", 
        type=str, 
        default="32,64,128,256,512",
        help="Comma-separated list of sizes to test"
    )
    benchmark_parser.add_argument(
        "--iterations", 
        type=int, 
        default=100,
        help="Number of iterations per benchmark"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="System information")
    info_parser.add_argument(
        "--show-architecture", 
        action="store_true",
        help="Show system architecture diagram"
    )
    info_parser.add_argument(
        "--show-config", 
        action="store_true",
        help="Show current configuration"
    )
    
    return parser


def train_command(args) -> int:
    """
    Execute training command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print(f"Training with config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    
    # TODO: Implement training in Chapter 8
    print("ERROR: Training not yet implemented (will be added in Chapter 8)")
    return 1


def eval_command(args) -> int:
    """
    Execute evaluation command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    
    # TODO: Implement evaluation in Chapter 8
    print("ERROR: Evaluation not yet implemented (will be added in Chapter 8)")
    return 1


def benchmark_command(args) -> int:
    """
    Execute benchmark command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    if args.ops:
        ops = args.ops.split(",")
        print(f"Benchmarking operations: {ops}")
    else:
        print("Benchmarking all available operations")
    
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"Testing sizes: {sizes}")
    print(f"Iterations per test: {args.iterations}")
    
    # TODO: Implement benchmarking in Chapter 12
    print("ERROR: Benchmarking not yet implemented (will be added in Chapter 12)")
    return 1


def info_command(args) -> int:
    """
    Execute info command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print(f"Tiny🔥Torch ML System v{__version__}")
    print("=" * 50)
    
    if args.show_architecture:
        print("\nSystem Architecture:")
        print("""
        ┌─────────────────────────────────────────────────────────────┐
        │                   Tiny🔥Torch System                        │
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
    
    if args.show_config:
        print("\nCurrent Configuration:")
        print(f"  Python: {sys.version}")
        print(f"  Working Directory: {os.getcwd()}")
        print(f"  TinyTorch Path: {Path(__file__).parent}")
        
        # Check for optional dependencies
        try:
            import numpy as np
            print(f"  NumPy: {np.__version__}")
        except ImportError:
            print("  NumPy: Not installed")
            
        try:
            import numba
            print(f"  Numba: {numba.__version__}")
        except ImportError:
            print("  Numba: Not installed (optional)")
    
    # Show implementation status
    print("\nImplementation Status:")
    projects = [
        ("setup", "Environment setup & onboarding", "✅ Environment ready"),
        ("tensor", "Core tensor implementation", "✅ Basic structure"),
        ("mlp", "Multi-layer perceptron", "🚧 Not implemented"),
        ("cnn", "Convolutional neural networks", "🚧 Not implemented"), 
        ("config", "Configuration system", "🚧 Not implemented"),
        ("data", "Data pipeline & loading", "🚧 Not implemented"),
        ("autograd", "Automatic differentiation", "🚧 Not implemented"),
        ("training", "Training loop & optimization", "🚧 Not implemented"),
        ("profiling", "Performance profiling tools", "🚧 Not implemented"),
        ("compression", "Model compression techniques", "🚧 Not implemented"),
        ("kernels", "Custom compute kernels", "🚧 Not implemented"),
        ("benchmarking", "Performance benchmarking", "🚧 Not implemented"),
        ("mlops", "MLOps & production monitoring", "🚧 Not implemented"),
    ]
    
    for project, component, status in projects:
        print(f"  {project:12} {component:20} {status}")
    
    return 0


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to appropriate command handler
    if args.command == "train":
        return train_command(args)
    elif args.command == "eval":
        return eval_command(args)
    elif args.command == "benchmark":
        return benchmark_command(args)
    elif args.command == "info":
        return info_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 