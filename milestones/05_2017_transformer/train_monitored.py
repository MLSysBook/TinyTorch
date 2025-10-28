#!/usr/bin/env python3
"""
Monitored Training Script for TinyTalks
========================================

Features:
- Early stopping if loss doesn't improve
- Continuous progress monitoring
- Automatic experiment termination for bad runs
- Clear feedback on learning progress

Usage:
    python train_monitored.py --mode test    # 10 epochs, quick validation
    python train_monitored.py --mode full    # 30 epochs, full training
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

# Import TinyTorch components
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.text.tokenization import CharTokenizer

console = Console()

# Import TinyGPT and dataset classes
exec(open(project_root / "milestones/05_2017_transformer/tinytalks_gpt.py").read())


class TrainingMonitor:
    """Monitor training progress and implement early stopping"""
    
    def __init__(self, patience=5, min_delta=0.01):
        """
        Args:
            patience: Number of checks without improvement before stopping
            min_delta: Minimum change in loss to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.checks_without_improvement = 0
        self.losses = []
        
    def check(self, current_loss):
        """
        Check if training should continue
        
        Returns:
            (should_continue, message)
        """
        self.losses.append(current_loss)
        
        # Calculate improvement
        improvement = self.best_loss - current_loss
        
        if improvement > self.min_delta:
            # Significant improvement
            self.best_loss = current_loss
            self.checks_without_improvement = 0
            return True, f"✓ Loss improved by {improvement:.4f}"
        else:
            # No significant improvement
            self.checks_without_improvement += 1
            
            if self.checks_without_improvement >= self.patience:
                return False, f"✗ No improvement for {self.patience} checks. Stopping."
            else:
                return True, f"⚠ No improvement ({self.checks_without_improvement}/{self.patience})"
    
    def summary(self):
        """Get training summary"""
        if len(self.losses) < 2:
            return "Not enough data"
        
        initial = self.losses[0]
        final = self.losses[-1]
        best = min(self.losses)
        decrease = initial - final
        decrease_pct = (decrease / initial) * 100 if initial > 0 else 0
        
        return {
            'initial_loss': initial,
            'final_loss': final,
            'best_loss': best,
            'total_decrease': decrease,
            'decrease_percent': decrease_pct,
            'num_checks': len(self.losses)
        }


def train_with_monitoring(model, dataset, optimizer, criterion, config, monitor):
    """
    Train with continuous monitoring and early stopping
    
    Args:
        model: TinyGPT model
        dataset: TinyTalksDataset
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        config: Training configuration dict
        monitor: TrainingMonitor instance
    
    Returns:
        success: True if training completed successfully
    """
    epochs = config['epochs']
    batch_size = config['batch_size']
    check_interval = config.get('check_interval', 50)  # Check every N batches
    
    console.print(f"\n[bold cyan]Starting Training with Monitoring[/bold cyan]")
    console.print(f"  Check interval: Every {check_interval} batches")
    console.print(f"  Early stopping: {monitor.patience} checks without improvement\n")
    
    total_batches_processed = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        console.print(f"[bold]Epoch {epoch+1}/{epochs}[/bold]")
        
        # Create batches
        num_sequences = len(dataset)
        indices = np.random.permutation(num_sequences)
        
        for batch_start in range(0, num_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_sequences)
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch data
            batch_inputs = []
            batch_targets = []
            for idx in batch_indices:
                input_seq, target_seq = dataset[idx]
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
            
            # Convert to tensors
            batch_input = Tensor(np.array(batch_inputs))
            batch_target = Tensor(np.array(batch_targets))
            
            # Forward pass
            logits = model.forward(batch_input)
            
            # Reshape for loss
            batch_size_actual, seq_length, vocab_size = logits.shape
            logits_2d = logits.reshape(batch_size_actual * seq_length, vocab_size)
            targets_1d = batch_target.reshape(-1)
            
            # Compute loss
            loss = criterion.forward(logits_2d, targets_1d)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Track loss
            loss_value = float(loss.data)
            epoch_loss += loss_value
            batch_count += 1
            total_batches_processed += 1
            
            # Monitor progress at check intervals
            if total_batches_processed % check_interval == 0:
                avg_loss = epoch_loss / batch_count
                should_continue, message = monitor.check(avg_loss)
                
                elapsed = time.time() - start_time
                console.print(f"  Batch {total_batches_processed} | Loss: {avg_loss:.4f} | {message} | Time: {elapsed:.1f}s")
                
                if not should_continue:
                    console.print(f"\n[yellow]Early stopping triggered at epoch {epoch+1}, batch {batch_count}[/yellow]")
                    return False
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / batch_count
        epoch_time = time.time() - epoch_start
        console.print(f"  → Epoch {epoch+1} complete: Avg Loss = {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s\n")
    
    console.print(f"[green]✓ Training completed successfully![/green]\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Monitored TinyTalks Training')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                       help='Training mode: test (10 epochs) or full (30 epochs)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (checks without improvement)')
    parser.add_argument('--min-delta', type=float, default=0.01,
                       help='Minimum loss decrease to count as improvement')
    parser.add_argument('--check-interval', type=int, default=50,
                       help='Check progress every N batches')
    
    args = parser.parse_args()
    
    # Enable autograd
    enable_autograd()
    
    # Configuration based on mode
    if args.mode == 'test':
        config = {
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.001,
            'embed_dim': 128,
            'num_layers': 6,
            'num_heads': 8,
            'check_interval': args.check_interval,
            'mode': 'TEST (Quick Validation)'
        }
    else:  # full
        config = {
            'epochs': 30,
            'batch_size': 32,
            'lr': 0.001,
            'embed_dim': 128,
            'num_layers': 6,
            'num_heads': 8,
            'check_interval': args.check_interval,
            'mode': 'FULL (Complete Training)'
        }
    
    # Display configuration
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]    Monitored TinyTalks Training - Option C       [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")
    
    table = Table(box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Mode", config['mode'])
    table.add_row("Epochs", str(config['epochs']))
    table.add_row("Batch Size", str(config['batch_size']))
    table.add_row("Learning Rate", str(config['lr']))
    table.add_row("Model Size", f"{config['embed_dim']}d, {config['num_layers']}L, {config['num_heads']}H")
    table.add_row("Early Stopping Patience", str(args.patience))
    table.add_row("Min Delta", str(args.min_delta))
    table.add_row("Check Interval", f"Every {args.check_interval} batches")
    
    console.print(table)
    console.print()
    
    # Load dataset
    console.print("[bold]Loading TinyTalks dataset...[/bold]")
    dataset_path = project_root / "datasets/tinytalks/splits/train.txt"
    with open(dataset_path, 'r') as f:
        text = f.read()
    
    dataset = TinyTalksDataset(text, seq_length=64)
    console.print(f"  ✓ Loaded: {len(text):,} chars, {dataset.tokenizer.vocab_size} vocab\n")
    
    # Initialize model
    console.print("[bold]Initializing model...[/bold]")
    model = TinyGPT(
        vocab_size=dataset.tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=64
    )
    
    params = model.parameters()
    param_count = sum(p.data.size for p in params)
    console.print(f"  ✓ Model initialized: {param_count:,} parameters\n")
    
    # Initialize training components
    optimizer = Adam(params, lr=config['lr'])
    criterion = CrossEntropyLoss()
    monitor = TrainingMonitor(patience=args.patience, min_delta=args.min_delta)
    
    # Train
    console.print("[bold]Starting training...[/bold]\n")
    start_time = time.time()
    
    success = train_with_monitoring(model, dataset, optimizer, criterion, config, monitor)
    
    total_time = time.time() - start_time
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]              Training Summary                     [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")
    
    summary = monitor.summary()
    
    result_table = Table(box=box.ROUNDED)
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="yellow")
    
    result_table.add_row("Status", "✓ SUCCESS" if success else "⚠ EARLY STOP")
    result_table.add_row("Total Time", f"{total_time/60:.1f} minutes")
    result_table.add_row("Initial Loss", f"{summary['initial_loss']:.4f}")
    result_table.add_row("Final Loss", f"{summary['final_loss']:.4f}")
    result_table.add_row("Best Loss", f"{summary['best_loss']:.4f}")
    result_table.add_row("Total Decrease", f"{summary['total_decrease']:.4f} ({summary['decrease_percent']:.1f}%)")
    result_table.add_row("Checks Performed", str(summary['num_checks']))
    
    console.print(result_table)
    console.print()
    
    # Recommendation
    if success and summary['decrease_percent'] > 50:
        console.print("[bold green]✓ EXCELLENT: Model is learning well! Continue with full training.[/bold green]")
    elif success and summary['decrease_percent'] > 20:
        console.print("[bold yellow]⚠ MODERATE: Model is learning but slowly. Consider tuning hyperparameters.[/bold yellow]")
    elif success:
        console.print("[bold red]✗ POOR: Model not learning effectively. Needs hyperparameter adjustment.[/bold red]")
    else:
        console.print("[bold red]✗ FAILED: Training stopped early. Try different hyperparameters.[/bold red]")


if __name__ == "__main__":
    main()

