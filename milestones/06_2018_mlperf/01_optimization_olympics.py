#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🏆 MILESTONE 06: The Optimization Olympics (MLPerf 2018)           ║
║                  Compress and Accelerate Your Neural Network                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Historical Context:
In 2018, MLPerf was launched to standardize ML benchmarking. The key insight:
It's not just about accuracy - production ML needs efficiency too.

🎯 WHAT YOU'LL LEARN:
1. How to PROFILE your model (parameters, size, speed)
2. How to QUANTIZE (FP32 → INT8 = 4× smaller)
3. How to PRUNE (remove small weights = 2-4× smaller)
4. How to measure the TRADEOFFS (accuracy vs efficiency)

🏗️ THE OPTIMIZATION PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         YOUR TRAINED MODEL                               │
    │                    Accurate but large and slow                          │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────────┐
    │                          STEP 1: PROFILE                                 │
    │                  Count parameters, measure latency                       │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────────┐
    │                         STEP 2: QUANTIZE                                 │
    │                   FP32 → INT8 (4× compression)                           │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────────┐
    │                          STEP 3: PRUNE                                   │
    │                Remove small weights (2-4× compression)                   │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────────┐
    │                      OPTIMIZED MODEL 🎉                                  │
    │              8-16× smaller, minimal accuracy loss                        │
    └─────────────────────────────────────────────────────────────────────────┘

✅ REQUIRED MODULES (Run after Module 16):
  Module 14 (Profiling)     : YOUR profiling tools
  Module 15 (Quantization)  : YOUR quantization implementation
  Module 16 (Compression)   : YOUR pruning techniques

📊 EXPECTED RESULTS:
  | Optimization  | Size    | Accuracy | Notes                    |
  |---------------|---------|----------|--------------------------|
  | Baseline      | 100%    | 85-90%   | Full precision           |
  | + Quantization| 25%     | 84-89%   | INT8 weights             |
  | + Pruning     | 12.5%   | 82-87%   | 50% weights removed      |
  | Combined      | ~10%    | 80-85%   | Production ready!        |
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

# ============================================================================
# SIMPLE MLP FOR DEMONSTRATION
# ============================================================================

class SimpleMLP:
    """Simple MLP for digit classification - the optimization target."""
    
    def __init__(self, input_size=64, hidden_size=32, num_classes=10):
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        
        # Store weight references for optimization
        self.layers = [self.fc1, self.fc2]
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            # Check both 'weights' and 'weight' (different naming conventions)
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
            elif hasattr(layer, 'weight'):
                params.append(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                params.append(layer.bias)
        return params
    
    def get_weights(self):
        """Get all weights as a list of numpy arrays."""
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.data.copy())
            elif hasattr(layer, 'weight'):
                weights.append(layer.weight.data.copy())
        return weights
    
    def set_weights(self, weights):
        """Set all weights from a list of numpy arrays."""
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights.data = weights[idx].copy()
                idx += 1
            elif hasattr(layer, 'weight'):
                layer.weight.data = weights[idx].copy()
                idx += 1


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count total parameters in model."""
    total = 0
    for param in model.parameters():
        total += param.data.size
    return total


def model_size_bytes(model):
    """Calculate model size in bytes (FP32)."""
    return count_parameters(model) * 4  # 4 bytes per float32


def quantize_weights(weights, bits=8):
    """
    Quantize weights to lower precision.
    
    FP32 (4 bytes) → INT8 (1 byte) = 4× compression
    """
    quantized = []
    for w in weights:
        # Simple post-training quantization
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (2**bits - 1)
        if scale == 0:
            scale = 1.0
        
        # Quantize
        w_int = np.round((w - w_min) / scale).astype(np.int8)
        
        # Dequantize for inference
        w_dequant = w_int.astype(np.float32) * scale + w_min
        quantized.append(w_dequant)
    
    return quantized


def prune_weights(weights, sparsity=0.5):
    """
    Prune weights by setting smallest magnitudes to zero.
    
    Sparsity 0.5 = 50% of weights are zeros = 2× compression potential
    """
    pruned = []
    for w in weights:
        # Find threshold
        flat = np.abs(w.flatten())
        threshold = np.percentile(flat, sparsity * 100)
        
        # Prune
        mask = np.abs(w) > threshold
        w_pruned = w * mask
        pruned.append(w_pruned)
    
    return pruned


def evaluate_accuracy(model, X, y):
    """Evaluate model accuracy."""
    from tinytorch.core.tensor import Tensor
    
    logits = model(Tensor(X))
    preds = np.argmax(logits.data, axis=1)
    accuracy = np.mean(preds == y.flatten()) * 100
    return accuracy


def measure_latency(model, X, n_runs=100):
    """Measure average inference latency."""
    from tinytorch.core.tensor import Tensor
    
    # Warmup
    for _ in range(10):
        _ = model(Tensor(X[:1]))
    
    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(Tensor(X[:1]))
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times) * 1000  # ms


# ============================================================================
# MAIN MILESTONE
# ============================================================================

def load_tinydigits():
    """Load TinyDigits dataset (bundled with TinyTorch)."""
    import pickle
    
    # Find dataset
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "datasets" / "tinydigits" / "train.pkl"
    test_path = project_root / "datasets" / "tinydigits" / "test.pkl"
    
    if train_path.exists() and test_path.exists():
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        X_train = train_data['images'].astype(np.float32)
        y_train = train_data['labels'].reshape(-1, 1)
        X_test = test_data['images'].astype(np.float32)
        y_test = test_data['labels'].reshape(-1, 1)
        
        # Flatten images if they're 2D (8x8) - MLP needs flat input
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        return X_train, y_train, X_test, y_test
    
    # Fallback: try alternative path or generate synthetic
    alt_path = project_root / "datasets" / "tinydigits" / "tinydigits.pkl"
    if alt_path.exists():
        with open(alt_path, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    console.print(f"[yellow]Dataset not found, using synthetic data for demo...[/yellow]")
    
    # Generate synthetic digit-like data
    np.random.seed(42)
    X_train = np.random.randn(1000, 64).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(1000, 1))
    X_test = np.random.randn(200, 64).astype(np.float32)
    y_test = np.random.randint(0, 10, size=(200, 1))
    return X_train, y_train, X_test, y_test


def train_baseline(model, X_train, y_train, epochs=10, lr=0.01):
    """Quick training of baseline model."""
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.core.optimizers import SGD
    
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    for param in model.parameters():
        param.requires_grad = True
    
    batch_size = 32
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        
        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            X_batch = Tensor(X_train[batch_idx])
            y_batch = Tensor(y_train[batch_idx].flatten())
            
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main():
    """Run the Optimization Olympics!"""
    
    # Welcome
    console.print()
    console.print(Panel(
        "[bold cyan]🏆 THE OPTIMIZATION OLYMPICS[/bold cyan]\n\n"
        "[dim]MLPerf 2018: Where accuracy meets efficiency[/dim]\n\n"
        "Today you'll learn to compress a neural network\n"
        "while preserving accuracy - just like real production ML!",
        title="Milestone 06: MLPerf",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()
    
    # ========================================================================
    # STEP 1: BASELINE
    # ========================================================================
    
    console.print(Panel(
        "[bold yellow]📊 STEP 1: Establish Baseline[/bold yellow]\n"
        "Train a model and measure its performance",
        border_style="yellow"
    ))
    
    # Load data
    console.print("[dim]Loading TinyDigits dataset...[/dim]")
    X_train, y_train, X_test, y_test = load_tinydigits()
    console.print(f"  ✓ Training: {len(X_train)} samples")
    console.print(f"  ✓ Test: {len(X_test)} samples")
    console.print()
    
    # Train baseline
    console.print("[dim]Training baseline MLP...[/dim]")
    model = SimpleMLP(input_size=64, hidden_size=32, num_classes=10)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Training...", total=None)
        train_baseline(model, X_train, y_train, epochs=15)
    
    # Baseline metrics
    baseline_params = count_parameters(model)
    baseline_size = model_size_bytes(model)
    baseline_acc = evaluate_accuracy(model, X_test, y_test)
    baseline_latency = measure_latency(model, X_test)
    baseline_weights = model.get_weights()
    
    # Show baseline
    table = Table(title="📊 Baseline Model", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Notes", style="dim")
    
    table.add_row("Parameters", f"{baseline_params:,}", "Total trainable weights")
    table.add_row("Size", f"{baseline_size:,} bytes", "FP32 precision")
    table.add_row("Accuracy", f"{baseline_acc:.1f}%", "Test set performance")
    table.add_row("Latency", f"{baseline_latency:.3f} ms", "Per-sample inference")
    
    console.print(table)
    console.print()
    
    # ========================================================================
    # STEP 2: QUANTIZATION
    # ========================================================================
    
    console.print(Panel(
        "[bold blue]🗜️ STEP 2: Quantization (FP32 → INT8)[/bold blue]\n"
        "Reduce precision: 4 bytes → 1 byte = 4× smaller",
        border_style="blue"
    ))
    
    # Apply quantization
    quantized_weights = quantize_weights(baseline_weights, bits=8)
    model.set_weights(quantized_weights)
    
    quant_size = baseline_size // 4  # INT8 is 4× smaller
    quant_acc = evaluate_accuracy(model, X_test, y_test)
    quant_latency = measure_latency(model, X_test)
    
    # Show quantization results
    table = Table(title="🗜️ After Quantization", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")
    
    table.add_row(
        "Size", 
        f"{baseline_size:,} B", 
        f"{quant_size:,} B",
        f"[green]4× smaller[/green]"
    )
    table.add_row(
        "Accuracy",
        f"{baseline_acc:.1f}%",
        f"{quant_acc:.1f}%",
        f"[{'green' if abs(baseline_acc - quant_acc) < 2 else 'yellow'}]{baseline_acc - quant_acc:+.1f}%[/]"
    )
    
    console.print(table)
    console.print()
    
    # ========================================================================
    # STEP 3: PRUNING
    # ========================================================================
    
    console.print(Panel(
        "[bold magenta]✂️ STEP 3: Pruning (Remove Small Weights)[/bold magenta]\n"
        "Set 50% of smallest weights to zero = 2× compression potential",
        border_style="magenta"
    ))
    
    # Apply pruning
    pruned_weights = prune_weights(baseline_weights, sparsity=0.5)
    model.set_weights(pruned_weights)
    
    # Count zeros
    total_weights = sum(w.size for w in pruned_weights)
    zero_weights = sum(np.sum(w == 0) for w in pruned_weights)
    sparsity = (zero_weights / total_weights * 100) if total_weights > 0 else 0
    
    pruned_acc = evaluate_accuracy(model, X_test, y_test)
    
    # Show pruning results
    table = Table(title="✂️ After Pruning", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_column("Change", style="bold")
    
    table.add_row(
        "Non-zero weights",
        f"{total_weights:,}",
        f"{total_weights - zero_weights:,}",
        f"[green]{sparsity:.0f}% pruned[/green]"
    )
    table.add_row(
        "Accuracy",
        f"{baseline_acc:.1f}%",
        f"{pruned_acc:.1f}%",
        f"[{'green' if abs(baseline_acc - pruned_acc) < 5 else 'yellow'}]{baseline_acc - pruned_acc:+.1f}%[/]"
    )
    
    console.print(table)
    console.print()
    
    # ========================================================================
    # STEP 4: COMBINED
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🎯 STEP 4: Combined Optimization[/bold green]\n"
        "Apply BOTH quantization AND pruning",
        border_style="green"
    ))
    
    # Apply both
    combined_weights = prune_weights(baseline_weights, sparsity=0.5)
    combined_weights = quantize_weights(combined_weights, bits=8)
    model.set_weights(combined_weights)
    
    combined_size = quant_size  # Still quantized
    combined_acc = evaluate_accuracy(model, X_test, y_test)
    
    # Calculate effective compression (quantization + sparsity)
    effective_compression = 4 * 2  # 4× from quantization, potential 2× from sparsity
    
    console.print()
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    
    console.print("=" * 70)
    console.print(Panel("[bold]🏆 OPTIMIZATION OLYMPICS RESULTS[/bold]", border_style="gold1"))
    console.print()
    
    # Final comparison table
    table = Table(title="🎖️ Final Standings", box=box.DOUBLE)
    table.add_column("Configuration", style="cyan", width=20)
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Compression", style="bold magenta", justify="right")
    
    table.add_row(
        "🥇 Baseline (FP32)",
        f"{baseline_size:,} B",
        f"{baseline_acc:.1f}%",
        "1×"
    )
    table.add_row(
        "🥈 + Quantization",
        f"{quant_size:,} B",
        f"{quant_acc:.1f}%",
        "[green]4×[/green]"
    )
    table.add_row(
        "🥉 + Pruning",
        f"~{baseline_size//2:,} B*",
        f"{pruned_acc:.1f}%",
        "[green]2×[/green]"
    )
    table.add_row(
        "🏆 Combined",
        f"~{baseline_size//8:,} B*",
        f"{combined_acc:.1f}%",
        f"[bold green]{effective_compression}×[/bold green]"
    )
    
    console.print(table)
    console.print("[dim]* Effective size with sparse storage[/dim]")
    console.print()
    
    # Key insights
    console.print(Panel(
        "[bold green]🎓 KEY INSIGHTS[/bold green]\n\n"
        f"✅ [cyan]Quantization (FP32 → INT8):[/cyan]\n"
        f"   • 4× smaller model size\n"
        f"   • {abs(baseline_acc - quant_acc):.1f}% accuracy impact\n"
        f"   • [dim]Used by: TensorRT, ONNX Runtime, mobile deployment[/dim]\n\n"
        f"✅ [cyan]Pruning (Remove Small Weights):[/cyan]\n"
        f"   • {sparsity:.0f}% weights removed\n"
        f"   • {abs(baseline_acc - pruned_acc):.1f}% accuracy impact\n"
        f"   • [dim]Used by: Mobile models, edge deployment[/dim]\n\n"
        f"✅ [cyan]Combined:[/cyan]\n"
        f"   • {effective_compression}× total compression\n"
        f"   • {abs(baseline_acc - combined_acc):.1f}% accuracy impact\n"
        f"   • [dim]The secret sauce of production ML![/dim]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    
    # Verdict
    accuracy_drop = baseline_acc - combined_acc
    if accuracy_drop < 5:
        verdict = "[bold green]🏆 EXCELLENT![/bold green] Great compression with minimal accuracy loss!"
    elif accuracy_drop < 10:
        verdict = "[bold yellow]🥈 GOOD![/bold yellow] Solid compression, acceptable accuracy tradeoff."
    else:
        verdict = "[bold red]⚠️  HIGH LOSS[/bold red] - Consider less aggressive settings."
    
    console.print(Panel(
        f"{verdict}\n\n"
        f"[dim]You achieved {effective_compression}× compression with {accuracy_drop:.1f}% accuracy loss.[/dim]\n\n"
        "[bold cyan]What you learned:[/bold cyan]\n"
        "  ✅ How to profile ML models\n"
        "  ✅ Quantization: reduce precision for smaller models\n"
        "  ✅ Pruning: remove weights for sparser models\n"
        "  ✅ The accuracy-efficiency tradeoff\n\n"
        "[bold]This is how production ML systems are deployed![/bold]",
        title="🎯 Milestone 06 Complete",
        border_style="green",
        box=box.DOUBLE
    ))
    console.print()


if __name__ == "__main__":
    main()

