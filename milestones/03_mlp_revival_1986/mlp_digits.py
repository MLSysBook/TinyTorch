#!/usr/bin/env python3
"""
MLP on Digits (1986) - Rumelhart, Hinton, Williams
==================================================

🎯 MILESTONE 3: MULTI-LAYER PERCEPTRON ON REAL DIGITS

The 1986 backpropagation paper proved multi-layer networks could solve
real-world problems. Let's recreate that breakthrough using YOUR TinyTorch!

✅ REQUIRED MODULES (Run after Module 08):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure
  Module 02 (Activations)   : YOUR ReLU activation  
  Module 03 (Layers)        : YOUR Linear layer
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (Autograd)      : YOUR automatic differentiation
  Module 06 (Optimizers)    : YOUR SGD optimizer
  Module 08 (DataLoader)    : YOUR data batching system
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE:
    ┌─────────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Input Image │    │ Flatten │    │ Linear  │    │ Linear  │
    │    8×8      │───▶│   64    │───▶│ 64→32   │───▶│ 32→10   │
    │   Pixels    │    │         │    │  +ReLU  │    │         │
    └─────────────┘    └─────────┘    └─────────┘    └─────────┘
                                      Hidden Layer   10 Classes

📊 DATASET: 8×8 Handwritten Digits
  - 1,797 real handwritten digits (from UCI)
  - 8×8 grayscale images (64 features)
  - 10 classes (digits 0-9)
  - Ships with TinyTorch (no download!)

🔥 THE BREAKTHROUGH:
  - Multi-layer networks learn hierarchical features
  - Hidden layer discovers useful digit patterns
  - Expected: 85-90% accuracy (excellent for MLP on small images!)
  
📌 NOTE: This is a BASELINE. CNN (Milestone 04) will show improvement!
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss, SGD
from tinytorch.data.loader import TensorDataset, DataLoader

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ============================================================================
# 🎓 STUDENT CODE: Multi-Layer Perceptron
# ============================================================================

class DigitMLP:
    """
    Multi-Layer Perceptron for digit classification.
    
    Architecture:
      Input (64) → Linear(64→32) → ReLU → Linear(32→10) → Output
    """
    
    def __init__(self, input_size=64, hidden_size=32, num_classes=10):
        console.print("🧠 Building Multi-Layer Perceptron...")
        
        # Hidden layer
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        
        # Output layer
        self.fc2 = Linear(hidden_size, num_classes)
        
        console.print(f"  ✓ Hidden layer: {input_size} → {hidden_size} (with ReLU)")
        console.print(f"  ✓ Output layer: {hidden_size} → {num_classes}")
        
        total_params = (input_size * hidden_size + hidden_size) + \
                      (hidden_size * num_classes + num_classes)
        console.print(f"  ✓ Total parameters: {total_params:,}\n")
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten if needed (8×8 → 64)
        if len(x.data.shape) > 2:
            batch_size = x.data.shape[0]
            x = Tensor(x.data.reshape(batch_size, -1))
        
        # Hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc2(x)
        return x
    
    def parameters(self):
        """Get all trainable parameters."""
        return [self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias]


def load_digit_dataset():
    """Load the 8×8 digits dataset."""
    console.print(Panel.fit(
        "[bold]Loading 8×8 Digit Dataset[/bold]\n"
        "Real handwritten digits from UCI repository",
        title="📊 Dataset",
        border_style="cyan"
    ))
    
    # Load from local data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'digits_8x8.npz')
    
    if not os.path.exists(data_path):
        console.print(f"[red]✗ Dataset not found at {data_path}[/red]")
        console.print("[yellow]Expected location: milestones/03_mlp_revival_1986/data/[/yellow]")
        sys.exit(1)
    
    data = np.load(data_path)
    images = data['images']  # (1797, 8, 8)
    labels = data['labels']  # (1797,)
    
    console.print(f"✓ Loaded {len(images)} digit images")
    console.print(f"✓ Image shape: {images[0].shape}")
    console.print(f"✓ Classes: {np.unique(labels)}")
    
    # Split into train/test (80/20)
    n_train = int(0.8 * len(images))
    
    train_images = Tensor(images[:n_train].astype(np.float32))
    train_labels = Tensor(labels[:n_train].astype(np.int64))
    test_images = Tensor(images[n_train:].astype(np.float32))
    test_labels = Tensor(labels[n_train:].astype(np.int64))
    
    console.print(f"\n📊 Split:")
    console.print(f"  Training: {len(train_images.data)} samples")
    console.print(f"  Testing:  {len(test_images.data)} samples\n")
    
    return train_images, train_labels, test_images, test_labels


def evaluate_accuracy(model, images, labels):
    """Compute classification accuracy."""
    # Forward pass
    logits = model.forward(images)
    
    # Get predictions (argmax)
    predictions = np.argmax(logits.data, axis=1)
    
    # Compare with labels
    correct = (predictions == labels.data).sum()
    total = len(labels.data)
    accuracy = 100.0 * correct / total
    
    return accuracy, predictions


def compare_batch_sizes(train_images, train_labels, test_images, test_labels):
    """
    Compare different batch sizes to show DataLoader's impact on training.
    
    This demonstrates a key systems trade-off in ML:
    - Larger batches: Faster throughput (fewer Python loops)
    - Smaller batches: More gradient updates per epoch
    """
    import time
    
    console.print(Panel.fit(
        "[bold cyan]🔬 Systems Experiment: Batch Size Impact[/bold cyan]\n\n"
        "[dim]Let's explore how batch size affects training speed and learning.\n"
        "This shows YOUR DataLoader in action![/dim]",
        title="⚙️ DataLoader Capabilities",
        border_style="yellow"
    ))
    
    batch_sizes = [16, 64, 256]
    epochs = 5  # Quick experiment
    results = []
    
    for batch_size in batch_sizes:
        console.print(f"\n[bold]Testing batch_size={batch_size}[/bold]")
        
        # Create DataLoader with this batch size
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        console.print(f"  Batches per epoch: {len(train_loader)}")
        
        # Create fresh model
        model = DigitMLP(input_size=64, hidden_size=32, num_classes=10)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()
        
        # Time the training
        start_time = time.time()
        
        for epoch in range(epochs):
            for batch_images, batch_labels in train_loader:
                logits = model.forward(batch_images)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        elapsed = time.time() - start_time
        
        # Evaluate
        final_acc, _ = evaluate_accuracy(model, test_images, test_labels)
        
        # Calculate throughput
        total_samples = len(train_dataset.features) * epochs
        samples_per_sec = total_samples / elapsed
        updates = len(train_loader) * epochs
        
        results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'accuracy': final_acc,
            'updates': updates,
            'throughput': samples_per_sec
        })
        
        console.print(f"  Time: {elapsed:.1f}s, Accuracy: {final_acc:.1f}%")
    
    # Show comparison table
    console.print("\n")
    table = Table(title="📊 Batch Size Comparison", box=box.ROUNDED)
    table.add_column("Batch Size", style="cyan", justify="center")
    table.add_column("Training Time", style="green")
    table.add_column("Gradient Updates", style="yellow", justify="center")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Throughput", style="blue")
    
    for r in results:
        table.add_row(
            str(r['batch_size']),
            f"{r['time']:.1f}s",
            str(r['updates']),
            f"{r['accuracy']:.1f}%",
            f"{r['throughput']:.0f} samples/s"
        )
    
    console.print(table)
    
    # Key insights
    console.print("\n")
    console.print(Panel.fit(
        "[bold]💡 Key Systems Insights:[/bold]\n\n"
        "[green]✓ Larger batches process data faster[/green] (fewer Python loops)\n"
        "[green]✓ Smaller batches give more gradient updates[/green] (more optimization steps)\n"
        "[green]✓ Throughput vs update frequency trade-off[/green]\n\n"
        "[bold]What This Shows:[/bold]\n"
        f"  • Batch 16:  Slowest but {results[0]['updates']} updates\n"
        f"  • Batch 64:  Balanced - {results[1]['updates']} updates\n"
        f"  • Batch 256: Fastest but only {results[2]['updates']} updates\n\n"
        "[bold]🚀 Production Tip:[/bold] In real systems, batch size is limited by:\n"
        "  • GPU memory (larger batches need more VRAM)\n"
        "  • Gradient noise (tiny batches → unstable training)\n"
        "  • Sweet spot: Usually 32-128 for most tasks\n\n"
        "[dim]YOUR DataLoader makes experimenting with this trivial -\n"
        "just change one number and the whole pipeline adapts![/dim]",
        title="⚙️ DataLoader Impact",
        border_style="cyan"
    ))


def train_mlp():
    """Train MLP on digit recognition task."""
    console.print(Panel.fit(
        "[bold cyan]Training Multi-Layer Perceptron on Real Digits[/bold cyan]\n\n"
        "[dim]Watch YOUR MLP learn to recognize handwritten digits!\n"
        "This is the same breakthrough that launched modern deep learning in 1986.[/dim]",
        title="🔥 1986 Backpropagation Revolution",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Load dataset
    train_images, train_labels, test_images, test_labels = load_digit_dataset()
    
    # Create DataLoader (Module 08!)
    console.print("[bold]Creating DataLoader...[/bold]")
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    console.print(f"✓ Batches per epoch: {len(train_loader)}\n")
    
    # Create model
    model = DigitMLP(input_size=64, hidden_size=32, num_classes=10)
    
    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Test BEFORE training
    console.print("[bold yellow]Step 1:[/bold yellow] Testing untrained network...")
    initial_acc, _ = evaluate_accuracy(model, test_images, test_labels)
    console.print(f"  Accuracy: [red]{initial_acc:.1f}%[/red] (random guessing ~10%)\n")
    
    # Training
    console.print("[bold yellow]Step 2:[/bold yellow] Training on real digits...\n")
    console.print("[bold cyan]🔥 Training Multi-Layer Perceptron[/bold cyan]")
    console.print("[dim](Using backpropagation through hidden layers!)[/dim]\n")
    
    epochs = 20
    initial_loss = None
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_images, batch_labels in train_loader:
            # Forward pass
            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.data
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        
        if initial_loss is None:
            initial_loss = avg_loss
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_acc, _ = evaluate_accuracy(model, test_images, test_labels)
            console.print(f"Epoch {epoch+1:2d}/{epochs}  "
                         f"Loss: [cyan]{avg_loss:.4f}[/cyan]  "
                         f"Test Accuracy: [green]{test_acc:.1f}%[/green]")
    
    console.print("\n[bold green]✅ Training Complete![/bold green]\n")
    
    # Final evaluation
    console.print("[bold yellow]Step 3:[/bold yellow] Final Evaluation...")
    final_acc, predictions = evaluate_accuracy(model, test_images, test_labels)
    
    # Show results table
    table = Table(title="🎯 Training Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Before Training", style="yellow")
    table.add_column("After Training", style="green")
    table.add_column("Improvement", style="magenta")
    
    table.add_row(
        "Loss",
        f"{initial_loss:.4f}",
        f"{avg_loss:.4f}",
        f"-{initial_loss - avg_loss:.4f}"
    )
    table.add_row(
        "Test Accuracy",
        f"{initial_acc:.1f}%",
        f"{final_acc:.1f}%",
        f"+{final_acc - initial_acc:.1f}%"
    )
    
    console.print(table)
    
    # Show sample predictions
    console.print("\n[bold]Sample Predictions:[/bold]")
    n_samples = 10
    for i in range(n_samples):
        true_label = test_labels.data[i]
        pred_label = predictions[i]
        status = "✓" if pred_label == true_label else "✗"
        color = "green" if pred_label == true_label else "red"
        console.print(f"  {status} True: {true_label}, Predicted: {pred_label}", style=color)
    
    # Historical context
    console.print()
    console.print(Panel.fit(
        "[bold green]🎉 Success! Your MLP Learned to Recognize Digits![/bold green]\n\n"
        f"Final accuracy: [bold]{final_acc:.1f}%[/bold]\n\n"
        "[bold]💡 What YOU Just Accomplished:[/bold]\n"
        "  • Built a multi-layer network with YOUR components\n"
        "  • Trained on REAL handwritten digits\n"
        "  • Used YOUR DataLoader for efficient batching\n"
        "  • Backprop through hidden layers works perfectly!\n\n"
        "[bold]📌 Note:[/bold] This is an MLP (fully-connected network).\n"
        "It flattens images, losing spatial structure.\n\n"
        "[dim]Next: Milestone 04 (CNN) will show how preserving\n"
        "spatial structure improves performance![/dim]",
        title="🌟 1986 Breakthrough Recreated",
        border_style="green",
        box=box.DOUBLE
    ))
    
    # Optional: Batch size experiment
    console.print("\n")
    run_experiment = input("\n🔬 Run batch size experiment? (y/n): ").lower().strip() == 'y'
    
    if run_experiment:
        compare_batch_sizes(train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    train_mlp()
