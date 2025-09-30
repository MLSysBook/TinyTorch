#!/usr/bin/env python3
"""
The Perceptron (1957) - Frank Rosenblatt [WITH TRAINING]
=========================================================

🎯 MILESTONE 1 PART 2: TRAINED PERCEPTRON (After Modules 01-07)

Now that you've completed training modules, let's see the SAME architecture
actually LEARN! Watch random weights → intelligent predictions through training.

✅ REQUIRED MODULES (Run after Module 07):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure
  Module 02 (Activations)   : YOUR sigmoid activation  
  Module 03 (Layers)        : YOUR Linear layer
  Module 04 (Losses)        : YOUR loss function (BinaryCrossEntropyLoss)
  Module 05 (Autograd)      : YOUR automatic differentiation
  Module 06 (Optimizers)    : YOUR SGD optimizer
  Module 07 (Training)      : YOUR training loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔥 THE TRANSFORMATION:
  Before Training:  50% accuracy (random guessing) ❌
  After Training:   95%+ accuracy (intelligent!)   ✅
  
  SAME architecture. SAME data. Just add LEARNING.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import Sigmoid

# For training (Modules 04-06)
try:
    from tinytorch.core.losses import BinaryCrossEntropyLoss
    from tinytorch.core.optimizers import SGD
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Training modules not available: {e}")
    print("Please complete Modules 04-06 first!")
    TRAINING_AVAILABLE = False
    sys.exit(1)

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ============================================================================
# 🎓 STUDENT CODE: Same Perceptron class from forward_pass.py
# ============================================================================

class Perceptron:
    """
    Simple perceptron: Linear + Sigmoid
    
    SAME as forward_pass.py - but now we'll TRAIN it!
    """
    
    def __init__(self, input_size=2, output_size=1):
        self.linear = Linear(input_size, output_size)
        self.activation = Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        """Return all trainable parameters."""
        return self.linear.parameters()


# ============================================================================
# 📊 DATA GENERATION: Linearly separable 2D data
# ============================================================================

def generate_data(n_samples=100, seed=None):
    """Generate linearly separable data."""
    if seed is not None:
        np.random.seed(seed)
    
    # Class 1: Top-right cluster (high x1, high x2)
    class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([3, 3])
    labels1 = np.ones((n_samples // 2, 1))
    
    # Class 0: Bottom-left cluster (low x1, low x2)
    class0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([1, 1])
    labels0 = np.zeros((n_samples // 2, 1))
    
    # Combine
    X = np.vstack([class1, class0])
    y = np.vstack([labels1, labels0])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return Tensor(X), Tensor(y)


# ============================================================================
# 🎯 TRAINING FUNCTION
# ============================================================================

def train_perceptron(model, X, y, epochs=100, lr=0.1):
    """Train the perceptron using SGD."""
    
    # Setup training components
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    console.print("\n[bold cyan]🔥 Starting Training...[/bold cyan]\n")
    
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = (pred_classes == y.data).mean()
        
        history["loss"].append(loss.data.item())
        history["accuracy"].append(accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")
    
    console.print("\n[bold green]✅ Training Complete![/bold green]\n")
    
    return history


# ============================================================================
# 📈 EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, X, y):
    """Evaluate trained model."""
    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    accuracy = (pred_classes == y.data).mean()
    
    # Get model weights
    weights = model.linear.weight.data
    bias = model.linear.bias.data
    
    return accuracy, weights, bias, predictions


def main():
    """Main training pipeline."""
    
    # Introduction
    console.print(Panel.fit(
        "[bold cyan]The Perceptron - WITH TRAINING[/bold cyan]\n\n"
        "[dim]Watch the SAME architecture learn through gradient descent![/dim]\n"
        "[dim]Random weights → Intelligent predictions in ~30 seconds[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Generate data
    console.print("\n[bold]Step 1:[/bold] Generating linearly separable data...")
    X, y = generate_data(n_samples=100, seed=42)  # Fixed seed for reproducibility
    console.print(f"  ✓ Created {len(X.data)} samples (50 per class)")
    
    # Create model
    console.print("\n[bold]Step 2:[/bold] Creating perceptron with random weights...")
    model = Perceptron(input_size=2, output_size=1)
    
    # Evaluate BEFORE training
    acc_before, w_before, b_before, _ = evaluate_model(model, X, y)
    console.print(f"  ❌ Accuracy BEFORE training: {acc_before:.1%} (random guessing)")
    
    # Train the model
    console.print("\n[bold]Step 3:[/bold] Training with gradient descent...")
    history = train_perceptron(model, X, y, epochs=100, lr=0.1)
    
    # Evaluate AFTER training
    acc_after, w_after, b_after, predictions = evaluate_model(model, X, y)
    
    # Show transformation
    console.print("\n")
    table = Table(title="🎯 The Transformation", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Before Training", style="red")
    table.add_column("After Training", style="green", justify="right")
    table.add_column("Improvement", justify="center")
    
    table.add_row(
        "Accuracy",
        f"{acc_before:.1%}",
        f"{acc_after:.1%}",
        f"+{(acc_after - acc_before):.1%}" if acc_after > acc_before else "—"
    )
    
    table.add_row(
        "Weight w₁",
        f"{w_before.flatten()[0]:.4f}",
        f"{w_after.flatten()[0]:.4f}",
        "→ Learned" if abs(w_after.flatten()[0] - w_before.flatten()[0]) > 0.1 else "—"
    )
    
    table.add_row(
        "Weight w₂",
        f"{w_before.flatten()[1]:.4f}",
        f"{w_after.flatten()[1]:.4f}",
        "→ Learned" if abs(w_after.flatten()[1] - w_before.flatten()[1]) > 0.1 else "—"
    )
    
    table.add_row(
        "Bias b",
        f"{b_before.flatten()[0]:.4f}",
        f"{b_after.flatten()[0]:.4f}",
        "→ Learned" if abs(b_after.flatten()[0] - b_before.flatten()[0]) > 0.1 else "—"
    )
    
    console.print(table)
    
    # Final message
    if acc_after >= 0.9:
        console.print(Panel.fit(
            "[bold green]🎉 SUCCESS![/bold green]\n\n"
            f"Your perceptron achieved {acc_after:.1%} accuracy!\n"
            "The model LEARNED to classify the data correctly.\n\n"
            "[dim]Same architecture as forward_pass.py - just add training![/dim]",
            border_style="green",
            box=box.DOUBLE
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Training in Progress[/bold yellow]\n\n"
            f"Current accuracy: {acc_after:.1%}\n"
            "Try more epochs or different learning rate.\n\n"
            "[dim]The model is learning but needs more training.[/dim]",
            border_style="yellow"
        ))
    
    # Key insight
    console.print("\n")
    console.print(Panel.fit(
        "[bold]💡 KEY INSIGHT: The Power of Learning[/bold]\n\n"
        "The architecture is trivial (~10 lines of code).\n"
        "The magic is in the TRAINING LOOP:\n\n"
        "  1️⃣  Forward pass: Make predictions\n"
        "  2️⃣  Compute loss: Measure error\n"
        "  3️⃣  Backward pass: Calculate gradients (Module 05)\n"
        "  4️⃣  Update weights: Move toward better solution (Module 06)\n"
        "  5️⃣  Repeat: Until convergence (Module 07)\n\n"
        "[dim]This same pattern scales from perceptrons → GPT-4![/dim]",
        border_style="blue"
    ))


if __name__ == "__main__":
    if not TRAINING_AVAILABLE:
        console.print("[red]Cannot run: Training modules not available[/red]")
        sys.exit(1)
    
    main()
