#!/usr/bin/env python3
"""
XOR Solved! Multi-Layer Networks (1986)
========================================

📚 HISTORICAL CONTEXT:
After the 1969 XOR crisis killed neural networks, research funding dried up for over 
a decade. Then in 1986, Rumelhart, Hinton, and Williams published the backpropagation 
algorithm for training multi-layer networks - and XOR became trivial!

🎯 MILESTONE 2 PART 2: THE SOLUTION (After Modules 01-07)

Watch a multi-layer network SOLVE the "impossible" XOR problem that stumped AI for 
17 years. The secret? Hidden layers + backpropagation (which YOU just built!).

✅ REQUIRED MODULES (Run after Module 07):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autodiff
  Module 02 (Activations)   : YOUR ReLU and Sigmoid (non-linearity!)
  Module 03 (Layers)        : YOUR Linear layers (multiple layers!)
  Module 04 (Losses)        : YOUR loss function  
  Module 05 (Autograd)      : YOUR backpropagation through hidden layers
  Module 06 (Optimizers)    : YOUR SGD optimizer
  Module 07 (Training)      : YOUR training loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ THE KEY INSIGHT - Hidden Layers Create New Features:

    Single Layer (FAILS):          Multi-Layer (SUCCEEDS):
    
    Input → Linear → Sigmoid        Input → Linear → ReLU → Linear → Sigmoid
               ↑                              ↑          ↑
          No hidden layer          Hidden Layer!  Non-linearity!

The hidden layer learns NEW features that make XOR linearly separable!

🔍 HOW IT WORKS - Feature Learning:

Original space (XOR not separable):
    
    1 │ 1    0          Hidden units learn:
      │                 • h₁: detects "x₁ AND NOT x₂"
    0 │ 0    1          • h₂: detects "x₂ AND NOT x₁"  
      └─────            • h₃: detects other patterns
        0    1          • h₄: etc.

New feature space (linearly separable!):
    
    The hidden layer creates a new representation where
    XOR becomes a simple linear decision boundary!

✅ EXPECTED RESULTS:
- Training time: ~30 seconds
- Accuracy: 95-100% (problem solved!)
- Loss decreases smoothly
- Perfect XOR predictions

This is the architecture that ended the AI Winter!
"""

import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch import Tensor, Linear, ReLU, Sigmoid, BinaryCrossEntropyLoss, SGD

console = Console()


# ============================================================================
# 🎲 DATA GENERATION
# ============================================================================

def generate_xor_data(n_samples=100):
    """Generate XOR dataset with slight noise."""
    console.print("\n[bold]Step 1:[/bold] Generating XOR dataset...")
    
    # Generate each XOR case with repetition
    samples_per_case = n_samples // 4
    
    # Case 1: (0,0) → 0
    x1 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([0.0, 0.0])
    y1 = np.zeros((samples_per_case, 1))
    
    # Case 2: (0,1) → 1
    x2 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([0.0, 1.0])
    y2 = np.ones((samples_per_case, 1))
    
    # Case 3: (1,0) → 1
    x3 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([1.0, 0.0])
    y3 = np.ones((samples_per_case, 1))
    
    # Case 4: (1,1) → 0
    x4 = np.random.randn(samples_per_case, 2) * 0.1 + np.array([1.0, 1.0])
    y4 = np.zeros((samples_per_case, 1))
    
    # Combine and shuffle
    X = np.vstack([x1, x2, x3, x4])
    y = np.vstack([y1, y2, y3, y4])
    
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    console.print(f"  ✓ Created [bold]{n_samples}[/bold] XOR samples")
    console.print(f"  ✓ Problem: [bold yellow]NOT linearly separable[/bold yellow]")
    console.print(f"  ✓ Solution: [bold green]Use hidden layers![/bold green]")
    
    return Tensor(X), Tensor(y)


# ============================================================================
# 🏗️ MULTI-LAYER NETWORK (The Solution!)
# ============================================================================

class XORNetwork:
    """
    Multi-layer network that SOLVES XOR!
    
    The hidden layer creates new features that make XOR linearly separable.
    This is the architecture that ended the AI Winter.
    """
    
    def __init__(self, hidden_size=4):
        # Hidden layer - THE KEY INNOVATION!
        self.hidden = Linear(2, hidden_size)
        self.relu = ReLU()  # Non-linearity is essential!
        
        # Output layer
        self.output = Linear(hidden_size, 1)
        self.sigmoid = Sigmoid()
    
    def __call__(self, x):
        """
        Forward pass through hidden layer.
        
        Input → Hidden Layer → ReLU → Output Layer → Sigmoid
        """
        # Hidden layer transforms input space
        h = self.hidden(x)
        h_activated = self.relu(h)
        
        # Output layer in new feature space
        logits = self.output(h_activated)
        output = self.sigmoid(logits)
        
        return output
    
    def parameters(self):
        """Return all trainable parameters."""
        return self.hidden.parameters() + self.output.parameters()


# ============================================================================
# 🔥 TRAINING FUNCTION (That Will SUCCEED on XOR!)
# ============================================================================

def train_network(model, X, y, epochs=500, lr=0.5):
    """
    Train multi-layer network on XOR.
    
    This WILL succeed - hidden layers solve the problem!
    """
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    console.print("\n[bold cyan]🔥 Training Multi-Layer Network...[/bold cyan]")
    console.print("[dim](This will work - hidden layers solve XOR!)[/dim]\n")
    
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        # Backward pass (through hidden layers!)
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = (pred_classes == y.data).mean()
        
        history["loss"].append(loss.data.item())
        history["accuracy"].append(accuracy)
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            console.print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {loss.data:.4f}  Accuracy: {accuracy:.1%}")
    
    console.print("\n[bold green]✅ Training Complete - XOR Solved![/bold green]\n")
    
    return history


# ============================================================================
# 📊 EVALUATION & CELEBRATION
# ============================================================================

def evaluate_and_celebrate(model, X, y, history):
    """Evaluate the successful model and celebrate the victory!"""
    
    predictions = model(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    final_accuracy = (pred_classes == y.data).mean()
    
    # Get metrics
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    initial_acc = history["accuracy"][0]
    final_acc = history["accuracy"][-1]
    
    # Show transformation
    table = Table(title="\n🎯 The Transformation", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Before Training", style="white")
    table.add_column("After Training", style="white")
    table.add_column("Improvement", style="bold green")
    
    loss_improvement = f"-{initial_loss - final_loss:.4f}"
    acc_improvement = f"+{final_acc - initial_acc:.1%}"
    
    table.add_row("Loss", f"{initial_loss:.4f}", f"{final_loss:.4f}", loss_improvement)
    table.add_row("Accuracy", f"{initial_acc:.1%}", f"{final_acc:.1%}", acc_improvement)
    
    console.print(table)
    
    # Celebrate success!
    if final_accuracy >= 0.9:
        console.print(Panel(
            "[bold green]🎉 SUCCESS! XOR Problem Solved![/bold green]\n\n"
            f"Final accuracy: {final_accuracy:.1%}\n"
            f"Final loss: {final_loss:.4f}\n\n"
            "[bold]The \"impossible\" problem is now trivial![/bold]\n"
            "Hidden layers + backpropagation = AI renaissance",
            title="✅ 1986 AI Revival",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[yellow]Accuracy: {final_accuracy:.1%}[/yellow]\n\n"
            "Try training longer or adjusting learning rate.",
            border_style="yellow"
        ))
    
    # Show XOR truth table vs predictions
    console.print("\n[bold]XOR Truth Table vs Model Predictions:[/bold]")
    test_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    test_preds = model(Tensor(test_inputs))
    
    truth_table = Table(show_header=True, border_style="green")
    truth_table.add_column("x₁", style="cyan")
    truth_table.add_column("x₂", style="cyan")
    truth_table.add_column("XOR (True)", style="green")
    truth_table.add_column("Predicted", style="yellow")
    truth_table.add_column("Correct?", style="white")
    
    all_correct = True
    for i, (x1, x2) in enumerate(test_inputs):
        true_xor = int(x1 != x2)
        pred_prob = test_preds.data[i, 0]
        pred = int(pred_prob > 0.5)
        correct = pred == true_xor
        all_correct = all_correct and correct
        
        truth_table.add_row(
            f"{int(x1)}", 
            f"{int(x2)}", 
            f"{true_xor}", 
            f"{pred} ({pred_prob:.3f})",
            "✅" if correct else "❌"
        )
    
    console.print(truth_table)
    
    if all_correct:
        console.print("\n[bold green]✨ Perfect! All XOR cases correctly predicted![/bold green]")


# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Demonstrate solving XOR with multi-layer networks."""
    
    console.print(Panel.fit(
        "[bold]XOR Solved! Multi-Layer Networks (1986)[/bold]\n\n"
        "[dim]Watch a multi-layer network SOLVE the problem that killed AI.[/dim]\n"
        "[dim]Hidden layers + backpropagation = The AI Renaissance![/dim]",
        border_style="green"
    ))
    
    # Generate data
    X, y = generate_xor_data(n_samples=100)
    
    # Create multi-layer network
    console.print("\n[bold]Step 2:[/bold] Creating multi-layer network...")
    model = XORNetwork(hidden_size=4)
    console.print("  ✓ Architecture: Input(2) → [bold green]Hidden(4)[/bold green] → ReLU → Output(1) → Sigmoid")
    console.print("  ✓ [bold green]Hidden layer is the KEY![/bold green] It learns new features.")
    console.print("  ✓ Total parameters: ~17 (vs 3 for single-layer)")
    
    # Check initial performance
    console.print("\n[bold]Initial Performance (random weights):[/bold]")
    initial_preds = model(X)
    initial_acc = ((initial_preds.data > 0.5).astype(int) == y.data).mean()
    console.print(f"  Accuracy: {initial_acc:.1%} (random guessing)")
    
    # Train the network
    console.print("\n[bold]Step 3:[/bold] Training on XOR...")
    history = train_network(model, X, y, epochs=500, lr=0.5)
    
    # Evaluate and celebrate
    evaluate_and_celebrate(model, X, y, history)
    
    # Historical context
    console.print(Panel(
        "[bold]💡 What You Just Accomplished[/bold]\n\n"
        "[bold red]1969:[/bold red] XOR crisis - single layers fail\n"
        "[bold yellow]1970-1986:[/bold yellow] AI Winter - 17 years of darkness\n"
        "[bold green]1986:[/bold green] Backprop + hidden layers solve it\n"
        "[bold cyan]TODAY:[/bold cyan] YOU solved it with YOUR TinyTorch!\n\n"
        "[bold]The Components YOU Built:[/bold]\n"
        "  • Tensor with autograd (Module 01 + 05)\n"
        "  • Linear layers for transformations (Module 03)\n"
        "  • ReLU for non-linearity (Module 02)\n"
        "  • Backprop through multiple layers (Module 05)\n"
        "  • SGD for optimization (Module 06)\n\n"
        "[dim]This same pattern scales to GPT-4, AlphaGo, and beyond![/dim]",
        title="🎓 Educational Significance",
        border_style="blue"
    ))
    
    console.print(Panel(
        "[bold cyan]🚀 Next Steps[/bold cyan]\n\n"
        "You've solved the problem that stumped AI for 17 years!\n\n"
        "[bold]Ready for more?[/bold]\n"
        "  • Milestone 03: Train deeper networks on real data\n"
        "  • Module 08: DataLoaders for batch processing\n"
        "  • Module 09: CNNs for image recognition\n"
        "  • And beyond: Transformers, attention, etc.\n\n"
        "[dim]Every modern AI architecture builds on what you just learned![/dim]",
        title="🌟 Your Journey",
        border_style="cyan"
    ))


if __name__ == "__main__":
    main()
