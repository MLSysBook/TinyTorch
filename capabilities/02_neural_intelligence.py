#!/usr/bin/env python3
"""
🚀 CAPABILITY SHOWCASE: Neural Intelligence
After Module 03 (Activations)

"Look what you built!" - Your activations make networks intelligent!
"""

import sys
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.align import Align

# Import from YOUR TinyTorch implementation
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh
except ImportError:
    print("❌ TinyTorch activations not found. Make sure you've completed Module 03 (Activations)!")
    sys.exit(1)

console = Console()

def visualize_activation_function(activation_class, name, x_range=(-5, 5), color="cyan"):
    """Visualize an activation function with ASCII art."""
    console.print(Panel.fit(f"📊 {name} ACTIVATION FUNCTION", style=f"bold {color}"))
    
    # Create input range
    x_vals = np.linspace(x_range[0], x_range[1], 21)
    x_tensor = Tensor([x_vals.tolist()])
    
    # Apply activation
    activation = activation_class()
    y_tensor = activation.forward(x_tensor)
    y_vals = np.array(y_tensor.data[0])
    
    # Create ASCII plot
    console.print(f"\n🎯 {name}(x) for x in [{x_range[0]}, {x_range[1]}]:")
    
    # Normalize y values for plotting
    y_min, y_max = y_vals.min(), y_vals.max()
    height = 10
    
    for i in range(height, -1, -1):
        line = f"{y_max - i*(y_max-y_min)/height:5.1f} │"
        for j, y in enumerate(y_vals):
            normalized_y = (y - y_min) / (y_max - y_min) * height
            if abs(normalized_y - i) < 0.5:
                line += "●"
            else:
                line += " "
        console.print(line)
    
    # X axis
    console.print("      └" + "─" * len(x_vals))
    console.print(f"       {x_range[0]:>2}        0        {x_range[1]:>2}")
    
    return x_vals, y_vals

def demonstrate_nonlinearity():
    """Show why nonlinearity is crucial for intelligence."""
    console.print(Panel.fit("🧠 WHY NONLINEARITY CREATES INTELLIGENCE", style="bold green"))
    
    console.print("🔍 Let's see what happens with and without activations...")
    
    # Linear transformation only
    console.print("\n📈 [bold]Without Activations (Linear Only):[/bold]")
    console.print("   Input: [1, 2, 3] → Linear → [4, 10, 16]")
    console.print("   Input: [2, 4, 6] → Linear → [8, 20, 32]")
    console.print("   📊 Output is just a scaled version of input!")
    console.print("   🚫 Cannot learn complex patterns (XOR, image recognition, etc.)")
    
    # With activations
    console.print("\n🎯 [bold]With ReLU Activation:[/bold]")
    
    # Example computation
    inputs1 = Tensor([[1, -2, 3]])
    inputs2 = Tensor([[2, -4, 6]]) 
    
    relu = ReLU()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Computing with YOUR ReLU...", total=None)
        time.sleep(1)
        
        output1 = relu.forward(inputs1)
        output2 = relu.forward(inputs2)
        
        progress.update(task, description="✅ Nonlinear magic complete!")
        time.sleep(0.5)
    
    console.print(f"   Input:  [1, -2, 3] → ReLU → {output1.data[0]}")
    console.print(f"   Input:  [2, -4, 6] → ReLU → {output2.data[0]}")
    console.print("   ✨ Non-linear transformation enables complex learning!")

def demonstrate_decision_boundaries():
    """Show how activations create decision boundaries."""
    console.print(Panel.fit("🎯 DECISION BOUNDARIES", style="bold yellow"))
    
    console.print("🔍 How your activations help networks make decisions:")
    
    # Simulate a simple decision problem
    test_points = [
        ((-1.5, "Negative input"), "red"),
        ((-0.1, "Small negative"), "red"), 
        ((0.0, "Zero"), "yellow"),
        ((0.1, "Small positive"), "green"),
        ((2.5, "Large positive"), "green")
    ]
    
    activations = [
        (ReLU(), "ReLU", "cyan"),
        (Sigmoid(), "Sigmoid", "magenta"),
        (Tanh(), "Tanh", "blue")
    ]
    
    table = Table(title="Decision Boundaries with YOUR Activations")
    table.add_column("Input", style="white")
    for _, name, color in activations:
        table.add_column(name, style=color)
    
    for (input_val, desc), point_color in test_points:
        row = [f"{input_val:6.1f} ({desc})"]
        
        for activation, _, _ in activations:
            input_tensor = Tensor([[input_val]])
            output = activation.forward(input_tensor)
            row.append(f"{output.data[0][0]:6.3f}")
        
        table.add_row(*row)
    
    console.print(table)
    
    console.print("\n💡 Key Insights:")
    console.print("   🎯 ReLU: Sharp cutoff at zero (great for sparse features)")
    console.print("   🎯 Sigmoid: Smooth probability-like output (0 to 1)")
    console.print("   🎯 Tanh: Centered output (-1 to 1, zero-centered gradients)")

def simulate_xor_problem():
    """Demonstrate the famous XOR problem that requires nonlinearity."""
    console.print(Panel.fit("🔢 THE FAMOUS XOR PROBLEM", style="bold red"))
    
    console.print("🧩 XOR cannot be solved by linear models alone!")
    console.print("    But with YOUR activations, it's possible!")
    
    # XOR truth table
    xor_table = Table(title="XOR Truth Table")
    xor_table.add_column("Input A", style="cyan")
    xor_table.add_column("Input B", style="cyan")
    xor_table.add_column("XOR Output", style="yellow")
    xor_table.add_column("Linear?", style="red")
    
    xor_data = [
        ("0", "0", "0", "✓"),
        ("0", "1", "1", "?"),
        ("1", "0", "1", "?"),
        ("1", "1", "0", "✗")
    ]
    
    for row in xor_data:
        xor_table.add_row(*row)
    
    console.print(xor_table)
    
    console.print("\n🚫 [bold red]Linear models fail:[/bold red]")
    console.print("   No single line can separate the XOR pattern!")
    
    console.print("\n✅ [bold green]With activations (coming in Module 05):[/bold green]")
    console.print("   Your ReLU enables hidden layers that can solve XOR!")
    console.print("   This is the foundation of ALL neural network intelligence!")

def show_training_preview():
    """Preview how activations will be used in training."""
    console.print(Panel.fit("🔮 COMING SOON: GRADIENT MAGIC", style="bold magenta"))
    
    console.print("🎯 In Module 09 (Autograd), your activations will:")
    console.print("   📊 Compute forward pass (what you just saw)")
    console.print("   ⬅️ Compute backward pass (gradients for learning)")
    console.print("   🔄 Enable networks to learn from mistakes")
    
    console.print("\n🧠 Each activation has different gradient properties:")
    
    gradient_table = Table(title="Gradient Characteristics (Preview)")
    gradient_table.add_column("Activation", style="cyan")
    gradient_table.add_column("Gradient Property", style="yellow")
    gradient_table.add_column("Best For", style="green")
    
    gradient_table.add_row("ReLU", "0 or 1 (sparse)", "Deep networks, CNNs")
    gradient_table.add_row("Sigmoid", "Always positive", "Binary classification")
    gradient_table.add_row("Tanh", "Centered around 0", "RNNs, hidden layers")
    
    console.print(gradient_table)

def main():
    """Main showcase function."""
    console.clear()
    
    # Header
    header = Panel.fit(
        "[bold cyan]🚀 CAPABILITY SHOWCASE: NEURAL INTELLIGENCE[/bold cyan]\n"
        "[yellow]After Module 03 (Activations)[/yellow]\n\n"
        "[green]\"Look what you built!\" - Your activations make networks intelligent![/green]",
        border_style="bright_blue"
    )
    console.print(Align.center(header))
    console.print()
    
    try:
        # Demonstrate activation functions
        visualize_activation_function(ReLU, "ReLU", color="cyan")
        console.print("\n" + "="*60)
        
        visualize_activation_function(Sigmoid, "Sigmoid", color="magenta") 
        console.print("\n" + "="*60)
        
        visualize_activation_function(Tanh, "Tanh", color="blue")
        console.print("\n" + "="*60)
        
        demonstrate_nonlinearity()
        console.print("\n" + "="*60)
        
        demonstrate_decision_boundaries()
        console.print("\n" + "="*60)
        
        simulate_xor_problem()
        console.print("\n" + "="*60)
        
        show_training_preview()
        
        # Celebration
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 ACTIVATION MASTERY ACHIEVED! 🎉[/bold green]\n\n"
            "[cyan]You've implemented the SECRET of neural network intelligence![/cyan]\n"
            "[white]Without activations: Just linear algebra (boring)[/white]\n"
            "[white]With YOUR activations: Universal function approximation! 🤯[/white]\n\n"
            "[yellow]Next up: Layers (Module 04) - Combining tensors and activations![/yellow]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error running showcase: {e}")
        console.print("💡 Make sure you've completed Module 03 and your activation functions work!")

if __name__ == "__main__":
    main()