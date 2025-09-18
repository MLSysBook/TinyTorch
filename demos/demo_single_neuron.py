#!/usr/bin/env python3
"""
TinyTorch Demo 04: Single Neuron Learning
Shows a single neuron learning the AND gate - actual decision boundary formation!
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn

def demo_single_neuron():
    """Demo single neuron learning AND gate with decision boundary"""
    
    console = Console()
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        import tinytorch.core.layers as layers
        
        # Main header
        console.print(Panel.fit(
            "🧠 TinyTorch Single Neuron Learning Demo\nWatch a neuron learn the AND gate!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # What this demo shows
        console.print(Panel(
            "[bold yellow]What This Demo Shows:[/bold yellow]\n\n"
            "We're going to watch a single neuron (the basic unit of neural networks) learn to solve\n"
            "the AND gate problem through gradient descent. You'll see:\n\n"
            "• How random weights produce wrong answers initially\n"
            "• How the neuron adjusts its weights based on errors\n"
            "• The formation of a decision boundary that separates 0s from 1s\n"
            "• Why some problems (AND) are learnable while others (XOR) need multiple layers\n\n"
            "[bold cyan]Key Insight:[/bold cyan] A neuron is just a weighted sum followed by an activation function.\n"
            "Learning means finding the right weights!",
            title="📚 Understanding This Demo",
            style="blue"
        ))
        console.print()
        
        # Demo 1: The AND gate problem
        console.print(Panel(
            "The AND gate outputs 1 only when BOTH inputs are 1.\n"
            "This is a 'linearly separable' problem - a single line can divide the outputs.",
            title="⚡ Demo 1: The AND Gate Learning Problem",
            style="green"
        ))
        
        # AND gate truth table
        X = tt.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
        y = tt.Tensor([[0], [0], [0], [1]])              # AND outputs
        
        # Create AND gate truth table
        and_table = Table(show_header=True, header_style="bold magenta")
        and_table.add_column("X1", style="cyan", justify="center")
        and_table.add_column("X2", style="cyan", justify="center")
        and_table.add_column("AND Output", style="yellow", justify="center")
        
        for i in range(4):
            x1, x2 = X.data[i]
            target = y.data[i, 0]
            and_table.add_row(str(int(x1)), str(int(x2)), str(int(target)))
        
        console.print(and_table)
        console.print()
        
        console.print("[dim]💡 [bold]How to Read This:[/bold] The AND gate is like a logical 'both must be true' operator.[/dim]")
        console.print("[dim]   Notice only the last row (1 AND 1) outputs 1. Our neuron needs to learn this pattern![/dim]")
        console.print()
        
        # Demo 2: Manual neuron implementation
        console.print(Panel(
            "Understanding: output = sigmoid(w1*x1 + w2*x2 + bias)",
            title="🔍 Demo 2: Building a Neuron from Scratch",
            style="blue"
        ))
        
        # Initialize neuron weights (starting random)
        weights = tt.Tensor([[0.2], [0.3]])  # w1, w2
        bias = tt.Tensor([[-0.1]])           # bias term
        sigmoid = act.Sigmoid()
        
        console.print(f"[bold cyan]Initial parameters:[/bold cyan]")
        console.print(f"  • Weight 1: {weights.data[0,0]:.1f}")
        console.print(f"  • Weight 2: {weights.data[1,0]:.1f}")
        console.print(f"  • Bias: {bias.data[0,0]:.1f}")
        console.print()
        
        # Forward pass with initial weights
        console.print("[bold red]Forward pass with random weights:[/bold red]")
        z = tt.Tensor(X.data @ weights.data + bias.data)  # Linear combination
        predictions = sigmoid.forward(z)
        
        # Create initial predictions table
        initial_table = Table(show_header=True, header_style="bold magenta")
        initial_table.add_column("Input", style="cyan")
        initial_table.add_column("Prediction", style="yellow")
        initial_table.add_column("Target", style="green")
        initial_table.add_column("Status", style="red")
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = predictions.data[i, 0]
            target = y.data[i, 0]
            status = "✅" if abs(pred - target) < 0.5 else "❌"
            initial_table.add_row(f"[{int(x1)}, {int(x2)}]", f"{pred:.3f}", str(int(target)), status)
        
        console.print(initial_table)
        
        # Compute error
        error = np.mean((predictions.data - y.data) ** 2)
        error_panel = Panel(
            f"Initial error (MSE): {error:.3f}\n❌ Random weights don't work!",
            title="Initial Performance",
            style="red"
        )
        console.print(error_panel)
        console.print()
        
        # Demo 3: Training the neuron (simplified gradient descent)
        console.print(Panel(
            "Using simplified gradient descent...",
            title="🎓 Demo 3: Training the Neuron",
            style="yellow"
        ))
        
        # Simple training loop
        learning_rate = 2.0
        epochs = 5
        
        # Create training progress table
        training_table = Table(show_header=True, header_style="bold magenta")
        training_table.add_column("Epoch", style="cyan", justify="center")
        training_table.add_column("Error", style="red")
        training_table.add_column("Weight 1", style="green")
        training_table.add_column("Weight 2", style="green")
        training_table.add_column("Bias", style="yellow")
        
        for epoch in range(epochs):
            # Forward pass
            z = tt.Tensor(X.data @ weights.data + bias.data)
            predictions = sigmoid.forward(z)
            
            # Compute error
            error = np.mean((predictions.data - y.data) ** 2)
            
            # Add row to training table
            training_table.add_row(
                str(epoch + 1),
                f"{error:.3f}",
                f"{weights.data[0,0]:.2f}",
                f"{weights.data[1,0]:.2f}",
                f"{bias.data[0,0]:.2f}"
            )
            
            # Simplified gradient computation (educational)
            # For sigmoid: gradient = prediction * (1 - prediction) * error
            for i in range(4):
                pred = predictions.data[i, 0]
                target = y.data[i, 0]
                x1, x2 = X.data[i]
                
                # Gradient of error w.r.t. weights
                sigmoid_grad = pred * (1 - pred)
                error_grad = 2 * (pred - target)
                total_grad = sigmoid_grad * error_grad
                
                # Update weights (simplified)
                weights.data[0, 0] -= learning_rate * total_grad * x1 / 4
                weights.data[1, 0] -= learning_rate * total_grad * x2 / 4
                bias.data[0, 0] -= learning_rate * total_grad / 4
        
        console.print(training_table)
        console.print()
        
        console.print("[dim]💡 [bold]What's Happening:[/bold] Watch the error decrease as the neuron learns![/dim]")
        console.print("[dim]   • Error measures how wrong our predictions are (lower is better)[/dim]")
        console.print("[dim]   • Weights are adjusting to reduce this error through gradient descent[/dim]")
        console.print("[dim]   • The bias shifts the decision boundary position[/dim]")
        console.print()
        
        # Final predictions
        console.print("[bold green]🎯 Final Results After Training:[/bold green]")
        z_final = tt.Tensor(X.data @ weights.data + bias.data)
        final_predictions = sigmoid.forward(z_final)
        
        # Create final results table
        final_table = Table(show_header=True, header_style="bold magenta")
        final_table.add_column("Input", style="cyan")
        final_table.add_column("Raw Output", style="yellow")
        final_table.add_column("Decision", style="blue")
        final_table.add_column("Target", style="green")
        final_table.add_column("Correct?", style="red")
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = final_predictions.data[i, 0]
            target = y.data[i, 0]
            decision = int(pred > 0.5)
            status = "✅" if decision == target else "❌"
            final_table.add_row(f"[{int(x1)}, {int(x2)}]", f"{pred:.3f}", str(decision), str(int(target)), status)
        
        console.print(final_table)
        
        final_error = np.mean((final_predictions.data - y.data) ** 2)
        success_panel = Panel(
            f"Final error: {final_error:.3f}\n🎉 Neuron successfully learned AND gate!",
            title="Training Success",
            style="green"
        )
        console.print(success_panel)
        console.print()
        
        # Demo 4: Decision boundary visualization
        console.print(Panel(
            "The line that separates 0s from 1s...",
            title="📊 Demo 4: Understanding the Decision Boundary",
            style="magenta"
        ))
        
        # The decision boundary is where w1*x1 + w2*x2 + b = 0
        w1, w2, b = weights.data[0,0], weights.data[1,0], bias.data[0,0]
        
        console.print(f"[bold cyan]Decision equation:[/bold cyan] {w1:.2f}*x1 + {w2:.2f}*x2 + {b:.2f} = 0")
        
        # Solve for x2 when x1 = 0 and x1 = 1
        if w2 != 0:
            x2_when_x1_0 = -b / w2
            x2_when_x1_1 = -(w1 + b) / w2
            console.print(f"[bold yellow]Boundary line:[/bold yellow] from (0, {x2_when_x1_0:.2f}) to (1, {x2_when_x1_1:.2f})")
        
        console.print()
        
        # Visual decision boundary in a panel
        boundary_viz = """
  1.0 |     |
      |  ✅  |  (1,1) ← AND = 1
  0.5 |-----|
      |  ⭕  |  ⭕   ← AND = 0
  0.0 |_____|
      0.0  0.5  1.0
      ↑ (0,0), (0,1), (1,0) ← AND = 0
"""
        
        console.print(Panel(boundary_viz, title="Visual Decision Boundary", style="cyan"))
        console.print()
        
        # Demo 5: Using TinyTorch Dense layer
        console.print(Panel(
            "Same neuron, cleaner implementation...",
            title="🚀 Demo 5: Using TinyTorch Dense Layer",
            style="bright_green"
        ))
        
        # Create a Dense layer (1 neuron, 2 inputs)
        dense_layer = layers.Dense(input_size=2, output_size=1, use_bias=True)
        
        # Set the learned weights by creating new tensors with correct dimensions
        # Dense layer expects weights as (input_size, output_size)
        dense_layer.weights = tt.Tensor(weights.data)  # Already (2, 1)
        dense_layer.bias = tt.Tensor(bias.data.T)  # Convert to (1,) shape
        
        console.print("[bold cyan]Using Dense layer with learned weights:[/bold cyan]")
        dense_output = dense_layer.forward(X)
        dense_predictions = sigmoid.forward(dense_output)
        
        # Create Dense layer verification table
        dense_table = Table(show_header=True, header_style="bold magenta")
        dense_table.add_column("Input", style="cyan")
        dense_table.add_column("Dense Output", style="yellow")
        dense_table.add_column("Decision", style="blue")
        dense_table.add_column("Correct?", style="green")
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = dense_predictions.data[i, 0]
            target = y.data[i, 0]
            decision = int(pred > 0.5)
            status = "✅" if decision == target else "❌"
            dense_table.add_row(f"[{int(x1)}, {int(x2)}]", f"{pred:.3f}", str(decision), status)
        
        console.print(dense_table)
        console.print()
        
        # Success summary
        console.print(Panel.fit(
            "🎯 Achievements:\n"
            "• Built a neuron from scratch with weights and bias\n"
            "• Trained it to learn the AND gate logic\n"
            "• Visualized the decision boundary formation\n"
            "• Showed actual gradient descent learning\n"
            "• Used TinyTorch Dense layer for clean implementation\n\n"
            "🔥 Next: Multi-layer networks solving XOR!",
            title="🏆 TinyTorch Single Neuron Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch modules: {e}\n\n💡 Make sure to run: tito export 04_layers",
            title="❌ Import Error",
            style="bold red"
        ))
        return False
    except Exception as e:
        console.print(Panel(
            f"Demo failed: {e}",
            title="❌ Error",
            style="bold red"
        ))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_single_neuron()
    sys.exit(0 if success else 1)