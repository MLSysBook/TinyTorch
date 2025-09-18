#!/usr/bin/env python3
"""
TinyTorch Demo 03: Activation Functions - The Key to Intelligence
Shows how nonlinear functions enable neural networks to learn complex patterns
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

def demo_activations():
    """Demo activation functions with real function approximation"""
    
    console = Console()
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        
        # Main header
        console.print(Panel.fit(
            "📈 TinyTorch Activation Functions Demo\nDiscover how nonlinearity creates intelligence!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # Demo 1: Function shapes visualization
        console.print(Panel(
            "Comparing linear vs nonlinear transformations...",
            title="🎨 Demo 1: Activation Function Shapes",
            style="green"
        ))
        
        # Create test inputs
        x_data = np.linspace(-3, 3, 11)  # -3 to 3 in steps
        x = tt.Tensor(x_data.reshape(-1, 1))
        
        console.print(f"[bold cyan]Input values:[/bold cyan] {x_data}")
        console.print()
        
        # Test different activations
        relu = act.ReLU()
        sigmoid = act.Sigmoid()
        softmax = act.Softmax()
        
        # Create activation comparison table
        activation_table = Table(show_header=True, header_style="bold magenta")
        activation_table.add_column("Function", style="cyan")
        activation_table.add_column("Output", style="yellow")
        activation_table.add_column("Key Property", style="green")
        
        # ReLU transformation
        relu_output = relu.forward(x)
        relu_str = "[" + ", ".join(f"{val:.1f}" for val in relu_output.data.flatten()) + "]"
        activation_table.add_row("ReLU(x)", relu_str, "Cuts off negative values → sparse representations")
        
        # Sigmoid transformation  
        sigmoid_output = sigmoid.forward(x)
        sigmoid_str = "[" + ", ".join(f"{val:.2f}" for val in sigmoid_output.data.flatten()) + "]"
        activation_table.add_row("Sigmoid(x)", sigmoid_str, "Squashes to (0,1) → probability-like outputs")
        
        console.print(activation_table)
        console.print()
        
        # Demo 2: The XOR Problem Setup
        console.print(Panel(
            "Showing why we NEED nonlinear activations...",
            title="⚡ Demo 2: Why Linearity Fails - The XOR Problem",
            style="yellow"
        ))
        
        # XOR truth table
        xor_inputs = tt.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_outputs = tt.Tensor([[0], [1], [1], [0]])
        
        # Create XOR truth table
        xor_table = Table(show_header=True, header_style="bold magenta")
        xor_table.add_column("X1", style="cyan", justify="center")
        xor_table.add_column("X2", style="cyan", justify="center")
        xor_table.add_column("XOR Output", style="yellow", justify="center")
        
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            y = xor_outputs.data[i, 0]
            xor_table.add_row(str(int(x1)), str(int(x2)), str(int(y)))
        
        console.print(xor_table)
        console.print()
        
        # Try linear transformation (will fail)
        console.print("[bold red]🔍 Testing Linear Transformation:[/bold red]")
        linear_weights = tt.Tensor([[1.0], [1.0]])  # Simple linear combination
        linear_output = tt.Tensor(xor_inputs.data @ linear_weights.data)
        
        # Create linear test results table
        linear_table = Table(show_header=True, header_style="bold magenta")
        linear_table.add_column("Input", style="cyan")
        linear_table.add_column("Linear Output", style="yellow")
        linear_table.add_column("Expected", style="green")
        linear_table.add_column("Status", style="red")
        
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            linear_pred = linear_output.data[i, 0]
            actual = xor_outputs.data[i, 0]
            status = "✅" if abs(linear_pred - actual) < 0.5 else "❌"
            linear_table.add_row(f"[{int(x1)}, {int(x2)}]", f"{linear_pred:.1f}", str(int(actual)), status)
        
        console.print(linear_table)
        
        # Failure explanation
        failure_panel = Panel(
            "❌ Linear transformation cannot solve XOR!\n   (No single line can separate XOR classes)",
            title="Linear Limitation",
            style="red"
        )
        console.print(failure_panel)
        console.print()
        
        # Show how nonlinearity helps
        console.print("[bold green]✨ Adding Nonlinearity (ReLU):[/bold green]")
        
        # First layer: create useful features
        W1 = tt.Tensor([[1.0, 1.0], [-1.0, -1.0]])  # 2 neurons
        b1 = tt.Tensor([[-0.5], [1.5]])  # Biases
        
        # Forward pass through first layer + ReLU
        z1 = tt.Tensor(xor_inputs.data @ W1.data + b1.data.T)
        a1 = relu.forward(z1)
        
        # Create ReLU transformation table
        relu_table = Table(show_header=True, header_style="bold magenta")
        relu_table.add_column("Input", style="cyan")
        relu_table.add_column("After ReLU", style="green")
        relu_table.add_column("Linearly Separable?", style="yellow")
        
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            features = a1.data[i]
            separable = "✅" if (features[0] > 0 or features[1] > 0) else "❌"
            relu_table.add_row(f"[{int(x1)}, {int(x2)}]", f"[{features[0]:.1f}, {features[1]:.1f}]", separable)
        
        console.print(relu_table)
        
        success_panel = Panel(
            "🎯 ReLU created linearly separable features!",
            title="Nonlinearity Success",
            style="green"
        )
        console.print(success_panel)
        console.print()
        
        # Demo 3: Softmax for classification
        console.print(Panel(
            "Converting raw scores to probabilities...",
            title="🎲 Demo 3: Softmax for Multi-Class Classification",
            style="blue"
        ))
        
        # Simulate classifier outputs for 3 classes
        raw_scores = tt.Tensor([[2.0, 1.0, 0.1],    # Confident class 0
                               [0.5, 2.8, 0.2],    # Confident class 1  
                               [1.0, 1.1, 1.05]])  # Uncertain
        
        # Apply softmax
        probabilities = softmax.forward(raw_scores)
        
        # Create softmax comparison table
        softmax_table = Table(show_header=True, header_style="bold magenta")
        softmax_table.add_column("Sample", style="cyan")
        softmax_table.add_column("Raw Scores", style="yellow")
        softmax_table.add_column("Probabilities", style="green")
        softmax_table.add_column("Prediction", style="red")
        
        for i in range(3):
            scores = raw_scores.data[i]
            probs = probabilities.data[i]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
            raw_str = f"[{scores[0]:.1f}, {scores[1]:.1f}, {scores[2]:.2f}]"
            prob_str = f"[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]"
            pred_str = f"Class {predicted_class} ({confidence:.1%})"
            
            softmax_table.add_row(f"Sample {i+1}", raw_str, prob_str, pred_str)
        
        console.print(softmax_table)
        console.print()
        
        # Demo 4: Activation combinations
        console.print(Panel(
            "Combining linear transformations + activations...",
            title="🧠 Demo 4: Building Neural Network Layers",
            style="magenta"
        ))
        
        # Simulate a 2-layer network: input → hidden (ReLU) → output (Sigmoid)
        input_data = tt.Tensor([[0.5], [0.8], [-0.3]])
        
        # Layer 1: Linear + ReLU
        W1 = tt.Tensor([[0.6, -0.4], [0.2, 0.9], [-0.1, 0.3]])  # 3→2
        hidden = relu.forward(tt.Tensor(W1.data.T @ input_data.data))
        
        # Layer 2: Linear + Sigmoid
        W2 = tt.Tensor([[0.7], [0.5]])  # 2→1
        output = sigmoid.forward(tt.Tensor(W2.data.T @ hidden.data))
        
        # Create neural network flow table
        nn_table = Table(show_header=True, header_style="bold magenta")
        nn_table.add_column("Layer", style="cyan")
        nn_table.add_column("Values", style="yellow")
        nn_table.add_column("Activation", style="green")
        
        input_str = f"[{', '.join(f'{val:.1f}' for val in input_data.data.flatten())}]"
        hidden_str = f"[{', '.join(f'{val:.2f}' for val in hidden.data.flatten())}]"
        output_str = f"{output.data.flatten()[0]:.3f}"
        
        nn_table.add_row("Input", input_str, "None")
        nn_table.add_row("Hidden", hidden_str, "ReLU")
        nn_table.add_row("Output", output_str, "Sigmoid")
        
        console.print(nn_table)
        
        network_panel = Panel(
            "🎯 This is a complete neural network forward pass!",
            title="Neural Network Success",
            style="green"
        )
        console.print(network_panel)
        console.print()
        
        # Success summary
        console.print(Panel.fit(
            "🎯 Achievements:\n"
            "• Visualized how activation functions shape data\n"
            "• Proved why linearity fails on XOR problem\n"
            "• Showed how ReLU creates learnable features\n"
            "• Used Softmax for probability classification\n"
            "• Built complete neural network layers\n\n"
            "🔥 Next: Single layer networks with decision boundaries!",
            title="🏆 TinyTorch Activations Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch modules: {e}\n\n💡 Make sure to run: tito export 03_activations",
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
    success = demo_activations()
    sys.exit(0 if success else 1)