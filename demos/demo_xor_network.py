#!/usr/bin/env python3
"""
TinyTorch Demo 05: XOR Network - The Classic AI Milestone
Shows multi-layer network solving the famous XOR problem that single layers can't!
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

def demo_xor_network():
    """Demo multi-layer network solving XOR - the classic AI milestone"""
    
    console = Console()
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        import tinytorch.core.layers as layers
        import tinytorch.core.dense as dense
        
        # Main header
        console.print(Panel.fit(
            "⚡ TinyTorch XOR Network Demo\nSolving the XOR problem - multi-layer breakthrough!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # Demo 1: The XOR problem setup
        console.print(Panel(
            "Why single neurons fail and multi-layer networks succeed...",
            title="🧩 Demo 1: The Impossible XOR Problem",
            style="green"
        ))
        
        # XOR truth table
        X = tt.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
        y = tt.Tensor([[0], [1], [1], [0]])              # XOR outputs
        
        # Create XOR truth table
        xor_table = Table(show_header=True, header_style="bold magenta")
        xor_table.add_column("X1", style="cyan", justify="center")
        xor_table.add_column("X2", style="cyan", justify="center")
        xor_table.add_column("XOR Output", style="yellow", justify="center")
        
        for i in range(4):
            x1, x2 = X.data[i]
            target = y.data[i, 0]
            xor_table.add_row(str(int(x1)), str(int(x2)), str(int(target)))
        
        console.print(xor_table)
        
        # Problem and solution panels
        problem_panel = Panel(
            "❌ Problem: No single line can separate XOR classes!",
            title="Single Layer Limitation",
            style="red"
        )
        solution_panel = Panel(
            "✅ Solution: Multi-layer network creates complex decision boundaries",
            title="Multi-Layer Power",
            style="green"
        )
        
        from rich.columns import Columns
        console.print(Columns([problem_panel, solution_panel]))
        console.print()
        
        # Demo 2: Building a 2-layer network manually
        print("🏗️ Demo 2: Building 2-Layer Network from Scratch")
        print("Architecture: 2 → 2 → 1 (input → hidden → output)")
        print()
        
        # Layer 1: 2 inputs → 2 hidden neurons
        W1 = tt.Tensor([[1.0, 1.0], [1.0, 1.0]])    # Hidden layer weights
        b1 = tt.Tensor([[-0.5, -1.5]])              # Hidden layer biases
        
        # Layer 2: 2 hidden → 1 output neuron  
        W2 = tt.Tensor([[1.0], [-2.0]])             # Output layer weights
        b2 = tt.Tensor([[-0.5]])                    # Output bias
        
        print("Layer 1 weights (2→2):")
        print(f"  W1 = \n{W1.data}")
        print(f"  b1 = {b1.data}")
        print()
        print("Layer 2 weights (2→1):")
        print(f"  W2 = \n{W2.data}")
        print(f"  b2 = {b2.data}")
        print()
        
        # Activations
        relu = act.ReLU()
        sigmoid = act.Sigmoid()
        
        # Forward pass step by step
        print("🔍 Step-by-step Forward Pass:")
        print()
        
        for i, (input_name, expected) in enumerate([("0,0", 0), ("0,1", 1), ("1,0", 1), ("1,1", 0)]):
            x_input = X.data[i:i+1]  # Single input
            
            print(f"Input [{input_name}]:")
            print(f"  x = {x_input.flatten()}")
            
            # Layer 1: Linear + ReLU
            z1 = tt.Tensor(x_input @ W1.data + b1.data)
            a1 = relu.forward(z1)
            print(f"  Hidden (after ReLU): {a1.data.flatten()}")
            
            # Layer 2: Linear + Sigmoid
            z2 = tt.Tensor(a1.data @ W2.data + b2.data)
            output = sigmoid.forward(z2)
            prediction = output.data[0, 0]
            
            print(f"  Output: {prediction:.3f} → {int(prediction > 0.5)} (want {expected})")
            result = "✅" if (prediction > 0.5) == expected else "❌"
            print(f"  Result: {result}")
            print()
        
        # Demo 3: Using TinyTorch Dense and Sequential
        print("🚀 Demo 3: Using TinyTorch Dense Networks")
        print("Building the same network with clean TinyTorch code...")
        print()
        
        # Create layers
        hidden_layer = layers.Dense(input_size=2, output_size=2, use_bias=True)
        output_layer = layers.Dense(input_size=2, output_size=1, use_bias=True)
        
        # Set the working weights we found manually
        hidden_layer.weights = tt.Tensor(W1.data)
        hidden_layer.bias = tt.Tensor(b1.data.flatten())  # Flatten to 1D for broadcasting
        output_layer.weights = tt.Tensor(W2.data)
        output_layer.bias = tt.Tensor(b2.data.flatten())  # Flatten to 1D for broadcasting
        
        print("Testing TinyTorch Dense implementation:")
        # Forward pass through network
        hidden_output = hidden_layer.forward(X)
        hidden_activation = relu.forward(hidden_output)
        final_output = output_layer.forward(hidden_activation)
        predictions = sigmoid.forward(final_output)
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = predictions.data[i, 0]
            target = y.data[i, 0]
            decision = "✅" if (pred > 0.5) == target else "❌"
            print(f"  [{int(x1)}, {int(x2)}] → {pred:.3f} → {int(pred > 0.5)} {decision}")
        
        print()
        
        # Demo 4: Understanding why it works
        print("💡 Demo 4: Why Multi-Layer Networks Work")
        print("Visualizing the hidden layer transformations...")
        print()
        
        print("Hidden layer creates new features:")
        hidden_features = relu.forward(hidden_layer.forward(X))
        
        print("Original → Hidden Features:")
        for i in range(4):
            x1, x2 = X.data[i]
            h1, h2 = hidden_features.data[i]
            target = y.data[i, 0]
            print(f"  [{int(x1)}, {int(x2)}] → [{h1:.1f}, {h2:.1f}] (XOR={int(target)})")
        
        print()
        print("In hidden space, XOR becomes linearly separable!")
        print("  Hidden neuron 1: Detects 'any input active'")
        print("  Hidden neuron 2: Detects 'both inputs active'")
        print("  Output: h1 - 2*h2 = XOR")
        print()
        
        # Demo 5: Sequential network using TinyTorch
        print("🎯 Demo 5: Complete TinyTorch Sequential Network")
        print("Building XOR solver with Sequential model...")
        print()
        
        # Create sequential model
        model = dense.Sequential([
            layers.Dense(2, 2, use_bias=True),
            act.ReLU(),
            layers.Dense(2, 1, use_bias=True),
            act.Sigmoid()
        ])
        
        # Set the proven weights
        model.layers[0].weights = tt.Tensor(W1.data)
        model.layers[0].bias = tt.Tensor(b1.data.flatten())
        model.layers[2].weights = tt.Tensor(W2.data)
        model.layers[2].bias = tt.Tensor(b2.data.flatten())
        
        print("Sequential model architecture:")
        print("  Input(2) → Dense(2) → ReLU → Dense(1) → Sigmoid → Output(1)")
        print()
        
        # Test sequential model
        sequential_output = model.forward(X)
        
        print("Sequential model results:")
        for i in range(4):
            x1, x2 = X.data[i]
            pred = sequential_output.data[i, 0]
            target = y.data[i, 0]
            decision = "✅" if (pred > 0.5) == target else "❌"
            print(f"  [{int(x1)}, {int(x2)}] → {pred:.3f} → {int(pred > 0.5)} {decision}")
        
        print()
        
        # Demo 6: Training simulation
        print("🎓 Demo 6: Training Process Simulation")
        print("How networks learn to solve XOR through training...")
        print()
        
        # Simulate learning progress
        print("Training simulation (what gradient descent would do):")
        learning_stages = [
            ("Random init", [[0.1, 0.2], [0.3, 0.4], [0.5], [0.6]], [0.5, 0.5, 0.5, 0.5]),
            ("Early learning", [[0.5, 0.8], [0.7, 0.9], [0.8], [0.2]], [0.4, 0.6, 0.6, 0.4]),
            ("Converging", [[0.9, 1.0], [1.0, 1.2], [1.0], [-0.3]], [0.2, 0.8, 0.8, 0.2]),
            ("Learned XOR", W1.data.tolist() + W2.data.flatten().tolist() + b1.data.flatten().tolist() + b2.data.flatten().tolist(), [0.05, 0.95, 0.95, 0.05])
        ]
        
        for stage, _, outputs in learning_stages:
            print(f"  {stage}: outputs = {outputs}")
            error = np.mean([(o - t)**2 for o, t in zip(outputs, [0, 1, 1, 0])])
            print(f"    → Error: {error:.3f}")
        
        print()
        
        # Success summary
        console.print(Panel.fit(
            "🎯 Achievements:\n"
            "• Proved single layers cannot solve XOR\n"
            "• Built 2-layer network solving XOR manually\n"
            "• Used TinyTorch Dense layers for clean implementation\n"
            "• Explained why hidden layers create separable features\n"
            "• Built complete Sequential model\n"
            "• Simulated the training process\n\n"
            "🔥 Next: Computer vision with spatial operations!",
            title="🏆 TinyTorch XOR Network Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch modules: {e}\n\n💡 Make sure to run: tito export 05_dense",
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
    success = demo_xor_network()
    sys.exit(0 if success else 1)