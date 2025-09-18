#!/usr/bin/env python3
"""
TinyTorch Demo 04: Single Neuron Learning
Shows a single neuron learning the AND gate - actual decision boundary formation!
"""

import sys
import numpy as np

def demo_single_neuron():
    """Demo single neuron learning AND gate with decision boundary"""
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        import tinytorch.core.layers as layers
        
        print("🧠 TinyTorch Single Neuron Learning Demo")
        print("=" * 50)
        print("Watch a neuron learn the AND gate!")
        print()
        
        # Demo 1: The AND gate problem
        print("⚡ Demo 1: The AND Gate Learning Problem")
        print("Teaching a neuron digital logic...")
        print()
        
        # AND gate truth table
        X = tt.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
        y = tt.Tensor([[0], [0], [0], [1]])              # AND outputs
        
        print("AND Gate Truth Table:")
        for i in range(4):
            x1, x2 = X.data[i]
            target = y.data[i, 0]
            print(f"  {int(x1)} AND {int(x2)} = {int(target)}")
        print()
        
        # Demo 2: Manual neuron implementation
        print("🔍 Demo 2: Building a Neuron from Scratch")
        print("Understanding: output = sigmoid(w1*x1 + w2*x2 + bias)")
        print()
        
        # Initialize neuron weights (starting random)
        weights = tt.Tensor([[0.2], [0.3]])  # w1, w2
        bias = tt.Tensor([[-0.1]])           # bias term
        sigmoid = act.Sigmoid()
        
        print(f"Initial weights: w1={weights.data[0,0]:.1f}, w2={weights.data[1,0]:.1f}")
        print(f"Initial bias: {bias.data[0,0]:.1f}")
        print()
        
        # Forward pass with initial weights
        print("Forward pass with random weights:")
        z = tt.Tensor(X.data @ weights.data + bias.data)  # Linear combination
        predictions = sigmoid.forward(z)
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = predictions.data[i, 0]
            target = y.data[i, 0]
            print(f"  [{int(x1)}, {int(x2)}] → {pred:.3f} (want {int(target)})")
        
        # Compute error
        error = np.mean((predictions.data - y.data) ** 2)
        print(f"Initial error (MSE): {error:.3f}")
        print("❌ Random weights don't work!")
        print()
        
        # Demo 3: Training the neuron (simplified gradient descent)
        print("🎓 Demo 3: Training the Neuron")
        print("Using simplified gradient descent...")
        print()
        
        # Simple training loop
        learning_rate = 2.0
        epochs = 5
        
        print("Training progress:")
        for epoch in range(epochs):
            # Forward pass
            z = tt.Tensor(X.data @ weights.data + bias.data)
            predictions = sigmoid.forward(z)
            
            # Compute error
            error = np.mean((predictions.data - y.data) ** 2)
            
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
            
            print(f"  Epoch {epoch+1}: Error = {error:.3f}, "
                  f"w1={weights.data[0,0]:.2f}, w2={weights.data[1,0]:.2f}, b={bias.data[0,0]:.2f}")
        
        print()
        
        # Final predictions
        print("🎯 Final Results After Training:")
        z_final = tt.Tensor(X.data @ weights.data + bias.data)
        final_predictions = sigmoid.forward(z_final)
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = final_predictions.data[i, 0]
            target = y.data[i, 0]
            decision = "✅" if (pred > 0.5) == target else "❌"
            print(f"  [{int(x1)}, {int(x2)}] → {pred:.3f} → {int(pred > 0.5)} {decision}")
        
        final_error = np.mean((final_predictions.data - y.data) ** 2)
        print(f"Final error: {final_error:.3f}")
        print()
        
        # Demo 4: Decision boundary visualization
        print("📊 Demo 4: Understanding the Decision Boundary")
        print("The line that separates 0s from 1s...")
        print()
        
        # The decision boundary is where w1*x1 + w2*x2 + b = 0
        w1, w2, b = weights.data[0,0], weights.data[1,0], bias.data[0,0]
        
        print(f"Decision equation: {w1:.2f}*x1 + {w2:.2f}*x2 + {b:.2f} = 0")
        
        # Solve for x2 when x1 = 0 and x1 = 1
        if w2 != 0:
            x2_when_x1_0 = -b / w2
            x2_when_x1_1 = -(w1 + b) / w2
            print(f"Boundary line: from (0, {x2_when_x1_0:.2f}) to (1, {x2_when_x1_1:.2f})")
        
        print()
        print("Visual decision boundary:")
        print("  1.0 |     |")
        print("      |  ❌  |  (1,1) ← AND = 1")
        print("  0.5 |-----|")  
        print("      |  ⭕  |  ⭕   ← AND = 0")
        print("  0.0 |_____|")
        print("      0.0  0.5  1.0")
        print("      ↑ (0,0), (0,1), (1,0) ← AND = 0")
        print()
        
        # Demo 5: Using TinyTorch Dense layer
        print("🚀 Demo 5: Using TinyTorch Dense Layer")
        print("Same neuron, cleaner implementation...")
        print()
        
        # Create a Dense layer (1 neuron, 2 inputs)
        dense_layer = layers.Dense(input_size=2, output_size=1, use_bias=True)
        
        # Set the learned weights by creating new tensors with correct dimensions
        # Dense layer expects weights as (input_size, output_size)
        dense_layer.weights = tt.Tensor(weights.data)  # Already (2, 1)
        dense_layer.bias = tt.Tensor(bias.data.T)  # Convert to (1,) shape
        
        print("Using Dense layer with learned weights:")
        dense_output = dense_layer.forward(X)
        dense_predictions = sigmoid.forward(dense_output)
        
        for i in range(4):
            x1, x2 = X.data[i]
            pred = dense_predictions.data[i, 0]
            target = y.data[i, 0]
            decision = "✅" if (pred > 0.5) == target else "❌"
            print(f"  [{int(x1)}, {int(x2)}] → {pred:.3f} → {int(pred > 0.5)} {decision}")
        
        print()
        
        print("🏆 TinyTorch Single Neuron Demo Complete!")
        print("🎯 Achievements:")
        print("  • Built a neuron from scratch with weights and bias")
        print("  • Trained it to learn the AND gate logic")
        print("  • Visualized the decision boundary formation")
        print("  • Showed actual gradient descent learning")
        print("  • Used TinyTorch Dense layer for clean implementation")
        print()
        print("🔥 Next: Multi-layer networks solving XOR!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import TinyTorch modules: {e}")
        print("💡 Make sure to run: tito export 04_layers")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_single_neuron()
    sys.exit(0 if success else 1)