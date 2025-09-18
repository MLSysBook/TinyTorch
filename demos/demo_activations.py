#!/usr/bin/env python3
"""
TinyTorch Demo 03: Activation Functions - The Key to Intelligence
Shows how nonlinear functions enable neural networks to learn complex patterns
"""

import sys
import numpy as np

def demo_activations():
    """Demo activation functions with real function approximation"""
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        
        print("📈 TinyTorch Activation Functions Demo")
        print("=" * 50)
        print("Discover how nonlinearity creates intelligence!")
        print()
        
        # Demo 1: Function shapes visualization
        print("🎨 Demo 1: Activation Function Shapes")
        print("Comparing linear vs nonlinear transformations...")
        print()
        
        # Create test inputs
        x_data = np.linspace(-3, 3, 11)  # -3 to 3 in steps
        x = tt.Tensor(x_data.reshape(-1, 1))
        
        print(f"Input values: {x_data}")
        print()
        
        # Test different activations
        relu = act.ReLU()
        sigmoid = act.Sigmoid()
        softmax = act.Softmax()
        
        # ReLU transformation
        relu_output = relu.forward(x)
        print(f"ReLU(x):     {relu_output.data.flatten()}")
        print("           ↳ Cuts off negative values → sparse representations")
        
        # Sigmoid transformation  
        sigmoid_output = sigmoid.forward(x)
        print(f"Sigmoid(x):  {sigmoid_output.data.flatten()}")
        print("           ↳ Squashes to (0,1) → probability-like outputs")
        print()
        
        # Demo 2: The XOR Problem Setup
        print("⚡ Demo 2: Why Linearity Fails - The XOR Problem")
        print("Showing why we NEED nonlinear activations...")
        print()
        
        # XOR truth table
        xor_inputs = tt.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_outputs = tt.Tensor([[0], [1], [1], [0]])
        
        print("XOR Truth Table:")
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            y = xor_outputs.data[i, 0]
            print(f"  {int(x1)} XOR {int(x2)} = {int(y)}")
        print()
        
        # Try linear transformation (will fail)
        print("🔍 Testing Linear Transformation:")
        linear_weights = tt.Tensor([[1.0], [1.0]])  # Simple linear combination
        linear_output = tt.Tensor(xor_inputs.data @ linear_weights.data)
        
        print("Linear: W @ [x1, x2]")
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            linear_pred = linear_output.data[i, 0]
            actual = xor_outputs.data[i, 0]
            print(f"  [{int(x1)}, {int(x2)}] → {linear_pred:.1f} (want {int(actual)})")
        
        print("❌ Linear transformation cannot solve XOR!")
        print("   (No single line can separate XOR classes)")
        print()
        
        # Show how nonlinearity helps
        print("✨ Adding Nonlinearity (ReLU):")
        
        # First layer: create useful features
        W1 = tt.Tensor([[1.0, 1.0], [-1.0, -1.0]])  # 2 neurons
        b1 = tt.Tensor([[-0.5], [1.5]])  # Biases
        
        # Forward pass through first layer + ReLU
        z1 = tt.Tensor(xor_inputs.data @ W1.data + b1.data.T)
        a1 = relu.forward(z1)
        
        print("After ReLU layer:")
        for i in range(4):
            x1, x2 = xor_inputs.data[i]
            features = a1.data[i]
            print(f"  [{int(x1)}, {int(x2)}] → [{features[0]:.1f}, {features[1]:.1f}]")
        
        print("🎯 ReLU created linearly separable features!")
        print()
        
        # Demo 3: Softmax for classification
        print("🎲 Demo 3: Softmax for Multi-Class Classification")
        print("Converting raw scores to probabilities...")
        print()
        
        # Simulate classifier outputs for 3 classes
        raw_scores = tt.Tensor([[2.0, 1.0, 0.1],    # Confident class 0
                               [0.5, 2.8, 0.2],    # Confident class 1  
                               [1.0, 1.1, 1.05]])  # Uncertain
        
        print("Raw classifier scores:")
        for i in range(3):
            scores = raw_scores.data[i]
            print(f"  Sample {i+1}: [{scores[0]:.1f}, {scores[1]:.1f}, {scores[2]:.2f}]")
        print()
        
        # Apply softmax
        probabilities = softmax.forward(raw_scores)
        
        print("After Softmax (probabilities):")
        for i in range(3):
            probs = probabilities.data[i]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            print(f"  Sample {i+1}: [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")
            print(f"           → Predicts class {predicted_class} ({confidence:.1%} confidence)")
        print()
        
        # Demo 4: Activation combinations
        print("🧠 Demo 4: Building Neural Network Layers")
        print("Combining linear transformations + activations...")
        print()
        
        # Simulate a 2-layer network: input → hidden (ReLU) → output (Sigmoid)
        input_data = tt.Tensor([[0.5], [0.8], [-0.3]])
        
        # Layer 1: Linear + ReLU
        W1 = tt.Tensor([[0.6, -0.4], [0.2, 0.9], [-0.1, 0.3]])  # 3→2
        hidden = relu.forward(tt.Tensor(W1.data.T @ input_data.data))
        
        # Layer 2: Linear + Sigmoid
        W2 = tt.Tensor([[0.7], [0.5]])  # 2→1
        output = sigmoid.forward(tt.Tensor(W2.data.T @ hidden.data))
        
        print(f"Input:   {input_data.data.flatten()}")
        print(f"Hidden:  {hidden.data.flatten()} (after ReLU)")
        print(f"Output:  {output.data.flatten()[0]:.3f} (after Sigmoid)")
        print()
        print("🎯 This is a complete neural network forward pass!")
        print()
        
        print("🏆 TinyTorch Activations Demo Complete!")
        print("🎯 Achievements:")
        print("  • Visualized how activation functions shape data")
        print("  • Proved why linearity fails on XOR problem")
        print("  • Showed how ReLU creates learnable features")
        print("  • Used Softmax for probability classification")
        print("  • Built complete neural network layers")
        print()
        print("🔥 Next: Single layer networks with decision boundaries!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import TinyTorch modules: {e}")
        print("💡 Make sure to run: tito export 03_activations")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_activations()
    sys.exit(0 if success else 1)