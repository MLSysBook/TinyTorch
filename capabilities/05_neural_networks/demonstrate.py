#!/usr/bin/env python3
"""
Capability Demonstration: Neural Networks Can Learn!
This runs AFTER integration tests pass, showing the "Holy Shit" moment
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def demonstrate_xor_learning():
    """Show that neural networks can solve XOR - the classic non-linear problem."""
    
    print("\n" + "="*70)
    print("🧠 CAPABILITY UNLOCKED: Neural Networks That Learn!")
    print("="*70)
    
    print("\n📖 Historical Context:")
    print("In 1969, Minsky & Papert showed that single neurons CANNOT solve XOR.")
    print("In 1986, Rumelhart, Hinton & Williams proved that hidden layers CAN!")
    print("Today, YOU have built the framework that makes this possible.\n")
    
    # Import the student's own TinyTorch code
    try:
        import tinytorch as tt
        from tinytorch.core.layers import Dense
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor
        import numpy as np
        
        print("✅ Your TinyTorch framework loaded successfully!")
        
    except ImportError as e:
        print(f"❌ Could not import TinyTorch: {e}")
        print("Make sure you've run: tito module complete 05_dense")
        return False
    
    print("\n🔬 Building XOR Network with YOUR Framework:")
    print("-" * 50)
    
    # Build the network using student's code
    print("Creating layers...")
    hidden_layer = Dense(2, 4, use_bias=True)
    output_layer = Dense(4, 1, use_bias=True)
    relu = ReLU()
    sigmoid = Sigmoid()
    
    print("✓ Hidden Layer: 2 → 4 neurons")
    print("✓ Output Layer: 4 → 1 neuron")
    print("✓ Activations: ReLU + Sigmoid")
    
    # XOR problem setup
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    print("\n🎯 The XOR Problem:")
    print("Inputs  | Target")
    print("--------|-------")
    for i in range(4):
        print(f"{X_data[i]} | {y_data[i][0]}")
    
    # Smart initialization for faster convergence (optional boost)
    np.random.seed(42)
    hidden_layer.weights = Tensor(np.random.randn(2, 4) * 2)
    hidden_layer.bias = Tensor(np.zeros(4))
    output_layer.weights = Tensor(np.random.randn(4, 1) * 2)
    output_layer.bias = Tensor(np.array([0.]))
    
    print("\n🚀 Training Neural Network...")
    print("-" * 50)
    
    # Visual training progress
    def print_progress_bar(iteration, total, loss):
        bar_length = 40
        progress = iteration / total
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\rEpoch {iteration}/{total} [{bar}] Loss: {loss:.4f}", end="")
    
    # Simple gradient descent training
    learning_rate = 0.5
    epochs = 1000
    
    for epoch in range(epochs):
        # Forward pass
        X = Tensor(X_data)
        y = Tensor(y_data)
        
        h = hidden_layer(X)
        h_activated = relu(h)
        output = output_layer(h_activated)
        predictions = sigmoid(output)
        
        # Compute loss (MSE)
        error = predictions.data - y_data
        loss = np.mean(error ** 2)
        
        # Backpropagation (simplified)
        d_output = error * predictions.data * (1 - predictions.data)
        d_hidden = d_output @ output_layer.weights.data.T
        d_hidden = d_hidden * (h.data > 0)  # ReLU derivative
        
        # Update weights (create new tensors since .data is read-only)
        output_layer.weights = Tensor(
            output_layer.weights.data - learning_rate * (h_activated.data.T @ d_output) / 4
        )
        output_layer.bias = Tensor(
            output_layer.bias.data - learning_rate * np.mean(d_output, axis=0)
        )
        hidden_layer.weights = Tensor(
            hidden_layer.weights.data - learning_rate * (X_data.T @ d_hidden) / 4
        )
        hidden_layer.bias = Tensor(
            hidden_layer.bias.data - learning_rate * np.mean(d_hidden, axis=0)
        )
        
        # Show progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print_progress_bar(epoch + 1, epochs, loss)
        
        # Early stopping if converged
        if loss < 0.01:
            print_progress_bar(epoch + 1, epochs, loss)
            print(f"\n✨ Converged at epoch {epoch + 1}!")
            break
    
    print("\n\n🎊 RESULTS - Your Network Learned XOR!")
    print("-" * 50)
    
    # Final predictions
    X_test = Tensor(X_data)
    h = hidden_layer(X_test)
    h_activated = relu(h)
    output = output_layer(h_activated)
    final_predictions = sigmoid(output)
    
    print("Input   | Target | Prediction | Correct?")
    print("--------|--------|------------|----------")
    
    all_correct = True
    for i in range(4):
        pred = final_predictions.data[i, 0]
        target = y_data[i, 0]
        correct = "✅" if abs(pred - target) < 0.5 else "❌"
        if abs(pred - target) >= 0.5:
            all_correct = False
        print(f"{X_data[i]} | {target:.1f}    | {pred:.4f}    | {correct}")
    
    if all_correct:
        print("\n" + "🌟"*35)
        print("🏆 ACHIEVEMENT UNLOCKED: Neural Networks Work!")
        print("🌟"*35)
        print("\n💡 What You Just Proved:")
        print("• Your Dense layers work correctly")
        print("• Your activation functions add non-linearity")
        print("• Multi-layer networks can solve non-linear problems")
        print("• YOU built a working deep learning framework!")
        
        print("\n🚀 Next Milestone: Convolutional Networks for Vision")
        print("Continue with: tito module complete 06_spatial")
    else:
        print("\n⚠️ Network didn't fully converge. This is normal!")
        print("The important thing is that YOUR framework runs!")
    
    print("\n" + "="*70)
    print("Remember: You didn't just learn about neural networks...")
    print("         YOU BUILT THE FRAMEWORK THAT MAKES THEM POSSIBLE!")
    print("="*70 + "\n")
    
    return all_correct


def main():
    """Run the capability demonstration."""
    try:
        success = demonstrate_xor_learning()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())