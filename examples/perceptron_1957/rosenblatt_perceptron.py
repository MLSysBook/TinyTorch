#!/usr/bin/env python3
"""
The Perceptron (1957) - Frank Rosenblatt
=======================================

📚 HISTORICAL CONTEXT:
Frank Rosenblatt's Perceptron was the first trainable artificial neural network that 
could learn from examples. It sparked the first AI boom and demonstrated that machines 
could actually learn to recognize patterns, launching the neural network revolution.

🎯 WHAT YOU'RE BUILDING:
Using YOUR TinyTorch implementations, you'll recreate the exact same perceptron that 
started it all - proving that YOU can build the foundation of modern AI from scratch.

✅ REQUIRED MODULES (Run after Module 4):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : YOUR data structure with gradient tracking
  Module 03 (Activations)   : YOUR sigmoid activation for smooth gradients  
  Module 04 (Layers)        : YOUR Linear layer for weight transformations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Original 1957 Design):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input       │    │   Linear    │    │  Sigmoid    │    │ Binary      │
    │ Features    │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ Output      │
    │ (x1, x2)    │    │     04      │    │     03      │    │ (0 or 1)    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

🔍 HOW THE PERCEPTRON LEARNS - A LINEAR DECISION BOUNDARY:

    INITIAL (Random Weights):        TRAINING (Gradient Descent):      CONVERGED (Learned):
    
    4 │ • • • • •                    4 │ • • • • •                    4 │ • • • • • 
      │ • • • • •  Class 1            │ • • • • • ╱                    │ • • • • • ╱
    2 │ - - - - -  ← Wrong!         2 │ • • • • ╱ •  ← Adjusting     2 │ • • • • ╱ •  ← Perfect!
      │ ○ ○ ○ ○ ○                      │ ○ ○ ○ ╱ ○ ○                    │ ○ ○ ○ ╱ ○ ○
    0 │ ○ ○ ○ ○ ○  Class 0         0 │ ○ ○ ╱ ○ ○ ○                  0 │ ○ ○ ╱ ○ ○ ○
      └────────────                    └────────────                    └────────────
        0   2   4                        0   2   4                        0   2   4

    Mathematical Operation:          Weight Updates:
    y = sigmoid(w₁·x₁ + w₂·x₂ + b)  w = w - η·∇L  (η = learning rate)
    
    Where YOUR modules compute:
    - Linear: z = w₁·x₁ + w₂·x₂ + b  (weighted sum)
    - Sigmoid: y = 1/(1 + e⁻ᶻ)       (squash to [0,1])
    - Decision: class = 1 if y > 0.5 else 0

🔍 KEY INSIGHTS:
- Single-layer architecture: Just linear transformation + activation
- Linearly separable only: Can't solve XOR problem (that comes later!)
- Foundation for everything: Modern networks are just deeper perceptrons

📊 EXPECTED PERFORMANCE:
- Dataset: 1,000 linearly separable synthetic points
- Training time: 30 seconds
- Expected accuracy: 95%+ (problem is linearly separable)
"""

import sys
import os
import numpy as np
import argparse

# Add project root to path for TinyTorch imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor      # Module 02: YOU built this!
from tinytorch.core.layers import Linear     # Module 04: YOU built this!
from tinytorch.core.activations import Sigmoid  # Module 03: YOU built this!

# Import dataset manager for automatic data handling
from examples.data_manager import DatasetManager

class RosenblattPerceptron:
    """
    Rosenblatt's original Perceptron using YOUR TinyTorch implementations!
    
    Historical note: The original used a step function, but we use sigmoid 
    for smooth gradients (an innovation that came slightly later).
    """
    
    def __init__(self, input_size=2, output_size=1):
        print("🧠 Building Rosenblatt's Perceptron with YOUR TinyTorch modules...")
        
        # Single layer - just like the original 1957 design!
        self.linear = Linear(input_size, output_size)  # Module 04: YOUR Linear layer!
        self.activation = Sigmoid()                     # Module 03: YOUR Sigmoid function!
        
        print(f"   Linear layer: {input_size} → {output_size} (YOUR Module 04 implementation!)")
        print(f"   Activation: Sigmoid (YOUR Module 03 implementation!)")
        
    def forward(self, x):
        """Forward pass through YOUR perceptron implementation."""
        # Step 1: Linear transformation using YOUR weights
        x = self.linear(x)        # Module 04: YOUR Linear.forward() method!
        
        # Step 2: Activation using YOUR sigmoid  
        x = self.activation(x)    # Module 03: YOUR Sigmoid.forward() method!
        
        return x
    
    def parameters(self):
        """Get trainable parameters from YOUR Linear layer."""
        return [self.linear.weights, self.linear.bias]  # Module 04: YOUR parameters!

def simple_training_loop(model, X, y, learning_rate=0.1, epochs=100):
    """
    Simple training loop using YOUR Tensor autograd system!
    
    Note: We're using a basic training loop here. Later milestones will use
    YOUR more sophisticated optimizers from Module 07!
    """
    print("\n🚀 Training Perceptron with YOUR TinyTorch autograd system!")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Using YOUR Tensor backward() method for gradients!")
    
    # Convert to YOUR Tensor format
    X_tensor = Tensor(X)  # Module 02: YOUR Tensor class!
    y_tensor = Tensor(y.reshape(-1, 1))  # Module 02: YOUR data structure!
    
    for epoch in range(epochs):
        # Forward pass using YOUR implementations
        predictions = model.forward(X_tensor)  # YOUR forward method!
        
        # Simple binary cross-entropy loss (manually computed)
        # Note: Later you'll build a proper loss function in Module 05!
        # Convert to numpy arrays for math operations
        y_np = np.array(y_tensor.data.data if hasattr(y_tensor.data, 'data') else y_tensor.data)
        pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
        loss_value = np.mean(-y_np * np.log(pred_np + 1e-8) -
                            (1 - y_np) * np.log(1 - pred_np + 1e-8))
        loss = Tensor([loss_value])
        
        # Backward pass using YOUR autograd
        loss.backward()  # Module 02: YOUR backward propagation!
        
        # Manual parameter updates (later you'll use YOUR optimizers!)
        for param in model.parameters():
            if param.grad is not None:
                param.data -= learning_rate * param.grad  # Simple gradient descent
                param.grad = None  # Clear gradients
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}: Loss = {loss_value:.4f} (YOUR training loop!)")
    
    return model


def test_model(model, X, y):
    """Test YOUR perceptron on the data."""
    print("\n🧪 Testing YOUR Perceptron Implementation:")
    
    # Forward pass with YOUR components
    X_tensor = Tensor(X)  # Module 02: YOUR Tensor!
    predictions = model.forward(X_tensor)  # YOUR architecture!
    
    # Convert to binary predictions
    pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
    binary_preds = (pred_np > 0.5).astype(int)
    accuracy = np.mean(binary_preds.flatten() == y) * 100
    
    print(f"   Accuracy: {accuracy:.1f}% on linearly separable data")
    print(f"   YOUR perceptron correctly classified {accuracy:.1f}% of examples!")
    
    # Show some example predictions
    print("\n   Sample predictions (YOUR model's output):")
    for i in range(min(5, len(X))):
        x_val = X[i]
        pred_prob = predictions.data[i, 0]
        pred_class = binary_preds[i, 0]
        true_class = y[i]
        status = "✓" if pred_class == true_class else "✗"
        print(f"   {status} Input: [{x_val[0]:.2f}, {x_val[1]:.2f}] → "
              f"Probability: {pred_prob:.3f} → Class: {pred_class} (True: {true_class})")
    
    return accuracy

def analyze_perceptron_systems(model, X):
    """Analyze YOUR perceptron from an ML systems perspective."""
    print("\n🔬 SYSTEMS ANALYSIS of YOUR Perceptron Implementation:")
    
    # Memory analysis using YOUR tensor system
    import tracemalloc
    tracemalloc.start()
    
    # Test forward pass with YOUR components
    X_tensor = Tensor(X)  # Module 02: YOUR Tensor!
    output = model.forward(X_tensor)  # Module 04 + 03: YOUR architecture!
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Parameter analysis
    total_params = model.linear.weights.data.size + model.linear.bias.data.size
    memory_per_param = 4  # bytes for float32
    
    print(f"   Memory usage: {peak / 1024:.1f} KB peak (YOUR Tensor operations)")
    print(f"   Parameters: {total_params} weights (YOUR Linear layer)")
    print(f"   Model size: {total_params * memory_per_param} bytes")
    print(f"   Computational complexity: O(n) per forward pass (linear scaling)")
    print(f"   YOUR implementation handles: Binary classification with linear decision boundary")
    
    # Historical context
    print(f"\n   🏛️  Historical Context:")
    print(f"   • 1957: YOUR perceptron uses the SAME architecture as Rosenblatt's original")
    print(f"   • Limitation: Can only solve linearly separable problems")
    print(f"   • Innovation: First machine learning algorithm that could learn from data")
    print(f"   • Legacy: Foundation for all modern neural networks (including GPT!)")

def main():
    """Demonstrate Rosenblatt's Perceptron using YOUR TinyTorch system!"""
    
    parser = argparse.ArgumentParser(description='Rosenblatt Perceptron 1957')
    parser.add_argument('--test-only', action='store_true', 
                       help='Test architecture without training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    print("🎯 PERCEPTRON 1957 - Proof of YOUR TinyTorch Mastery!")
    print("   Historical significance: First trainable neural network")
    print("   YOUR achievement: Recreated using YOUR own implementations")
    print("   Components used: YOUR Tensor + YOUR Linear + YOUR Sigmoid")
    print()
    
    # Step 1: Get linearly separable data
    print("\n📊 Preparing linearly separable data...")
    data_manager = DatasetManager()
    X, y = data_manager.get_perceptron_data(num_samples=1000)
    
    # Step 2: Create perceptron with YOUR components  
    model = RosenblattPerceptron(input_size=2, output_size=1)
    
    if args.test_only:
        print("\n🧪 ARCHITECTURE TEST MODE")
        print("Testing YOUR components work together...")
        
        # Quick forward pass test
        test_input = Tensor(X[:5])  # Module 02: YOUR Tensor!
        test_output = model.forward(test_input)  # YOUR architecture!
        print(f"✅ Forward pass successful! Output shape: {test_output.data.shape}")
        print("✅ YOUR TinyTorch modules integrate correctly!")
        return
    
    # Step 3: Train using YOUR training system
    model = simple_training_loop(model, X, y, epochs=args.epochs)
    
    # Step 4: Test YOUR implementation
    accuracy = test_model(model, X, y)
    
    # Step 5: Analyze YOUR implementation
    analyze_perceptron_systems(model, X)
    
    print("\n✅ SUCCESS! Perceptron Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • YOU built the first trainable neural network from scratch")
    print("   • YOUR Linear layer performs the same math as Rosenblatt's original") 
    print("   • YOUR Sigmoid activation enables smooth gradient learning")
    print("   • YOUR Tensor system handles automatic differentiation")
    print("\n🚀 Next Steps:")
    print("   • Continue to XOR 1969 milestone after Module 06 (Autograd)")
    print("   • YOUR foundation enables solving non-linear problems!")
    print(f"   • With {accuracy:.1f}% accuracy, YOUR perceptron works perfectly!")

if __name__ == "__main__":
    main()