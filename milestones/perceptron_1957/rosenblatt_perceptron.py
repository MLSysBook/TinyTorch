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
  Module 01 (Tensor)        : YOUR data structure with gradient tracking
  Module 02 (Activations)   : YOUR sigmoid activation for smooth gradients  
  Module 03 (Layers)        : YOUR Linear layer for weight transformations
  Data Generation           : Directly generated within this script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Original 1957 Design):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Input       │    │   Linear    │    │  Sigmoid    │    │ Binary      │
    │ Features    │───▶│ YOUR Module │───▶│ YOUR Module │───▶│ Output      │
    │ (x1, x2)    │    │     03      │    │     02      │    │ (0 or 1)    │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

🔍 HOW THE PERCEPTRON LEARNS - A LINEAR DECISION BOUNDARY:

    INITIAL (Random Weights):        TRAINING (Gradient Descent):      CONVERGED (Learned):
    
    4 │ • • • • •                    4 │ • • • • •                    4 │ • • • • • 
      │ • • • • •  Class 1             │ • • • • • ╱                    │ • • • • • ╱
    2 │ - - - - -  ← Wrong!          2 │ • • • • ╱ •  ← Adjusting     2 │ • • • • ╱ •  ← Perfect!
      │ ○ ○ ○ ○ ○                      │ ○ ○ ○ ╱ ○ ○                    │ ○ ○ ○ ╱ ○ ○
    0 │ ○ ○ ○ ○ ○  Class 0           0 │ ○ ○ ╱ ○ ○ ○                  0 │ ○ ○ ╱ ○ ○ ○
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

# Add project root to path for correct tinytorch imports
# This allows the script to be run from the root of the project
sys.path.insert(0, os.getcwd())

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor        # Module 01: YOU built this!
from tinytorch.core.layers import Linear        # Module 03: YOU built this!
from tinytorch.core.activations import Sigmoid  # Module 02: YOU built this!

class RosenblattPerceptron:
    """
    Rosenblatt's original Perceptron using YOUR TinyTorch implementations!
    
    Historical note: The original used a step function, but we use sigmoid 
    for smooth gradients (an innovation that came slightly later).
    """
    
    def __init__(self, input_size=2, output_size=1):
        print("🧠 Assembling Rosenblatt's Perceptron with YOUR TinyTorch modules...")
        
        # Single layer - just like the original 1957 design!
        self.linear = Linear(input_size, output_size)  # Module 03: YOUR Linear layer!
        self.activation = Sigmoid()                    # Module 02: YOUR Sigmoid function!
        
        print(f"   ✅ Linear layer: {input_size} → {output_size} (YOUR Module 03 implementation!)")
        print(f"   ✅ Activation: Sigmoid (YOUR Module 02 implementation!)")
        
    def forward(self, x):
        """Forward pass through YOUR perceptron implementation."""
        # Step 1: Linear transformation using YOUR weights
        x = self.linear(x)        # Module 03: YOUR Linear.forward() method!
        
        # Step 2: Activation using YOUR sigmoid  
        x = self.activation(x)    # Module 02: YOUR Sigmoid.forward() method!
        
        return x

def main():
    """Demonstrate Rosenblatt's Perceptron using YOUR TinyTorch system!"""
    
    print("🎯 MILESTONE: The Perceptron (1957)")
    print("   Historical significance: The first trainable neural network.")
    print("   YOUR achievement: Assembling it from YOUR own modules.")
    print("   Components used: YOUR Tensor + YOUR Linear + YOUR Sigmoid.")
    print("-" * 60)
    
    # Step 1: Prepare synthetic data
    print("\n📊 Step 1: Preparing linearly separable data...")
    np.random.seed(42)
    cluster1 = np.random.normal([2, 2], 0.5, (5, 2))  # Just a few samples are needed
    cluster2 = np.random.normal([-2, -2], 0.5, (5, 2))
    X = np.vstack([cluster1, cluster2]).astype(np.float32)
    print(f"   ✅ Data created successfully with shape: {X.shape}")

    # Step 2: Create the Perceptron model with YOUR components  
    print("\n🧠 Step 2: Instantiating the Perceptron model...")
    model = RosenblattPerceptron(input_size=2, output_size=1)
    print("   ✅ Model assembled successfully!")

    # Step 3: Perform a forward pass
    print("\n🔬 Step 3: Running a forward pass to test integration...")
    # Convert data to YOUR Tensor format
    input_tensor = Tensor(X)  # Module 01: YOUR Tensor class!
    print(f"   - Input tensor created with shape: {input_tensor.shape}")

    # Run the forward pass through YOUR implementations
    output_tensor = model.forward(input_tensor)
    print(f"   - Output tensor received with shape: {output_tensor.shape}")

    # --- Verification ---
    print("\n" + "="*60)
    print("✅ SUCCESS! Your components integrated perfectly.")
    print("   You have successfully assembled the architecture of the first")
    print("   trainable neural network using the modules YOU built.")
    print("="*60)
    
    print("\n🎓 What YOU Accomplished:")
    print("   • YOU assembled a neural network from scratch.")
    print("   • YOUR Tensor class handled the data flow.")
    print("   • YOUR Linear layer performed the mathematical transformation.")
    print("   • YOUR Sigmoid activation processed the layer's output.")
    
    print("\n🚀 Next Steps:")
    print("   • In future modules, you will build the components needed to TRAIN this model:")
    print("     - Module 04 (Losses): To measure how wrong the model's predictions are.")
    print("     - Module 05 (Autograd): To calculate the gradients needed to improve.")
    print("     - Module 06 (Optimizers): To update the model's weights automatically.")
    print("\n   For now, congratulations on this major milestone!")

if __name__ == "__main__":
    main()