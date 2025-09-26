#!/usr/bin/env python3
"""
[MILESTONE NAME] ([YEAR]) - [HISTORICAL FIGURE]
===============================================

📚 HISTORICAL CONTEXT:
[2-3 sentences about the historical significance and why this was a breakthrough]

🎯 WHAT YOU'RE BUILDING:
[1-2 sentences about what students will demonstrate with their own implementations]

✅ REQUIRED MODULES (Run after Module [X]):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : [Brief description of how it's used]
  Module 03 (Activations)   : [Brief description of how it's used]
  Module 04 (Layers)        : [Brief description of how it's used]
  Module XX (YYY)           : [Additional modules as needed]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE:
    [ASCII diagram showing the network architecture]

🔍 KEY INSIGHTS:
- [Bullet point about what this demonstrates]
- [Bullet point about why this architecture works]  
- [Bullet point about production relevance]

📊 EXPECTED PERFORMANCE:
- [Dataset info]: [Performance metric]
- [Training time]: [Approximate time]
- [Memory usage]: [Approximate memory]
"""

import sys
import os
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor      # Module 02: YOU built this data structure!
from tinytorch.core.layers import Linear     # Module 04: YOU built these transformations!
from tinytorch.core.activations import ReLU  # Module 03: YOU built this nonlinearity!
# [Add other imports as needed with YOU BUILT comments]

def download_dataset():
    """
    Download and prepare dataset for this milestone.
    
    This function handles all dataset logistics so you can focus on 
    demonstrating the ML system you built!
    """
    print("📥 Downloading dataset...")
    # [Dataset download logic]
    print("✅ Dataset ready!")
    return data_loader

def create_model():
    """Build the model using YOUR TinyTorch implementations!"""
    
    class MilestoneModel:
        def __init__(self):
            # YOU built these components in the modules!
            self.layer1 = Linear(input_size, hidden_size)   # Module 04: YOUR Linear layer!
            self.activation = ReLU()                        # Module 03: YOUR ReLU function!
            self.layer2 = Linear(hidden_size, output_size)  # Module 04: YOUR weight matrices!
        
        def forward(self, x):
            # Forward pass using YOUR implementations
            x = self.layer1(x)        # Module 04: YOUR Linear.forward()!
            x = self.activation(x)    # Module 03: YOUR ReLU activation!
            x = self.layer2(x)        # Module 04: YOUR final transformation!
            return x
    
    return MilestoneModel()

def train_model(model, data_loader):
    """Train using YOUR optimization and loss implementations!"""
    
    # Set up training using YOUR TinyTorch modules
    optimizer = YourOptimizer(model.parameters())  # Module XX: YOU built this optimizer!
    loss_fn = YourLossFunction()                    # Module XX: YOU built this loss!
    
    print("🚀 Training with YOUR TinyTorch implementation!")
    print("   [Brief description of what's happening]")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_data, batch_labels in data_loader:  # Module XX: YOUR DataLoader!
            # Forward pass with YOUR components
            outputs = model.forward(batch_data)         # YOUR model architecture!
            loss = loss_fn(outputs, batch_labels)       # YOUR loss computation!
            
            # Backward pass with YOUR autograd
            loss.backward()                             # Module XX: YOUR autodiff!
            optimizer.step()                            # Module XX: YOUR optimization!
            optimizer.zero_grad()                       # Module XX: YOUR gradient reset!
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f} (YOUR training loop!)")

def analyze_performance(model):
    """Analyze the system YOU built from an ML systems perspective."""
    
    print("\n🔬 SYSTEMS ANALYSIS of YOUR Implementation:")
    
    # Memory analysis using YOUR tensor system
    import tracemalloc
    tracemalloc.start()
    
    # Test forward pass
    test_input = Tensor(np.random.randn(batch_size, input_size))  # YOUR Tensor!
    output = model.forward(test_input)                            # YOUR architecture!
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"   Memory usage: {peak / 1024 / 1024:.2f} MB peak")
    
    # Parameter analysis
    total_params = sum(layer.weight.size for layer in [model.layer1, model.layer2])
    print(f"   Parameters: {total_params:,} weights (YOUR Linear layers!)")
    
    # Performance characteristics
    print(f"   Computational complexity: O([complexity]) per forward pass")
    print(f"   YOUR implementation handles: [capability description]")

def main():
    """Demonstrate the complete milestone using YOUR TinyTorch system!"""
    
    print("🎯 [MILESTONE NAME] - Proof of YOUR Mastery!")
    print("   Historical significance: [Brief context]")
    print("   YOUR achievement: [What they've built]")
    print()
    
    # Step 1: Get dataset
    data_loader = download_dataset()
    
    # Step 2: Create model with YOUR components  
    model = create_model()
    
    # Step 3: Train using YOUR training system
    train_model(model, data_loader)
    
    # Step 4: Analyze YOUR implementation
    analyze_performance(model)
    
    print("\n✅ SUCCESS! Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • [Specific achievement 1 using YOUR modules]")
    print("   • [Specific achievement 2 using YOUR implementations]") 
    print("   • [Connection to modern ML systems]")
    print("\n🚀 Next Steps:")
    print("   • Continue to [next milestone] after Module [X]")
    print("   • YOUR foundation enables: [future capabilities]")

if __name__ == "__main__":
    main()