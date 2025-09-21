#!/usr/bin/env python3
"""
Simple CIFAR-10 training test - minimal example to isolate the broadcasting issue.
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset
from tinytorch.core.training import MeanSquaredError as MSELoss
from tinytorch.core.autograd import Variable

def test_simple_training():
    """Test minimal training loop to isolate broadcasting issue."""
    print("🧪 Simple CIFAR-10 Training Test")
    print("=" * 50)
    
    # Load small batch
    dataset = CIFAR10Dataset(root="./data", train=False, download=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Fixed batch size
    
    # Create simple model
    model = Dense(3072, 10)  # Direct 3072 → 10 (simplest case)
    softmax = Softmax()
    
    # Convert to Variables
    model.weights = Variable(model.weights, requires_grad=True)
    model.bias = Variable(model.bias, requires_grad=True)
    
    print(f"✅ Model created: weights {model.weights.data.shape}, bias {model.bias.data.shape}")
    
    # Loss function
    loss_fn = MSELoss()
    
    # Get one batch
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"\n🔄 Batch {batch_idx}: {images.shape}")
        
        # Check shapes before forward
        print(f"  Before forward - bias shape: {model.bias.data.shape}")
        
        # Flatten images carefully
        batch_size = images.shape[0]
        flattened = images.data.reshape(batch_size, -1)  # Just numpy reshape
        x = Variable(Tensor(flattened), requires_grad=True)
        
        print(f"  Input to model: {x.data.shape}")
        
        try:
            # Forward pass
            output = model.forward(x)
            print(f"  ✅ Forward pass: {output.data.shape}")
            print(f"  After forward - bias shape: {model.bias.data.shape}")
            
            # Apply softmax
            output = softmax.forward(output)
            print(f"  ✅ Softmax: {output.data.shape}")
            
            # Create target (one-hot)
            targets = np.zeros((batch_size, 10))
            for i in range(batch_size):
                targets[i, labels.data[i]] = 1
            target_var = Variable(Tensor(targets), requires_grad=False)
            
            # Compute loss
            loss = loss_fn(output, target_var)
            print(f"  ✅ Loss computed: {loss.data}")
            
            # Try backward (this might be where it breaks)
            if hasattr(loss, 'backward'):
                print("  🔄 Attempting backward pass...")
                loss.backward()
                print("  ✅ Backward pass succeeded!")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"  Debug - bias shape when failed: {model.bias.data.shape}")
            print(f"  Debug - weights shape: {model.weights.data.shape}")
            return False
        
        if batch_idx >= 2:  # Test a few batches
            break
    
    print("\n🎉 Simple training test completed successfully!")
    return True

if __name__ == "__main__":
    test_simple_training()