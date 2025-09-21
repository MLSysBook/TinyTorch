#!/usr/bin/env python3
"""
Test the bias shape fix directly.
"""

import numpy as np
import sys
import os
sys.path.append('/Users/VJ/GitHub/TinyTorch')

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.autograd import Variable
from tinytorch.core.optimizers import Adam

class SimpleLoss:
    """Simple MSE loss for testing."""
    def __call__(self, pred, target):
        diff = pred.data.data - target.data.data
        loss_data = np.mean(diff ** 2)
        
        # Create a Variable for the loss
        loss_var = Variable(Tensor(np.array(loss_data)), requires_grad=True)
        
        # Simple backward implementation
        def backward():
            # Compute gradient w.r.t. prediction
            grad = 2 * diff / diff.size
            if pred.grad is None:
                pred.grad = Variable(Tensor(grad))
            else:
                pred.grad.data.data += grad
        
        loss_var.backward = backward
        return loss_var

def test_bias_shape_fix():
    """Test that bias shapes are preserved with variable batch sizes."""
    print("🔍 Testing Bias Shape Fix")
    print("=" * 50)
    
    # Create a simple model
    layer = Dense(10, 3)
    activation = ReLU()
    
    # Convert to Variables
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    print(f"Initial bias shape: {layer.bias.data.shape}")
    
    # Create optimizer
    optimizer = Adam([layer.weights, layer.bias], learning_rate=0.001)
    loss_fn = SimpleLoss()
    
    # Test multiple batch sizes
    batch_sizes = [32, 16, 8, 4, 1]
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n--- Iteration {i+1}: Batch size {batch_size} ---")
        
        # Create data
        x_data = np.random.randn(batch_size, 10).astype(np.float32)
        x = Variable(Tensor(x_data), requires_grad=True)
        
        y_data = np.random.randn(batch_size, 3).astype(np.float32)
        y = Variable(Tensor(y_data), requires_grad=False)
        
        print(f"Before forward - bias shape: {layer.bias.data.shape}")
        
        # Forward pass
        z = layer.forward(x)
        output = activation.forward(z)
        
        print(f"After forward - bias shape: {layer.bias.data.shape}")
        
        # Compute loss
        loss = loss_fn(output, y)
        print(f"Loss: {loss.data.data}")
        
        # Backward pass
        optimizer.zero_grad()
        
        print(f"Before backward - bias shape: {layer.bias.data.shape}")
        try:
            loss.backward()
            print(f"After backward - bias shape: {layer.bias.data.shape}")
            
            # Optimizer step (this was corrupting shapes before fix)
            print(f"Before optimizer step - bias shape: {layer.bias.data.shape}")
            optimizer.step()
            print(f"✅ After optimizer step - bias shape: {layer.bias.data.shape}")
            
            # Verify shape is still correct
            expected_shape = (3,)
            actual_shape = layer.bias.data.shape
            if actual_shape == expected_shape:
                print(f"✅ Shape preserved: {actual_shape}")
            else:
                print(f"❌ Shape corrupted: expected {expected_shape}, got {actual_shape}")
                return False, i, batch_size
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Bias shape when error occurred: {layer.bias.data.shape}")
            return False, i, batch_size
    
    print(f"\n🎉 All batch sizes completed successfully!")
    print(f"Final bias shape: {layer.bias.data.shape}")
    return True, None, None

if __name__ == "__main__":
    success, fail_iter, fail_batch = test_bias_shape_fix()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    if success:
        print("✅ BIAS SHAPE FIX SUCCESSFUL!")
        print("Variable batch sizes now work correctly!")
    else:
        print(f"❌ Test failed at iteration {fail_iter}, batch size {fail_batch}")
        print("The bias shape corruption issue still exists.")