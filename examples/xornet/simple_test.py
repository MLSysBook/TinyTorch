#!/usr/bin/env python3
"""
Simple XOR test using the exact pattern from the working autograd test
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.autograd import Variable

def test_xor_simple():
    """Test XOR using the exact working pattern from autograd tests"""
    
    # Simple model
    fc1 = Dense(2, 4)  # 2 inputs -> 4 hidden
    fc2 = Dense(4, 1)  # 4 hidden -> 1 output
    
    # Initialize with reasonable values (from working test)
    for layer in [fc1, fc2]:
        fan_in = layer.weights.shape[0]
        std = np.sqrt(2.0 / fan_in)
        layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
        layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
        
        layer.weights = Variable(layer.weights, requires_grad=True)
        layer.bias = Variable(layer.bias, requires_grad=True)
    
    # Optimizer
    params = [fc1.weights, fc1.bias, fc2.weights, fc2.bias]
    optimizer = SGD(params, learning_rate=0.1)
    
    # XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    print("Training XOR with working pattern...")
    print("Initial test:")
    
    # Track losses
    losses = []
    
    for i in range(100):
        # Forward (exact pattern from working test)
        x_var = Variable(Tensor(X), requires_grad=True)
        h = fc1(x_var)
        relu = ReLU()
        h = relu(h)
        out = fc2(h)
        
        # Loss
        y_var = Variable(Tensor(y), requires_grad=False)
        loss_fn = MeanSquaredError()
        loss = loss_fn(out, y_var)
        
        if hasattr(loss.data, 'data'):
            loss_val = float(loss.data.data)
        else:
            loss_val = float(loss.data._data)
        losses.append(loss_val)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Fix bias gradients if needed (from working test)
        for layer in [fc1, fc2]:
            if layer.bias.grad is not None:
                if hasattr(layer.bias.grad.data, 'data'):
                    grad = layer.bias.grad.data.data
                else:
                    grad = layer.bias.grad.data
                
                if len(grad.shape) == 2:
                    # Sum over batch dimension
                    layer.bias.grad = Variable(Tensor(np.sum(grad, axis=0)))
        
        # Update
        optimizer.step()
        
        if i % 20 == 0:
            print(f"  Iteration {i:2d}: Loss = {loss_val:.4f}")
    
    # Final test
    x_var = Variable(Tensor(X), requires_grad=False)
    h = fc1(x_var)
    h = relu(h)
    predictions = fc2(h)
    
    print("\nFinal results:")
    pred_data = predictions.data._data
    for i in range(4):
        prediction = pred_data[i, 0]
        target = y[i, 0]
        correct = "✅" if abs(prediction - target) < 0.5 else "❌"
        print(f"  {X[i]} -> {prediction:.3f} (want {target}) {correct}")
    
    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"\nLoss change: {initial_loss:.4f} -> {final_loss:.4f}")
    if final_loss < initial_loss * 0.9:
        print("✅ Learning happened!")
        return True
    else:
        print("❌ No learning detected")
        return False

if __name__ == "__main__":
    success = test_xor_simple()