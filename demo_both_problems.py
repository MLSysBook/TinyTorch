#!/usr/bin/env python3
"""
TinyTorch Complete Solution Demo

Demonstrates that TinyTorch now has a complete working training pipeline by solving:
1. Linear Regression (simple, linear relationship)
2. XOR Learning (complex, requires nonlinearity)

Both problems train successfully, proving the pipeline works end-to-end.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Linear
from tinytorch.core.activations import Tanh, Sigmoid
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import SGD, Adam

def demo_linear_regression():
    """Demonstrate linear regression training."""
    print("🔸 Problem 1: Linear Regression")
    print("Task: Learn y = 2x + 1 from noisy data")
    
    # Generate training data: y = 2x + 1 + noise
    np.random.seed(42)
    X_train = np.random.randn(100, 1) * 2
    y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)
    
    # Simple linear model (no hidden layers needed)
    model = Linear(1, 1)
    loss_fn = MeanSquaredError()
    optimizer = SGD([model.weights, model.bias], learning_rate=0.01)
    
    print(f"Training on {len(X_train)} samples...")
    
    # Training loop
    for epoch in range(200):
        X_var = Variable(X_train, requires_grad=False)
        y_var = Variable(y_train, requires_grad=False)
        
        predictions = model(X_var)
        loss = loss_fn(predictions, y_var)
        
        # Reset gradients
        model.weights.grad = None
        model.bias.grad = None
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if epoch % 50 == 0:
            loss_val = loss.data.data if hasattr(loss.data, 'data') else loss.data
            print(f"  Epoch {epoch:3d}: Loss = {loss_val:.6f}")
    
    # Check learned parameters
    learned_weight = model.weights.data[0, 0]
    learned_bias = model.bias.data[0]
    
    print(f"Results:")
    print(f"  True parameters:    weight=2.000, bias=1.000")
    print(f"  Learned parameters: weight={learned_weight:.3f}, bias={learned_bias:.3f}")
    
    success = abs(learned_weight - 2.0) < 0.2 and abs(learned_bias - 1.0) < 0.2
    print(f"  Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success

def demo_xor_learning():
    """Demonstrate XOR learning with neural network."""
    print("\\n🔸 Problem 2: XOR Learning")  
    print("Task: Learn XOR function (requires nonlinearity)")
    
    # XOR data
    X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = np.array([[0.0], [1.0], [1.0], [0.0]])
    
    # Neural network with hidden layer
    layer1 = Linear(2, 4)
    activation1 = Tanh()
    layer2 = Linear(4, 1)
    activation2 = Sigmoid()
    
    # Collect all parameters
    all_params = layer1.parameters() + layer2.parameters()
    optimizer = Adam(all_params, learning_rate=0.01)
    loss_fn = MeanSquaredError()
    
    def forward(x):
        """Forward pass through network."""
        x = layer1(x)
        x = activation1(x)
        x = layer2(x)
        x = activation2(x)
        return x
    
    def zero_grad():
        """Reset all gradients."""
        for param in all_params:
            param.grad = None
    
    print(f"Network: 2 → 4 (Tanh) → 1 (Sigmoid)")
    print("Training...")
    
    # Training loop
    for epoch in range(500):
        X_var = Variable(X_train, requires_grad=False)
        y_var = Variable(y_train, requires_grad=False)
        
        predictions = forward(X_var)
        loss = loss_fn(predictions, y_var)
        
        zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            loss_val = loss.data.data if hasattr(loss.data, 'data') else loss.data
            print(f"  Epoch {epoch:3d}: Loss = {loss_val:.6f}")
    
    # Test final predictions
    final_preds = forward(Variable(X_train, requires_grad=False))
    pred_data = final_preds.data.data if hasattr(final_preds.data, 'data') else final_preds.data
    
    print("Results:")
    print("  Input  → Expected | Predicted")
    correct = 0
    for i in range(len(X_train)):
        expected = y_train[i, 0]
        predicted = pred_data[i, 0]
        predicted_class = 1.0 if predicted > 0.5 else 0.0
        is_correct = abs(predicted_class - expected) < 0.1
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"  {X_train[i]} → {expected:.0f}       | {predicted:.3f} {status}")
    
    accuracy = correct / len(X_train) * 100
    success = accuracy == 100.0
    print(f"  Accuracy: {accuracy:.0f}%")
    print(f"  Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success

def main():
    """Run both training demos."""
    print("🔥 TinyTorch Complete Training Pipeline Demo")
    print("=" * 60)
    
    success1 = demo_linear_regression()
    success2 = demo_xor_learning()
    
    print("\\n" + "=" * 60)
    print("📊 SUMMARY")
    print(f"Linear Regression: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"XOR Learning:      {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\\n🎉 COMPLETE SUCCESS!")
        print("TinyTorch has a fully working training pipeline:")
        print("  ✅ Linear layers maintain gradient connections")
        print("  ✅ Activations work with Variables")
        print("  ✅ Loss functions support autograd")
        print("  ✅ Optimizers update parameters correctly")
        print("  ✅ Can solve both linear AND nonlinear problems")
        print("  ✅ End-to-end training works perfectly")
    else:
        print("\\nSome issues remain, but core functionality is working.")
    
    return success1 and success2

if __name__ == "__main__":
    main()