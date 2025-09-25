#!/usr/bin/env python
"""
Complete TinyTorch Training Solution
====================================
The working implementation that solves the original problem.
"""

import numpy as np
import sys

sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable, add, multiply, matmul, subtract

class WorkingLinear:
    """Working Linear layer that maintains gradient connections."""
    
    def __init__(self, in_features, out_features):
        # Parameters with requires_grad=True
        self.weights = Parameter(np.random.randn(in_features, out_features) * 0.1)
        self.bias = Parameter(np.random.randn(out_features) * 0.1)  # 1D bias
    
    def forward(self, x):
        """Forward pass maintaining gradient chain."""
        # Convert input to Variable if needed
        x_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        
        # Convert parameters to Variables to maintain gradient connections
        weight_var = Variable(self.weights)
        bias_var = Variable(self.bias)
        
        # Linear transformation: x @ weights + bias
        output = matmul(x_var, weight_var)
        
        # Handle bias addition with broadcasting
        # If bias is 1D and output is 2D, we need to make them compatible
        if len(output.shape) == 2 and len(bias_var.shape) == 1:
            # Create 2D bias for broadcasting
            bias_2d = Variable(self.bias.data.reshape(1, -1))  # (1, out_features)
            bias_var = bias_2d
        
        output = add(output, bias_var)
        return output
    
    def parameters(self):
        """Return parameters for optimizer."""
        return [self.weights, self.bias]
    
    def __call__(self, x):
        return self.forward(x)


def sigmoid_variable(x):
    """Sigmoid activation for Variables."""
    if not isinstance(x, Variable):
        x = Variable(x)
    
    # Forward pass with numerical stability
    data = np.clip(x.data.data, -500, 500)
    sig_data = 1.0 / (1.0 + np.exp(-data))
    
    # Backward pass
    def grad_fn(grad_output):
        grad = sig_data * (1 - sig_data) * grad_output.data.data
        x.backward(Variable(grad))
    
    return Variable(sig_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


def relu_variable(x):
    """ReLU activation for Variables."""
    if not isinstance(x, Variable):
        x = Variable(x)
    
    # Forward pass
    relu_data = np.maximum(0, x.data.data)
    
    # Backward pass
    def grad_fn(grad_output):
        grad = (x.data.data > 0) * grad_output.data.data
        x.backward(Variable(grad))
    
    return Variable(relu_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


class WorkingSGD:
    """Working SGD optimizer."""
    
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad.data


def mse_loss_simple(pred, target):
    """Simple MSE loss using the computational graph approach."""
    # Ensure Variables
    pred_var = pred if isinstance(pred, Variable) else Variable(pred)
    target_var = Variable(target, requires_grad=False)
    
    # MSE = mean((pred - target)^2)
    diff = subtract(pred_var, target_var)
    squared = multiply(diff, diff)
    
    # For simplicity, return sum instead of mean (adjust learning rate accordingly)
    loss_data = np.sum(squared.data.data)
    
    # Create loss Variable that will trigger backward through the graph
    loss = Variable(loss_data, requires_grad=True)
    
    def loss_grad_fn(grad_output):
        # Start the backward chain by calling backward on squared
        squared.backward(Variable(np.ones_like(squared.data.data)))
    
    loss._grad_fn = loss_grad_fn
    return loss


def test_linear_regression_working():
    """Test linear regression with working implementation."""
    print("="*60)
    print("LINEAR REGRESSION - WORKING IMPLEMENTATION")
    print("="*60)
    
    # Data: y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([[3.0], [5.0], [7.0], [9.0]], dtype=np.float32)
    
    # Model
    model = WorkingLinear(1, 1)
    print(f"Initial: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")
    
    # Training setup
    optimizer = WorkingSGD(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(100):
        # Forward pass
        output = model(Variable(X))
        loss = mse_loss_simple(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients (first epoch only)
        if epoch == 0:
            print("Gradient check:")
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.data)
                    print(f"  Parameter {i}: grad_norm = {grad_norm:.4f}")
                else:
                    print(f"  Parameter {i}: NO GRADIENT!")
        
        # Update
        optimizer.step()
        
        if epoch % 25 == 0:
            loss_val = float(loss.data.data)
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    print(f"Final: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")
    print(f"Target: weight=2.000, bias=1.000")
    
    # Check convergence
    w_err = abs(model.weights.data[0,0] - 2.0)
    b_err = abs(model.bias.data[0] - 1.0)
    
    if w_err < 0.2 and b_err < 0.2:
        print("✅ Linear regression converged!")
        return True
    else:
        print("❌ Linear regression failed to converge")
        return False


def test_xor_working():
    """Test XOR with working implementation."""
    print("\n" + "="*60)
    print("XOR TRAINING - WORKING IMPLEMENTATION") 
    print("="*60)
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Network
    layer1 = WorkingLinear(2, 8)
    layer2 = WorkingLinear(8, 1)
    
    # Training setup
    params = layer1.parameters() + layer2.parameters()
    optimizer = WorkingSGD(params, lr=0.5)
    
    print(f"Total parameters: {len(params)}")
    
    # Training loop
    for epoch in range(500):
        # Forward pass
        h1 = layer1(Variable(X))
        h1_act = relu_variable(h1)
        h2 = layer2(h1_act)
        output = sigmoid_variable(h2)
        
        # Loss
        loss = mse_loss_simple(output, y)
        loss_val = float(loss.data.data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients (first epoch only)
        if epoch == 0:
            print("Gradient check:")
            grad_count = 0
            for i, param in enumerate(params):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.data)
                    print(f"  Parameter {i}: grad_norm = {grad_norm:.4f}")
                    grad_count += 1
                else:
                    print(f"  Parameter {i}: NO GRADIENT!")
            
            if grad_count == len(params):
                print("✅ All parameters have gradients!")
            else:
                print(f"❌ Only {grad_count}/{len(params)} parameters have gradients!")
        
        # Update
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    # Test predictions
    print("\nFinal predictions:")
    h1 = layer1(Variable(X))
    h1_act = relu_variable(h1)
    h2 = layer2(h1_act)
    predictions = sigmoid_variable(h2)
    
    pred_vals = predictions.data.data
    for x_val, pred, target in zip(X, pred_vals, y):
        print(f"  {x_val} → {pred[0]:.3f} (target: {target[0]})")
    
    # Accuracy
    binary_preds = (pred_vals > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y)
    print(f"\nAccuracy: {accuracy*100:.0f}%")
    
    if accuracy >= 0.75:
        print("✅ XOR training successful!")
        return True
    else:
        print("❌ XOR training failed")
        return False


if __name__ == "__main__":
    print("COMPLETE TINYTORCH TRAINING SOLUTION")
    print("Based on PyTorch's architectural lessons")
    print()
    
    # Test linear regression
    linear_success = test_linear_regression_working()
    
    # Test XOR
    xor_success = test_xor_working()
    
    print("\n" + "="*60)
    print("SOLUTION RESULTS")
    print("="*60)
    print(f"Linear Regression: {'✅ SUCCESS' if linear_success else '❌ FAILED'}")
    print(f"XOR Training:      {'✅ SUCCESS' if xor_success else '❌ FAILED'}")
    
    if linear_success and xor_success:
        print("\n🎉 COMPLETE SUCCESS!")
        print("\n" + "="*60)
        print("WHAT WE FIXED")
        print("="*60)
        print("1. ✅ Added __matmul__ operator to Variable class")
        print("2. ✅ Fixed Variable initialization for different Tensor types")
        print("3. ✅ Implemented matmul() and divide() functions with gradients")
        print("4. ✅ Updated Linear layers to convert Parameters to Variables")
        print("5. ✅ Ensured gradient flow from Variables back to Parameters")
        print("6. ✅ Built computational graph through individual operations")
        print()
        print("🎯 KEY INSIGHT:")
        print("The solution maintains TinyTorch's educational Tensor/Variable separation")
        print("while ensuring proper gradient flow through the _source_tensor mechanism.")
        print("This mirrors PyTorch's early architecture before Tensor/Variable unification.")
        print()
        print("Students can now train real neural networks with TinyTorch!")
        
    else:
        print("\n⚠️ Solution incomplete. Check failing tests.")
        
    print("\n" + "="*60)
    print("USAGE FOR STUDENTS")
    print("="*60)
    print("To use this in TinyTorch training:")
    print("1. Use Parameter() for trainable weights")
    print("2. Convert to Variable() in forward pass")
    print("3. Build loss using autograd operations (add, multiply, subtract)")
    print("4. Call loss.backward() to compute gradients")
    print("5. Use optimizer.step() to update parameters")
    print()
    print("The gradient flow works: Parameter → Variable → Operations → Loss → Backward")