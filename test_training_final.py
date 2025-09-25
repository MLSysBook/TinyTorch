#!/usr/bin/env python
"""
Final Training Test - Complete solution using fixed TinyTorch
============================================================
"""

import numpy as np
import sys

# Import our modules
sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable, add, multiply, matmul

class SimpleLinear:
    """Simple linear layer using our fixed Variable system."""
    
    def __init__(self, in_features, out_features):
        # Parameters with requires_grad=True
        self.weights = Parameter(np.random.randn(in_features, out_features) * 0.1)
        self.bias = Parameter(np.random.randn(out_features, 1) * 0.1)  # Column vector for broadcasting
    
    def forward(self, x):
        # Convert to Variables for gradient tracking
        weight_var = Variable(self.weights)
        bias_var = Variable(self.bias)
        
        # Ensure input is Variable
        x_var = x if isinstance(x, Variable) else Variable(x, requires_grad=False)
        
        # Linear transformation: x @ W + b
        output = matmul(x_var, weight_var)
        output = add(output, bias_var)
        return output
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def __call__(self, x):
        return self.forward(x)


class SimpleMSELoss:
    """MSE loss that works with Variables and maintains computational graph."""
    
    def __call__(self, pred, target):
        # Ensure both are Variables
        pred_var = pred if isinstance(pred, Variable) else Variable(pred)
        target_var = Variable(target, requires_grad=False)
        
        # MSE = mean((pred - target)^2)
        # Use subtract operation from autograd to maintain graph
        from autograd_dev import subtract
        diff = subtract(pred_var, target_var)  # This maintains the computational graph
        squared = multiply(diff, diff)
        
        # Compute sum (we'll treat as mean by scaling learning rate)
        loss_data = np.sum(squared.data.data)
        
        # Create loss Variable with proper gradient function that triggers the graph
        loss = Variable(loss_data, requires_grad=True)
        
        def loss_grad_fn(grad_output=Variable(1.0)):
            # Simply pass gradient of 1 to start the backward chain
            # The subtract and multiply operations will handle their own gradients
            squared.backward(Variable(np.ones_like(squared.data.data)))
        
        loss._grad_fn = loss_grad_fn
        return loss


class SimpleSGD:
    """Simple SGD optimizer."""
    
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        for p in self.params:
            if p.grad is not None:
                # Update: param = param - lr * grad
                p.data = p.data - self.lr * p.grad.data


def test_linear_regression():
    """Test linear regression y = 2x + 1"""
    print("="*60)
    print("TESTING LINEAR REGRESSION WITH COMPLETE SOLUTION")
    print("="*60)
    
    # Data: y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)  # (4, 1)
    y = np.array([[3.0], [5.0], [7.0], [9.0]], dtype=np.float32)  # (4, 1)
    
    # Model
    model = SimpleLinear(1, 1)
    print(f"Initial: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0,0]:.3f}")
    
    # Training setup
    optimizer = SimpleSGD(model.parameters(), lr=0.01)
    criterion = SimpleMSELoss()
    
    # Training loop
    losses = []
    for epoch in range(100):
        # Forward pass
        output = model(Variable(X))
        loss = criterion(output, y)
        
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
        losses.append(float(loss.data.data))
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d}: Loss = {losses[-1]:.4f}")
    
    print(f"Final: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0,0]:.3f}")
    print(f"Target: weight=2.000, bias=1.000")
    
    # Check convergence
    w_err = abs(model.weights.data[0,0] - 2.0)
    b_err = abs(model.bias.data[0,0] - 1.0)
    
    if w_err < 0.2 and b_err < 0.2:
        print("✅ Linear regression converged!")
        return True
    else:
        print("❌ Linear regression failed to converge")
        print(f"Errors: weight={w_err:.3f}, bias={b_err:.3f}")
        return False


def sigmoid(x):
    """Sigmoid activation for Variables."""
    if not isinstance(x, Variable):
        x = Variable(x)
    
    # Forward pass with numerical stability
    data = np.clip(x.data.data, -500, 500)  # Prevent overflow
    sig_data = 1.0 / (1.0 + np.exp(-data))
    
    # Backward pass
    def grad_fn(grad_output):
        grad = sig_data * (1 - sig_data) * grad_output.data.data
        x.backward(Variable(grad))
    
    return Variable(sig_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


def relu(x):
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


def test_xor_training():
    """Test XOR training with complete solution."""
    print("\n" + "="*60)
    print("TESTING XOR TRAINING WITH COMPLETE SOLUTION")
    print("="*60)
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Network
    layer1 = SimpleLinear(2, 4)
    layer2 = SimpleLinear(4, 1)
    
    # Training setup
    params = layer1.parameters() + layer2.parameters()
    optimizer = SimpleSGD(params, lr=0.5)
    criterion = SimpleMSELoss()
    
    print(f"Total parameters: {len(params)}")
    
    # Training loop
    for epoch in range(300):
        # Forward pass
        h1 = layer1(Variable(X))
        h1_relu = relu(h1)
        h2 = layer2(h1_relu)
        output = sigmoid(h2)
        
        # Loss
        loss = criterion(output, y)
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
        
        if epoch % 75 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    # Test final predictions
    print("\nFinal predictions:")
    h1 = layer1(Variable(X))
    h1_relu = relu(h1)
    h2 = layer2(h1_relu)
    predictions = sigmoid(h2)
    
    pred_vals = predictions.data.data
    for x_val, pred, target in zip(X, pred_vals, y):
        print(f"  {x_val} → {pred[0]:.3f} (target: {target[0]})")
    
    # Check accuracy
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
    print("TESTING COMPLETE TINYTORCH TRAINING SOLUTION")
    print("Based on PyTorch's lessons learned from Tensor/Variable unification")
    print()
    
    # Test simple case first
    linear_success = test_linear_regression()
    
    # Test complex case
    xor_success = test_xor_training()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Linear Regression: {'✅ PASS' if linear_success else '❌ FAIL'}")
    print(f"XOR Training:      {'✅ PASS' if xor_success else '❌ FAIL'}")
    
    if linear_success and xor_success:
        print("\n🎉 SUCCESS! Training now works with TinyTorch!")
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        print("Key fixes implemented:")
        print("1. ✅ Added __matmul__ operator to Variable class")
        print("2. ✅ Fixed Variable initialization to handle different Tensor types")
        print("3. ✅ Added matmul, divide functions with proper gradients")
        print("4. ✅ Updated Linear layer to work with Variables")
        print("5. ✅ Gradient flow from Variables back to Parameters works")
        print()
        print("This solution maintains the educational Tensor/Variable separation")
        print("while enabling proper gradient flow for neural network training.")
        print("Students can now train real neural networks!")
        
    else:
        print("\n⚠️ Some tests failed. Check implementation.")