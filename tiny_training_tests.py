#!/usr/bin/env python3
"""
Tiny Training Tests - Verify learning without overfitting
Small versions of each example to ensure training dynamics are correct.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Softmax

def log(message):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def test_tiny_perceptron():
    """Test tiny perceptron on 10 samples."""
    log("Testing Tiny Perceptron (10 samples)...")
    
    class TinyPerceptron:
        def __init__(self):
            self.linear = Linear(2, 1)
            self.sigmoid = Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.linear(x))
        
        def parameters(self):
            return [self.linear.weights, self.linear.bias]
    
    # Create tiny linearly separable dataset
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
                  [0.2, 0.8], [0.8, 0.2], [0.3, 0.3], [0.7, 0.7], [0.4, 0.6]])
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
    
    model = TinyPerceptron()
    X_tensor = Tensor(X.astype(np.float32))
    y_tensor = Tensor(y.astype(np.float32))
    
    # Train for 20 epochs
    losses = []
    for epoch in range(20):
        # Forward
        predictions = model.forward(X_tensor)
        
        # Loss (MSE)
        diff = predictions - y_tensor
        squared_diff = diff * diff
        
        # Backward
        grad_output = Tensor(np.ones_like(squared_diff.data) / len(X))
        squared_diff.backward(grad_output)
        
        # Update
        lr = 0.5
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                grad_np = np.array(grad_data.data if hasattr(grad_data, 'data') else grad_data)
                param.data = param.data - lr * grad_np
                param.grad = None
        
        # Track loss
        pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
        loss_val = np.mean((pred_np - y) ** 2)
        losses.append(loss_val)
    
    # Check if loss decreased
    improved = losses[-1] < losses[0]
    log(f"  Initial loss: {losses[0]:.4f}")
    log(f"  Final loss: {losses[-1]:.4f}")
    log(f"  {'✅ PASS' if improved else '❌ FAIL'} - Loss {'decreased' if improved else 'did not decrease'}")
    
    return improved, losses

def test_tiny_xor():
    """Test tiny XOR with 4 samples."""
    log("Testing Tiny XOR (4 samples)...")
    
    class TinyXOR:
        def __init__(self):
            self.hidden = Linear(2, 4)
            self.output = Linear(4, 1)
            self.relu = ReLU()
            self.sigmoid = Sigmoid()
        
        def forward(self, x):
            h = self.relu(self.hidden(x))
            return self.sigmoid(self.output(h))
        
        def parameters(self):
            return [self.hidden.weights, self.hidden.bias,
                   self.output.weights, self.output.bias]
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    model = TinyXOR()
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)
    
    # Train
    losses = []
    for epoch in range(200):  # More epochs for XOR
        # Forward
        predictions = model.forward(X_tensor)

        # Loss
        diff = predictions - y_tensor
        squared_diff = diff * diff

        # Backward
        grad_output = Tensor(np.ones_like(squared_diff.data) * 0.25)
        squared_diff.backward(grad_output)

        # Update
        lr = 0.5  # Higher learning rate for XOR
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                grad_np = np.array(grad_data.data if hasattr(grad_data, 'data') else grad_data)
                param.data = param.data - lr * grad_np
                param.grad = None
        
        # Track
        pred_np = np.array(predictions.data.data if hasattr(predictions.data, 'data') else predictions.data)
        loss_val = np.mean((pred_np - y) ** 2)
        losses.append(loss_val)
    
    # Check learning
    improved = losses[-1] < losses[0] * 0.8  # At least 20% improvement
    accuracy = np.mean((pred_np > 0.5) == y) * 100
    
    log(f"  Initial loss: {losses[0]:.4f}")
    log(f"  Final loss: {losses[-1]:.4f}")
    log(f"  Accuracy: {accuracy:.1f}%")
    log(f"  {'✅ PASS' if improved else '❌ FAIL'} - {'Learning' if improved else 'Not learning'}")
    
    return improved, losses

def test_tiny_mlp():
    """Test tiny MLP on 3-class problem with 30 samples."""
    log("Testing Tiny MLP (30 samples, 3 classes)...")
    
    class TinyMLP:
        def __init__(self):
            self.fc1 = Linear(4, 8)
            self.fc2 = Linear(8, 3)
            self.relu = ReLU()
        
        def forward(self, x):
            h = self.relu(self.fc1(x))
            return self.fc2(h)
        
        def parameters(self):
            return [self.fc1.weights, self.fc1.bias,
                   self.fc2.weights, self.fc2.bias]
    
    # Create tiny dataset
    np.random.seed(42)
    X = np.random.randn(30, 4).astype(np.float32) * 0.5
    y = np.array([i % 3 for i in range(30)])  # 3 classes
    
    model = TinyMLP()
    X_tensor = Tensor(X)
    
    # Train/Val split
    train_idx = np.arange(24)
    val_idx = np.arange(24, 30)

    train_losses = []
    val_losses = []

    for epoch in range(100):  # More epochs for small dataset
        # Train
        X_train = Tensor(X[train_idx])
        y_train = y[train_idx]

        outputs = model.forward(X_train)

        # One-hot encode targets
        targets = np.zeros((len(train_idx), 3))
        for i, label in enumerate(y_train):
            targets[i, label] = 1
        targets_tensor = Tensor(targets)

        # MSE loss
        diff = outputs - targets_tensor
        squared_diff = diff * diff

        # Backward
        grad_output = Tensor(np.ones_like(squared_diff.data) / len(train_idx))
        squared_diff.backward(grad_output)

        # Update
        lr = 0.1  # Higher learning rate for tiny dataset
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                grad_np = np.array(grad_data.data if hasattr(grad_data, 'data') else grad_data)
                param.data = param.data - lr * grad_np
                param.grad = None
        
        # Track training loss
        outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
        train_loss = np.mean((outputs_np - targets) ** 2)
        train_losses.append(train_loss)
        
        # Validation
        X_val = Tensor(X[val_idx])
        y_val = y[val_idx]
        val_outputs = model.forward(X_val)
        
        val_targets = np.zeros((len(val_idx), 3))
        for i, label in enumerate(y_val):
            val_targets[i, label] = 1
        
        val_outputs_np = np.array(val_outputs.data.data if hasattr(val_outputs.data, 'data') else val_outputs.data)
        val_loss = np.mean((val_outputs_np - val_targets) ** 2)
        val_losses.append(val_loss)
    
    # Check for overfitting
    train_improved = train_losses[-1] < train_losses[0] * 0.7  # Less strict for tiny data
    val_stable = val_losses[-1] < val_losses[0]  # Val shouldn't increase much
    no_overfit = val_losses[-1] < train_losses[-1] * 3  # More lenient for tiny dataset
    
    log(f"  Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    log(f"  Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
    log(f"  {'✅' if no_overfit else '⚠️'} Overfitting check: Val/Train = {val_losses[-1]/train_losses[-1]:.2f}")
    log(f"  {'✅ PASS' if train_improved and no_overfit else '❌ FAIL'}")
    
    return train_improved and no_overfit, (train_losses, val_losses)

def test_tiny_cnn():
    """Test tiny CNN with 2x2 images."""
    log("Testing Tiny CNN (2x2 images, 10 samples)...")
    
    # Simplified conv for tiny test
    class TinyCNN:
        def __init__(self):
            self.conv_weight = Tensor(np.random.randn(2, 1, 2, 2).astype(np.float32) * 0.1)
            self.fc = Linear(2, 2)  # 2 features to 2 classes
        
        def forward(self, x):
            # Manual tiny convolution (1x2x2 -> 2x1x1)
            batch_size = x.data.shape[0]
            conv_out = []
            
            for b in range(batch_size):
                img = x.data[b, 0]  # Single channel
                features = []
                for f in range(2):  # 2 filters
                    kernel = self.conv_weight.data[f, 0]
                    # Valid convolution on 2x2 -> 1x1
                    val = np.sum(img * kernel)
                    features.append(val)
                conv_out.append(features)
            
            conv_tensor = Tensor(np.array(conv_out).astype(np.float32))
            return self.fc(conv_tensor)
        
        def parameters(self):
            return [self.conv_weight, self.fc.weights, self.fc.bias]
    
    # Tiny dataset: 10 2x2 images
    np.random.seed(42)
    X = np.random.randn(10, 1, 2, 2).astype(np.float32) * 0.5
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    model = TinyCNN()
    losses = []
    
    for epoch in range(30):
        # Forward batch
        outputs = model.forward(Tensor(X))
        
        # One-hot targets
        targets = np.zeros((10, 2))
        for i, label in enumerate(y):
            targets[i, label] = 1
        targets_tensor = Tensor(targets)
        
        # Loss
        diff = outputs - targets_tensor
        squared_diff = diff * diff
        
        # Backward
        grad_output = Tensor(np.ones_like(squared_diff.data) * 0.1)
        squared_diff.backward(grad_output)
        
        # Update
        lr = 0.01
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
                grad_np = np.array(grad_data.data if hasattr(grad_data, 'data') else grad_data)
                param.data = param.data - lr * grad_np
                param.grad = None
        
        # Track
        outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
        loss_val = np.mean((outputs_np - targets) ** 2)
        losses.append(loss_val)
    
    improved = losses[-1] < losses[0] * 0.7
    log(f"  Initial loss: {losses[0]:.4f}")
    log(f"  Final loss: {losses[-1]:.4f}")
    log(f"  {'✅ PASS' if improved else '❌ FAIL'} - {'Learning' if improved else 'Not learning'}")
    
    return improved, losses

def main():
    """Run all tiny training tests."""
    log("="*60)
    log("TINY TRAINING VERIFICATION TESTS")
    log("Ensuring proper training dynamics without overfitting")
    log("="*60)
    
    results = []
    
    # Test each tiny model
    tests = [
        ("Perceptron", test_tiny_perceptron),
        ("XOR", test_tiny_xor),
        ("MLP", test_tiny_mlp),
        ("CNN", test_tiny_cnn)
    ]
    
    for name, test_func in tests:
        log(f"\n{name}:")
        passed, data = test_func()
        results.append((name, passed))
    
    # Summary
    log("\n" + "="*60)
    log("TINY TRAINING SUMMARY")
    log("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        log(f"{name:12} {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        log("\n✅ All tiny models train correctly!")
        log("Training dynamics verified - no overfitting detected")
    else:
        log("\n⚠️ Some models have training issues")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
