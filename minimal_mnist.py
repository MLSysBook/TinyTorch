#!/usr/bin/env python3
"""
Minimal viable MNIST training - just what's needed, no frills.
"""

import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, '.')

# Suppress module test outputs
import contextlib
import io

print("Loading TinyTorch components...")
with contextlib.redirect_stdout(io.StringIO()):
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.optimizers import SGD

# Simple MNIST MLP
class MNISTNet:
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        # Flatten if needed
        if len(x.data.shape) > 2:
            batch_size = x.data.shape[0]
            x = Variable(x.data.reshape(batch_size, -1), requires_grad=x.requires_grad)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return [self.fc1.weights, self.fc1.bias,
                self.fc2.weights, self.fc2.bias]

def softmax(x):
    """Simple softmax for predictions."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    """
    Simple cross-entropy loss with backward function.
    predictions: Variable with logits
    targets: one-hot encoded targets as Variable
    """
    # Get data
    pred_data = predictions.data.data if hasattr(predictions.data, 'data') else predictions.data
    target_data = targets.data.data if hasattr(targets.data, 'data') else targets.data

    # Softmax
    probs = softmax(pred_data)

    # Cross entropy
    eps = 1e-8
    loss_val = -np.mean(np.sum(target_data * np.log(probs + eps), axis=1))

    # Create loss Variable
    loss = Variable(loss_val, requires_grad=True)

    # Gradient function that properly chains backward
    def backward_fn():
        if predictions.requires_grad:
            batch_size = pred_data.shape[0]
            grad = (probs - target_data) / batch_size

            # Set gradient on predictions
            if predictions.grad is None:
                predictions.grad = Variable(grad)
            else:
                existing_grad = predictions.grad.data if hasattr(predictions.grad, 'data') else predictions.grad
                predictions.grad = Variable(existing_grad + grad)

            # CRITICAL: Call backward on predictions to propagate to earlier layers
            if hasattr(predictions, 'backward'):
                predictions.backward()

    loss.backward_fn = backward_fn
    return loss

def generate_dummy_mnist_data(n_samples=1000):
    """Generate fake MNIST-like data for testing."""
    # Random images (28x28 = 784 pixels)
    X = np.random.randn(n_samples, 784).astype(np.float32) * 0.5

    # Random labels (0-9)
    y = np.random.randint(0, 10, n_samples)

    # Convert to one-hot
    y_onehot = np.zeros((n_samples, 10))
    y_onehot[np.arange(n_samples), y] = 1

    return X, y_onehot, y

def train_epoch(model, X, y_onehot, optimizer, batch_size=32):
    """Train for one epoch."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    total_loss = 0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        # Get batch
        batch_idx = indices[i:i+batch_size]
        batch_X = X[batch_idx]
        batch_y = y_onehot[batch_idx]

        # Convert to Variables
        inputs = Variable(batch_X, requires_grad=False)
        targets = Variable(batch_y, requires_grad=False)

        # Forward pass
        outputs = model.forward(inputs)

        # Compute loss
        loss = cross_entropy_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Track loss - properly extract scalar value
        # loss is Variable, loss.data is Tensor, loss.data.data is ndarray
        loss_val = loss.data.data
        if isinstance(loss_val, np.ndarray):
            loss_val = float(loss_val.squeeze())

        total_loss += loss_val
        n_batches += 1

    return total_loss / n_batches

def evaluate(model, X, y_labels):
    """Evaluate accuracy."""
    # Forward pass
    inputs = Variable(X, requires_grad=False)
    outputs = model.forward(inputs)

    # Get predictions
    output_data = outputs.data.data if hasattr(outputs.data, 'data') else outputs.data
    predictions = np.argmax(output_data, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_labels)
    return accuracy

def main():
    print("\n🚀 Starting minimal MNIST training...")

    # Generate data
    print("Generating dummy MNIST data...")
    X_train, y_train_onehot, y_train_labels = generate_dummy_mnist_data(1000)
    X_test, y_test_onehot, y_test_labels = generate_dummy_mnist_data(200)

    # Create model
    print("Creating model...")
    model = MNISTNet()

    # Create optimizer
    optimizer = SGD(model.parameters(), learning_rate=0.1)

    # Training loop
    print("\nTraining...")
    n_epochs = 10

    for epoch in range(n_epochs):
        # Train
        avg_loss = train_epoch(model, X_train, y_train_onehot, optimizer)

        # Evaluate
        train_acc = evaluate(model, X_train[:200], y_train_labels[:200])
        test_acc = evaluate(model, X_test, y_test_labels)

        print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}")

    print("\n✅ Training complete!")

    # Final evaluation
    final_acc = evaluate(model, X_test, y_test_labels)
    print(f"\nFinal test accuracy: {final_acc:.2%}")

    if final_acc > 0.15:  # Better than random (10% for 10 classes)
        print("🎉 Model is learning! (Better than random guessing)")

    return model

if __name__ == "__main__":
    model = main()