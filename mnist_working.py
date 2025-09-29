#!/usr/bin/env python3
"""
Working MNIST example - properly uses TinyTorch modules.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

# Suppress module outputs
import contextlib
import io

print("Loading TinyTorch...")
with contextlib.redirect_stdout(io.StringIO()):
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
    from tinytorch.core.layers import Linear
    from tinytorch.core.activations import ReLU
    from tinytorch.core.optimizers import SGD
    # Use the losses we created
    from tinytorch.core.losses import CrossEntropyLoss

class MNISTNet:
    """Simple MNIST network."""
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        # Handle both Variable and Tensor inputs
        if not isinstance(x, Variable):
            x = Variable(x.data if hasattr(x, 'data') else x, requires_grad=False)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return [self.fc1.weights, self.fc1.bias,
                self.fc2.weights, self.fc2.bias]

def generate_mnist_data(n_train=1000, n_test=200):
    """Generate dummy MNIST data."""
    # Training data
    X_train = np.random.randn(n_train, 784).astype(np.float32) * 0.5
    y_train = np.random.randint(0, 10, n_train)

    # Test data
    X_test = np.random.randn(n_test, 784).astype(np.float32) * 0.5
    y_test = np.random.randint(0, 10, n_test)

    return X_train, y_train, X_test, y_test

def train_epoch(model, X, y, loss_fn, optimizer, batch_size=32):
    """Train for one epoch."""
    n = len(X)
    indices = np.random.permutation(n)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X[batch_idx]
        batch_y = y[batch_idx]

        # Forward
        inputs = Variable(batch_X, requires_grad=False)
        outputs = model.forward(inputs)

        # Loss - CrossEntropyLoss expects integer labels
        targets = Variable(batch_y, requires_grad=False)
        loss = loss_fn(outputs, targets)

        # Backward
        if hasattr(loss, 'backward'):
            loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Track loss
        loss_val = loss.data.data
        if isinstance(loss_val, np.ndarray):
            loss_val = float(loss_val.squeeze())
        total_loss += loss_val
        n_batches += 1

    return total_loss / max(n_batches, 1)

def evaluate(model, X, y):
    """Evaluate accuracy."""
    # Forward pass
    outputs = model.forward(Variable(X, requires_grad=False))

    # Get predictions
    output_data = outputs.data.data if hasattr(outputs.data, 'data') else outputs.data
    predictions = np.argmax(output_data, axis=1)

    # Accuracy
    accuracy = np.mean(predictions == y)
    return accuracy

def main():
    print("\n🚀 Starting MNIST training...")

    # Generate data
    print("Generating data...")
    X_train, y_train, X_test, y_test = generate_mnist_data(1000, 200)

    # Model
    print("Creating model...")
    model = MNISTNet()

    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), learning_rate=0.1)

    # Training
    print("\nTraining...")
    n_epochs = 10

    for epoch in range(n_epochs):
        # Train
        avg_loss = train_epoch(model, X_train, y_train, loss_fn, optimizer)

        # Evaluate
        train_acc = evaluate(model, X_train[:200], y_train[:200])
        test_acc = evaluate(model, X_test, y_test)

        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.1%}, Test Acc={test_acc:.1%}")

    print("\n✅ Training complete!")

    # Final accuracy
    final_acc = evaluate(model, X_test, y_test)
    print(f"Final test accuracy: {final_acc:.1%}")

    if final_acc > 0.15:
        print("🎉 Model is learning! (Better than random)")

    return model

if __name__ == "__main__":
    model = main()