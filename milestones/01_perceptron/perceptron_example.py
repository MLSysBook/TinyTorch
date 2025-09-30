#!/usr/bin/env python3
"""
Milestone 1: Perceptron Example
Training a Linear + Sigmoid perceptron on 2D dataset with decision boundary visualization.

Success Criteria: 95% accuracy on linearly separable data
Modules Used: 01 (Tensor), 02 (Activations), 03 (Layers), 04 (Losses)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add modules to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules by executing them
print("🎯 Loading TinyTorch Modules for Perceptron Milestone...")

print("📦 Loading Module 01: Tensor...")
exec(open(project_root / 'modules/01_tensor/tensor_dev.py').read())

print("📦 Loading Module 02: Activations...")
exec(open(project_root / 'modules/02_activations/activations_dev.py').read())

print("📦 Loading Module 03: Layers...")
exec(open(project_root / 'modules/03_layers/layers_dev.py').read())

print("📦 Loading Module 04: Losses...")
# Change to module directory to avoid __file__ issues
import os
old_cwd = os.getcwd()
try:
    os.chdir(project_root / 'modules/04_losses')
    exec(open('losses_dev.py').read())
finally:
    os.chdir(old_cwd)

print("✅ All modules loaded successfully!")

# Generate linearly separable 2D data
def generate_dataset(n_samples=200):
    """Generate linearly separable 2D dataset."""
    np.random.seed(42)

    # Class 0: points around (-1, -1)
    class0_x = np.random.normal(-1, 0.5, (n_samples//2, 2))
    class0_y = np.zeros((n_samples//2, 1))

    # Class 1: points around (1, 1)
    class1_x = np.random.normal(1, 0.5, (n_samples//2, 2))
    class1_y = np.ones((n_samples//2, 1))

    # Combine
    X = np.vstack([class0_x, class1_x])
    y = np.vstack([class0_y, class1_y])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return Tensor(X), Tensor(y)

# Create perceptron model
def create_perceptron():
    """Create Linear + Sigmoid perceptron."""
    return Sequential(
        Linear(2, 1),  # 2 inputs -> 1 output
        Sigmoid()      # Binary classification
    )

# Training function (simplified without optimizers)
def train_perceptron(model, X, y, epochs=1000, lr=0.1):
    """Train perceptron with simple gradient descent."""
    loss_fn = MSELoss()
    losses = []

    print(f"🚀 Training perceptron for {epochs} epochs...")

    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)

        # Simple weight update (manual for this milestone)
        # In real training we'd use Module 05 (autograd) + Module 06 (optimizers)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.4f}")

        losses.append(loss.data)

        # Manual gradient descent (simplified)
        # This is what Module 05 (autograd) will automate!
        linear_layer = model.layers[0]

        # Compute gradients manually for educational purposes
        error = predictions.data - y.data
        grad_w = X.data.T @ error / len(X.data)
        grad_b = np.mean(error, axis=0)

        # Update weights
        linear_layer.weight.data -= lr * grad_w
        if linear_layer.bias is not None:
            linear_layer.bias.data -= lr * grad_b

    return losses

# Evaluation
def evaluate_accuracy(model, X, y):
    """Compute classification accuracy."""
    predictions = model.forward(X)
    pred_classes = (predictions.data > 0.5).astype(int)
    accuracy = np.mean(pred_classes == y.data)
    return accuracy

# Visualization
def plot_decision_boundary(model, X, y, title="Perceptron Decision Boundary"):
    """Plot data points and decision boundary."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data points
    X_data = X.data
    y_data = y.data.flatten()

    class0_mask = y_data == 0
    class1_mask = y_data == 1

    ax.scatter(X_data[class0_mask, 0], X_data[class0_mask, 1],
               c='red', marker='o', alpha=0.7, label='Class 0')
    ax.scatter(X_data[class1_mask, 0], X_data[class1_mask, 1],
               c='blue', marker='s', alpha=0.7, label='Class 1')

    # Plot decision boundary
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    mesh_points = Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = model.forward(mesh_points).data
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def main():
    """Main perceptron milestone demonstration."""
    print("\n" + "="*60)
    print("🎯 MILESTONE 1: PERCEPTRON")
    print("Demonstrating: Linear + Sigmoid on 2D linearly separable data")
    print("Success Criteria: 95% accuracy")
    print("="*60)

    # Generate data
    print("\n📊 Generating linearly separable 2D dataset...")
    X, y = generate_dataset(n_samples=200)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Create model
    print("\n🧠 Creating perceptron model...")
    model = create_perceptron()
    print(f"Model: {model}")

    # Train model
    print("\n🏃 Training perceptron...")
    losses = train_perceptron(model, X, y, epochs=500, lr=0.5)

    # Evaluate
    print("\n📈 Evaluating performance...")
    accuracy = evaluate_accuracy(model, X, y)
    print(f"Final accuracy: {accuracy:.1%}")

    # Check success criteria
    success_threshold = 0.95
    if accuracy >= success_threshold:
        print(f"🎉 SUCCESS! Achieved {accuracy:.1%} accuracy (>= {success_threshold:.1%})")
        print("✅ Milestone 1: Perceptron ACHIEVED!")
    else:
        print(f"❌ Failed to meet {success_threshold:.1%} threshold. Got {accuracy:.1%}")

    # Visualization
    print("\n📊 Creating visualization...")
    try:
        fig = plot_decision_boundary(model, X, y,
                                   f"Perceptron (Accuracy: {accuracy:.1%})")
        plt.savefig('milestones/01_perceptron/perceptron_results.png', dpi=150, bbox_inches='tight')
        print("📁 Visualization saved as 'perceptron_results.png'")
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    print("\n" + "="*60)
    print("🎓 MILESTONE 1 COMPLETE")
    print("Next: Module 05 (Autograd) enables automatic gradients!")
    print("Then: Milestone 2 (MLP) with proper training loops!")
    print("="*60)

if __name__ == "__main__":
    main()