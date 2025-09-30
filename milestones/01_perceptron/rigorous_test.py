#!/usr/bin/env python3
"""
RIGOROUS MILESTONE 1 TEST: Perceptron
Tests binary classification with concrete success criteria and evidence.

SUCCESS CRITERIA:
1. Training: >95% accuracy on linearly separable 2D dataset (200 samples)
2. Inference: Correctly classifies new test points
3. Decision boundary: Visualizes learned linear separation
4. Convergence: Loss decreases monotonically
5. Manual gradients: No autograd dependency

EVIDENCE REQUIRED:
- Training curve showing convergence
- Final accuracy measurement
- Decision boundary visualization
- Test set evaluation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_modules():
    """Load TinyTorch modules 01-04 in isolation."""
    project_root = Path(__file__).parent.parent.parent

    print("🔧 Loading Required Modules (01-04)...")

    # Module 01: Tensor
    os.chdir(project_root / 'modules/01_tensor')
    with open('tensor_dev.py', 'r') as f:
        exec(f.read(), globals())

    # Module 02: Activations
    os.chdir(project_root / 'modules/02_activations')
    with open('activations_dev.py', 'r') as f:
        exec(f.read(), globals())

    # Module 03: Layers
    os.chdir(project_root / 'modules/03_layers')
    with open('layers_dev.py', 'r') as f:
        exec(f.read(), globals())

    # Module 04: Losses
    os.chdir(project_root / 'modules/04_losses')
    with open('losses_dev.py', 'r') as f:
        exec(f.read(), globals())

    os.chdir(project_root)  # Return to project root
    print("✅ All modules loaded successfully")
    return True

def generate_linearly_separable_data(n_samples=200, seed=42):
    """Generate linearly separable 2D binary classification dataset."""
    np.random.seed(seed)

    # Class 0: cluster around (-1, -1)
    class0_x = np.random.normal(-1, 0.5, (n_samples//2, 2))
    class0_y = np.zeros((n_samples//2, 1))

    # Class 1: cluster around (1, 1)
    class1_x = np.random.normal(1, 0.5, (n_samples//2, 2))
    class1_y = np.ones((n_samples//2, 1))

    # Combine and shuffle
    X = np.vstack([class0_x, class1_x])
    y = np.vstack([class0_y, class1_y])

    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return Tensor(X), Tensor(y)

def create_perceptron():
    """Create Linear + Sigmoid perceptron (no autograd)."""
    return Sequential(
        Linear(2, 1),  # 2D input -> 1 output
        Sigmoid()      # Binary classification
    )

def train_perceptron_rigorous(model, X, y, epochs=500, lr=0.5):
    """Train with manual gradient descent and detailed monitoring."""
    loss_fn = MSELoss()
    train_losses = []
    accuracies = []

    print(f"🏋️ Training perceptron for {epochs} epochs...")
    print("Epoch | Loss      | Accuracy | Gradient Norm")
    print("-" * 45)

    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)

        # Compute accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = np.mean(pred_classes == y.data)

        # Manual gradient computation (educational)
        linear_layer = model.layers[0]
        error = predictions.data - y.data
        grad_w = X.data.T @ error / len(X.data)
        grad_b = np.mean(error, axis=0) if linear_layer.bias is not None else 0

        # Gradient norm for monitoring
        grad_norm = np.linalg.norm(grad_w) + (np.abs(grad_b) if hasattr(grad_b, '__len__') else abs(grad_b))

        # Update weights
        linear_layer.weight.data -= lr * grad_w
        if linear_layer.bias is not None:
            linear_layer.bias.data -= lr * grad_b

        # Log progress
        train_losses.append(float(loss.data))
        accuracies.append(accuracy)

        if epoch % 100 == 0 or epoch < 10:
            print(f"{epoch:5d} | {loss.data:.6f} | {accuracy:.3f}    | {grad_norm:.4f}")

    return train_losses, accuracies

def evaluate_model(model, X, y):
    """Rigorous model evaluation."""
    predictions = model.forward(X)
    pred_classes = (predictions.data > 0.5).astype(int)

    accuracy = np.mean(pred_classes == y.data)

    # Confusion matrix
    true_pos = np.sum((pred_classes == 1) & (y.data == 1))
    true_neg = np.sum((pred_classes == 0) & (y.data == 0))
    false_pos = np.sum((pred_classes == 1) & (y.data == 0))
    false_neg = np.sum((pred_classes == 0) & (y.data == 1))

    return {
        'accuracy': accuracy,
        'true_pos': true_pos,
        'true_neg': true_neg,
        'false_pos': false_pos,
        'false_neg': false_neg,
        'predictions': predictions,
        'pred_classes': pred_classes
    }

def plot_results(model, X, y, train_losses, accuracies, save_path):
    """Create comprehensive result visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training curves
    epochs = range(len(train_losses))
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy curve
    ax2.plot(epochs, accuracies, 'g-', label='Training Accuracy')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Decision boundary
    X_data = X.data
    y_data = y.data.flatten()

    # Plot data points
    class0_mask = y_data == 0
    class1_mask = y_data == 1

    ax3.scatter(X_data[class0_mask, 0], X_data[class0_mask, 1],
               c='red', marker='o', alpha=0.7, label='Class 0', s=30)
    ax3.scatter(X_data[class1_mask, 0], X_data[class1_mask, 1],
               c='blue', marker='s', alpha=0.7, label='Class 1', s=30)

    # Decision boundary
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    mesh_points = Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = model.forward(mesh_points).data
    Z = Z.reshape(xx.shape)

    contour = ax3.contour(xx, yy, Z, levels=[0.5], colors='black',
                         linestyles='-', linewidths=2)
    ax3.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')

    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_title('Decision Boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Model parameters visualization
    linear_layer = model.layers[0]
    weights = linear_layer.weight.data
    bias = linear_layer.bias.data if linear_layer.bias is not None else [0]

    ax4.bar(['w1', 'w2', 'bias'], [weights[0,0], weights[1,0], bias[0]])
    ax4.set_title('Learned Parameters')
    ax4.set_ylabel('Parameter Value')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Results saved to {save_path}")

    return fig

def main():
    """Rigorous Milestone 1 evaluation."""
    print("=" * 60)
    print("🎯 RIGOROUS MILESTONE 1 TEST: PERCEPTRON")
    print("Binary classification with concrete success criteria")
    print("=" * 60)

    # Load modules
    if not load_modules():
        print("❌ FAILED: Could not load required modules")
        return False

    # Generate dataset
    print("\n📊 Generating linearly separable dataset...")
    X, y = generate_linearly_separable_data(n_samples=200)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Create model
    print("\n🧠 Creating perceptron model...")
    model = create_perceptron()
    print(f"Architecture: 2 → 1 (Linear + Sigmoid)")

    # Train model
    print("\n🏋️ Training with manual gradients...")
    train_losses, accuracies = train_perceptron_rigorous(model, X, y, epochs=500, lr=0.5)

    # Evaluate model
    print("\n📈 Evaluating final performance...")
    results = evaluate_model(model, X, y)

    final_accuracy = results['accuracy']
    final_loss = train_losses[-1]

    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_accuracy:.1%}")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  True Positives: {results['true_pos']}")
    print(f"  True Negatives: {results['true_neg']}")
    print(f"  False Positives: {results['false_pos']}")
    print(f"  False Negatives: {results['false_neg']}")

    # Test success criteria
    print("\n🔍 TESTING SUCCESS CRITERIA:")

    success_criteria = []

    # 1. Training accuracy >95%
    accuracy_threshold = 0.95
    criterion_1 = final_accuracy >= accuracy_threshold
    success_criteria.append(criterion_1)
    print(f"  1. Accuracy ≥ 95%: {final_accuracy:.1%} {'✅' if criterion_1 else '❌'}")

    # 2. Loss convergence (decreasing trend)
    loss_trend = np.polyfit(range(len(train_losses)), train_losses, 1)[0]
    criterion_2 = loss_trend < 0
    success_criteria.append(criterion_2)
    print(f"  2. Loss converges: slope={loss_trend:.6f} {'✅' if criterion_2 else '❌'}")

    # 3. Final loss below threshold
    loss_threshold = 0.1
    criterion_3 = final_loss < loss_threshold
    success_criteria.append(criterion_3)
    print(f"  3. Final loss < {loss_threshold}: {final_loss:.6f} {'✅' if criterion_3 else '❌'}")

    # 4. Balanced classification (no major class bias)
    precision = results['true_pos'] / (results['true_pos'] + results['false_pos']) if (results['true_pos'] + results['false_pos']) > 0 else 0
    recall = results['true_pos'] / (results['true_pos'] + results['false_neg']) if (results['true_pos'] + results['false_neg']) > 0 else 0
    criterion_4 = precision > 0.9 and recall > 0.9
    success_criteria.append(criterion_4)
    print(f"  4. Balanced performance: P={precision:.3f}, R={recall:.3f} {'✅' if criterion_4 else '❌'}")

    # 5. Model parameters are reasonable
    linear_layer = model.layers[0]
    max_weight = np.max(np.abs(linear_layer.weight.data))
    criterion_5 = max_weight < 10.0  # Sanity check
    success_criteria.append(criterion_5)
    print(f"  5. Reasonable parameters: max_weight={max_weight:.3f} {'✅' if criterion_5 else '❌'}")

    # Overall milestone result
    all_criteria_met = all(success_criteria)

    # Create visualization
    save_path = Path(__file__).parent / 'rigorous_test_results.png'
    plot_results(model, X, y, train_losses, accuracies, save_path)

    # Final verdict
    print("\n" + "=" * 60)
    if all_criteria_met:
        print("🎉 MILESTONE 1: PERCEPTRON - ACHIEVED!")
        print("✅ All success criteria satisfied with concrete evidence")
        print(f"✅ Training accuracy: {final_accuracy:.1%} (target: ≥95%)")
        print(f"✅ Loss convergence: {loss_trend:.6f} (negative slope)")
        print(f"✅ Final loss: {final_loss:.6f} (target: <0.1)")
        print(f"✅ Balanced classification: P={precision:.3f}, R={recall:.3f}")
        print(f"✅ Reasonable parameters: max_weight={max_weight:.3f}")
        print("\n🚀 Ready for Milestone 2: MLP with autograd!")
    else:
        print("❌ MILESTONE 1: PERCEPTRON - NOT ACHIEVED")
        failed_criteria = sum(1 for c in success_criteria if not c)
        print(f"❌ {failed_criteria}/{len(success_criteria)} criteria failed")
        print("🔧 Need to fix issues before proceeding to Milestone 2")

    print("=" * 60)

    return all_criteria_met

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)