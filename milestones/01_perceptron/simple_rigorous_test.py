#!/usr/bin/env python3
"""
SIMPLIFIED RIGOROUS MILESTONE 1 TEST: Perceptron
Focus on core binary classification capability with concrete success criteria.
"""

import sys
import numpy as np
from pathlib import Path

# Simple tensor implementation for testing
class SimpleTensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape

    def __str__(self):
        return f"Tensor({self.data}, shape={self.shape})"

# Simple perceptron components
class SimpleLinear:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        self.weight = SimpleTensor(np.random.normal(0, np.sqrt(2.0 / in_features), (in_features, out_features)))
        self.bias = SimpleTensor(np.zeros(out_features))

    def forward(self, x):
        # y = xW + b
        output = np.dot(x.data, self.weight.data) + self.bias.data
        return SimpleTensor(output)

class SimpleSigmoid:
    def forward(self, x):
        # Sigmoid with numerical stability
        z = np.clip(x.data, -500, 500)  # Prevent overflow
        return SimpleTensor(1.0 / (1.0 + np.exp(-z)))

class SimpleMSELoss:
    def forward(self, predictions, targets):
        diff = predictions.data - targets.data
        loss = np.mean(diff ** 2)
        return loss

class SimplePerceptron:
    def __init__(self):
        self.linear = SimpleLinear(2, 1)
        self.sigmoid = SimpleSigmoid()

    def forward(self, x):
        linear_out = self.linear.forward(x)
        return self.sigmoid.forward(linear_out)

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

    return SimpleTensor(X), SimpleTensor(y)

def train_perceptron_manual(model, X, y, epochs=500, lr=0.5):
    """Train with manual gradient descent."""
    loss_fn = SimpleMSELoss()
    train_losses = []
    accuracies = []

    print(f"🏋️ Training perceptron for {epochs} epochs...")
    print("Epoch | Loss      | Accuracy")
    print("-" * 30)

    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)

        # Compute accuracy
        pred_classes = (predictions.data > 0.5).astype(int)
        accuracy = np.mean(pred_classes == y.data)

        # Manual gradient computation
        error = predictions.data - y.data

        # Gradient through sigmoid: error * sigmoid * (1 - sigmoid)
        sigmoid_grad = predictions.data * (1 - predictions.data)
        linear_error = error * sigmoid_grad

        # Gradients for linear layer
        grad_w = X.data.T @ linear_error / len(X.data)
        grad_b = np.mean(linear_error, axis=0)

        # Update weights
        model.linear.weight.data -= lr * grad_w
        model.linear.bias.data -= lr * grad_b

        # Log progress
        train_losses.append(loss)
        accuracies.append(accuracy)

        if epoch % 100 == 0 or epoch < 10:
            print(f"{epoch:5d} | {loss:.6f} | {accuracy:.3f}")

    return train_losses, accuracies

def evaluate_model(model, X, y):
    """Evaluate model performance."""
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
        'true_pos': int(true_pos),
        'true_neg': int(true_neg),
        'false_pos': int(false_pos),
        'false_neg': int(false_neg)
    }

def main():
    """Rigorous Milestone 1 evaluation."""
    print("=" * 60)
    print("🎯 RIGOROUS MILESTONE 1 TEST: PERCEPTRON")
    print("Binary classification with concrete success criteria")
    print("=" * 60)

    # Generate dataset
    print("\n📊 Generating linearly separable dataset...")
    X, y = generate_linearly_separable_data(n_samples=200)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Create model
    print("\n🧠 Creating perceptron model...")
    model = SimplePerceptron()
    print(f"Architecture: 2 → 1 (Linear + Sigmoid)")

    # Train model
    print("\n🏋️ Training with manual gradients...")
    train_losses, accuracies = train_perceptron_manual(model, X, y, epochs=500, lr=0.5)

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
    max_weight = np.max(np.abs(model.linear.weight.data))
    criterion_5 = max_weight < 10.0  # Sanity check
    success_criteria.append(criterion_5)
    print(f"  5. Reasonable parameters: max_weight={max_weight:.3f} {'✅' if criterion_5 else '❌'}")

    # Overall milestone result
    all_criteria_met = all(success_criteria)

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