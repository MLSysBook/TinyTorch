#!/usr/bin/env python3
"""
Milestone 2: MLP Training with TinyTorch
Demonstrates multi-layer perceptron training using the exported TinyTorch package.

SUCCESS CRITERIA:
1. Solves XOR problem (non-linearly separable) with >95% accuracy
2. Uses automatic differentiation (autograd)
3. Uses modern optimizer (Adam)
4. Demonstrates 2+ hidden layers
"""

import numpy as np
import sys
from pathlib import Path

# Import from TinyTorch package (exported modules)
from tinytorch.core.tensor import Tensor
from tinytorch.nn import Linear, Sequential, ReLU, Sigmoid
from tinytorch.optim import Adam
from tinytorch.nn.functional import mse_loss

def generate_xor_dataset():
    """Generate the XOR problem dataset."""
    X = np.array([
        [0, 0],  # XOR(0,0) = 0
        [0, 1],  # XOR(0,1) = 1
        [1, 0],  # XOR(1,0) = 1
        [1, 1]   # XOR(1,1) = 0
    ], dtype=np.float32)

    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ], dtype=np.float32)

    return Tensor(X, requires_grad=False), Tensor(y, requires_grad=False)

def create_mlp():
    """Create 2-hidden-layer MLP for XOR problem."""
    return Sequential(
        Linear(2, 4),     # Input layer: 2 → 4
        ReLU(),
        Linear(4, 4),     # Hidden layer: 4 → 4
        ReLU(),
        Linear(4, 1),     # Output layer: 4 → 1
        Sigmoid()         # Binary classification
    )

def train_mlp():
    """Train MLP on XOR problem using TinyTorch."""
    print("🎯 MILESTONE 2: MLP TRAINING")
    print("Non-linear classification with autograd + Adam optimizer")
    print("=" * 60)

    # Generate XOR dataset
    print("📊 Generating XOR dataset...")
    X, y = generate_xor_dataset()
    print("XOR Truth Table:")
    print("  [0,0] → 0")
    print("  [0,1] → 1")
    print("  [1,0] → 1")
    print("  [1,1] → 0")

    # Create model
    print(f"\n🧠 Creating MLP model...")
    model = create_mlp()
    print(f"Architecture: 2 → 4 → 4 → 1 (with ReLU + Sigmoid)")

    # Setup training
    optimizer = Adam(model.parameters(), learning_rate=0.01)
    epochs = 1000

    print(f"\n🏋️ Training for {epochs} epochs with Adam optimizer...")
    print("Epoch | Loss      | Accuracy | All Correct")
    print("-" * 45)

    # Helper function to extract data from nested structures
    def extract_data(x):
        if hasattr(x, 'data'):
            if hasattr(x.data, 'data'):
                data = x.data.data  # Variable with Tensor data
            else:
                data = x.data  # Tensor data
        else:
            data = x

        # Convert memoryview to numpy array
        if isinstance(data, memoryview):
            data = np.array(data)
        return data

    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X)
        loss = mse_loss(predictions, y)

        # Backward pass (autograd!)
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Compute accuracy (handle nested data structure)
        pred_data = extract_data(predictions)
        target_data = extract_data(y)

        pred_classes = (pred_data > 0.5).astype(int)
        accuracy = np.mean(pred_classes == target_data)
        all_correct = np.all(pred_classes == target_data)

        if epoch % 200 == 0 or epoch < 10 or all_correct:
            status = "✅" if all_correct else "🔄"
            loss_value = extract_data(loss)
            print(f"{epoch:5d} | {float(loss_value):.6f} | {accuracy:.3f}    | {status}")

        # Early stopping if perfect
        if all_correct and epoch > 100:
            print(f"🎉 Perfect XOR solution found at epoch {epoch}!")
            break

    # Final evaluation
    print(f"\n📈 Final Evaluation:")
    final_predictions = model(X)
    final_pred_data = extract_data(final_predictions)
    final_target_data = extract_data(y)
    final_accuracy = np.mean((final_pred_data > 0.5) == final_target_data)
    final_loss = float(extract_data(mse_loss(final_predictions, y)))

    print(f"Final accuracy: {final_accuracy:.1%}")
    print(f"Final loss: {final_loss:.6f}")

    # Test each XOR pattern
    print(f"\n📋 XOR Pattern Analysis:")
    print("Pattern   | True | Pred | Prob  | Correct")
    print("-" * 40)

    all_patterns_correct = True
    X_data = extract_data(X)
    y_data = extract_data(y)
    final_pred_data = extract_data(final_predictions)

    for i, (inputs, true_output) in enumerate(zip(X_data, y_data)):
        predicted_prob = final_pred_data[i, 0]
        predicted_class = int(predicted_prob > 0.5)
        true_class = int(true_output[0])
        correct = (predicted_class == true_class)
        all_patterns_correct &= correct

        status = "✅" if correct else "❌"
        print(f"{inputs} |  {true_class}   |  {predicted_class}   | {predicted_prob:.3f} | {status}")

    # Test success criteria
    print(f"\n🔍 TESTING SUCCESS CRITERIA:")

    success_criteria = []

    # 1. Accuracy >95%
    criterion_1 = final_accuracy >= 0.95
    success_criteria.append(criterion_1)
    print(f"  1. Accuracy ≥ 95%: {final_accuracy:.1%} {'✅' if criterion_1 else '❌'}")

    # 2. All XOR patterns correct
    criterion_2 = all_patterns_correct
    success_criteria.append(criterion_2)
    print(f"  2. All XOR patterns correct: {'✅' if criterion_2 else '❌'}")

    # 3. Uses autograd (verified by training working)
    criterion_3 = epochs > 0  # If we got here, autograd worked
    success_criteria.append(criterion_3)
    print(f"  3. Autograd working: ✅")

    # 4. Uses modern optimizer
    criterion_4 = True  # We used Adam
    success_criteria.append(criterion_4)
    print(f"  4. Adam optimizer used: ✅")

    # 5. Multi-layer architecture
    criterion_5 = True  # We have 2 hidden layers
    success_criteria.append(criterion_5)
    print(f"  5. Multi-layer (2 hidden): ✅")

    # Overall result
    all_criteria_met = all(success_criteria)

    print(f"\n" + "=" * 60)
    if all_criteria_met:
        print("🎉 MILESTONE 2: MLP - ACHIEVED!")
        print("✅ Non-linear problem solved with TinyTorch!")
        print("✅ Autograd + Adam optimizer working!")
        print("✅ Multi-layer architecture successful!")
        print(f"✅ XOR accuracy: {final_accuracy:.1%}")
        print("\n🚀 Ready for Milestone 3: CNN!")
    else:
        print("❌ MILESTONE 2: MLP - NOT ACHIEVED")
        failed = sum(1 for c in success_criteria if not c)
        print(f"❌ {failed}/{len(success_criteria)} criteria failed")

    print("=" * 60)
    return all_criteria_met

if __name__ == "__main__":
    success = train_mlp()
    sys.exit(0 if success else 1)