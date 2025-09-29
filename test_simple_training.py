#!/usr/bin/env python3
"""
Simple training test to debug gradient flow.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'modules/05_autograd')
sys.path.insert(0, 'modules/03_layers')
sys.path.insert(0, 'modules/04_losses')

import numpy as np

# Import directly from the fixed modules
from autograd_dev import Variable
from layers_dev import Linear
from losses_dev import MSELoss

def test_simple_training_step():
    """Test a single training step end-to-end."""
    print("🔬 Testing Simple Training Step")
    print("=" * 40)

    # Create simple dataset: linear function y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

    print(f"Dataset: X = {X.ravel()}, y = {y.ravel()}")

    # Create simple linear model
    model = Linear(1, 1)
    loss_fn = MSELoss()

    print(f"Initial weights: {model.weights.data.data}")
    print(f"Initial bias: {model.bias.data.data}")

    # Single training step
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")

        # Forward pass
        X_var = Variable(X, requires_grad=False)
        y_var = Variable(y, requires_grad=False)

        output = model.forward(X_var)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output.data.data.ravel()}")

        # Compute loss
        loss = loss_fn(output, y_var)
        print(f"Loss: {loss.data.data}")

        # Check gradient setup
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn is not None}")
        print(f"Output requires_grad: {output.requires_grad}")
        print(f"Model weights requires_grad: {model.weights.requires_grad}")

        # Reset gradients
        model.weights.grad = None
        model.bias.grad = None

        # Backward pass
        print("Calling loss.backward()...")
        try:
            loss.backward()
            print("✅ Backward pass completed!")

            # Check gradients
            print(f"Weight grad exists: {model.weights.grad is not None}")
            print(f"Bias grad exists: {model.bias.grad is not None}")

            if model.weights.grad is not None:
                # Handle numpy array gradients properly
                weight_grad_data = np.array(model.weights.grad)
                bias_grad_data = np.array(model.bias.grad)
                print(f"Weight grad: {weight_grad_data}")
                print(f"Bias grad shape: {bias_grad_data.shape}")
                print(f"Bias param shape: {model.bias.data.data.shape}")
                print(f"Bias grad: {bias_grad_data}")

                # Simple gradient descent
                lr = 0.01
                model.weights.data.data -= lr * weight_grad_data

                # Sum the bias gradient to match bias parameter shape
                if bias_grad_data.shape != model.bias.data.data.shape:
                    bias_grad_summed = np.sum(bias_grad_data, axis=0)  # Sum across batch dimension
                    print(f"Summed bias grad: {bias_grad_summed} (shape: {bias_grad_summed.shape})")
                else:
                    bias_grad_summed = bias_grad_data

                model.bias.data.data -= lr * bias_grad_summed

                print(f"Updated weights: {model.weights.data.data}")
                print(f"Updated bias: {model.bias.data.data}")
            else:
                print("❌ No gradients computed!")
                break

        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            break

    # Test final prediction
    print(f"\n--- Final Test ---")
    test_input = Variable([[5.0]], requires_grad=False)  # Expected: 2*5 + 1 = 11
    test_output = model.forward(test_input)
    print(f"Input: 5.0, Expected: 11.0, Got: {test_output.data.data[0][0]}")

    return True

if __name__ == "__main__":
    test_simple_training_step()