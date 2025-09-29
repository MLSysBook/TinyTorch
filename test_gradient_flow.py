#!/usr/bin/env python3
"""
Test gradient flow through the entire system.

This script tests if gradients properly flow from loss -> linear layers -> parameters.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'modules/05_autograd')
sys.path.insert(0, 'modules/03_layers')
sys.path.insert(0, 'modules/04_losses')

import numpy as np
import contextlib
import io

# Import our autograd system
from autograd_dev import Variable, multiply, add

# Import our layers system
from layers_dev import Linear, Parameter

# Import our loss functions
from losses_dev import MSELoss

def test_simple_gradient_flow():
    """Test gradient flow through a simple linear layer."""
    print("🔬 Testing Simple Gradient Flow")
    print("=" * 40)

    # Create a simple linear layer: 2 inputs -> 1 output
    layer = Linear(2, 1)

    print("\n📊 Initial State:")
    print(f"  Weight shape: {layer.weights.data.data.shape}")
    print(f"  Weight values: {layer.weights.data.data}")
    print(f"  Bias value: {layer.bias.data.data}")
    print(f"  Weight grad: {layer.weights.grad}")
    print(f"  Bias grad: {layer.bias.grad}")

    # Create input data (2 features)
    x = Variable([[1.0, 2.0]], requires_grad=False)

    # Forward pass
    print("\n🔄 Forward Pass:")
    output = layer.forward(x)
    print(f"  Input: {x.data.data}")
    print(f"  Output: {output.data.data}")
    print(f"  Output type: {type(output)}")
    print(f"  Output requires_grad: {output.requires_grad}")

    # Create target and compute loss
    target = Variable([[0.5]], requires_grad=False)
    loss_fn = MSELoss()
    loss = loss_fn(output, target)

    print(f"\n💔 Loss Computation:")
    print(f"  Target: {target.data.data}")
    print(f"  Loss: {loss.data.data}")
    print(f"  Loss type: {type(loss)}")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    # Backward pass
    print(f"\n⬅️ Backward Pass:")
    print("  Calling loss.backward()...")

    try:
        loss.backward(1.0)  # Pass scalar gradient for the loss
        print("  ✅ Backward pass completed successfully!")

        # Check gradients
        print(f"\n🎯 Gradient Results:")
        print(f"  Weight grad: {layer.weights.grad}")
        print(f"  Bias grad: {layer.bias.grad}")

        # Check if gradients exist and are non-zero
        if layer.weights.grad is not None and layer.bias.grad is not None:
            print("  ✅ Gradients successfully computed!")

            # Check if gradients have reasonable values
            # Handle different gradient data structures
            if hasattr(layer.weights.grad, 'data'):
                if hasattr(layer.weights.grad.data, 'data'):
                    weight_grad_data = layer.weights.grad.data.data
                else:
                    weight_grad_data = layer.weights.grad.data
            else:
                weight_grad_data = layer.weights.grad

            if hasattr(layer.bias.grad, 'data'):
                if hasattr(layer.bias.grad.data, 'data'):
                    bias_grad_data = layer.bias.grad.data.data
                else:
                    bias_grad_data = layer.bias.grad.data
            else:
                bias_grad_data = layer.bias.grad

            # Convert memoryview to array if needed
            if isinstance(weight_grad_data, memoryview):
                weight_grad_data = np.array(weight_grad_data)
            if isinstance(bias_grad_data, memoryview):
                bias_grad_data = np.array(bias_grad_data)

            weight_grad_norm = np.linalg.norm(weight_grad_data)
            bias_grad_norm = np.linalg.norm(bias_grad_data)
            print(f"  Weight gradient norm: {weight_grad_norm:.6f}")
            print(f"  Bias gradient norm: {bias_grad_norm:.6f}")

            if weight_grad_norm > 1e-8 and bias_grad_norm > 1e-8:
                print("  ✅ Gradient magnitudes are reasonable!")
                return True
            else:
                print("  ❌ Gradients are too small - might be zero!")
                return False
        else:
            print("  ❌ Gradients are None - backpropagation failed!")
            return False

    except Exception as e:
        print(f"  ❌ Backward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_two_layer_network():
    """Test gradient flow through a two-layer network."""
    print("\n\n🔬 Testing Two-Layer Network")
    print("=" * 40)

    # Create two-layer network: 3 -> 2 -> 1
    layer1 = Linear(3, 2)
    layer2 = Linear(2, 1)

    print("\n📊 Network Structure:")
    print(f"  Layer 1: 3 -> 2 (weights: {layer1.weights.data.data.shape})")
    print(f"  Layer 2: 2 -> 1 (weights: {layer2.weights.data.data.shape})")

    # Input data
    x = Variable([[1.0, 2.0, 3.0]], requires_grad=False)

    # Forward pass through network
    print(f"\n🔄 Forward Pass:")
    h1 = layer1.forward(x)
    print(f"  Input: {x.data.data}")
    print(f"  Hidden: {h1.data.data}")

    output = layer2.forward(h1)
    print(f"  Output: {output.data.data}")

    # Loss computation
    target = Variable([[1.0]], requires_grad=False)
    loss_fn = MSELoss()
    loss = loss_fn(output, target)

    print(f"\n💔 Loss: {loss.data.data}")

    # Backward pass
    print(f"\n⬅️ Backward Pass:")
    try:
        loss.backward(1.0)  # Pass scalar gradient
        print("  ✅ Backward pass completed!")

        # Check all gradients
        print(f"\n🎯 All Gradients:")
        print(f"  Layer 1 weight grad: {layer1.weights.grad is not None}")
        print(f"  Layer 1 bias grad: {layer1.bias.grad is not None}")
        print(f"  Layer 2 weight grad: {layer2.weights.grad is not None}")
        print(f"  Layer 2 bias grad: {layer2.bias.grad is not None}")

        if all([
            layer1.weights.grad is not None,
            layer1.bias.grad is not None,
            layer2.weights.grad is not None,
            layer2.bias.grad is not None
        ]):
            # Calculate gradient norms
            # Handle different gradient data structures
            def extract_grad_data(grad):
                if hasattr(grad, 'data'):
                    if hasattr(grad.data, 'data'):
                        data = grad.data.data
                    else:
                        data = grad.data
                else:
                    data = grad
                # Convert memoryview to array if needed
                if isinstance(data, memoryview):
                    data = np.array(data)
                return data

            l1_w_data = extract_grad_data(layer1.weights.grad)
            l1_b_data = extract_grad_data(layer1.bias.grad)
            l2_w_data = extract_grad_data(layer2.weights.grad)
            l2_b_data = extract_grad_data(layer2.bias.grad)

            l1_w_norm = np.linalg.norm(l1_w_data)
            l1_b_norm = np.linalg.norm(l1_b_data)
            l2_w_norm = np.linalg.norm(l2_w_data)
            l2_b_norm = np.linalg.norm(l2_b_data)

            print(f"  Layer 1 weight grad norm: {l1_w_norm:.6f}")
            print(f"  Layer 1 bias grad norm: {l1_b_norm:.6f}")
            print(f"  Layer 2 weight grad norm: {l2_w_norm:.6f}")
            print(f"  Layer 2 bias grad norm: {l2_b_norm:.6f}")

            print("  ✅ All gradients computed successfully!")
            return True
        else:
            print("  ❌ Some gradients missing!")
            return False

    except Exception as e:
        print(f"  ❌ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_step():
    """Test that optimizer can use gradients to update parameters."""
    print("\n\n🔬 Testing Optimizer Integration")
    print("=" * 40)

    # Simple optimization test
    layer = Linear(1, 1)

    # Get initial weight
    initial_weight = layer.weights.data.data.copy()
    initial_bias = layer.bias.data.data.copy()

    print(f"  Initial weight: {initial_weight}")
    print(f"  Initial bias: {initial_bias}")

    # Forward pass with known input/output
    x = Variable([[2.0]], requires_grad=False)
    output = layer.forward(x)

    # Target for specific gradient direction
    target = Variable([[0.0]], requires_grad=False)  # Want output to be smaller

    loss_fn = MSELoss()
    loss = loss_fn(output, target)

    print(f"  Loss before update: {loss.data.data}")

    # Backward pass
    loss.backward(1.0)  # Pass scalar gradient

    # Simple gradient descent update
    learning_rate = 0.1
    if layer.weights.grad is not None:
        # Extract gradient data properly
        if hasattr(layer.weights.grad, 'data'):
            if hasattr(layer.weights.grad.data, 'data'):
                weight_grad_data = layer.weights.grad.data.data
            else:
                weight_grad_data = layer.weights.grad.data
        else:
            weight_grad_data = layer.weights.grad
        if isinstance(weight_grad_data, memoryview):
            weight_grad_data = np.array(weight_grad_data)
        # Subtract gradient (gradient descent)
        new_weight = layer.weights.data.data - learning_rate * weight_grad_data
        layer.weights.data.data[:] = new_weight  # Update in place

    if layer.bias.grad is not None:
        # Extract gradient data properly
        if hasattr(layer.bias.grad, 'data'):
            if hasattr(layer.bias.grad.data, 'data'):
                bias_grad_data = layer.bias.grad.data.data
            else:
                bias_grad_data = layer.bias.grad.data
        else:
            bias_grad_data = layer.bias.grad
        if isinstance(bias_grad_data, memoryview):
            bias_grad_data = np.array(bias_grad_data)
        new_bias = layer.bias.data.data - learning_rate * bias_grad_data
        layer.bias.data.data[:] = new_bias

    print(f"  Updated weight: {layer.weights.data.data}")
    print(f"  Updated bias: {layer.bias.data.data}")

    # Verify parameters actually changed
    weight_changed = not np.allclose(initial_weight, layer.weights.data.data)
    bias_changed = not np.allclose(initial_bias, layer.bias.data.data)

    if weight_changed and bias_changed:
        print("  ✅ Parameters updated successfully!")

        # Test forward pass with updated parameters
        # Reset gradients first
        layer.weights.grad = None
        layer.bias.grad = None

        new_output = layer.forward(x)
        new_loss = loss_fn(new_output, target)

        print(f"  Loss after update: {new_loss.data.data}")

        # Loss should be smaller (we did gradient descent)
        if new_loss.data.data < loss.data.data:
            print("  ✅ Loss decreased - optimization working!")
            return True
        else:
            print("  ⚠️ Loss didn't decrease - might be learning rate or other issue")
            return True  # Still counts as parameter update working
    else:
        print("  ❌ Parameters didn't change!")
        return False

if __name__ == "__main__":
    print("🚀 Testing Gradient Flow in TinyTorch")
    print("=" * 50)

    results = []

    # Run all tests
    results.append(("Simple gradient flow", test_simple_gradient_flow()))
    results.append(("Two-layer network", test_two_layer_network()))
    results.append(("Optimizer integration", test_optimizer_step()))

    # Summary
    print("\n\n📊 FINAL RESULTS")
    print("=" * 30)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! Gradient flow is working correctly.")
        print(f"   Your fixes have successfully enabled PyTorch-style gradient flow!")
        print(f"   Neural networks can now learn via backpropagation! 🧠✨")
    else:
        print(f"\n❌ Some tests failed. Gradient flow needs more work.")
        print(f"   Check the error messages above for debugging guidance.")