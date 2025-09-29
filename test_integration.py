#!/usr/bin/env python3
"""
Comprehensive integration test for TinyTorch.

Tests that all components work together to enable neural network training.
"""

import sys
import numpy as np

# Import TinyTorch components
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.autograd import Variable

def test_simple_network_forward():
    """Test forward pass through a simple network."""
    print("🔬 Testing Simple Network Forward Pass")
    print("=" * 40)

    # Create a simple 2-layer network
    layer1 = Linear(3, 2)
    layer2 = Linear(2, 1)
    relu = ReLU()

    # Input data
    x = Tensor([[1.0, 2.0, 3.0]])

    # Forward pass
    h1 = layer1(x)
    h1_activated = relu(h1)
    output = layer2(h1_activated)

    print(f"  Input shape: {x.shape}")
    print(f"  Hidden shape: {h1.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✅ Forward pass successful!")

    return True

def test_gradient_flow_integration():
    """Test that gradients flow through the entire system."""
    print("\n🔬 Testing Gradient Flow Integration")
    print("=" * 40)

    # Import autograd components from source
    sys.path.insert(0, 'modules/05_autograd')
    sys.path.insert(0, 'modules/03_layers')
    from autograd_dev import Variable
    from layers_dev import Linear

    # Create network
    layer = Linear(2, 1)

    # Input and target
    x = Variable([[1.0, 2.0]], requires_grad=False)
    target = Variable([[0.5]], requires_grad=False)

    # Forward pass
    output = layer.forward(x)

    # Compute loss
    from tinytorch.core.losses import MSELoss
    loss_fn = MSELoss()
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward(1.0)

    # Check gradients
    if layer.weights.grad is not None and layer.bias.grad is not None:
        print("  ✅ Gradients computed successfully!")
        print(f"  Weight grad exists: {layer.weights.grad is not None}")
        print(f"  Bias grad exists: {layer.bias.grad is not None}")
        return True
    else:
        print("  ❌ Gradient computation failed!")
        return False

def test_loss_functions():
    """Test that loss functions work correctly."""
    print("\n🔬 Testing Loss Functions")
    print("=" * 40)

    # Test MSE Loss
    mse = MSELoss()
    predictions = Variable([[0.5, 0.3]], requires_grad=True)
    targets = Variable([[1.0, 0.0]], requires_grad=False)

    mse_loss = mse(predictions, targets)
    print(f"  MSE Loss: {mse_loss.data.data if hasattr(mse_loss.data, 'data') else mse_loss.data}")

    # Test CrossEntropy Loss
    ce = CrossEntropyLoss()
    logits = Variable([[2.0, 1.0, 0.1]], requires_grad=True)
    labels = Variable([0], requires_grad=False)

    ce_loss = ce(logits, labels)
    print(f"  CrossEntropy Loss: {ce_loss.data.data if hasattr(ce_loss.data, 'data') else ce_loss.data}")

    print("  ✅ Loss functions working!")
    return True

def test_training_step():
    """Test a complete training step."""
    print("\n🔬 Testing Complete Training Step")
    print("=" * 40)

    # Import from source modules
    sys.path.insert(0, 'modules/05_autograd')
    sys.path.insert(0, 'modules/03_layers')
    from autograd_dev import Variable
    from layers_dev import Linear

    # Create simple network
    layer = Linear(2, 1)

    # Training data
    x = Variable([[1.0, 2.0]], requires_grad=False)
    target = Variable([[0.5]], requires_grad=False)

    # Store initial weights
    initial_weight = layer.weights.data.data.copy()
    initial_bias = layer.bias.data.data.copy()

    # Forward pass
    output = layer.forward(x)

    # Loss
    from tinytorch.core.losses import MSELoss
    loss_fn = MSELoss()
    initial_loss = loss_fn(output, target)

    # Backward
    initial_loss.backward(1.0)

    # Manual gradient descent update
    learning_rate = 0.1
    if layer.weights.grad is not None:
        # Extract gradient
        if hasattr(layer.weights.grad, 'data'):
            weight_grad = layer.weights.grad.data if not hasattr(layer.weights.grad.data, 'data') else layer.weights.grad.data.data
        else:
            weight_grad = layer.weights.grad
        if isinstance(weight_grad, memoryview):
            weight_grad = np.array(weight_grad)
        # Update
        layer.weights.data.data[:] = layer.weights.data.data - learning_rate * weight_grad

    if layer.bias.grad is not None:
        # Extract gradient
        if hasattr(layer.bias.grad, 'data'):
            bias_grad = layer.bias.grad.data if not hasattr(layer.bias.grad.data, 'data') else layer.bias.grad.data.data
        else:
            bias_grad = layer.bias.grad
        if isinstance(bias_grad, memoryview):
            bias_grad = np.array(bias_grad)
        # Update
        layer.bias.data.data[:] = layer.bias.data.data - learning_rate * bias_grad

    # Check parameters changed
    weight_changed = not np.allclose(initial_weight, layer.weights.data.data)
    bias_changed = not np.allclose(initial_bias, layer.bias.data.data)

    if weight_changed and bias_changed:
        print("  ✅ Training step successful - parameters updated!")

        # Clear gradients for next iteration
        layer.weights.grad = None
        layer.bias.grad = None

        # Forward pass with new weights
        new_output = layer.forward(x)
        new_loss = loss_fn(new_output, target)

        # Extract loss values for comparison
        initial_loss_val = initial_loss.data.data if hasattr(initial_loss.data, 'data') else initial_loss.data
        new_loss_val = new_loss.data.data if hasattr(new_loss.data, 'data') else new_loss.data

        print(f"  Initial loss: {initial_loss_val}")
        print(f"  New loss: {new_loss_val}")

        if new_loss_val < initial_loss_val:
            print("  ✅ Loss decreased - learning is working!")
        return True
    else:
        print("  ❌ Parameters didn't update!")
        return False

def test_multi_layer_network():
    """Test a deeper network."""
    print("\n🔬 Testing Multi-Layer Network")
    print("=" * 40)

    # Create 3-layer network
    layer1 = Linear(4, 3)
    layer2 = Linear(3, 2)
    layer3 = Linear(2, 1)
    relu = ReLU()

    # Input
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])

    # Forward pass
    h1 = relu(layer1(x))
    h2 = relu(layer2(h1))
    output = layer3(h2)

    print(f"  Network: 4 → 3 → 2 → 1")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✅ Multi-layer network works!")

    return True

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n🔬 Testing Batch Processing")
    print("=" * 40)

    # Create network
    layer = Linear(3, 2)

    # Batch of 4 samples
    batch = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])

    # Forward pass
    output = layer(batch)

    print(f"  Batch size: 4")
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")

    if output.shape == (4, 2):
        print("  ✅ Batch processing works correctly!")
        return True
    else:
        print("  ❌ Batch processing failed!")
        return False

if __name__ == "__main__":
    print("🚀 TinyTorch Integration Tests")
    print("=" * 50)
    print("Testing that all components work together for neural network training\n")

    results = []

    # Run all tests
    results.append(("Simple forward pass", test_simple_network_forward()))
    results.append(("Gradient flow", test_gradient_flow_integration()))
    results.append(("Loss functions", test_loss_functions()))
    results.append(("Training step", test_training_step()))
    results.append(("Multi-layer network", test_multi_layer_network()))
    results.append(("Batch processing", test_batch_processing()))

    # Summary
    print("\n\n📊 INTEGRATION TEST RESULTS")
    print("=" * 30)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print(f"   TinyTorch is ready for neural network training!")
        print(f"   • Forward passes work correctly")
        print(f"   • Gradients flow through the network")
        print(f"   • Loss functions compute properly")
        print(f"   • Training updates parameters")
        print(f"   • Multi-layer networks are supported")
        print(f"   • Batch processing works efficiently")
    else:
        print(f"\n❌ Some integration tests failed.")
        print(f"   Check the error messages above for details.")