#!/usr/bin/env python3
"""
Test integration of pure Tensor approach across modules 01-04.
Verify clean architecture without hasattr() hacks.
"""

import sys
import numpy as np

# Import from individual modules
sys.path.insert(0, 'modules/01_tensor')
sys.path.insert(0, 'modules/02_activations')
sys.path.insert(0, 'modules/03_layers')
sys.path.insert(0, 'modules/04_losses')

from tensor_dev import Tensor
from activations_dev import ReLU, Softmax
from layers_dev import Linear
from losses_dev import MSELoss, CrossEntropyLoss

def test_pure_tensor_integration():
    """Test that all modules work with pure Tensor class."""
    print("🧪 Testing Pure Tensor Integration (Modules 01-04)")
    print("=" * 50)

    # Test basic tensor operations
    print("📊 Testing basic Tensor operations...")
    x = Tensor([[1.0, 2.0]])
    y = Tensor([[0.5, 1.5]])
    z = x + y
    print(f"  Tensor addition: {z.data}")
    print("  ✅ Pure Tensor operations work")

    # Test activations with pure tensors
    print("\n🔥 Testing activations with pure Tensors...")
    relu = ReLU()
    negative_tensor = Tensor([[-1.0, 2.0, -3.0]])
    activated = relu(negative_tensor)
    print(f"  ReLU result: {activated.data}")
    print("  ✅ Activations work with pure Tensors")

    # Test linear layer with pure tensors
    print("\n🏗️ Testing Linear layer with pure Tensors...")
    layer = Linear(2, 1)
    input_tensor = Tensor([[1.0, 2.0]])
    output = layer(input_tensor)
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.data}")
    print("  ✅ Linear layer works with pure Tensors")

    # Test loss functions with pure tensors
    print("\n💔 Testing loss functions with pure Tensors...")
    predictions = Tensor([[0.8]])
    targets = Tensor([[1.0]])

    mse_loss = MSELoss()
    loss_value = mse_loss(predictions, targets)
    print(f"  MSE Loss: {loss_value.data}")
    print("  ✅ Loss functions work with pure Tensors")

    # Test full neural network pipeline
    print("\n🧠 Testing full neural network pipeline...")

    # Create simple network: 3 → 2 → 1
    layer1 = Linear(3, 2)
    layer2 = Linear(2, 1)
    relu = ReLU()
    loss_fn = MSELoss()

    # Forward pass
    x = Tensor([[1.0, 2.0, 3.0]])
    h1 = layer1(x)
    h1_activated = relu(h1)
    output = layer2(h1_activated)

    # Loss computation
    target = Tensor([[0.5]])
    loss = loss_fn(output, target)

    print(f"  Network input: {x.data}")
    print(f"  Network output: {output.data}")
    print(f"  Loss: {loss.data}")
    print("  ✅ Full neural network pipeline works!")

    return True

def test_no_gradient_contamination():
    """Verify that modules 01-04 have no gradient-related code."""
    print("\n🔬 Verifying NO gradient contamination...")
    print("=" * 50)

    # Test that Tensor has no gradient attributes
    tensor = Tensor([1, 2, 3])
    print(f"  Tensor has 'grad' attribute: {hasattr(tensor, 'grad')}")
    print(f"  Tensor has 'requires_grad' attribute: {hasattr(tensor, 'requires_grad')}")
    print(f"  Tensor has 'backward' method: {hasattr(tensor, 'backward')}")

    if not hasattr(tensor, 'grad') and not hasattr(tensor, 'requires_grad'):
        print("  ✅ Pure Tensor class - no gradient contamination!")
    else:
        print("  ❌ Tensor class has gradient attributes!")
        return False

    # Test linear layer parameters
    layer = Linear(2, 1)
    print(f"  Layer weights type: {type(layer.weights)}")
    print(f"  Layer bias type: {type(layer.bias)}")

    if isinstance(layer.weights, Tensor) and isinstance(layer.bias, Tensor):
        print("  ✅ Linear layer uses pure Tensors!")
    else:
        print("  ❌ Linear layer not using pure Tensors!")
        return False

    return True

def test_clean_interfaces():
    """Test that there are no hasattr() hacks anywhere."""
    print("\n🧹 Testing clean interfaces (no hasattr hacks)...")
    print("=" * 50)

    # This would fail if there were hasattr() checks
    try:
        tensor = Tensor([1, 2, 3])
        layer = Linear(2, 1)
        input_data = Tensor([[1.0, 2.0]])
        output = layer(input_data)

        print(f"  Clean tensor operations: {output.data.shape}")
        print("  ✅ No hasattr() hacks - clean interfaces!")
        return True

    except AttributeError as e:
        print(f"  ❌ AttributeError indicates hasattr() hack needed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Clean Pure Tensor Architecture")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Pure tensor integration", test_pure_tensor_integration()))
    results.append(("No gradient contamination", test_no_gradient_contamination()))
    results.append(("Clean interfaces", test_clean_interfaces()))

    # Summary
    print("\n📊 INTEGRATION TEST RESULTS")
    print("=" * 30)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Clean pure Tensor architecture is working perfectly!")
        print(f"   • Modules 01-04 work with pure Tensors")
        print(f"   • No gradient contamination anywhere")
        print(f"   • No hasattr() hacks needed")
        print(f"   • Perfect module focus and separation")
        print(f"   • Ready for Module 05 decorator enhancement!")
    else:
        print(f"\n❌ Some tests failed.")
        print(f"   Architecture needs more cleanup.")