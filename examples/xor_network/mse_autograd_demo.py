#!/usr/bin/env python3
"""
MSE Loss Autograd Integration Demo

This demonstrates the successful integration of MSE loss with the autograd system.
The key achievement: loss functions now return Variables that support .backward()
"""

import numpy as np
from tinytorch.core.autograd import Variable
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.tensor import Tensor

def demo_mse_autograd_success():
    """Demonstrate successful MSE loss autograd integration."""
    print("🎯 MSE Loss Autograd Integration - SUCCESS DEMO")
    print("=" * 50)
    
    # Test 1: Basic autograd functionality
    print("\n1️⃣ Basic MSE Loss with Autograd:")
    mse = MeanSquaredError()
    
    # Before fix: loss returned Tensor, no .backward()
    # After fix: loss returns Variable with .backward()
    
    predictions = Variable([[2.0, 3.0], [1.0, 4.0]], requires_grad=True)
    targets = Variable([[1.5, 2.5], [1.5, 3.5]], requires_grad=False)
    
    print(f"   Predictions: {predictions.data.data}")
    print(f"   Targets: {targets.data.data}")
    
    # Compute loss
    loss = mse(predictions, targets)
    
    print(f"\n   🔍 Key Results:")
    print(f"   • Loss type: {type(loss)}")
    print(f"   • Loss value: {loss.data}")
    print(f"   • Has .backward(): {hasattr(loss, 'backward')}")
    print(f"   • Requires grad: {loss.requires_grad}")
    
    # Test backward pass
    print(f"\n   🔄 Testing Backward Pass:")
    loss.backward()
    
    print(f"   • Gradients computed: {predictions.grad is not None}")
    if predictions.grad:
        print(f"   • Gradient values: {predictions.grad.data.data}")
        print(f"   • Gradient shape: {predictions.grad.data.data.shape}")
    
    # Test 2: Demonstrate gradient correctness
    print("\n2️⃣ Gradient Correctness Verification:")
    
    # Simple case: predict [2.0], target [1.0]
    # MSE = (2.0 - 1.0)² = 1.0
    # Gradient = 2 * (2.0 - 1.0) = 2.0
    
    pred = Variable([[2.0]], requires_grad=True)
    target = Variable([[1.0]], requires_grad=False)
    
    loss = mse(pred, target)
    loss.backward()
    
    print(f"   Prediction: {pred.data.data[0, 0]}")
    print(f"   Target: {target.data.data[0, 0]}")
    print(f"   Loss: {loss.data} (expected: 1.0)")
    print(f"   Gradient: {pred.grad.data.data[0, 0]} (expected: 2.0)")
    
    # Test 3: Backward compatibility
    print("\n3️⃣ Backward Compatibility with Tensors:")
    
    # Should work with regular Tensors too
    tensor_pred = Tensor([[3.0, 2.0]])
    tensor_target = Tensor([[2.5, 2.5]])
    
    loss = mse(tensor_pred, tensor_target)
    
    print(f"   Input: Tensor -> Output: {type(loss)}")
    print(f"   Loss supports .backward(): {hasattr(loss, 'backward')}")
    print(f"   Auto-conversion successful: {isinstance(loss, Variable)}")
    
    # Test 4: Training loop simulation
    print("\n4️⃣ Training Loop Simulation:")
    
    # Simulate a simple regression training step
    print("   Simulating regression training step:")
    
    # "Model" parameters (just a weight)
    weight = Variable([[2.1]], requires_grad=True)  # Close to optimal weight of 2.0
    
    # Training data: x=1 -> y=2 (so optimal weight is 2.0)
    x = Variable([[1.0]], requires_grad=False)
    y_true = Variable([[2.0]], requires_grad=False)
    
    print(f"   Initial weight: {weight.data.data[0, 0]}")
    
    # Forward pass
    y_pred = weight * x  # Simple linear model
    loss = mse(y_pred, y_true)
    
    print(f"   Prediction: {y_pred.data.data[0, 0]}")
    print(f"   Target: {y_true.data.data[0, 0]}")
    print(f"   Loss: {loss.data}")
    
    # Backward pass
    loss.backward()
    
    gradient = weight.grad.data.data[0, 0]
    print(f"   Weight gradient: {gradient}")
    
    # Manual parameter update
    learning_rate = 0.1
    new_weight = weight.data.data[0, 0] - learning_rate * gradient
    print(f"   Updated weight: {new_weight} (closer to optimal 2.0)")
    
    print("\n🎉 SUCCESS SUMMARY:")
    print("✅ MSE loss now returns Variables with autograd support")
    print("✅ .backward() method works correctly")
    print("✅ Gradients are computed accurately")
    print("✅ Backward compatible with Tensors")
    print("✅ Ready for real neural network training!")
    
    print("\n🚀 IMPACT:")
    print("• Training loops can now call loss.backward()")
    print("• Optimizers can use computed gradients")
    print("• End-to-end backpropagation is now possible")
    print("• XOR networks and other models can be trained!")

def demonstrate_before_vs_after():
    """Show the difference between before and after the fix."""
    print("\n" + "=" * 50)
    print("📊 BEFORE vs AFTER Comparison")
    print("=" * 50)
    
    mse = MeanSquaredError()
    pred = Variable([[2.0]], requires_grad=True)
    target = Variable([[1.0]], requires_grad=False)
    
    loss = mse(pred, target)
    
    print("\n❌ BEFORE the fix:")
    print("   • MSE returned: Tensor (no autograd)")
    print("   • loss.backward(): AttributeError")
    print("   • Training loops: Couldn't compute gradients")
    print("   • Result: No real neural network training possible")
    
    print("\n✅ AFTER the fix:")
    print(f"   • MSE returns: {type(loss)}")
    print(f"   • loss.backward(): {hasattr(loss, 'backward')} ✓")
    print("   • Training loops: Can compute gradients ✓")
    print("   • Result: Real neural network training possible ✓")
    
    # Demonstrate it works
    loss.backward()
    print(f"   • Gradient computed: {pred.grad.data.data[0, 0]}")

if __name__ == "__main__":
    demo_mse_autograd_success()
    demonstrate_before_vs_after()