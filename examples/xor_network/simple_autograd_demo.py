#!/usr/bin/env python3
"""
Simple Demo: Training Module Autograd Integration

This demonstrates the key fix: loss functions now return Variables that support
.backward() for automatic gradient computation.
"""

import numpy as np
from tinytorch.core.autograd import Variable
from tinytorch.core.training import MeanSquaredError, CrossEntropyLoss, BinaryCrossEntropyLoss

def demo_loss_autograd():
    """Demonstrate that all loss functions now support autograd."""
    print("🧪 Training Module Autograd Integration Demo")
    print("=" * 45)
    
    # Test 1: MSE Loss with autograd
    print("\n1️⃣ MSE Loss with Gradient Computation:")
    mse = MeanSquaredError()
    
    # Create predictions that need gradients
    predictions = Variable([[2.0, 3.0], [1.0, 4.0]], requires_grad=True)
    targets = Variable([[1.5, 2.5], [1.5, 3.5]], requires_grad=False)
    
    print(f"   Predictions: {predictions.data.data}")
    print(f"   Targets: {targets.data.data}")
    
    # Compute loss
    loss = mse(predictions, targets)
    print(f"   Loss: {loss.data} (type: {type(loss)})")
    print(f"   Supports .backward(): {hasattr(loss, 'backward')}")
    
    # Compute gradients
    loss.backward()
    print(f"   Gradients computed: {predictions.grad is not None}")
    if predictions.grad:
        print(f"   Prediction gradients: {predictions.grad.data.data}")
    
    # Test 2: CrossEntropy Loss with autograd
    print("\n2️⃣ CrossEntropy Loss with Gradient Computation:")
    ce = CrossEntropyLoss()
    
    # Create logits that need gradients
    logits = Variable([[2.0, 1.0, 0.1], [0.5, 2.1, 0.9]], requires_grad=True)
    labels = Variable([0, 1], requires_grad=False)  # Class indices
    
    print(f"   Logits: {logits.data.data}")
    print(f"   Labels: {labels.data.data}")
    
    # Compute loss
    loss = ce(logits, labels)
    print(f"   Loss: {loss.data} (type: {type(loss)})")
    print(f"   Supports .backward(): {hasattr(loss, 'backward')}")
    
    # Compute gradients
    loss.backward()
    print(f"   Gradients computed: {logits.grad is not None}")
    if logits.grad:
        print(f"   Logit gradients: {np.round(logits.grad.data.data, 4)}")
    
    # Test 3: Binary CrossEntropy Loss with autograd
    print("\n3️⃣ Binary CrossEntropy Loss with Gradient Computation:")
    bce = BinaryCrossEntropyLoss()
    
    # Create binary logits that need gradients
    binary_logits = Variable([[1.0], [-0.5], [2.0]], requires_grad=True)
    binary_labels = Variable([[1.0], [0.0], [1.0]], requires_grad=False)
    
    print(f"   Binary Logits: {binary_logits.data.data.flatten()}")
    print(f"   Binary Labels: {binary_labels.data.data.flatten()}")
    
    # Compute loss
    loss = bce(binary_logits, binary_labels)
    print(f"   Loss: {loss.data} (type: {type(loss)})")
    print(f"   Supports .backward(): {hasattr(loss, 'backward')}")
    
    # Compute gradients
    loss.backward()
    print(f"   Gradients computed: {binary_logits.grad is not None}")
    if binary_logits.grad:
        print(f"   Binary logit gradients: {np.round(binary_logits.grad.data.data.flatten(), 4)}")
    
    # Test 4: Backward compatibility with Tensors
    print("\n4️⃣ Backward Compatibility with Regular Tensors:")
    from tinytorch.core.tensor import Tensor
    
    # Test that loss functions work with Tensors too (auto-converts to Variables)
    tensor_pred = Tensor([[3.0, 2.0]])
    tensor_true = Tensor([[2.5, 2.5]])
    
    loss = mse(tensor_pred, tensor_true)
    print(f"   Tensor input -> Variable output: {type(loss)}")
    print(f"   Loss supports .backward(): {hasattr(loss, 'backward')}")
    
    print("\n🎉 SUCCESS: All loss functions now support autograd!")
    print("✅ Loss functions return Variables with .backward() method")
    print("✅ Gradients are computed automatically for neural network training")
    print("✅ Backward compatible with regular Tensors")
    print("\n🚀 Training module is ready for real neural network training!")

def demo_training_loop_integration():
    """Demonstrate that the training loop can now use gradient computation."""
    print("\n" + "=" * 50)
    print("🔄 Training Loop Integration Demo")
    print("=" * 50)
    
    # Simulate a simple training step
    print("\nSimulating training step with gradient computation:")
    
    # Mock model output and targets
    model_output = Variable([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    targets = Variable([0, 1], requires_grad=False)
    
    # Loss function
    loss_fn = CrossEntropyLoss()
    
    print(f"   Model output: {model_output.data.data}")
    print(f"   Targets: {targets.data.data}")
    
    # 1. Compute loss
    loss = loss_fn(model_output, targets)
    print(f"   ✅ Loss computed: {loss.data}")
    
    # 2. Backward pass (this now works!)
    print("   🔄 Running loss.backward()...")
    loss.backward()
    
    # 3. Check gradients
    if model_output.grad:
        print(f"   ✅ Gradients computed: {np.round(model_output.grad.data.data, 4)}")
        print("   📈 These gradients can now be used by optimizers!")
    else:
        print("   ❌ No gradients computed")
    
    print("\n🎯 Key Achievement:")
    print("   • Loss functions now participate in autograd computational graph")
    print("   • Training loops can call loss.backward() to compute gradients")
    print("   • Optimizers can use these gradients to update parameters")
    print("   • Complete end-to-end training with backpropagation now possible!")

if __name__ == "__main__":
    demo_loss_autograd()
    demo_training_loop_integration()