#!/usr/bin/env python3
"""
Comprehensive tests for autograd gradient computation and shapes.

These tests catch the real bugs we discovered during CIFAR-10 training.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.training import MeanSquaredError, CrossEntropyLoss
from tinytorch.core.optimizers import SGD


def test_gradient_shapes():
    """Test that gradients have correct shapes after backward pass."""
    print("=" * 60)
    print("TEST: Gradient Shapes")
    print("=" * 60)
    
    # Create a simple layer
    layer = Dense(10, 5)
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    print(f"Weight shape: {layer.weights.shape}")
    print(f"Bias shape: {layer.bias.shape}")
    
    # Different batch sizes to test
    batch_sizes = [1, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Reset gradients
        layer.weights.grad = None
        layer.bias.grad = None
        
        # Forward pass
        x = Variable(Tensor(np.random.randn(batch_size, 10)), requires_grad=True)
        y = layer(x)
        
        # Create loss
        target = Variable(Tensor(np.random.randn(batch_size, 5)), requires_grad=False)
        loss_fn = MeanSquaredError()
        loss = loss_fn(y, target)
        
        # Backward pass
        if hasattr(loss, 'backward'):
            loss.backward()
            print("✅ Backward pass completed")
        else:
            print("❌ Loss doesn't have backward method")
            return False
        
        # Check gradient shapes
        success = True
        
        # Weight gradient
        if layer.weights.grad is not None:
            if hasattr(layer.weights.grad.data, 'data'):
                weight_grad_shape = layer.weights.grad.data.data.shape
            else:
                weight_grad_shape = layer.weights.grad.data.shape
            
            if weight_grad_shape == layer.weights.shape:
                print(f"✅ Weight gradient shape correct: {weight_grad_shape}")
            else:
                print(f"❌ Weight gradient shape WRONG: {weight_grad_shape} != {layer.weights.shape}")
                success = False
        else:
            print("❌ No weight gradient!")
            success = False
        
        # Bias gradient
        if layer.bias.grad is not None:
            if hasattr(layer.bias.grad.data, 'data'):
                bias_grad_data = layer.bias.grad.data.data
            else:
                bias_grad_data = layer.bias.grad.data
            
            # Check if bias gradient needs aggregation
            if len(bias_grad_data.shape) == 2:
                print(f"⚠️  Bias gradient has batch dimension: {bias_grad_data.shape}")
                # Should be summed over batch
                correct_shape = (bias_grad_data.shape[1],)
                print(f"   Should be: {correct_shape}")
                success = False
            elif bias_grad_data.shape == layer.bias.shape:
                print(f"✅ Bias gradient shape correct: {bias_grad_data.shape}")
            else:
                print(f"❌ Bias gradient shape WRONG: {bias_grad_data.shape} != {layer.bias.shape}")
                success = False
        else:
            print("❌ No bias gradient!")
            success = False
        
        if not success:
            print("\n❌ FAILED: Gradient shapes are incorrect!")
            return False
    
    print("\n✅ PASSED: All gradient shapes correct!")
    return True


def test_bias_gradient_aggregation():
    """Test that bias gradients are correctly aggregated over batch dimension."""
    print("\n" + "=" * 60)
    print("TEST: Bias Gradient Aggregation")
    print("=" * 60)
    
    # Simple 2-layer network
    fc1 = Dense(10, 5)
    fc2 = Dense(5, 3)
    
    # Make trainable
    fc1.weights = Variable(fc1.weights, requires_grad=True)
    fc1.bias = Variable(fc1.bias, requires_grad=True)
    fc2.weights = Variable(fc2.weights, requires_grad=True)
    fc2.bias = Variable(fc2.bias, requires_grad=True)
    
    # Forward with batch
    batch_size = 4
    x = Variable(Tensor(np.random.randn(batch_size, 10)), requires_grad=True)
    
    # Network forward
    h = fc1(x)
    relu = ReLU()
    h = relu(h)
    y = fc2(h)
    
    # Loss
    target = Variable(Tensor(np.random.randn(batch_size, 3)), requires_grad=False)
    loss_fn = MeanSquaredError()
    loss = loss_fn(y, target)
    
    # Backward
    loss.backward()
    
    # Check all bias gradients
    success = True
    
    for layer_name, layer in [("fc1", fc1), ("fc2", fc2)]:
        if layer.bias.grad is not None:
            if hasattr(layer.bias.grad.data, 'data'):
                grad_shape = layer.bias.grad.data.data.shape
            else:
                grad_shape = layer.bias.grad.data.shape
            
            expected_shape = layer.bias.shape
            
            if grad_shape == expected_shape:
                print(f"✅ {layer_name}.bias gradient shape: {grad_shape}")
            else:
                print(f"❌ {layer_name}.bias gradient shape WRONG: {grad_shape} != {expected_shape}")
                if len(grad_shape) == 2:
                    print(f"   Gradient has batch dimension that wasn't aggregated!")
                success = False
        else:
            print(f"❌ {layer_name}.bias has no gradient!")
            success = False
    
    if success:
        print("\n✅ PASSED: Bias gradients correctly aggregated!")
    else:
        print("\n❌ FAILED: Bias gradient aggregation is broken!")
    
    return success


def test_optimizer_with_gradients():
    """Test that optimizer can update parameters with computed gradients."""
    print("\n" + "=" * 60)
    print("TEST: Optimizer Parameter Updates")
    print("=" * 60)
    
    # Create layer
    layer = Dense(10, 5)
    layer.weights = Variable(layer.weights, requires_grad=True)
    layer.bias = Variable(layer.bias, requires_grad=True)
    
    # Store initial values
    initial_weights = np.copy(layer.weights.data._data)
    initial_bias = np.copy(layer.bias.data._data)
    
    # Create optimizer
    optimizer = SGD([layer.weights, layer.bias], learning_rate=0.1)
    
    # Forward pass
    x = Variable(Tensor(np.random.randn(8, 10)), requires_grad=True)
    y = layer(x)
    
    # Loss
    target = Variable(Tensor(np.random.randn(8, 5)), requires_grad=False)
    loss_fn = MeanSquaredError()
    loss = loss_fn(y, target)
    
    print(f"Initial loss: {loss.data}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    has_weight_grad = layer.weights.grad is not None
    has_bias_grad = layer.bias.grad is not None
    
    print(f"Weight gradient exists: {has_weight_grad}")
    print(f"Bias gradient exists: {has_bias_grad}")
    
    if not (has_weight_grad and has_bias_grad):
        print("❌ FAILED: No gradients computed!")
        return False
    
    # Try to step
    try:
        optimizer.step()
        print("✅ Optimizer step succeeded")
    except Exception as e:
        print(f"❌ Optimizer step failed: {e}")
        
        # Debug the shapes
        if hasattr(layer.bias.grad.data, 'data'):
            bias_grad_shape = layer.bias.grad.data.data.shape
        else:
            bias_grad_shape = layer.bias.grad.data.shape
        
        print(f"   Bias grad shape: {bias_grad_shape}")
        print(f"   Bias param shape: {layer.bias.shape}")
        return False
    
    # Check if parameters changed
    weights_changed = not np.allclose(initial_weights, layer.weights.data._data)
    bias_changed = not np.allclose(initial_bias, layer.bias.data._data)
    
    print(f"Weights updated: {weights_changed}")
    print(f"Bias updated: {bias_changed}")
    
    if weights_changed and bias_changed:
        print("\n✅ PASSED: Optimizer successfully updates parameters!")
        return True
    else:
        print("\n❌ FAILED: Parameters didn't update!")
        return False


def test_learning_happens():
    """Integration test: Train a small model and verify loss decreases."""
    print("\n" + "=" * 60)
    print("TEST: End-to-End Learning")
    print("=" * 60)
    
    # Simple model
    fc1 = Dense(10, 5)
    fc2 = Dense(5, 2)
    
    # Initialize with reasonable values
    for layer in [fc1, fc2]:
        fan_in = layer.weights.shape[0]
        std = np.sqrt(2.0 / fan_in)
        layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
        layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
        
        layer.weights = Variable(layer.weights, requires_grad=True)
        layer.bias = Variable(layer.bias, requires_grad=True)
    
    # Optimizer
    params = [fc1.weights, fc1.bias, fc2.weights, fc2.bias]
    optimizer = SGD(params, learning_rate=0.1)
    
    # Training data (simple XOR-like problem)
    # Create 4 samples with 10 features each
    X = np.random.randn(4, 10).astype(np.float32)
    # Simple binary targets for 2 classes
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    
    # Track losses
    losses = []
    
    print("Training for 20 iterations...")
    for i in range(20):
        # Forward
        x_var = Variable(Tensor(X), requires_grad=True)
        h = fc1(x_var)
        relu = ReLU()
        h = relu(h)
        out = fc2(h)
        
        # Loss
        y_var = Variable(Tensor(y), requires_grad=False)
        loss_fn = MeanSquaredError()
        loss = loss_fn(out, y_var)
        
        if hasattr(loss.data, 'data'):
            loss_val = float(loss.data.data)
        else:
            loss_val = float(loss.data)
        losses.append(loss_val)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Fix bias gradients if needed
        for layer in [fc1, fc2]:
            if layer.bias.grad is not None:
                if hasattr(layer.bias.grad.data, 'data'):
                    grad = layer.bias.grad.data.data
                else:
                    grad = layer.bias.grad.data
                
                if len(grad.shape) == 2:
                    # Sum over batch dimension
                    layer.bias.grad = Variable(Tensor(np.sum(grad, axis=0)))
        
        # Update
        optimizer.step()
        
        if i % 5 == 0:
            print(f"  Iteration {i:2d}: Loss = {loss_val:.4f}")
    
    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Improvement:  {initial_loss - final_loss:.4f}")
    
    if final_loss < initial_loss * 0.9:  # At least 10% improvement
        print("\n✅ PASSED: Model is learning! Loss decreased significantly.")
        return True
    else:
        print("\n❌ FAILED: Model is not learning! Loss didn't decrease enough.")
        return False


def test_crossentropy_gradients():
    """Test CrossEntropy loss gradient computation."""
    print("\n" + "=" * 60)
    print("TEST: CrossEntropy Gradients")
    print("=" * 60)
    
    # Create logits
    batch_size = 4
    num_classes = 3
    logits = Variable(Tensor(np.random.randn(batch_size, num_classes)), requires_grad=True)
    
    # Create labels
    labels = Variable(Tensor(np.array([0, 1, 2, 1])), requires_grad=False)
    
    # Compute loss
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    print(f"Loss value: {loss.data}")
    print(f"Loss has backward: {hasattr(loss, 'backward')}")
    
    if not hasattr(loss, 'backward'):
        print("❌ FAILED: CrossEntropy loss doesn't support backward!")
        return False
    
    # Backward
    loss.backward()
    
    # Check if logits got gradients
    if logits.grad is not None:
        print("✅ Logits received gradients")
        if hasattr(logits.grad.data, 'data'):
            grad_shape = logits.grad.data.data.shape
        else:
            grad_shape = logits.grad.data.shape
        
        if grad_shape == (batch_size, num_classes):
            print(f"✅ Gradient shape correct: {grad_shape}")
            return True
        else:
            print(f"❌ Gradient shape wrong: {grad_shape}")
            return False
    else:
        print("❌ FAILED: No gradients computed for logits!")
        return False


def run_all_tests():
    """Run all autograd tests."""
    print("=" * 60)
    print("AUTOGRAD GRADIENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Gradient Shapes", test_gradient_shapes),
        ("Bias Gradient Aggregation", test_bias_gradient_aggregation),
        ("Optimizer Updates", test_optimizer_with_gradients),
        ("CrossEntropy Gradients", test_crossentropy_gradients),
        ("End-to-End Learning", test_learning_happens),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some tests failed! The autograd system has bugs.")
        print("The bias gradient aggregation issue needs to be fixed.")
    else:
        print("\n🎉 All tests passed! The autograd system is working correctly.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()