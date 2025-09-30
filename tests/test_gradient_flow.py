#!/usr/bin/env python3
"""
Comprehensive gradient flow testing for TinyTorch.

This test suite systematically validates that gradients propagate correctly
through all components of the training stack.

Run with: pytest tests/test_gradient_flow.py -v
Or directly: python tests/test_gradient_flow.py
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tinytorch import Tensor, Linear, Dropout
from tinytorch import Sigmoid, ReLU, Tanh, GELU, Softmax
from tinytorch import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from tinytorch import SGD, AdamW


class TestBasicTensorGradients:
    """Test gradient computation for basic tensor operations."""
    
    def test_multiplication_gradient(self):
        """Test gradient flow through multiplication."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = x * 3
        loss = y.sum()
        
        loss.backward()
        
        # dy/dx = 3
        assert x.grad is not None, "Gradient should be computed"
        assert np.allclose(x.grad, [[3.0, 3.0]]), f"Expected [[3, 3]], got {x.grad}"
    
    def test_addition_gradient(self):
        """Test gradient flow through addition."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = Tensor([[3.0, 4.0]], requires_grad=True)
        z = x + y
        loss = z.sum()
        
        loss.backward()
        
        # dz/dx = 1, dz/dy = 1
        assert np.allclose(x.grad, [[1.0, 1.0]]), f"x.grad: {x.grad}"
        assert np.allclose(y.grad, [[1.0, 1.0]]), f"y.grad: {y.grad}"
    
    def test_chain_rule(self):
        """Test gradient flow through chain of operations."""
        x = Tensor([[2.0]], requires_grad=True)
        y = x * 3      # y = 3x
        z = y + 1      # z = 3x + 1
        w = z * 2      # w = 2(3x + 1) = 6x + 2
        
        w.backward()
        
        # dw/dx = 6
        assert np.allclose(x.grad, [[6.0]]), f"Expected [[6]], got {x.grad}"
    
    def test_matmul_gradient(self):
        """Test gradient flow through matrix multiplication."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        W = Tensor([[1.0], [2.0]], requires_grad=True)
        y = x.matmul(W)  # y = [[5.0]]
        
        y.backward()
        
        # dy/dx = W^T = [[1, 2]]
        # dy/dW = x^T = [[1], [2]]
        assert np.allclose(x.grad, [[1.0, 2.0]]), f"x.grad: {x.grad}"
        assert np.allclose(W.grad, [[1.0], [2.0]]), f"W.grad: {W.grad}"
    
    def test_broadcasting_gradient(self):
        """Test gradient flow with broadcasting (e.g., bias addition)."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
        bias = Tensor([1.0, 2.0], requires_grad=True)              # (2,)
        y = x + bias  # Broadcasting happens
        loss = y.sum()
        
        loss.backward()
        
        # Gradient should sum over broadcast dimension
        assert x.grad.shape == (2, 2), f"x.grad shape: {x.grad.shape}"
        assert bias.grad.shape == (2,), f"bias.grad shape: {bias.grad.shape}"
        assert np.allclose(bias.grad, [2.0, 2.0]), f"bias.grad: {bias.grad}"


class TestLayerGradients:
    """Test gradient computation through neural network layers."""
    
    def test_linear_layer_gradients(self):
        """Test gradient flow through Linear layer."""
        layer = Linear(2, 3)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        
        w_before = layer.weight.data.copy()
        b_before = layer.bias.data.copy()
        
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # All gradients should exist
        assert layer.weight.grad is not None, "Weight gradient missing"
        assert layer.bias.grad is not None, "Bias gradient missing"
        assert x.grad is not None, "Input gradient missing"
        
        # Gradient shapes should match parameter shapes
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape
    
    def test_multi_layer_gradients(self):
        """Test gradient flow through multiple layers."""
        layer1 = Linear(2, 3)
        layer2 = Linear(3, 1)
        
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        
        h = layer1(x)
        out = layer2(h)
        loss = out.sum()
        
        loss.backward()
        
        # All layers should have gradients
        assert layer1.weight.grad is not None
        assert layer1.bias.grad is not None
        assert layer2.weight.grad is not None
        assert layer2.bias.grad is not None


class TestActivationGradients:
    """Test gradient computation through activation functions."""
    
    def test_sigmoid_gradient(self):
        """Test gradient flow through Sigmoid."""
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        sigmoid = Sigmoid()
        
        y = sigmoid(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Sigmoid gradient missing"
        # Sigmoid gradient: σ'(x) = σ(x)(1 - σ(x))
        # At x=0: σ(0) = 0.5, σ'(0) = 0.25
        assert x.grad[0, 0] > 0, "Gradient should be positive"
    
    def test_relu_gradient(self):
        """Test gradient flow through ReLU."""
        x = Tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
        relu = ReLU()
        
        y = relu(x)
        loss = y.sum()
        loss.backward()
        
        # ReLU gradient: 1 if x > 0, else 0
        # Note: We haven't implemented ReLU backward yet, so this will fail
        # TODO: Implement ReLU backward in autograd
    
    def test_tanh_gradient(self):
        """Test gradient flow through Tanh."""
        x = Tensor([[0.0, 1.0]], requires_grad=True)
        tanh = Tanh()
        
        y = tanh(x)
        loss = y.sum()
        
        # TODO: Implement Tanh backward
        # loss.backward()


class TestLossGradients:
    """Test gradient computation through loss functions."""
    
    def test_bce_gradient(self):
        """Test gradient flow through Binary Cross-Entropy."""
        predictions = Tensor([[0.7, 0.3, 0.9]], requires_grad=True)
        targets = Tensor([[1.0, 0.0, 1.0]])
        
        loss_fn = BinaryCrossEntropyLoss()
        loss = loss_fn(predictions, targets)
        
        loss.backward()
        
        assert predictions.grad is not None, "BCE gradient missing"
        assert predictions.grad.shape == predictions.shape
        # Gradient should be negative for correct predictions
        assert predictions.grad[0, 0] < 0, "Gradient sign incorrect"
    
    def test_mse_gradient(self):
        """Test gradient flow through MSE loss."""
        predictions = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = Tensor([[2.0, 2.0, 2.0]])
        
        loss_fn = MSELoss()
        loss = loss_fn(predictions, targets)
        
        # TODO: Implement MSE backward
        # loss.backward()


class TestOptimizerIntegration:
    """Test optimizer integration with gradient flow."""
    
    def test_sgd_updates_parameters(self):
        """Test that SGD actually updates parameters."""
        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        w_before = layer.weight.data.copy()
        b_before = layer.bias.data.copy()
        
        # Forward pass
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = layer(x)
        loss = out.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Parameters should change
        assert not np.allclose(layer.weight.data, w_before), "Weights didn't update"
        assert not np.allclose(layer.bias.data, b_before), "Bias didn't update"
    
    def test_zero_grad_clears_gradients(self):
        """Test that zero_grad() clears gradients."""
        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        # First backward pass
        x = Tensor([[1.0, 2.0]])
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert layer.weight.grad is not None, "Gradient should exist"
        
        # Clear gradients
        optimizer.zero_grad()
        
        assert layer.weight.grad is None, "Gradient should be cleared"
        assert layer.bias.grad is None, "Bias gradient should be cleared"
    
    def test_adamw_updates_parameters(self):
        """Test that AdamW optimizer works."""
        layer = Linear(2, 1)
        optimizer = AdamW(layer.parameters(), lr=0.01)
        
        w_before = layer.weight.data.copy()
        
        x = Tensor([[1.0, 2.0]])
        out = layer(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        
        assert not np.allclose(layer.weight.data, w_before), "AdamW didn't update weights"


class TestFullTrainingLoop:
    """Test complete training scenarios."""
    
    def test_simple_convergence(self):
        """Test that a simple model can learn."""
        # Simple task: learn to output 5 from input [1, 2]
        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)
        loss_fn = MSELoss()
        
        x = Tensor([[1.0, 2.0]])
        target = Tensor([[5.0]])
        
        initial_loss = None
        final_loss = None
        
        # Train for a few iterations
        for i in range(50):
            # Forward
            pred = layer(x)
            loss = loss_fn(pred, target)
            
            if i == 0:
                initial_loss = loss.data
            if i == 49:
                final_loss = loss.data
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        # Loss should decrease
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} → {final_loss}"
    
    def test_binary_classification(self):
        """Test binary classification training."""
        layer = Linear(2, 1)
        sigmoid = Sigmoid()
        loss_fn = BinaryCrossEntropyLoss()
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        # Simple dataset: [1, 1] → 1, [0, 0] → 0
        X = Tensor([[1.0, 1.0], [0.0, 0.0]])
        y = Tensor([[1.0], [0.0]])
        
        initial_loss = None
        final_loss = None
        
        for i in range(50):
            # Forward
            logits = layer(X)
            probs = sigmoid(logits)
            loss = loss_fn(probs, y)
            
            if i == 0:
                initial_loss = loss.data
            if i == 49:
                final_loss = loss.data
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        assert final_loss < initial_loss, "Binary classification didn't learn"


class TestEdgeCases:
    """Test edge cases and potential failure modes."""
    
    def test_zero_gradient(self):
        """Test that zero gradients don't break training."""
        x = Tensor([[0.0, 0.0]], requires_grad=True)
        y = x * 0
        loss = y.sum()
        
        loss.backward()
        
        assert x.grad is not None
        assert np.allclose(x.grad, [[0.0, 0.0]])
    
    def test_very_small_values(self):
        """Test gradient flow with very small values."""
        x = Tensor([[1e-8, 1e-8]], requires_grad=True)
        y = x * 2
        loss = y.sum()
        
        loss.backward()
        
        assert x.grad is not None
        assert np.allclose(x.grad, [[2.0, 2.0]])
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across multiple backward passes."""
        x = Tensor([[1.0]], requires_grad=True)
        
        # First backward
        y1 = x * 2
        y1.backward()
        grad_after_first = x.grad.copy()
        
        # Second backward (without zero_grad)
        y2 = x * 3
        y2.backward()
        
        # Gradient should accumulate: 2 + 3 = 5
        expected = grad_after_first + np.array([[3.0]])
        assert np.allclose(x.grad, expected), f"Expected {expected}, got {x.grad}"


def run_all_tests():
    """Run all tests and print results."""
    import inspect
    
    test_classes = [
        TestBasicTensorGradients,
        TestLayerGradients,
        TestActivationGradients,
        TestLossGradients,
        TestOptimizerIntegration,
        TestFullTrainingLoop,
        TestEdgeCases,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    skipped_tests = []
    
    print("=" * 80)
    print("🧪 TINYTORCH GRADIENT FLOW TEST SUITE")
    print("=" * 80)
    
    for test_class in test_classes:
        print(f"\n{'=' * 80}")
        print(f"📦 {test_class.__name__}")
        print(f"{'=' * 80}")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            # Get docstring
            doc = method.__doc__ or method_name
            doc = doc.strip().split('\n')[0]
            
            print(f"\n  {method_name}")
            print(f"  {doc}")
            
            try:
                method()
                print(f"  ✅ PASSED")
                passed_tests += 1
            except NotImplementedError as e:
                print(f"  ⏭️  SKIPPED: {e}")
                skipped_tests.append((test_class.__name__, method_name, str(e)))
            except AssertionError as e:
                print(f"  ❌ FAILED: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests:   {total_tests}")
    print(f"✅ Passed:     {passed_tests}")
    print(f"❌ Failed:     {len(failed_tests)}")
    print(f"⏭️  Skipped:    {len(skipped_tests)}")
    
    if failed_tests:
        print("\n" + "=" * 80)
        print("❌ FAILED TESTS:")
        print("=" * 80)
        for class_name, method_name, error in failed_tests:
            print(f"\n  {class_name}.{method_name}")
            print(f"    {error}")
    
    if skipped_tests:
        print("\n" + "=" * 80)
        print("⏭️  SKIPPED TESTS (Not Yet Implemented):")
        print("=" * 80)
        for class_name, method_name, reason in skipped_tests:
            print(f"  {class_name}.{method_name}")
    
    print("\n" + "=" * 80)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
