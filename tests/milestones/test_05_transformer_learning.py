#!/usr/bin/env python3
"""
Transformer Learning Verification Test

This test systematically verifies that the transformer ACTUALLY LEARNS:
1. Forward pass produces correct shapes
2. Loss computation works
3. Backward pass computes gradients for ALL parameters
4. Optimizer updates ALL parameters
5. Loss decreases after updates
6. Model can overfit a single batch

This is a CRITICAL test - if this fails, the model cannot learn.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.models.transformer import GPT

# Enable autograd
enable_autograd()


def test_transformer_forward_pass():
    """Test 1: Forward pass produces correct output shapes."""
    print("\n" + "="*70)
    print("TEST 1: Forward Pass Shape Verification")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Create input
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    # Forward pass
    logits = model.forward(x)
    
    expected_shape = (batch_size, seq_len, vocab_size)
    actual_shape = logits.shape
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output: {expected_shape}")
    print(f"Actual output: {actual_shape}")
    
    assert logits.shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"
    print("✅ Forward pass shapes correct")
    
    return True


def test_transformer_loss_computation():
    """Test 2: Loss computation works and produces scalar."""
    print("\n" + "="*70)
    print("TEST 2: Loss Computation")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Create data
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    # Forward pass
    logits = model.forward(x)
    
    # Compute loss
    loss_fn = CrossEntropyLoss()
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Loss value: {loss.data}")
    print(f"Loss shape: {loss.shape}")
    print(f"Loss is scalar: {loss.data.size == 1}")
    print(f"Loss has _grad_fn: {hasattr(loss, '_grad_fn') and loss._grad_fn is not None}")
    
    assert loss.data.size == 1, "Loss should be scalar"
    assert hasattr(loss, '_grad_fn'), "Loss should have gradient function"
    print("✅ Loss computation works")
    
    return True


def test_transformer_gradient_computation():
    """Test 3: Backward pass computes gradients for ALL parameters."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Computation for All Parameters")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Set requires_grad for all parameters
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    print(f"Total parameters: {len(params)}")
    
    # Create data
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    # Forward pass
    logits = model.forward(x)
    
    # Compute loss
    loss_fn = CrossEntropyLoss()
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Loss before backward: {loss.data:.4f}")
    
    # Backward pass
    loss.backward(np.ones_like(loss.data))
    
    # Check gradients
    params_with_grads = 0
    params_without_grads = []
    
    for i, param in enumerate(params):
        if param.grad is not None:
            params_with_grads += 1
        else:
            params_without_grads.append(i)
    
    print(f"Parameters with gradients: {params_with_grads}/{len(params)}")
    
    if params_without_grads:
        print(f"❌ Parameters WITHOUT gradients: {params_without_grads}")
        assert False, f"{len(params_without_grads)} parameters have no gradients"
    
    print("✅ All parameters have gradients")
    
    return True


def test_transformer_parameter_updates():
    """Test 4: Optimizer actually updates parameters."""
    print("\n" + "="*70)
    print("TEST 4: Parameter Updates via Optimizer")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Set requires_grad and create optimizer
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.001)
    
    # Save initial parameter values
    initial_values = [param.data.copy() for param in params]
    
    # Create data
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    # Forward pass
    logits = model.forward(x)
    
    # Compute loss
    loss_fn = CrossEntropyLoss()
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    loss = loss_fn.forward(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward(np.ones_like(loss.data))
    
    # Update parameters
    optimizer.step()
    
    # Check which parameters changed
    params_changed = 0
    params_unchanged = []
    
    for i, (param, initial_val) in enumerate(zip(params, initial_values)):
        if not np.allclose(param.data, initial_val):
            params_changed += 1
        else:
            params_unchanged.append(i)
    
    print(f"Parameters changed: {params_changed}/{len(params)}")
    
    if params_unchanged:
        print(f"❌ Parameters UNCHANGED: {params_unchanged}")
        assert False, f"{len(params_unchanged)} parameters did not update"
    
    print("✅ All parameters updated by optimizer")
    
    return True


def test_transformer_loss_decreases():
    """Test 5: Loss decreases after multiple updates."""
    print("\n" + "="*70)
    print("TEST 5: Loss Decrease Verification")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Set requires_grad and create optimizer
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)  # Higher LR for faster convergence
    
    # Create FIXED data (same batch every time)
    np.random.seed(42)
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    loss_fn = CrossEntropyLoss()
    
    # Initial loss
    logits = model.forward(x)
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    initial_loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Initial loss: {initial_loss.data:.4f}")
    
    # Train for 10 steps
    for step in range(10):
        # Zero gradients
        for param in params:
            param.grad = None
        
        # Forward
        logits = model.forward(x)
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        # Backward
        loss.backward(np.ones_like(loss.data))
        
        # Update
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: Loss = {loss.data:.4f}")
    
    # Final loss
    logits = model.forward(x)
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    final_loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Final loss: {final_loss.data:.4f}")
    
    loss_decrease = initial_loss.data - final_loss.data
    percent_decrease = (loss_decrease / initial_loss.data) * 100
    
    print(f"Loss decrease: {loss_decrease:.4f} ({percent_decrease:.1f}%)")
    
    assert final_loss.data < initial_loss.data, \
        f"Loss did not decrease! Initial: {initial_loss.data:.4f}, Final: {final_loss.data:.4f}"
    
    print("✅ Loss decreased - model is learning!")
    
    return True


def test_transformer_single_batch_overfit():
    """Test 6: Model can overfit a single batch (critical capability test)."""
    print("\n" + "="*70)
    print("TEST 6: Single Batch Overfitting (Critical Learning Test)")
    print("="*70)
    
    vocab_size = 20
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    # Set requires_grad and create optimizer
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    
    # Create FIXED simple pattern
    np.random.seed(123)
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    loss_fn = CrossEntropyLoss()
    
    # Get initial loss
    logits = model.forward(x)
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    initial_loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Initial loss: {initial_loss.data:.4f}")
    print(f"Training for 50 steps to overfit single batch...")
    
    # Train for 50 steps
    for step in range(50):
        # Zero gradients
        for param in params:
            param.grad = None
        
        # Forward
        logits = model.forward(x)
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_len)
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        # Backward
        loss.backward(np.ones_like(loss.data))
        
        # Update
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}: Loss = {loss.data:.4f}")
    
    # Final loss
    logits = model.forward(x)
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    final_loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Final loss: {final_loss.data:.4f}")
    
    improvement = (initial_loss.data - final_loss.data) / initial_loss.data * 100
    print(f"Improvement: {improvement:.1f}%")
    
    # Should achieve at least 50% improvement on single batch
    assert improvement > 50, \
        f"Model not learning well enough! Only {improvement:.1f}% improvement (need >50%)"
    
    print("✅ Model can overfit single batch - learning capability verified!")
    
    return True


def run_all_tests():
    """Run all learning verification tests."""
    print("\n" + "="*70)
    print("TRANSFORMER LEARNING VERIFICATION TEST SUITE")
    print("="*70)
    print("\nThis suite verifies that the transformer can actually LEARN.")
    print("If any test fails, the model cannot train properly.\n")
    
    tests = [
        ("Forward Pass", test_transformer_forward_pass),
        ("Loss Computation", test_transformer_loss_computation),
        ("Gradient Computation", test_transformer_gradient_computation),
        ("Parameter Updates", test_transformer_parameter_updates),
        ("Loss Decrease", test_transformer_loss_decreases),
        ("Single Batch Overfit", test_transformer_single_batch_overfit),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n{'='*70}")
            print(f"✅ {test_name}: PASS")
            print(f"{'='*70}")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"❌ {test_name}: FAIL")
            print(f"Error: {e}")
            print(f"{'='*70}")
            import traceback
            traceback.print_exc()
            failed += 1
            break  # Stop on first failure to debug systematically
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The transformer is properly configured and CAN LEARN.")
        print("Ready for full Shakespeare training!")
    else:
        print(f"\n❌ {failed} test(s) failed")
        print("The transformer has issues that prevent learning.")
        print("Fix the failing test before proceeding to full training.")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

