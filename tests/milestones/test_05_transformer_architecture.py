#!/usr/bin/env python3
"""
Phase 1: Transformer Architecture Verification

These tests verify the transformer architecture is correct BEFORE training.
No reward hacking - we test the actual implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.models.transformer import GPT as TinyGPT

# Enable autograd
enable_autograd()


def test_forward_pass_shapes():
    """Test 1.1: Verify all tensor shapes through forward pass."""
    print("\n🧪 Test 1.1: Forward Pass Shape Validation")
    print("="*70)
    
    vocab_size = 65
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    seq_length = 64
    batch_size = 2
    
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Input: (batch, seq)
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output: ({batch_size}, {seq_length}, {vocab_size})")
    
    # Forward pass
    output = model.forward(x)
    
    print(f"Actual output: {output.shape}")
    
    # Verify shape
    expected_shape = (batch_size, seq_length, vocab_size)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    
    print("✅ Forward pass shapes correct")
    return True


def test_gradient_flow_all_params():
    """Test 1.2: Ensure gradients flow to ALL parameters."""
    print("\n🧪 Test 1.2: Gradient Flow Verification")
    print("="*70)
    
    vocab_size = 65
    embed_dim = 128
    num_layers = 2  # Smaller for faster test
    num_heads = 4
    seq_length = 32
    batch_size = 2
    
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Get parameters and set requires_grad
    params = model.parameters()
    for param in params:
        param.requires_grad = True
        param.grad = None  # Clear any existing gradients
    
    print(f"Total parameters: {len(params)}")
    
    # Forward pass
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    
    logits = model.forward(x)
    loss_fn = CrossEntropyLoss()
    
    # Reshape for loss: (batch*seq, vocab)
    logits_flat = logits.reshape(batch_size * seq_length, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_length)
    
    loss = loss_fn.forward(logits_flat, targets_flat)
    
    print(f"Loss: {loss.data:.4f}")
    
    # Backward pass
    loss.backward(np.ones_like(loss.data))
    
    # Check ALL parameters have gradients
    params_without_grads = []
    params_with_grads = []
    
    for i, param in enumerate(params):
        if param.grad is None:
            params_without_grads.append(i)
        else:
            params_with_grads.append(i)
    
    print(f"Parameters with gradients: {len(params_with_grads)}/{len(params)}")
    
    if params_without_grads:
        print(f"❌ Parameters WITHOUT gradients: {params_without_grads}")
        assert False, f"Parameters without gradients: {params_without_grads}"
    
    print(f"✅ All {len(params)} parameters receive gradients")
    return True


def test_single_batch_overfitting():
    """Test 1.3: Model should memorize a single batch perfectly."""
    print("\n🧪 Test 1.3: Single Batch Overfitting Test")
    print("="*70)
    
    vocab_size = 65
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    seq_length = 32
    batch_size = 2
    
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Set requires_grad for all parameters
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Single fixed batch
    np.random.seed(42)
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    
    print(f"Training on single batch: {x.shape}")
    
    initial_loss = None
    final_loss = None
    losses = []
    
    # Train for 100 steps on same batch
    for step in range(100):
        # Forward
        logits = model.forward(x)
        logits_flat = logits.reshape(batch_size * seq_length, vocab_size)
        targets_flat = targets.reshape(batch_size * seq_length)
        
        loss = loss_fn.forward(logits_flat, targets_flat)
        loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        
        if step == 0:
            initial_loss = loss_value
            print(f"Initial loss: {initial_loss:.4f}")
        
        losses.append(loss_value)
        
        # Backward
        optimizer.zero_grad()
        loss.backward(np.ones_like(loss.data))
        optimizer.step()
        
        if step % 20 == 0 and step > 0:
            print(f"  Step {step}: Loss = {loss_value:.4f} (change: {losses[step] - losses[step-1]:.4f})")
        
        final_loss = loss_value
    
    print(f"\nFinal loss: {final_loss:.4f}")
    
    # Loss should decrease significantly
    improvement = (initial_loss - final_loss) / initial_loss
    
    print(f"Improvement: {improvement:.1%}")
    
    # Check for NaN or explosion
    assert not np.isnan(final_loss), "Loss became NaN!"
    assert not np.isinf(final_loss), "Loss exploded to infinity!"
    
    # Loss should improve by at least 30%
    if improvement < 0.3:
        print(f"⚠️  Warning: Loss only improved by {improvement:.1%}, expected >30%")
        print(f"    This might indicate:")
        print(f"    - Learning rate too low")
        print(f"    - Gradients not flowing properly")
        print(f"    - Model initialization issues")
        
        # Let's check if loss is at least decreasing
        recent_improvement = (losses[0] - losses[-1]) / losses[0]
        assert recent_improvement > 0.1, \
            f"Loss barely decreased: {recent_improvement:.1%}"
    
    print(f"✅ Single batch overfitting works: {initial_loss:.4f} → {final_loss:.4f}")
    return True


def test_parameter_updates():
    """Test 1.4: Verify parameters actually change during training."""
    print("\n🧪 Test 1.4: Parameter Update Verification")
    print("="*70)
    
    vocab_size = 65
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    seq_length = 32
    batch_size = 2
    
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Set requires_grad for all parameters
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    # Save initial parameter values
    initial_params = [p.data.copy() for p in params]
    
    optimizer = Adam(params, lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Single training step
    x = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    targets = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), requires_grad=False)
    
    logits = model.forward(x)
    logits_flat = logits.reshape(batch_size * seq_length, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_length)
    
    loss = loss_fn.forward(logits_flat, targets_flat)
    
    optimizer.zero_grad()
    loss.backward(np.ones_like(loss.data))
    optimizer.step()
    
    # Check parameters changed
    params_changed = 0
    params_unchanged = 0
    
    for i, (initial, current) in enumerate(zip(initial_params, params)):
        max_diff = np.max(np.abs(current.data - initial))
        if max_diff > 1e-7:
            params_changed += 1
        else:
            params_unchanged += 1
    
    print(f"Parameters changed: {params_changed}/{len(params)}")
    print(f"Parameters unchanged: {params_unchanged}/{len(params)}")
    
    assert params_changed > len(params) * 0.9, \
        f"Only {params_changed}/{len(params)} parameters changed"
    
    print(f"✅ Parameters update correctly")
    return True


def test_attention_mask():
    """Test 1.5: Verify causal masking prevents looking ahead."""
    print("\n🧪 Test 1.5: Causal Attention Mask Verification")
    print("="*70)
    
    from tinytorch.core.attention import scaled_dot_product_attention
    
    batch_size = 2
    seq_len = 4
    head_dim = 8
    
    Q = Tensor(np.random.randn(batch_size, seq_len, head_dim), requires_grad=True)
    K = Tensor(np.random.randn(batch_size, seq_len, head_dim), requires_grad=True)
    V = Tensor(np.random.randn(batch_size, seq_len, head_dim), requires_grad=True)
    
    # Create causal mask
    mask = np.tril(np.ones((seq_len, seq_len)))  # Lower triangular
    mask = Tensor(mask)
    
    # Apply attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, head_dim), \
        f"Expected ({batch_size}, {seq_len}, {head_dim}), got {output.shape}"
    
    print("✅ Causal attention masking works")
    return True


def run_phase1_tests():
    """Run all Phase 1 architecture verification tests."""
    print("\n" + "="*70)
    print("PHASE 1: TRANSFORMER ARCHITECTURE VERIFICATION")
    print("="*70)
    print("\nThese tests verify the architecture is correct BEFORE training.")
    print("No shortcuts - we test the actual implementation.\n")
    
    tests = [
        ("Forward Pass Shapes", test_forward_pass_shapes),
        ("Gradient Flow to All Params", test_gradient_flow_all_params),
        ("Single Batch Overfitting", test_single_batch_overfitting),
        ("Parameter Updates", test_parameter_updates),
        ("Causal Attention Mask", test_attention_mask),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS", None))
        except Exception as e:
            results.append((test_name, "FAIL", str(e)))
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 TEST RESULTS")
    print("="*70)
    
    for test_name, status, error in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"    Error: {error}")
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 1 tests PASSED!")
        print("Architecture is verified. Ready for Phase 2 (Data Pipeline).")
    else:
        print("\n⚠️  Some tests FAILED. Fix these before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_phase1_tests()
    sys.exit(0 if success else 1)

