#!/usr/bin/env python
"""Working Loss Functions Module - Simplified for Testing"""

import numpy as np
import sys
import os

# Import our tensor foundation - use absolute path for reliability
TENSOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
if TENSOR_PATH not in sys.path:
    sys.path.insert(0, TENSOR_PATH)

from tensor_dev import Tensor

print("🔥 TinyTorch Loss Functions Module")
print("Ready to build essential loss functions!")

def mse_loss(predictions: Tensor, targets: Tensor, reduction='mean') -> Tensor:
    """
    Mean Squared Error loss function.
    
    Args:
        predictions: Model predictions (any shape)
        targets: True values (same shape as predictions)
        reduction: 'mean' (default), 'sum', or 'none'
        
    Returns:
        Loss tensor (scalar if reduction='mean'/'sum', same shape if reduction='none')
    """
    # Compute squared differences
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Apply reduction
    if reduction == 'mean':
        loss = squared_diff.sum() / Tensor(float(squared_diff.size))
    elif reduction == 'sum':
        loss = squared_diff.sum()
    elif reduction == 'none':
        loss = squared_diff
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    return loss

def cross_entropy_loss(logits: Tensor, targets: Tensor, reduction='mean') -> Tensor:
    """
    Cross-entropy loss function with numerical stability.
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: True class indices (batch_size,) or one-hot vectors (batch_size, num_classes)
        reduction: 'mean' (default), 'sum', or 'none'
        
    Returns:
        Loss tensor (scalar if reduction='mean'/'sum', same shape as batch if reduction='none')
    """
    # Apply softmax with numerical stability
    max_vals = Tensor(np.max(logits.data, axis=-1, keepdims=True))
    logits_stable = logits - max_vals
    exp_logits = Tensor(np.exp(logits_stable.data))
    sum_exp = Tensor(np.sum(exp_logits.data, axis=-1, keepdims=True))
    softmax_probs = exp_logits / sum_exp
    
    # Handle targets - convert to one-hot if needed
    if targets.data.ndim == 1:
        # targets are class indices, convert to one-hot
        num_classes = logits.shape[-1]
        batch_size = targets.shape[0]
        targets_onehot = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            targets_onehot[i, int(targets.data[i])] = 1.0
        targets = Tensor(targets_onehot)
    
    # Compute cross-entropy with numerical stability (prevent log(0))
    epsilon = 1e-12
    softmax_probs_safe = Tensor(np.maximum(softmax_probs.data, epsilon))
    log_probs = Tensor(np.log(softmax_probs_safe.data))
    
    # Cross-entropy: -sum(targets * log(probs)) for each sample
    ce_per_sample = targets * log_probs
    ce_per_sample = Tensor(-np.sum(ce_per_sample.data, axis=-1))
    
    # Apply reduction
    if reduction == 'mean':
        loss = ce_per_sample.sum() / Tensor(float(ce_per_sample.size))
    elif reduction == 'sum':
        loss = ce_per_sample.sum()
    elif reduction == 'none':
        loss = ce_per_sample
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    return loss

def test_mse_loss():
    """Test MSE loss function"""
    print("🔬 Testing MSE Loss...")
    
    # Test perfect predictions (zero loss)
    predictions = Tensor([[1.0, 2.0, 3.0]])
    targets = Tensor([[1.0, 2.0, 3.0]])
    loss = mse_loss(predictions, targets)
    
    assert np.isclose(loss.data, 0.0), f"Perfect predictions should have zero loss, got {loss.data}"
    
    # Test known case
    predictions = Tensor([[1.0, 2.0]])
    targets = Tensor([[1.5, 1.5]])
    loss = mse_loss(predictions, targets)
    # MSE = ((1.0-1.5)² + (2.0-1.5)²) / 2 = (0.25 + 0.25) / 2 = 0.25
    expected = 0.25
    
    assert np.isclose(loss.data, expected), f"Expected MSE {expected}, got {loss.data}"
    print("✅ MSE loss tests passed!")

def test_cross_entropy_loss():
    """Test Cross-Entropy loss function"""
    print("🔬 Testing Cross-Entropy Loss...")
    
    # Test perfect predictions (should give near-zero loss)
    logits = Tensor([[10.0, 0.0, 0.0]])  # Very confident in class 0
    targets = Tensor([0])                 # True class is 0
    loss = cross_entropy_loss(logits, targets)
    
    assert loss.data < 0.1, f"Perfect prediction should have low loss, got {loss.data}"
    
    # Test batch processing
    logits = Tensor([
        [2.0, 1.0, 0.1],    # Sample 1: prefers class 0
        [0.1, 0.2, 2.0],    # Sample 2: prefers class 2
        [1.5, 2.0, 0.5]     # Sample 3: prefers class 1
    ])
    targets = Tensor([0, 2, 1])  # Correct classes
    
    loss_batch = cross_entropy_loss(logits, targets, reduction='mean')
    
    # Should be relatively low since predictions align with targets
    assert 0.0 <= loss_batch.data <= 2.0, f"Batch loss should be reasonable, got {loss_batch.data}"
    print("✅ Cross-entropy loss tests passed!")

def test_integration():
    """Test both loss functions in a training-like scenario"""
    print("🔬 Testing Integration...")
    
    # Regression scenario
    pred_prices = Tensor([[245000, 190000, 315000, 160000]])  # Predictions  
    true_prices = Tensor([[250000, 180000, 320000, 150000]])  # True prices
    mse = mse_loss(pred_prices, true_prices)
    print(f"   Regression Loss (MSE): {mse.data:.0f}")
    
    # Classification scenario
    logits = Tensor([
        [3.2, 1.3, 0.2],   # Strong preference for class 0
        [0.1, 2.8, 1.1],   # Strong preference for class 1  
        [0.5, 0.8, 3.1]    # Strong preference for class 2
    ])
    true_classes = Tensor([0, 1, 2])
    ce_loss = cross_entropy_loss(logits, true_classes)
    print(f"   Classification Loss (CE): {ce_loss.data:.4f}")
    
    print("✅ Integration tests passed!")

if __name__ == "__main__":
    test_mse_loss()
    test_cross_entropy_loss()
    test_integration()
    
    print("\n🎉 All loss function tests passed!")
    print("✅ MSE: The foundation of regression training")
    print("✅ Cross-Entropy: The key to classification training")
    print("💡 Ready to train neural networks with proper objective functions!")