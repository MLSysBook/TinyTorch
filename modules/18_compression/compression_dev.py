# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Compression - Neural Network Pruning for Edge Deployment

Welcome to the Compression module! You'll implement pruning techniques that remove 70% of neural network parameters while maintaining accuracy, enabling deployment on resource-constrained edge devices.

## Connection from Quantization (Module 17)
In Module 17, you learned quantization - reducing precision from FP32 to INT8. But even quantized models can be too large for edge devices! Compression attacks the problem differently: instead of making numbers smaller, we **remove numbers entirely** through strategic pruning.

## Learning Goals
- Systems understanding: How neural network redundancy enables massive parameter reduction without accuracy loss
- Core implementation skill: Build magnitude-based pruning systems that identify and remove unimportant weights
- Pattern recognition: Understand when structured vs unstructured pruning optimizes for different hardware constraints
- Framework connection: See how your implementation mirrors production sparse inference systems
- Performance insight: Learn why 70% sparsity often provides optimal accuracy vs size tradeoffs

## Build → Profile → Optimize
1. **Build**: Magnitude-based pruners that remove small weights, discover massive redundancy in neural networks
2. **Profile**: Measure model size reduction, accuracy impact, and sparse computation efficiency
3. **Optimize**: Implement structured pruning for hardware-friendly sparsity patterns

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how neural networks contain massive redundancy that can be exploited for compression
- Practical capability to prune real CNNs and MLPs while maintaining 95%+ of original accuracy
- Systems insight into why pruning enables deployment scenarios impossible with dense models
- Performance consideration of when sparse computation provides real speedups vs theoretical ones
- Connection to production systems where pruning enables edge AI applications

## Systems Reality Check
💡 **Production Context**: Apple's Neural Engine, Google's Edge TPU, and mobile inference frameworks heavily rely on sparsity for efficient computation
⚡ **Performance Note**: 70% sparsity provides 3-5x model compression with <2% accuracy loss, but speedup depends on hardware sparse computation support
"""

# %% nbgrader={"grade": false, "grade_id": "compression-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp compression

#| export
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

# %% [markdown]
"""
## Part 1: Understanding Neural Network Redundancy

Before implementing pruning, let's understand the fundamental insight: **neural networks are massively over-parametrized**. Most weights contribute little to the final output and can be removed without significant accuracy loss.

### The Redundancy Discovery
- **Research insight**: Networks often have 80-90% redundant parameters
- **Lottery Ticket Hypothesis**: Sparse subnetworks can match dense network performance
- **Practical reality**: 70% sparsity typically loses <2% accuracy
- **Systems opportunity**: Massive compression enables edge deployment
"""

# %% nbgrader={"grade": false, "grade_id": "redundancy-analysis", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def analyze_weight_redundancy(weights: np.ndarray, title: str = "Weight Analysis"):
    """
    Analyze weight distributions to understand pruning opportunities.
    
    This function reveals the natural sparsity and redundancy patterns
    in neural network weights that make pruning effective.
    """
    # Flatten weights for analysis
    w_flat = weights.flatten()
    w_abs = np.abs(w_flat)
    
    print(f"📊 {title}")
    print("=" * 50)
    print(f"Total parameters: {len(w_flat):,}")
    print(f"Mean absolute weight: {w_abs.mean():.6f}")
    print(f"Weight standard deviation: {w_abs.std():.6f}")
    
    # Analyze weight distribution percentiles
    percentiles = [50, 70, 80, 90, 95, 99]
    print(f"\nWeight Magnitude Percentiles:")
    for p in percentiles:
        val = np.percentile(w_abs, p)
        smaller_count = np.sum(w_abs <= val)
        print(f"  {p:2d}%: {val:.6f} ({smaller_count:,} weights ≤ this value)")
    
    # Show natural sparsity (near-zero weights)
    zero_threshold = w_abs.mean() * 0.1  # 10% of mean as "near-zero"
    near_zero_count = np.sum(w_abs <= zero_threshold)
    natural_sparsity = near_zero_count / len(w_flat) * 100
    
    print(f"\nNatural Sparsity Analysis:")
    print(f"  Threshold (10% of mean): {zero_threshold:.6f}")
    print(f"  Near-zero weights: {near_zero_count:,} ({natural_sparsity:.1f}%)")
    print(f"  Already sparse without pruning!")
    
    return {
        'total_params': len(w_flat),
        'mean_abs': w_abs.mean(),
        'std': w_abs.std(),
        'natural_sparsity': natural_sparsity,
        'percentiles': {p: np.percentile(w_abs, p) for p in percentiles}
    }

# %% [markdown]
"""
### Test: Weight Redundancy Analysis

Let's verify our redundancy analysis works on realistic neural network weights.
"""

# %% nbgrader={"grade": true, "grade_id": "test-redundancy-analysis", "locked": false, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_redundancy_analysis():
    """Test weight redundancy analysis on sample networks."""
    print("Testing weight redundancy analysis...")
    
    # Create realistic CNN weights with natural sparsity
    np.random.seed(42)
    conv_weights = np.random.normal(0, 0.02, (64, 32, 3, 3))  # Conv layer
    fc_weights = np.random.normal(0, 0.01, (1000, 512))       # FC layer
    
    # Analyze both layer types
    conv_stats = analyze_weight_redundancy(conv_weights, "Conv2D Layer Weights")
    fc_stats = analyze_weight_redundancy(fc_weights, "Dense Layer Weights")
    
    # Verify analysis produces reasonable results
    assert conv_stats['total_params'] == 64*32*3*3, "Conv param count mismatch"
    assert fc_stats['total_params'] == 1000*512, "FC param count mismatch"
    assert conv_stats['natural_sparsity'] > 0, "Should detect some natural sparsity"
    assert fc_stats['natural_sparsity'] > 0, "Should detect some natural sparsity"
    
    print("✅ Weight redundancy analysis test passed!")

test_redundancy_analysis()

# %% [markdown]
"""
## Part 2: Magnitude-Based Pruning - The Foundation

The simplest and most effective pruning technique: **remove the smallest weights**. The intuition is that small weights contribute little to the network's computation, so removing them should have minimal impact on accuracy.

### Magnitude Pruning Algorithm
1. **Calculate importance**: Use absolute weight magnitude as importance metric
2. **Rank weights**: Sort all weights by absolute value
3. **Set threshold**: Choose magnitude threshold for desired sparsity level
4. **Create mask**: Zero out weights below threshold
5. **Apply mask**: Element-wise multiplication to enforce sparsity
"""

# %% nbgrader={"grade": false, "grade_id": "magnitude-pruning", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MagnitudePruner:
    """
    Magnitude-based pruning for neural network compression.
    
    This class implements the core pruning algorithm used in production
    systems: remove weights with smallest absolute values.
    """
    
    def __init__(self):
        # BEGIN SOLUTION
        self.pruning_masks = {}
        self.original_weights = {}
        self.pruning_stats = {}
        # END SOLUTION
    
    def calculate_threshold(self, weights: np.ndarray, sparsity: float) -> float:
        """
        Calculate magnitude threshold for desired sparsity level.
        
        Args:
            weights: Network weights to analyze
            sparsity: Fraction of weights to remove (0.0 to 1.0)
            
        Returns:
            threshold: Magnitude below which weights should be pruned
        """
        # BEGIN SOLUTION
        # Flatten weights and get absolute values
        w_flat = weights.flatten()
        w_abs = np.abs(w_flat)
        
        # Calculate percentile threshold
        # sparsity=0.7 means remove 70% of weights (keep top 30%)
        percentile = sparsity * 100
        threshold = np.percentile(w_abs, percentile)
        
        return threshold
        # END SOLUTION
    
    def create_mask(self, weights: np.ndarray, threshold: float) -> np.ndarray:
        """
        Create binary mask for pruning weights below threshold.
        
        Args:
            weights: Original weights
            threshold: Magnitude threshold for pruning
            
        Returns:
            mask: Binary mask (1=keep, 0=prune)
        """
        # BEGIN SOLUTION
        # Create mask: keep weights with absolute value >= threshold
        mask = (np.abs(weights) >= threshold).astype(np.float32)
        return mask
        # END SOLUTION
    
    def prune(self, weights: np.ndarray, sparsity: float = 0.7) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prune network weights using magnitude-based pruning.
        
        Args:
            weights: Original dense weights
            sparsity: Fraction of weights to prune (default: 70%)
            
        Returns:
            pruned_weights: Weights with small values set to zero
            mask: Binary pruning mask
            stats: Pruning statistics
        """
        # BEGIN SOLUTION
        # Store original weights
        original_shape = weights.shape
        original_size = weights.size
        
        # Calculate threshold for desired sparsity
        threshold = self.calculate_threshold(weights, sparsity)
        
        # Create pruning mask
        mask = self.create_mask(weights, threshold)
        
        # Apply pruning
        pruned_weights = weights * mask
        
        # Calculate statistics
        actual_sparsity = np.sum(mask == 0) / mask.size
        remaining_params = np.sum(mask == 1)
        compression_ratio = original_size / remaining_params if remaining_params > 0 else float('inf')
        
        stats = {
            'target_sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': threshold,
            'original_params': original_size,
            'remaining_params': int(remaining_params),
            'pruned_params': int(original_size - remaining_params),
            'compression_ratio': compression_ratio
        }
        
        return pruned_weights, mask, stats
        # END SOLUTION
    
    def measure_accuracy_impact(self, original_weights: np.ndarray, pruned_weights: np.ndarray) -> Dict:
        """
        Measure the impact of pruning on weight statistics.
        
        This gives us a proxy for accuracy impact before running full evaluation.
        """
        # BEGIN SOLUTION
        # Calculate difference statistics
        weight_diff = np.abs(original_weights - pruned_weights)
        
        # Normalize by original weight magnitude for relative comparison
        original_abs = np.abs(original_weights)
        relative_error = weight_diff / (original_abs + 1e-8)  # Avoid division by zero
        
        return {
            'mean_absolute_error': weight_diff.mean(),
            'max_absolute_error': weight_diff.max(),
            'mean_relative_error': relative_error.mean(),
            'weight_norm_preservation': np.linalg.norm(pruned_weights) / np.linalg.norm(original_weights)
        }
        # END SOLUTION

# %% [markdown]
"""
### Test: Magnitude-Based Pruning Implementation

Let's verify our magnitude pruning works correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-magnitude-pruning", "locked": false, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_magnitude_pruning():
    """Test magnitude-based pruning implementation."""
    print("Testing magnitude-based pruning...")
    
    pruner = MagnitudePruner()
    
    # Test case 1: Simple weights with known distribution
    weights = np.array([
        [0.5, 0.1, 0.8],
        [0.05, 0.9, 0.2],
        [0.3, 0.02, 0.7]
    ])
    
    # Test 50% sparsity (should keep 4.5 ≈ 4-5 weights)
    pruned, mask, stats = pruner.prune(weights, sparsity=0.5)
    
    print(f"Original weights:")
    print(weights)
    print(f"Pruning mask:")
    print(mask)
    print(f"Pruned weights:")
    print(pruned)
    print(f"Statistics: {stats}")
    
    # Verify sparsity is approximately correct
    actual_sparsity = stats['actual_sparsity']
    assert 0.4 <= actual_sparsity <= 0.6, f"Sparsity should be ~50%, got {actual_sparsity:.1%}"
    
    # Verify mask is binary
    assert np.all((mask == 0) | (mask == 1)), "Mask should be binary"
    
    # Verify pruned weights match mask
    expected_pruned = weights * mask
    np.testing.assert_array_equal(pruned, expected_pruned, "Pruned weights should match mask application")
    
    # Test case 2: High sparsity pruning
    large_weights = np.random.normal(0, 0.1, (100, 50))
    pruned_large, mask_large, stats_large = pruner.prune(large_weights, sparsity=0.8)
    
    assert 0.75 <= stats_large['actual_sparsity'] <= 0.85, "High sparsity should be approximately correct"
    assert stats_large['compression_ratio'] >= 4.0, "80% sparsity should give ~5x compression"
    
    # Test accuracy impact measurement
    accuracy_impact = pruner.measure_accuracy_impact(large_weights, pruned_large)
    assert 'mean_relative_error' in accuracy_impact, "Should measure relative error"
    assert accuracy_impact['weight_norm_preservation'] > 0, "Should preserve some weight norm"
    
    print("✅ Magnitude-based pruning test passed!")

test_magnitude_pruning()

# %% [markdown]
"""
## Part 3: Structured vs Unstructured Pruning

So far we've implemented **unstructured pruning** - removing individual weights anywhere. But this creates irregular sparsity patterns that are hard for hardware to accelerate. **Structured pruning** removes entire channels, filters, or blocks - creating regular patterns that map well to hardware.

### Structured Pruning Benefits:
- **Hardware friendly**: Regular patterns enable efficient sparse computation
- **Memory layout**: Removes entire rows/columns, reducing memory footprint  
- **Inference speed**: Actually accelerates computation (vs theoretical speedup)
- **Implementation simple**: No special sparse kernels needed
"""

# %% nbgrader={"grade": false, "grade_id": "structured-pruning", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def prune_conv_filters(conv_weights: np.ndarray, sparsity: float = 0.5) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Structured pruning for convolutional layers - remove entire filters.
    
    Args:
        conv_weights: Conv weights shaped (out_channels, in_channels, H, W)
        sparsity: Fraction of filters to remove
        
    Returns:
        pruned_weights: Weights with filters removed
        kept_filters: Indices of filters that were kept
        stats: Pruning statistics
    """
    # BEGIN SOLUTION
    # Calculate importance score for each output filter
    # Use L2 norm of entire filter as importance measure
    out_channels = conv_weights.shape[0]
    filter_norms = []
    
    for i in range(out_channels):
        filter_weights = conv_weights[i]  # Shape: (in_channels, H, W)
        l2_norm = np.linalg.norm(filter_weights)
        filter_norms.append(l2_norm)
    
    filter_norms = np.array(filter_norms)
    
    # Determine how many filters to keep
    num_filters_to_keep = int(out_channels * (1 - sparsity))
    num_filters_to_keep = max(1, num_filters_to_keep)  # Keep at least 1 filter
    
    # Find indices of top filters to keep
    top_filter_indices = np.argsort(filter_norms)[-num_filters_to_keep:]
    top_filter_indices.sort()  # Keep original ordering
    
    # Create pruned weights by selecting only top filters
    pruned_weights = conv_weights[top_filter_indices]
    
    # Calculate statistics
    actual_sparsity = 1 - (num_filters_to_keep / out_channels)
    
    stats = {
        'original_filters': out_channels,
        'remaining_filters': num_filters_to_keep,
        'pruned_filters': out_channels - num_filters_to_keep,
        'target_sparsity': sparsity,
        'actual_sparsity': actual_sparsity,
        'compression_ratio': out_channels / num_filters_to_keep,
        'filter_norms': filter_norms,
        'kept_filter_indices': top_filter_indices.tolist()
    }
    
    return pruned_weights, top_filter_indices.tolist(), stats
    # END SOLUTION

def compare_structured_vs_unstructured(conv_weights: np.ndarray, sparsity: float = 0.5):
    """
    Compare structured vs unstructured pruning on the same layer.
    """
    print("🔬 Structured vs Unstructured Pruning Comparison")
    print("=" * 60)
    
    # Unstructured pruning
    pruner = MagnitudePruner()
    unstructured_pruned, unstructured_mask, unstructured_stats = pruner.prune(conv_weights, sparsity)
    
    # Structured pruning  
    structured_pruned, kept_filters, structured_stats = prune_conv_filters(conv_weights, sparsity)
    
    print("Unstructured Pruning:")
    print(f"  Original shape: {conv_weights.shape}")
    print(f"  Pruned shape: {unstructured_pruned.shape} (same)")
    print(f"  Sparsity: {unstructured_stats['actual_sparsity']:.1%}")
    print(f"  Compression: {unstructured_stats['compression_ratio']:.1f}x")
    print(f"  Zero elements: {np.sum(unstructured_pruned == 0):,}")
    
    print("\nStructured Pruning:")
    print(f"  Original shape: {conv_weights.shape}")
    print(f"  Pruned shape: {structured_pruned.shape}")
    print(f"  Sparsity: {structured_stats['actual_sparsity']:.1%}")
    print(f"  Compression: {structured_stats['compression_ratio']:.1f}x")
    print(f"  Filters removed: {structured_stats['pruned_filters']}")
    
    print(f"\n💡 Key Differences:")
    print(f"   • Unstructured: Irregular sparsity, requires sparse kernels")
    print(f"   • Structured: Regular reduction, standard dense computation")
    print(f"   • Hardware: Structured pruning provides actual speedup")
    print(f"   • Memory: Structured pruning reduces memory footprint")
    
    return {
        'unstructured': (unstructured_pruned, unstructured_stats),
        'structured': (structured_pruned, structured_stats)
    }

# %% [markdown]
"""
### Test: Structured Pruning Implementation

Let's verify structured pruning works correctly and compare it with unstructured pruning.
"""

# %% nbgrader={"grade": true, "grade_id": "test-structured-pruning", "locked": false, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_structured_pruning():
    """Test structured pruning implementation."""
    print("Testing structured pruning...")
    
    # Create sample conv weights: (out_channels, in_channels, H, W)
    np.random.seed(42)
    conv_weights = np.random.normal(0, 0.1, (8, 4, 3, 3))
    
    # Test structured pruning
    pruned_weights, kept_filters, stats = prune_conv_filters(conv_weights, sparsity=0.5)
    
    print(f"Original shape: {conv_weights.shape}")
    print(f"Pruned shape: {pruned_weights.shape}")
    print(f"Kept filters: {kept_filters}")
    print(f"Stats: {stats}")
    
    # Verify output shape is correct
    expected_filters = int(8 * (1 - 0.5))  # 50% sparsity = keep 50% of filters
    assert pruned_weights.shape[0] == expected_filters, f"Should keep {expected_filters} filters"
    assert pruned_weights.shape[1:] == conv_weights.shape[1:], "Other dimensions should match"
    
    # Verify kept filters are the strongest ones
    filter_norms = [np.linalg.norm(conv_weights[i]) for i in range(8)]
    top_indices = np.argsort(filter_norms)[-expected_filters:]
    top_indices.sort()
    
    for i, kept_idx in enumerate(kept_filters):
        # Verify the pruned weight matches original filter
        np.testing.assert_array_equal(
            pruned_weights[i], 
            conv_weights[kept_idx],
            f"Filter {i} should match original filter {kept_idx}"
        )
    
    # Test comparison function
    comparison = compare_structured_vs_unstructured(conv_weights, 0.5)
    
    # Verify both methods produce different results
    unstructured_result = comparison['unstructured'][0]
    structured_result = comparison['structured'][0]
    
    assert unstructured_result.shape == conv_weights.shape, "Unstructured keeps same shape"
    assert structured_result.shape[0] < conv_weights.shape[0], "Structured reduces filters"
    
    print("✅ Structured pruning test passed!")

test_structured_pruning()

# %% [markdown]
"""
## Part 4: Sparse Neural Networks - Efficient Computation

Pruning creates sparse networks, but how do we compute with them efficiently? We need sparse linear layers that skip computation for zero weights.

### Sparse Computation Challenges:
- **Memory layout**: How to store only non-zero weights efficiently
- **Computation patterns**: Skip multiply-add operations for zero weights  
- **Hardware support**: Most hardware isn't optimized for arbitrary sparsity
- **Software optimization**: Need specialized sparse kernels for speedup
"""

# %% nbgrader={"grade": false, "grade_id": "sparse-computation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SparseLinear:
    """
    Sparse linear layer that efficiently computes with pruned weights.
    
    This demonstrates how to build sparse computation systems
    that actually achieve speedup from sparsity.
    """
    
    def __init__(self, in_features: int, out_features: int):
        # BEGIN SOLUTION
        self.in_features = in_features
        self.out_features = out_features
        
        # Dense weights (will be pruned)
        self.dense_weights = None
        self.bias = None
        
        # Sparse representation
        self.sparse_weights = None
        self.mask = None
        self.sparsity = 0.0
        
        # Performance tracking
        self.dense_ops = 0
        self.sparse_ops = 0
        # END SOLUTION
    
    def load_dense_weights(self, weights: np.ndarray, bias: Optional[np.ndarray] = None):
        """Load dense weights before pruning."""
        # BEGIN SOLUTION
        assert weights.shape == (self.out_features, self.in_features), f"Weight shape mismatch"
        self.dense_weights = weights.copy()
        self.bias = bias.copy() if bias is not None else np.zeros(self.out_features)
        # END SOLUTION
    
    def prune_weights(self, sparsity: float = 0.7):
        """Prune weights using magnitude-based pruning."""
        # BEGIN SOLUTION
        if self.dense_weights is None:
            raise ValueError("Must load dense weights before pruning")
        
        # Use magnitude pruner
        pruner = MagnitudePruner()
        self.sparse_weights, self.mask, stats = pruner.prune(self.dense_weights, sparsity)
        self.sparsity = stats['actual_sparsity']
        
        print(f"✂️  Pruned {self.sparsity:.1%} of weights")
        print(f"   Compression: {stats['compression_ratio']:.1f}x")
        # END SOLUTION
    
    def forward_dense(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using dense weights (reference)."""
        # BEGIN SOLUTION
        if self.dense_weights is None:
            raise ValueError("Dense weights not loaded")
        
        # Count operations
        self.dense_ops = self.in_features * self.out_features
        
        # Standard matrix multiply: y = x @ W^T + b
        output = np.dot(x, self.dense_weights.T) + self.bias
        return output
        # END SOLUTION
    
    def forward_sparse_naive(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using sparse weights (naive implementation)."""
        # BEGIN SOLUTION
        if self.sparse_weights is None:
            raise ValueError("Weights not pruned yet")
        
        # Count actual operations (skip zero weights)
        self.sparse_ops = np.sum(self.mask)
        
        # Naive sparse computation: still do full matrix multiply
        # (Real sparse implementations would use CSR/CSC formats)
        output = np.dot(x, self.sparse_weights.T) + self.bias
        return output
        # END SOLUTION
    
    def forward_sparse_optimized(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using optimized sparse computation."""
        # BEGIN SOLUTION
        if self.sparse_weights is None:
            raise ValueError("Weights not pruned yet")
        
        # Find non-zero weights
        nonzero_indices = np.nonzero(self.sparse_weights)
        
        # Count actual operations
        self.sparse_ops = len(nonzero_indices[0])
        
        # Optimized sparse computation (simulated)
        # In practice, this would use specialized sparse matrix libraries
        output = np.zeros((x.shape[0], self.out_features))
        
        # Only compute for non-zero weights
        for i in range(len(nonzero_indices[0])):
            row = nonzero_indices[0][i]
            col = nonzero_indices[1][i]
            weight = self.sparse_weights[row, col]
            
            # Accumulate: output[batch, row] += input[batch, col] * weight
            output[:, row] += x[:, col] * weight
        
        # Add bias
        output += self.bias
        
        return output
        # END SOLUTION
    
    def benchmark_speedup(self, batch_size: int = 32, iterations: int = 100) -> Dict:
        """Benchmark sparse vs dense computation speedup."""
        # BEGIN SOLUTION
        import time
        
        # Create test input
        x = np.random.normal(0, 1, (batch_size, self.in_features))
        
        # Benchmark dense forward pass
        start_time = time.time()
        for _ in range(iterations):
            _ = self.forward_dense(x)
        dense_time = time.time() - start_time
        
        # Benchmark sparse forward pass
        start_time = time.time()
        for _ in range(iterations):
            _ = self.forward_sparse_naive(x)
        sparse_time = time.time() - start_time
        
        # Calculate speedup metrics
        theoretical_speedup = self.dense_ops / self.sparse_ops if self.sparse_ops > 0 else 1
        actual_speedup = dense_time / sparse_time if sparse_time > 0 else 1
        
        return {
            'dense_time_ms': dense_time * 1000,
            'sparse_time_ms': sparse_time * 1000,
            'dense_ops': self.dense_ops,
            'sparse_ops': self.sparse_ops,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'sparsity': self.sparsity,
            'efficiency': actual_speedup / theoretical_speedup
        }
        # END SOLUTION

# %% [markdown]
"""
### Test: Sparse Neural Network Implementation

Let's verify our sparse neural network works correctly and measure performance.
"""

# %% nbgrader={"grade": true, "grade_id": "test-sparse-neural-network", "locked": false, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_sparse_neural_network():
    """Test sparse neural network implementation."""
    print("Testing sparse neural network...")
    
    # Create sparse linear layer
    sparse_layer = SparseLinear(256, 128)
    
    # Load random weights
    np.random.seed(42)
    weights = np.random.normal(0, 0.1, (128, 256))
    bias = np.random.normal(0, 0.01, 128)
    sparse_layer.load_dense_weights(weights, bias)
    
    # Prune weights
    sparse_layer.prune_weights(sparsity=0.8)  # 80% sparsity
    
    # Test forward passes
    x = np.random.normal(0, 1, (4, 256))  # Batch of 4
    
    # Compare outputs
    output_dense = sparse_layer.forward_dense(x)
    output_sparse_naive = sparse_layer.forward_sparse_naive(x)
    output_sparse_opt = sparse_layer.forward_sparse_optimized(x)
    
    print(f"Output shapes:")
    print(f"  Dense: {output_dense.shape}")
    print(f"  Sparse naive: {output_sparse_naive.shape}")
    print(f"  Sparse optimized: {output_sparse_opt.shape}")
    
    # Verify outputs have correct shape
    expected_shape = (4, 128)
    assert output_dense.shape == expected_shape, "Dense output shape incorrect"
    assert output_sparse_naive.shape == expected_shape, "Sparse naive output shape incorrect"
    assert output_sparse_opt.shape == expected_shape, "Sparse optimized output shape incorrect"
    
    # Verify sparse outputs match expected computation
    # Sparse naive should match dense computation on pruned weights
    np.testing.assert_allclose(
        output_sparse_naive, output_sparse_opt, rtol=1e-5,
        err_msg="Sparse naive and optimized should produce same results"
    )
    
    # The outputs shouldn't be identical (due to pruning) but should be reasonably close
    relative_error = np.mean(np.abs(output_dense - output_sparse_naive)) / np.mean(np.abs(output_dense))
    print(f"Relative error from pruning: {relative_error:.3%}")
    # With 80% sparsity, relative error can be substantial but model should still function
    assert relative_error < 1.0, "Error from pruning shouldn't completely destroy the model"
    
    # Benchmark performance
    benchmark = sparse_layer.benchmark_speedup(batch_size=32, iterations=50)
    
    print(f"\nPerformance Benchmark:")
    print(f"  Sparsity: {benchmark['sparsity']:.1%}")
    print(f"  Dense ops: {benchmark['dense_ops']:,}")
    print(f"  Sparse ops: {benchmark['sparse_ops']:,}")
    print(f"  Theoretical speedup: {benchmark['theoretical_speedup']:.1f}x")
    print(f"  Actual speedup: {benchmark['actual_speedup']:.1f}x")
    print(f"  Efficiency: {benchmark['efficiency']:.1%}")
    
    # Verify operation counting
    expected_dense_ops = 256 * 128
    assert benchmark['dense_ops'] == expected_dense_ops, "Dense op count incorrect"
    assert benchmark['sparse_ops'] < benchmark['dense_ops'], "Sparse should use fewer ops"
    
    print("✅ Sparse neural network test passed!")

test_sparse_neural_network()

# %% [markdown]
"""
## Part 5: Model Compression Pipeline - End-to-End Pruning

Now let's build a complete model compression pipeline that can prune entire neural networks layer by layer, maintaining the overall architecture while reducing parameters.

### Production Compression Pipeline:
1. **Model analysis**: Identify pruneable layers and sensitivity
2. **Layer-wise pruning**: Apply different sparsity levels per layer
3. **Accuracy validation**: Ensure pruning doesn't degrade performance  
4. **Performance benchmarking**: Measure actual compression benefits
5. **Export for deployment**: Package compressed model for inference
"""

# %% nbgrader={"grade": false, "grade_id": "compression-pipeline", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class ModelCompressor:
    """
    Complete model compression pipeline for neural networks.
    
    This class implements production-ready compression workflows
    that can handle complex models with mixed layer types.
    """
    
    def __init__(self):
        # BEGIN SOLUTION
        self.original_model = {}
        self.compressed_model = {}
        self.compression_stats = {}
        self.layer_sensitivities = {}
        # END SOLUTION
    
    def analyze_model_for_compression(self, model_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze model structure to determine compression strategy.
        
        Args:
            model_weights: Dictionary mapping layer names to weight arrays
            
        Returns:
            analysis: Compression analysis and recommendations
        """
        # BEGIN SOLUTION
        analysis = {
            'layers': {},
            'total_params': 0,
            'compressible_params': 0,
            'recommendations': {}
        }
        
        print("🔍 Model Compression Analysis")
        print("=" * 50)
        print("Layer        | Type    | Parameters | Natural Sparsity | Recommendation")
        print("-" * 70)
        
        for layer_name, weights in model_weights.items():
            layer_analysis = analyze_weight_redundancy(weights, f"Layer {layer_name}")
            
            # Determine layer type from shape
            if len(weights.shape) == 4:  # Conv layer: (out, in, H, W)
                layer_type = "Conv2D"
                recommended_sparsity = 0.6  # Conservative for conv layers
            elif len(weights.shape) == 2:  # Dense layer: (out, in)  
                layer_type = "Dense"
                recommended_sparsity = 0.8  # Aggressive for dense layers
            else:
                layer_type = "Other"
                recommended_sparsity = 0.5  # Safe default
            
            analysis['layers'][layer_name] = {
                'type': layer_type,
                'shape': weights.shape,
                'parameters': weights.size,
                'natural_sparsity': layer_analysis['natural_sparsity'],
                'recommended_sparsity': recommended_sparsity
            }
            
            analysis['total_params'] += weights.size
            if layer_type in ['Conv2D', 'Dense']:
                analysis['compressible_params'] += weights.size
            
            print(f"{layer_name:12} | {layer_type:7} | {weights.size:10,} | "
                  f"{layer_analysis['natural_sparsity']:12.1f}% | {recommended_sparsity:.0%}")
        
        # Calculate overall compression potential
        compression_potential = analysis['compressible_params'] / analysis['total_params']
        
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {analysis['total_params']:,}")
        print(f"   Compressible parameters: {analysis['compressible_params']:,}")
        print(f"   Compression potential: {compression_potential:.1%}")
        
        analysis['compression_potential'] = compression_potential
        return analysis
        # END SOLUTION
    
    def compress_model(self, model_weights: Dict[str, np.ndarray], 
                      layer_sparsities: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compress entire model using layer-wise pruning.
        
        Args:
            model_weights: Dictionary mapping layer names to weights
            layer_sparsities: Optional per-layer sparsity targets
            
        Returns:
            compressed_model: Compressed weights and statistics
        """
        # BEGIN SOLUTION
        if layer_sparsities is None:
            # Use default sparsities based on layer analysis
            analysis = self.analyze_model_for_compression(model_weights)
            layer_sparsities = {
                name: info['recommended_sparsity'] 
                for name, info in analysis['layers'].items()
            }
        
        print(f"\n⚙️  Compressing Model Layers")
        print("=" * 50)
        
        compressed_weights = {}
        total_original_params = 0
        total_remaining_params = 0
        
        for layer_name, weights in model_weights.items():
            sparsity = layer_sparsities.get(layer_name, 0.7)  # Default 70%
            
            print(f"\n🔧 Compressing {layer_name} (target: {sparsity:.0%} sparsity)...")
            
            # Apply magnitude-based pruning
            pruner = MagnitudePruner()
            pruned_weights, mask, stats = pruner.prune(weights, sparsity)
            
            compressed_weights[layer_name] = {
                'weights': pruned_weights,
                'mask': mask,
                'original_shape': weights.shape,
                'stats': stats
            }
            
            total_original_params += stats['original_params']
            total_remaining_params += stats['remaining_params']
            
            print(f"   Sparsity achieved: {stats['actual_sparsity']:.1%}")
            print(f"   Compression: {stats['compression_ratio']:.1f}x")
        
        # Calculate overall compression
        overall_compression = total_original_params / total_remaining_params if total_remaining_params > 0 else 1
        overall_sparsity = 1 - (total_remaining_params / total_original_params)
        
        self.compressed_model = compressed_weights
        self.compression_stats = {
            'total_original_params': total_original_params,
            'total_remaining_params': total_remaining_params,
            'overall_sparsity': overall_sparsity,
            'overall_compression': overall_compression,
            'layer_sparsities': layer_sparsities
        }
        
        print(f"\n✅ Model Compression Complete!")
        print(f"   Original parameters: {total_original_params:,}")
        print(f"   Remaining parameters: {total_remaining_params:,}")
        print(f"   Overall sparsity: {overall_sparsity:.1%}")
        print(f"   Overall compression: {overall_compression:.1f}x")
        
        return compressed_weights
        # END SOLUTION
    
    def validate_compression_quality(self, original_weights: Dict[str, np.ndarray], 
                                   compressed_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that compression doesn't degrade model too much.
        
        This is a simplified validation - in practice you'd run full model evaluation.
        """
        # BEGIN SOLUTION
        validation_results = {
            'layer_quality': {},
            'overall_quality': {},
            'quality_score': 0.0
        }
        
        print(f"\n✅ Validating Compression Quality")
        print("=" * 50)
        print("Layer        | Weight Error | Norm Preservation | Quality")
        print("-" * 55)
        
        layer_scores = []
        
        for layer_name in original_weights.keys():
            original = original_weights[layer_name]
            compressed_info = compressed_model[layer_name]
            compressed = compressed_info['weights']
            
            # Calculate quality metrics
            weight_diff = np.abs(original - compressed)
            mean_error = weight_diff.mean()
            max_error = weight_diff.max()
            
            # Norm preservation
            orig_norm = np.linalg.norm(original)
            comp_norm = np.linalg.norm(compressed)
            norm_preservation = comp_norm / orig_norm if orig_norm > 0 else 1.0
            
            # Simple quality score (higher is better)
            # Penalize high error, reward norm preservation
            quality_score = norm_preservation * (1 - mean_error / (np.abs(original).mean() + 1e-8))
            quality_score = max(0, min(1, quality_score))  # Clamp to [0, 1]
            
            validation_results['layer_quality'][layer_name] = {
                'mean_error': mean_error,
                'max_error': max_error,
                'norm_preservation': norm_preservation,
                'quality_score': quality_score
            }
            
            layer_scores.append(quality_score)
            
            print(f"{layer_name:12} | {mean_error:.6f} | {norm_preservation:13.3f} | {quality_score:.3f}")
        
        # Overall quality
        overall_quality_score = np.mean(layer_scores)
        validation_results['overall_quality'] = {
            'mean_quality_score': overall_quality_score,
            'quality_std': np.std(layer_scores),
            'min_quality': np.min(layer_scores),
            'max_quality': np.max(layer_scores)
        }
        validation_results['quality_score'] = overall_quality_score
        
        print(f"\n🎯 Overall Quality Score: {overall_quality_score:.3f}")
        if overall_quality_score > 0.8:
            print("   ✅ Excellent compression quality!")
        elif overall_quality_score > 0.6:
            print("   ⚠️  Acceptable compression quality")  
        else:
            print("   ❌ Poor compression quality - consider lower sparsity")
        
        return validation_results
        # END SOLUTION

# %% [markdown]
"""
### Test: Model Compression Pipeline

Let's verify our complete compression pipeline works on a multi-layer model.
"""

# %% nbgrader={"grade": true, "grade_id": "test-compression-pipeline", "locked": false, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_compression_pipeline():
    """Test complete model compression pipeline."""
    print("Testing model compression pipeline...")
    
    # Create sample multi-layer model
    np.random.seed(42)
    model_weights = {
        'conv1': np.random.normal(0, 0.02, (32, 3, 3, 3)),    # Conv: 32 filters, 3 input channels
        'conv2': np.random.normal(0, 0.02, (64, 32, 3, 3)),   # Conv: 64 filters, 32 input channels
        'fc1': np.random.normal(0, 0.01, (512, 1024)),        # Dense: 512 → 1024
        'fc2': np.random.normal(0, 0.01, (10, 512)),          # Dense: 10 → 512 (output layer)
    }
    
    # Create compressor
    compressor = ModelCompressor()
    
    # Step 1: Analyze model
    analysis = compressor.analyze_model_for_compression(model_weights)
    
    assert analysis['total_params'] > 0, "Should count total parameters"
    assert len(analysis['layers']) == 4, "Should analyze all 4 layers"
    assert 'conv1' in analysis['layers'], "Should analyze conv1"
    assert 'fc1' in analysis['layers'], "Should analyze fc1"
    
    # Verify layer type detection
    assert analysis['layers']['conv1']['type'] == 'Conv2D', "Should detect conv layers"
    assert analysis['layers']['fc1']['type'] == 'Dense', "Should detect dense layers"
    
    # Step 2: Compress model with custom sparsities
    custom_sparsities = {
        'conv1': 0.5,  # Conservative for first conv layer
        'conv2': 0.6,  # Moderate for second conv layer
        'fc1': 0.8,    # Aggressive for large dense layer
        'fc2': 0.3     # Conservative for output layer
    }
    
    compressed_model = compressor.compress_model(model_weights, custom_sparsities)
    
    # Verify compression results
    assert len(compressed_model) == 4, "Should compress all layers"
    for layer_name in model_weights.keys():
        assert layer_name in compressed_model, f"Missing compressed {layer_name}"
        compressed_info = compressed_model[layer_name]
        assert 'weights' in compressed_info, "Should have compressed weights"
        assert 'mask' in compressed_info, "Should have pruning mask"
        assert 'stats' in compressed_info, "Should have compression stats"
    
    # Verify compression statistics
    stats = compressor.compression_stats
    assert stats['overall_compression'] > 2.0, "Should achieve significant compression"
    assert 0.5 <= stats['overall_sparsity'] <= 0.8, "Overall sparsity should be reasonable"
    
    # Step 3: Validate compression quality
    validation = compressor.validate_compression_quality(model_weights, compressed_model)
    
    assert 'layer_quality' in validation, "Should validate each layer"
    assert 'overall_quality' in validation, "Should have overall quality metrics"
    assert 0 <= validation['quality_score'] <= 1, "Quality score should be normalized"
    
    # Each layer should have quality metrics
    for layer_name in model_weights.keys():
        assert layer_name in validation['layer_quality'], f"Missing quality for {layer_name}"
        layer_quality = validation['layer_quality'][layer_name]
        assert 'norm_preservation' in layer_quality, "Should measure norm preservation"
        assert layer_quality['norm_preservation'] > 0, "Norm preservation should be positive"
    
    # Test that compressed weights are actually sparse
    for layer_name, compressed_info in compressed_model.items():
        compressed_weights = compressed_info['weights']
        sparsity = np.sum(compressed_weights == 0) / compressed_weights.size
        expected_sparsity = custom_sparsities[layer_name]
        
        # Allow some tolerance in sparsity
        assert abs(sparsity - expected_sparsity) < 0.1, f"{layer_name} sparsity mismatch"
    
    print("✅ Model compression pipeline test passed!")

test_compression_pipeline()

# %% [markdown]
"""
## Part 6: Systems Analysis - Memory, Performance, and Deployment Impact

Let's analyze compression from a systems engineering perspective, measuring the real-world impact on memory usage, inference speed, and deployment scenarios.

### ML Systems Analysis: Why Pruning Enables Edge AI

**Memory Complexity**: O(N × sparsity) storage reduction where N = original parameters
**Computational Complexity**: Theoretical O(N × sparsity) speedup, actual depends on hardware
**Cache Efficiency**: Smaller models fit in cache, reducing memory bandwidth bottlenecks  
**Energy Efficiency**: Fewer operations = lower power consumption for mobile devices
**Deployment Enablement**: Makes models fit where they couldn't before
"""

# %% nbgrader={"grade": false, "grade_id": "compression-systems-analysis", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def profile_compression_memory():
    """
    Profile memory usage patterns during model compression.
    
    This function demonstrates how compression affects memory footprint
    and enables deployment on resource-constrained devices.
    """
    import tracemalloc
    
    print("🔬 Memory Profiling: Model Compression")
    print("=" * 50)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create large model (simulating real CNN)
    print("Creating large model weights...")
    model_weights = {
        'conv1': np.random.normal(0, 0.02, (128, 64, 3, 3)),     # ~0.3M parameters
        'conv2': np.random.normal(0, 0.02, (256, 128, 3, 3)),    # ~1.2M parameters  
        'fc1': np.random.normal(0, 0.01, (1024, 4096)),          # ~4.2M parameters
        'fc2': np.random.normal(0, 0.01, (10, 1024)),            # ~10K parameters
    }
    
    snapshot1 = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    print(f"After model creation: {current / 1024 / 1024:.1f} MB current, {peak / 1024 / 1024:.1f} MB peak")
    
    # Calculate original model size
    original_params = sum(w.size for w in model_weights.values())
    original_size_mb = sum(w.nbytes for w in model_weights.values()) / (1024 * 1024)
    
    print(f"Original model: {original_params:,} parameters, {original_size_mb:.1f} MB")
    
    # Compress model
    print("\nCompressing model...")
    compressor = ModelCompressor()
    compressed_model = compressor.compress_model(model_weights)
    
    snapshot2 = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    print(f"After compression: {current / 1024 / 1024:.1f} MB current, {peak / 1024 / 1024:.1f} MB peak")
    
    # Calculate compressed model size
    compressed_params = sum(
        np.sum(info['weights'] != 0) 
        for info in compressed_model.values()
    )
    
    # Estimate compressed storage (could use sparse formats)
    compressed_size_mb = original_size_mb * (compressed_params / original_params)
    
    print(f"\n💾 Storage Analysis:")
    print(f"   Original: {original_params:,} parameters ({original_size_mb:.1f} MB)")
    print(f"   Compressed: {compressed_params:,} parameters ({compressed_size_mb:.1f} MB)")
    print(f"   Compression ratio: {original_params / compressed_params:.1f}x")
    print(f"   Size reduction: {original_size_mb / compressed_size_mb:.1f}x")
    print(f"   Storage savings: {original_size_mb - compressed_size_mb:.1f} MB")
    
    tracemalloc.stop()
    
    return {
        'original_params': original_params,
        'compressed_params': compressed_params,
        'original_size_mb': original_size_mb,
        'compressed_size_mb': compressed_size_mb,
        'compression_ratio': original_params / compressed_params,
        'size_reduction': original_size_mb / compressed_size_mb
    }

def analyze_deployment_scenarios():
    """Analyze how compression enables different deployment scenarios."""
    print("\n🚀 Compression Deployment Impact Analysis")
    print("=" * 60)
    
    # Define deployment constraints
    scenarios = [
        {
            'name': 'Mobile Phone',
            'memory_limit_mb': 100,
            'compute_limit_gflops': 10,
            'power_sensitive': True,
            'description': 'On-device inference for camera apps'
        },
        {
            'name': 'IoT Device',
            'memory_limit_mb': 20,
            'compute_limit_gflops': 1,
            'power_sensitive': True,
            'description': 'Smart sensor with microcontroller'
        },
        {
            'name': 'Edge Server',
            'memory_limit_mb': 1000,
            'compute_limit_gflops': 100,
            'power_sensitive': False,
            'description': 'Local inference server for privacy'
        },
        {
            'name': 'Wearable',
            'memory_limit_mb': 10,
            'compute_limit_gflops': 0.5,
            'power_sensitive': True,
            'description': 'Smartwatch health monitoring'
        }
    ]
    
    # Model sizes at different compression levels
    model_configs = [
        {'name': 'Dense Model', 'size_mb': 200, 'gflops': 50, 'accuracy': 95.0},
        {'name': '50% Sparse', 'size_mb': 100, 'gflops': 25, 'accuracy': 94.5},
        {'name': '70% Sparse', 'size_mb': 60, 'gflops': 15, 'accuracy': 93.8},
        {'name': '90% Sparse', 'size_mb': 20, 'gflops': 5, 'accuracy': 91.2},
    ]
    
    print("Scenario       | Memory | Compute | Dense | 50% | 70% | 90% | Best Option")
    print("-" * 80)
    
    for scenario in scenarios:
        name = scenario['name']
        mem_limit = scenario['memory_limit_mb']
        compute_limit = scenario['compute_limit_gflops']
        
        # Check which model configurations fit
        viable_models = []
        for config in model_configs:
            fits_memory = config['size_mb'] <= mem_limit
            fits_compute = config['gflops'] <= compute_limit
            
            if fits_memory and fits_compute:
                viable_models.append(config['name'])
        
        # Determine best option
        if not viable_models:
            best_option = "None fit!"
        else:
            # Choose highest accuracy among viable options
            viable_configs = [c for c in model_configs if c['name'] in viable_models]
            best_config = max(viable_configs, key=lambda x: x['accuracy'])
            best_option = f"{best_config['name']} ({best_config['accuracy']:.1f}%)"
        
        # Show fit status for each compression level
        fit_status = []
        for config in model_configs:
            fits_mem = config['size_mb'] <= mem_limit
            fits_comp = config['gflops'] <= compute_limit
            if fits_mem and fits_comp:
                status = "✅"
            elif fits_mem:
                status = "⚡"  # Memory OK, compute too high
            elif fits_comp:
                status = "💾"  # Compute OK, memory too high
            else:
                status = "❌"
            fit_status.append(status)
        
        print(f"{name:14} | {mem_limit:4d}MB | {compute_limit:5.1f}G | "
              f"{fit_status[0]:3} | {fit_status[1]:3} | {fit_status[2]:3} | {fit_status[3]:3} | {best_option}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Compression often determines deployment feasibility")
    print(f"   • Edge devices require 70-90% sparsity for deployment")
    print(f"   • Mobile devices can use moderate compression (50-70%)")
    print(f"   • Power constraints favor sparse models (fewer operations)")
    print(f"   • Memory limits are often more restrictive than compute limits")

def benchmark_sparse_inference_speedup():
    """Benchmark actual vs theoretical speedup from sparsity."""
    print("\n⚡ Sparse Inference Speedup Analysis")
    print("=" * 50)
    
    import time
    
    # Test different model sizes and sparsity levels
    configs = [
        {'size': (256, 512), 'sparsity': 0.5},
        {'size': (512, 1024), 'sparsity': 0.7},
        {'size': (1024, 2048), 'sparsity': 0.8},
        {'size': (2048, 4096), 'sparsity': 0.9},
    ]
    
    print("Model Size    | Sparsity | Theoretical | Actual | Efficiency | Notes")
    print("-" * 70)
    
    for config in configs:
        size = config['size']
        sparsity = config['sparsity']
        
        # Create sparse layer
        sparse_layer = SparseLinear(size[0], size[1])
        
        # Load and prune weights
        weights = np.random.normal(0, 0.1, (size[1], size[0]))
        sparse_layer.load_dense_weights(weights)
        sparse_layer.prune_weights(sparsity)
        
        # Benchmark
        benchmark = sparse_layer.benchmark_speedup(batch_size=16, iterations=100)
        
        theoretical = benchmark['theoretical_speedup']
        actual = benchmark['actual_speedup'] 
        efficiency = benchmark['efficiency']
        
        # Determine bottleneck
        if efficiency > 0.8:
            notes = "CPU bound"
        elif efficiency > 0.5:
            notes = "Memory bound"
        else:
            notes = "Framework overhead"
        
        print(f"{size[0]}x{size[1]:4} | {sparsity:6.0%} | {theoretical:9.1f}x | "
              f"{actual:5.1f}x | {efficiency:8.1%} | {notes}")
    
    print(f"\n🎯 Speedup Reality Check:")
    print(f"   • Theoretical speedup assumes perfect sparse hardware")
    print(f"   • Actual speedup limited by memory bandwidth and overhead")
    print(f"   • High sparsity (>80%) shows diminishing returns") 
    print(f"   • Production sparse hardware (GPUs, TPUs) achieve better efficiency")

# %% [markdown]
"""
### Test: Systems Analysis Implementation

Let's verify our systems analysis provides valuable performance insights.
"""

# %% nbgrader={"grade": true, "grade_id": "test-systems-analysis", "locked": false, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_systems_analysis():
    """Test systems analysis and profiling functions."""
    print("Testing systems analysis...")
    
    # Test memory profiling
    memory_results = profile_compression_memory()
    assert memory_results['compression_ratio'] > 2.0, "Should show significant compression"
    assert memory_results['original_size_mb'] > memory_results['compressed_size_mb'], "Should reduce size"
    
    # Test deployment analysis
    analyze_deployment_scenarios()
    
    # Test speedup benchmarking
    benchmark_sparse_inference_speedup()
    
    # All functions should run without errors
    print("✅ Systems analysis test passed!")

test_systems_analysis()

# %% [markdown]
"""
## Part 7: Production Context - Real-World Pruning Systems

Let's explore how pruning is used in production ML systems and connect our implementation to real frameworks and deployment platforms.

### Production Pruning Systems:
1. **PyTorch Pruning**: `torch.nn.utils.prune` for magnitude and structured pruning
2. **TensorFlow Model Optimization**: Pruning API with gradual sparsity
3. **NVIDIA TensorRT**: Structured pruning for inference acceleration
4. **OpenVINO**: Intel's optimization toolkit with pruning support
5. **Edge TPU**: Google's quantization + pruning for mobile inference
6. **Apple Neural Engine**: Hardware-accelerated sparse computation
"""

# %% nbgrader={"grade": false, "grade_id": "production-context", "locked": false, "schema_version": 3, "solution": true, "task": false}
def compare_with_production_pruning():
    """
    Compare our implementation with production pruning systems.
    
    This function explains how real ML frameworks handle pruning
    and where our implementation fits in the broader ecosystem.
    """
    print("🏭 Production Pruning Systems Comparison")
    print("=" * 70)
    
    frameworks = {
        'PyTorch': {
            'pruning_methods': ['Magnitude', 'Random', 'Structured', 'Custom'],
            'sparsity_support': ['Unstructured', 'Structured (channel)', '2:4 sparsity'],
            'deployment': 'TorchScript, ONNX export with sparse ops',
            'hardware_acceleration': 'Limited - mostly research focused',
            'our_similarity': 'High - similar magnitude-based approach'
        },
        'TensorFlow': {
            'pruning_methods': ['Magnitude', 'Gradual', 'Structured'],
            'sparsity_support': ['Unstructured', 'Block sparse', 'Structured'],
            'deployment': 'TensorFlow Lite with sparse inference',
            'hardware_acceleration': 'XLA optimization, mobile acceleration',
            'our_similarity': 'High - magnitude pruning with calibration'
        },
        'TensorRT': {
            'pruning_methods': ['Structured only', 'Channel pruning'],
            'sparsity_support': ['2:4 structured sparsity', 'Channel removal'],
            'deployment': 'Optimized inference engine with sparse kernels',
            'hardware_acceleration': 'GPU Tensor Cores, specialized sparse ops',
            'our_similarity': 'Medium - focuses on structured pruning'
        },
        'OpenVINO': {
            'pruning_methods': ['Magnitude', 'Structured', 'Mixed precision'],
            'sparsity_support': ['Unstructured', 'Block sparse', 'Channel wise'],
            'deployment': 'Intel CPU/GPU optimization with sparse support',
            'hardware_acceleration': 'Intel VPU, CPU vectorization',
            'our_similarity': 'High - comprehensive pruning toolkit'
        },
        'Our TinyTorch': {
            'pruning_methods': ['Magnitude-based', 'Structured filter pruning'],
            'sparsity_support': ['Unstructured', 'Structured (filter removal)'],
            'deployment': 'Educational sparse computation simulation',
            'hardware_acceleration': 'Educational - simulated speedups',
            'our_similarity': 'Reference implementation for learning'
        }
    }
    
    print("Framework | Methods | Hardware Support | Deployment | Similarity")
    print("-" * 70)
    
    for name, specs in frameworks.items():
        methods_str = specs['pruning_methods'][0]  # Primary method
        hw_str = specs['hardware_acceleration'][:20] + "..." if len(specs['hardware_acceleration']) > 20 else specs['hardware_acceleration']
        deploy_str = specs['deployment'][:20] + "..." if len(specs['deployment']) > 20 else specs['deployment']
        sim_str = specs['our_similarity'][:15] + "..." if len(specs['our_similarity']) > 15 else specs['our_similarity']
        
        print(f"{name:9} | {methods_str:12} | {hw_str:16} | {deploy_str:12} | {sim_str}")
    
    print(f"\n🎯 Key Production Insights:")
    print(f"   • Our magnitude approach is industry standard")
    print(f"   • Production systems emphasize structured pruning for hardware")
    print(f"   • Real frameworks integrate pruning with quantization")
    print(f"   • Hardware acceleration requires specialized sparse kernels")
    print(f"   • Mobile deployment drives most production pruning adoption")

def demonstrate_pruning_applications():
    """Show real-world applications where pruning enables deployment."""
    print("\n🌟 Real-World Pruning Applications")
    print("=" * 50)
    
    applications = [
        {
            'domain': 'Mobile Photography',
            'model': 'Portrait segmentation CNN',
            'constraints': '< 10MB, < 100ms inference',
            'pruning_strategy': '70% unstructured + quantization',
            'outcome': 'Real-time portrait mode on phone cameras',
            'example': 'Google Pixel, iPhone portrait mode'
        },
        {
            'domain': 'Autonomous Vehicles', 
            'model': 'Object detection (YOLO)',
            'constraints': '< 500MB, < 50ms inference, safety critical',
            'pruning_strategy': '50% structured pruning for latency',
            'outcome': 'Real-time object detection for ADAS',
            'example': 'Tesla FSD, Waymo perception stack'
        },
        {
            'domain': 'Smart Home',
            'model': 'Voice keyword detection',
            'constraints': '< 1MB, always-on, battery powered',
            'pruning_strategy': '90% sparsity + 8-bit quantization',
            'outcome': 'Always-listening wake word detection',
            'example': 'Alexa, Google Assistant edge processing'
        },
        {
            'domain': 'Medical Imaging',
            'model': 'X-ray diagnosis CNN',
            'constraints': 'Edge deployment, <1GB memory',
            'pruning_strategy': '60% structured pruning + knowledge distillation',
            'outcome': 'Portable medical AI for remote clinics',
            'example': 'Google AI for radiology, Zebra Medical'
        },
        {
            'domain': 'Augmented Reality',
            'model': 'Hand tracking and gesture recognition',
            'constraints': '< 50MB, 60fps, mobile GPU',
            'pruning_strategy': 'Channel pruning + mobile-optimized architecture',
            'outcome': 'Real-time hand tracking for AR experiences',
            'example': 'Apple ARKit, Google ARCore, Meta Quest'
        }
    ]
    
    print("Domain              | Model Type | Pruning Strategy | Outcome")
    print("-" * 75)
    
    for app in applications:
        domain_str = app['domain'][:18]
        model_str = app['model'][:15] + "..." if len(app['model']) > 15 else app['model']
        strategy_str = app['pruning_strategy'][:20] + "..." if len(app['pruning_strategy']) > 20 else app['pruning_strategy']
        outcome_str = app['outcome'][:25] + "..." if len(app['outcome']) > 25 else app['outcome']
        
        print(f"{domain_str:18} | {model_str:10} | {strategy_str:16} | {outcome_str}")
        print(f"                   Example: {app['example']}")
        print()
    
    print("💡 Common Patterns in Production Pruning:")
    print("   • Latency-critical apps use structured pruning (regular sparsity)")  
    print("   • Memory-constrained devices use aggressive unstructured pruning")
    print("   • Safety-critical systems use conservative pruning with validation")
    print("   • Mobile apps combine pruning + quantization for maximum compression")
    print("   • Edge AI enables privacy (on-device processing) through compression")

# %% [markdown]
"""
### Test: Production Context Analysis

Let's verify our production context analysis provides valuable insights.
"""

# %% nbgrader={"grade": true, "grade_id": "test-production-context", "locked": false, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_production_context():
    """Test production context analysis."""
    print("Testing production context analysis...")
    
    # Test framework comparison
    compare_with_production_pruning()
    
    # Test applications demonstration
    demonstrate_pruning_applications()
    
    # Both functions should run without errors and provide insights
    print("✅ Production context analysis test passed!")

test_production_context()

# %% [markdown]
"""
## Comprehensive Testing

Let's run a comprehensive test of all compression functionality to ensure everything works together correctly.
"""

# %% nbgrader={"grade": false, "grade_id": "comprehensive-testing", "locked": false, "schema_version": 3, "solution": false, "task": false}
def run_all_tests():
    """Run comprehensive test suite for compression module."""
    print("🧪 Running Comprehensive Compression Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Weight Redundancy Analysis", test_redundancy_analysis),
        ("Magnitude-Based Pruning", test_magnitude_pruning),
        ("Structured Pruning", test_structured_pruning),
        ("Sparse Neural Network", test_sparse_neural_network),
        ("Model Compression Pipeline", test_compression_pipeline),
        ("Systems Analysis", test_systems_analysis),
        ("Production Context", test_production_context)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compression tests passed! Module implementation complete.")
        
        # Show final demo
        print(f"\n🚀 Final Compression Demo:")
        print("=" * 50)
        
        # Create a realistic model and compress it
        np.random.seed(42)
        demo_model = {
            'backbone_conv': np.random.normal(0, 0.02, (128, 64, 3, 3)),
            'classifier_fc': np.random.normal(0, 0.01, (10, 2048)),
        }
        
        compressor = ModelCompressor()
        compressed = compressor.compress_model(demo_model, {'backbone_conv': 0.7, 'classifier_fc': 0.8})
        
        original_params = sum(w.size for w in demo_model.values())
        compressed_params = sum(np.sum(info['weights'] != 0) for info in compressed.values())
        
        print(f"🎯 FINAL RESULT:")
        print(f"   Original model: {original_params:,} parameters")
        print(f"   Compressed model: {compressed_params:,} parameters")
        print(f"   Compression achieved: {original_params/compressed_params:.1f}x smaller")
        print(f"   Size reduction: {(1-compressed_params/original_params)*100:.1f}% of parameters removed")
        print(f"   ✅ Ready for edge deployment!")
        
    else:
        print(f"⚠️  {total - passed} tests failed. Review implementation.")

if __name__ == "__main__":
    run_all_tests()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented neural network pruning, let's reflect on the systems engineering principles and production deployment considerations.

**Instructions**: Think through these questions based on your implementation experience. Consider both the technical details and the broader systems implications.
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-1", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
**Question 1: Pruning Strategy Analysis**

You implemented both magnitude-based and structured pruning in your `MagnitudePruner` and `prune_conv_filters()` functions:

a) Why does magnitude-based pruning work so well for neural networks? What does the effectiveness of this simple heuristic tell us about neural network weight distributions?

b) In your structured vs unstructured comparison, structured pruning achieved lower compression ratios but is preferred for deployment. Explain this tradeoff in terms of hardware efficiency and inference speed.

c) Your compression pipeline used different sparsity targets per layer (conv: 60%, dense: 80%). Why do dense layers typically tolerate higher sparsity than convolutional layers?

**Your Answer:**

<!-- BEGIN SOLUTION -->
a) Magnitude-based pruning works because:
- Neural networks exhibit natural redundancy with many small, unimportant weights
- Weight magnitude correlates with importance - small weights contribute little to output
- Networks are over-parametrized, so removing low-magnitude weights has minimal accuracy impact
- The success reveals that weight distributions have long tails - most weights are small, few are large
- This natural sparsity suggests networks learn efficient representations despite overparametrization

b) The structured vs unstructured tradeoff:
- Unstructured: Higher compression (removes individual weights) but irregular sparsity patterns
- Structured: Lower compression (removes entire filters/channels) but regular, hardware-friendly patterns
- Hardware prefers structured because: dense computation on smaller tensors is faster than sparse computation
- Memory access: structured removal reduces tensor sizes, improving cache efficiency
- No need for specialized sparse kernels - can use standard GEMM operations
- Inference speed: structured pruning provides actual speedup, unstructured often theoretical only

c) Layer-specific sparsity tolerance:
- Dense layers: High redundancy, many parameters, more overparametrized → tolerate 80% sparsity
- Conv layers: Fewer parameters, each filter captures important spatial features → more sensitive
- First layers: Extract low-level features (edges, textures) → very sensitive to pruning
- Later layers: More abstract features with redundancy → can handle moderate pruning
- Output layers: Critical for final predictions → require conservative pruning
<!-- END SOLUTION -->
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-2", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
**Question 2: Sparse Computation and Hardware Efficiency**

Your `SparseLinear` class demonstrated the challenges of actually accelerating sparse computation:

a) Why did your sparse computation benchmarks show lower actual speedup compared to theoretical speedup? What are the main bottlenecks preventing sparse computation from achieving theoretical gains?

b) In your deployment analysis, mobile devices required 70-90% sparsity while edge servers could use 50%. Explain how hardware constraints drive pruning requirements differently across deployment targets.

c) You found that structured pruning provides better real-world performance than unstructured pruning. How would you design a neural network architecture that's naturally "pruning-friendly" from the start?

**Your Answer:**

<!-- BEGIN SOLUTION -->
a) Lower actual speedup due to multiple bottlenecks:
- Memory bandwidth: Sparse computation is often memory-bound, not compute-bound
- Framework overhead: PyTorch/NumPy not optimized for arbitrary sparsity patterns
- Cache inefficiency: Irregular sparse patterns hurt cache locality compared to dense operations
- Vectorization loss: SIMD instructions work best on dense, regular data patterns
- Index overhead: Storing and accessing sparse indices adds computational cost
- Hardware mismatch: Most CPUs/GPUs optimized for dense linear algebra, not sparse

b) Hardware-driven pruning requirements:
- Mobile: Strict memory (4GB total), battery, thermal constraints → need aggressive 70-90% sparsity
- Edge servers: More memory (16GB+), power, cooling → moderate 50% sparsity sufficient
- Cloud: Abundant resources → pruning for cost optimization, not necessity
- Embedded/IoT: Extreme constraints (MB not GB) → need structured pruning + quantization
- Different hardware accelerators: Edge TPU loves sparsity, standard GPUs don't benefit much

c) Pruning-friendly architecture design:
- Use more, smaller layers rather than fewer, large layers (easier to prune entire channels)
- Design with skip connections (allows aggressive pruning of individual branches)
- Separate feature extraction from classification (different pruning sensitivities)
- Use group convolutions (natural structured pruning boundaries)
- Design with mobile-first mindset (efficient from start, not compressed afterward)
- Consider lottery ticket initialization (start with good sparse subnetwork)
<!-- END SOLUTION -->
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-3", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
**Question 3: Model Compression Pipeline and Production Deployment**

Your `ModelCompressor` implemented a complete compression pipeline with analysis, compression, and validation:

a) Your pipeline analyzed each layer to recommend sparsity levels. In production deployment, how would you extend this to handle dynamic workloads where the optimal sparsity might change based on accuracy requirements or latency constraints?

b) You implemented quality validation by comparing weight preservation. But in production, what matters is end-to-end accuracy and latency. How would you design a compression validation system that ensures deployment success?

c) Looking at your production applications analysis, why is pruning often combined with other optimizations (quantization, knowledge distillation) rather than used alone? What are the complementary benefits?

**Your Answer:**

<!-- BEGIN SOLUTION -->
a) Dynamic compression for production:
- A/B testing framework: gradually adjust sparsity based on accuracy metrics in production
- Multi-model serving: maintain models at different compression levels (70%, 80%, 90% sparse)
- Dynamic switching: use less compressed models during high-accuracy periods, more during low-latency needs
- Feedback loop: monitor accuracy degradation and automatically adjust compression
- User-specific models: different compression for different user segments or use cases
- Time-based adaptation: more compression during peak load, less during quality-critical periods
- Canary deployments: test compression changes on small traffic percentage first

b) End-to-end validation system:
- Task-specific metrics: measure final accuracy, F1, BLEU - whatever matters for the application
- Latency benchmarking: measure actual inference time on target hardware
- A/B testing: compare compressed vs uncompressed models on real user traffic
- Regression testing: ensure compression doesn't break edge cases or specific inputs
- Hardware-specific validation: test on actual deployment hardware, not just development machines
- Load testing: verify performance under realistic concurrent inference loads
- Accuracy monitoring: continuous validation in production with automatic rollback triggers

c) Why pruning is combined with other optimizations:
- Pruning + quantization: attack both parameter count and parameter size (4x + 4x = 16x compression)
- Pruning + knowledge distillation: maintain accuracy while compressing (teacher-student training)
- Complementary bottlenecks: pruning reduces compute, quantization reduces memory bandwidth
- Different deployment needs: mobile needs both size and speed, cloud needs cost optimization
- Diminishing returns: 90% pruning alone may hurt accuracy, but 70% pruning + quantization achieves same compression with better accuracy
- Hardware optimization: different techniques work better on different hardware (GPU vs mobile CPU)
<!-- END SOLUTION -->
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-4", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
**Question 4: Edge AI and Deployment Enablement**

Based on your systems analysis and deployment scenarios:

a) Your memory profiling showed that pruning enables deployment where dense models won't fit. But pruning also changes the computational characteristics of models. How does this affect the entire ML systems stack, from training to serving?

b) In your production applications analysis, you saw pruning enabling privacy-preserving on-device AI. Explain how compression techniques like pruning change the fundamental economics and capabilities of AI deployment.

c) Looking forward, how do you think the relationship between model architectures, hardware capabilities, and compression techniques will evolve? What are the implications for ML systems engineering?

**Your Answer:**

<!-- BEGIN SOLUTION -->
a) Pruning affects the entire ML systems stack:
- Training: Need pruning-aware training, gradual sparsity increases, specialized optimizers
- Model versioning: Track both dense and compressed versions, compression parameters
- Serving infrastructure: Need sparse computation support, different batching strategies
- Monitoring: Different performance characteristics, need sparsity-aware metrics
- Debugging: Sparse models behave differently, need specialized debugging tools
- Hardware utilization: Lower compute utilization but different memory access patterns
- Load balancing: Sparse models have different latency profiles, affects request routing

b) Compression changes AI deployment economics:
- Democratizes AI: Enables AI on devices that couldn't run dense models (phones, IoT, wearables)
- Privacy transformation: On-device processing eliminates need to send data to cloud
- Cost structure shift: Reduces cloud compute costs, shifts processing to edge devices
- Latency improvement: Local processing eliminates network round-trips
- Offline capability: Compressed models enable AI without internet connectivity
- Market expansion: Creates new use cases impossible with cloud-only AI
- Energy efficiency: Critical for battery-powered devices, enables always-on AI

c) Future evolution predictions:
- Hardware-software co-design: Chips designed specifically for sparse computation (like Edge TPU)
- Architecture evolution: Networks designed for compression from scratch, not post-hoc optimization
- Automatic compression: ML systems that automatically find optimal compression for deployment targets
- Dynamic compression: Models that adapt compression level based on runtime constraints
- Compression-aware training: End-to-end training that considers deployment constraints
- Standardization: Common sparse formats and APIs across frameworks and hardware
- New paradigms: Mixture of experts, early exit networks - architecturally sparse models
- The future is compression-first design, not compression as afterthought
<!-- END SOLUTION -->
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Compression - Neural Network Pruning for Edge Deployment

### What You Accomplished

In this module, you built a complete **neural network compression system** using pruning techniques that remove 70% of parameters while maintaining 95%+ accuracy. You learned to:

**🔧 Core Implementation Skills:**
- **Magnitude-based pruning**: Identified and removed unimportant weights using simple yet effective heuristics
- **Structured vs unstructured pruning**: Built both approaches and understood their hardware tradeoffs
- **Sparse computation**: Implemented efficient sparse linear layers and benchmarked real vs theoretical speedups
- **End-to-end compression pipeline**: Created production-ready model compression with analysis, validation, and optimization

**📊 Systems Engineering Insights:**
- **Neural network redundancy**: Discovered that networks contain 70-90% redundant parameters that can be safely removed
- **Hardware efficiency tradeoffs**: Understood why structured pruning provides actual speedup while unstructured gives theoretical speedup
- **Memory vs compute optimization**: Learned how pruning reduces both memory footprint and computational requirements
- **Deployment enablement**: Saw how compression makes models fit where they previously couldn't run

**🏭 Production Understanding:**
- **Edge deployment scenarios**: Analyzed how pruning enables mobile, IoT, and embedded AI applications
- **Compression pipeline design**: Built systems that analyze, compress, and validate models for production deployment
- **Hardware-aware optimization**: Understood how different deployment targets require different pruning strategies
- **Quality assurance**: Implemented validation systems to ensure compression doesn't degrade model performance

### ML Systems Engineering Connection

This module demonstrates that **compression is fundamentally about enabling deployment**, not just reducing model size. You learned:

- **Why redundancy exists**: Neural networks are over-parametrized, creating massive compression opportunities
- **Hardware drives strategy**: Structured vs unstructured pruning choice depends on target hardware capabilities
- **Compression enables privacy**: On-device processing becomes possible when models are small enough
- **Systems thinking**: Compression affects the entire ML stack from training to serving

### Real-World Impact

Your compression implementation mirrors production systems used by:
- **Mobile AI**: Apple's Neural Engine, Google's Edge TPU leverage sparsity for efficient inference
- **Autonomous vehicles**: Tesla FSD uses pruning for real-time object detection
- **Smart devices**: Alexa, Google Assistant use extreme compression for always-on wake word detection
- **Medical AI**: Portable diagnostic systems enabled by compressed models

The techniques you built make the difference between AI that runs in the cloud versus AI that runs in your pocket - enabling privacy, reducing latency, and creating entirely new application categories.

**Next**: This completes our ML Systems engineering journey! You've now built the complete stack from tensors to production deployment, understanding how each component contributes to building real-world AI systems that scale.
"""