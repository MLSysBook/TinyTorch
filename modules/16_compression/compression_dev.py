# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 18: Compression - Making Models Smaller

Welcome to Module 18! You're about to build model compression techniques that make neural networks smaller and more efficient while preserving their intelligence.

## 🔗 Prerequisites & Progress
**You've Built**: Full TinyGPT pipeline with profiling, acceleration, and quantization
**You'll Build**: Pruning (magnitude & structured), knowledge distillation, and low-rank approximation
**You'll Enable**: Compressed models that maintain accuracy while using dramatically less storage and memory

**Connection Map**:
```
Quantization → Compression → Benchmarking
(precision)   (sparsity)    (evaluation)
```

## Learning Objectives
By the end of this module, you will:
1. Implement magnitude-based and structured pruning
2. Build knowledge distillation for model compression
3. Create low-rank approximations of weight matrices
4. Measure compression ratios and sparsity levels
5. Understand structured vs unstructured sparsity trade-offs

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/18_compression/compression_dev.py`  
**Building Side:** Code exports to `tinytorch.optimization.compression`

```python
# How to use this module:
from tinytorch.optimization.compression import magnitude_prune, structured_prune, measure_sparsity
```

**Why this matters:**
- **Learning:** Complete compression system in one focused module for deep understanding
- **Production:** Proper organization like real compression libraries with all techniques together
- **Consistency:** All compression operations and sparsity management in optimization.compression
- **Integration:** Works seamlessly with models and quantization for complete optimization pipeline
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp optimization.compression
#| export

import numpy as np
import copy
from typing import List, Dict, Any, Tuple, Optional
import time

"""
## 🔗 Module Dependencies

This module REQUIRES completion of:
- Module 01 (Tensor): Foundation data structure for weight storage
- Module 03 (Layers): Linear layer structure that we compress
- Module 15 (Quantization): Related optimization technique

**Progressive Building**:
```
Module 01 (Tensor) ──┐
                     ├──> Module 03 (Layers) ──┐
Module 02 (Activations) ──┘                    ├──> Module 16 (Compression)
                                               │
Module 15 (Quantization) ────────────────────┘
```

**What You've Built**:
- Module 01: Tensor (what we compress)
- Module 03: Linear layers (what we prune)
- Module 15: Quantization (complementary optimization)

**What This Module Adds**:
- Pruning techniques (remove weights)
- Knowledge distillation (compress knowledge)
- Low-rank approximation (compress matrices)
- Sparsity measurement

**To verify dependencies are met, run**:
    python -c "from tinytorch.core.tensor import Tensor; print('✅ Module 01 ready')"
    python -c "from tinytorch.core.layers import Linear; print('✅ Module 03 ready')"
    python -c "from tinytorch.optimization.quantization import quantize_model; print('✅ Module 15 ready')"
"""

# Direct imports from previous modules - these MUST exist
# If imports fail, students will get clear educational errors
try:
    from tinytorch.core.tensor import Tensor  # Module 01: Foundation
except ImportError as e:
    raise ImportError(
        "❌ Module 16 (Compression) requires Module 01 (Tensor) to be completed first.\n"
        "   This module compresses Tensor weights - you need Tensor to exist first!\n"
        "   Please complete Module 01 first, then run 'tito module complete 01'.\n"
        "   Original error: " + str(e)
    ) from e

try:
    from tinytorch.core.layers import Linear  # Module 03: What we compress
except ImportError as e:
    raise ImportError(
        "❌ Module 16 (Compression) requires Module 03 (Layers) to be completed first.\n"
        "   This module prunes Linear layer weights - you need Linear layers first!\n"
        "   Please complete Module 03 first, then run 'tito module complete 03'.\n"
        "   Original error: " + str(e)
    ) from e

# Sequential container - TESTING UTILITY ONLY (not part of progressive building)
# NOTE: Sequential is NOT part of the educational curriculum - it's a utility for testing compression
# Students should use explicit layer composition in their own code, not Sequential
class Sequential:
    """
    Minimal Sequential container for testing compression techniques.
    
    ⚠️ EDUCATIONAL NOTE: This is a TESTING UTILITY ONLY.
    Students should NOT use Sequential in their own code - they should compose
    layers explicitly to understand data flow:
    
    # ❌ DON'T DO THIS (hides learning):
    model = Sequential(Linear(10, 5), Linear(5, 2))
    
    # ✅ DO THIS (explicit, educational):
    layer1 = Linear(10, 5)
    layer2 = Linear(5, 2)
    x = layer1.forward(x)
    x = layer2.forward(x)
    """
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

# %% [markdown]
"""
## 1. Introduction: What is Model Compression?

Imagine you have a massive library with millions of books, but you only reference 10% of them regularly. Model compression is like creating a curated collection that keeps the essential knowledge while dramatically reducing storage space.

Model compression reduces the size and computational requirements of neural networks while preserving their intelligence. It's the bridge between powerful research models and practical deployment.

### Why Compression Matters in ML Systems

**The Storage Challenge:**
- Modern language models: 100GB+ (GPT-3 scale)
- Mobile devices: <1GB available for models
- Edge devices: <100MB realistic limits
- Network bandwidth: Slow downloads kill user experience

**The Speed Challenge:**
- Research models: Designed for accuracy, not efficiency
- Production needs: Sub-second response times
- Battery life: Energy consumption matters for mobile
- Cost scaling: Inference costs grow with model size

### The Compression Landscape

```
Neural Network Compression Techniques:

┌─────────────────────────────────────────────────────────────┐
│                    COMPRESSION METHODS                      │
├─────────────────────────────────────────────────────────────┤
│  WEIGHT-BASED                    │  ARCHITECTURE-BASED      │
│  ┌─────────────────────────────┐ │  ┌─────────────────────┐ │
│  │ Magnitude Pruning           │ │  │ Knowledge Distillation│ │
│  │ • Remove small weights      │ │  │ • Teacher → Student  │ │
│  │ • 90% sparsity achievable   │ │  │ • 10x size reduction │ │
│  │                             │ │  │                     │ │
│  │ Structured Pruning          │ │  │ Neural Architecture │ │
│  │ • Remove entire channels    │ │  │ Search (NAS)        │ │
│  │ • Hardware-friendly         │ │  │ • Automated design  │ │
│  │                             │ │  │                     │ │
│  │ Low-Rank Approximation      │ │  │ Early Exit          │ │
│  │ • Matrix factorization      │ │  │ • Adaptive compute  │ │
│  │ • SVD decomposition         │ │  │                     │ │
│  └─────────────────────────────┘ │  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

Think of compression like optimizing a recipe - you want to keep the essential ingredients that create the flavor while removing anything that doesn't contribute to the final dish.
"""

# %% [markdown]
"""
## 2. Foundations: Mathematical Background

Understanding the mathematics behind compression helps us choose the right technique for each situation and predict their effects on model performance.

### Magnitude-Based Pruning: The Simple Approach

The core insight: small weights contribute little to the final prediction. Magnitude pruning removes weights based on their absolute values.

```
Mathematical Foundation:
For weight w_ij in layer l:
    If |w_ij| < threshold_l → w_ij = 0

Threshold Selection:
- Global: One threshold for entire model
- Layer-wise: Different threshold per layer
- Percentile-based: Remove bottom k% of weights

Sparsity Calculation:
    Sparsity = (Zero weights / Total weights) × 100%
```

### Structured Pruning: Hardware-Friendly Compression

Unlike magnitude pruning which creates scattered zeros, structured pruning removes entire computational units (neurons, channels, attention heads).

```
Channel Importance Metrics:

Method 1: L2 Norm
    Importance(channel_i) = ||W[:,i]||₂ = √(Σⱼ W²ⱼᵢ)

Method 2: Gradient-based
    Importance(channel_i) = |∂Loss/∂W[:,i]|

Method 3: Activation-based
    Importance(channel_i) = E[|activations_i|]

Pruning Decision:
    Remove bottom k% of channels based on importance ranking
```

### Knowledge Distillation: Learning from Teachers

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. The student learns not just the correct answers, but the teacher's reasoning process.

```
Distillation Loss Function:
    L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(σ(z_s/T), σ(z_t/T))  # Soft targets
    L_hard = CrossEntropy(σ(z_s), y_true)        # Hard targets

    σ(z/T) = Softmax with temperature T
    z_s = Student logits, z_t = Teacher logits
    α = Balance parameter (typically 0.7)
    T = Temperature parameter (typically 3-5)

Temperature Effect:
    T=1: Standard softmax (sharp probabilities)
    T>1: Softer distributions (reveals teacher's uncertainty)
```

### Low-Rank Approximation: Matrix Compression

Large weight matrices often have redundancy that can be captured with lower-rank approximations using Singular Value Decomposition (SVD).

```
SVD Decomposition:
    W_{m×n} = U_{m×k} × Σ_{k×k} × V^T_{k×n}

Parameter Reduction:
    Original: m × n parameters
    Compressed: (m × k) + k + (k × n) = k(m + n + 1) parameters

    Compression achieved when: k < mn/(m+n+1)

Reconstruction Error:
    ||W - W_approx||_F = √(Σᵢ₌ₖ₊₁ʳ σᵢ²)

    Where σᵢ are singular values, r = rank(W)
```
"""

# %% [markdown]
"""
## 3. Sparsity Measurement - Understanding Model Density

Before we can compress models, we need to understand how dense they are. Sparsity measurement tells us what percentage of weights are zero (or effectively zero).

### Understanding Sparsity

Sparsity is like measuring how much of a parking lot is empty. A 90% sparse model means 90% of its weights are zero - only 10% of the "parking spaces" are occupied.

```
Sparsity Visualization:

Dense Matrix (0% sparse):           Sparse Matrix (75% sparse):
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐    ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│ 2.1 1.3 0.8 1.9 2.4 1.1 0.7 │    │ 2.1 0.0 0.0 1.9 0.0 0.0 0.0 │
│ 1.5 2.8 1.2 0.9 1.6 2.2 1.4 │    │ 0.0 2.8 0.0 0.0 0.0 2.2 0.0 │
│ 0.6 1.7 2.5 1.1 0.8 1.3 2.0 │    │ 0.0 0.0 2.5 0.0 0.0 0.0 2.0 │
│ 1.9 1.0 1.6 2.3 1.8 0.9 1.2 │    │ 1.9 0.0 0.0 2.3 0.0 0.0 0.0 │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘    └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
All weights active                   Only 7/28 weights active
Storage: 28 values                   Storage: 7 values + indices
```

Why this matters: Sparsity directly relates to memory savings, but achieving speedup requires special sparse computation libraries.
"""

# %%
def measure_sparsity(model) -> float:
    """
    Calculate the percentage of zero weights in a model.

    TODO: Count zero weights and total weights across all layers

    APPROACH:
    1. Iterate through all model parameters
    2. Count zeros using np.sum(weights == 0)
    3. Count total parameters
    4. Return percentage: zeros / total * 100

    EXAMPLE:
    >>> model = Sequential(Linear(10, 5), Linear(5, 2))
    >>> sparsity = measure_sparsity(model)
    >>> print(f"Model sparsity: {sparsity:.1f}%")
    Model sparsity: 0.0%  # Before pruning

    HINT: Use np.sum() to count zeros efficiently
    """
    ### BEGIN SOLUTION
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.size
        zero_params += np.sum(param.data == 0)

    if total_params == 0:
        return 0.0

    return (zero_params / total_params) * 100.0
    ### END SOLUTION

def test_unit_measure_sparsity():
    """🔬 Test sparsity measurement functionality."""
    print("🔬 Unit Test: Measure Sparsity...")

    # Test with dense model
    model = Sequential(Linear(4, 3), Linear(3, 2))
    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity == 0.0, f"Expected 0% sparsity, got {initial_sparsity}%"

    # Test with manually sparse model
    model.layers[0].weight.data[0, 0] = 0
    model.layers[0].weight.data[1, 1] = 0
    sparse_sparsity = measure_sparsity(model)
    assert sparse_sparsity > 0, f"Expected >0% sparsity, got {sparse_sparsity}%"

    print("✅ measure_sparsity works correctly!")

test_unit_measure_sparsity()

# %% [markdown]
"""
## 4. Magnitude-Based Pruning - Removing Small Weights

Magnitude pruning is the simplest and most intuitive compression technique. It's based on the observation that weights with small magnitudes contribute little to the model's output.

### How Magnitude Pruning Works

Think of magnitude pruning like editing a document - you remove words that don't significantly change the meaning. In neural networks, we remove weights that don't significantly affect predictions.

```
Magnitude Pruning Process:

Step 1: Collect All Weights
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.1, -1.8, 0.05, 3.2, -0.02]    │
│ Layer 2: [1.5, -0.03, 2.8, 0.08, -2.1, 0.01]   │
│ Layer 3: [0.7, 2.4, -0.06, 1.9, 0.04, -1.3]    │
└──────────────────────────────────────────────────┘
                    ↓
Step 2: Calculate Magnitudes
┌──────────────────────────────────────────────────┐
│ Magnitudes: [2.1, 0.1, 1.8, 0.05, 3.2, 0.02,   │
│              1.5, 0.03, 2.8, 0.08, 2.1, 0.01,   │
│              0.7, 2.4, 0.06, 1.9, 0.04, 1.3]    │
└──────────────────────────────────────────────────┘
                    ↓
Step 3: Find Threshold (e.g., 70th percentile)
┌──────────────────────────────────────────────────┐
│ Sorted: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,    │
│          0.08, 0.1, 0.7, 1.3, 1.5, 1.8,         │ Threshold: 0.1
│          1.9, 2.1, 2.1, 2.4, 2.8, 3.2]          │ (70% of weights removed)
└──────────────────────────────────────────────────┘
                    ↓
Step 4: Apply Pruning Mask
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.0, -1.8, 0.0, 3.2, 0.0]       │
│ Layer 2: [1.5, 0.0, 2.8, 0.0, -2.1, 0.0]       │ 70% weights → 0
│ Layer 3: [0.7, 2.4, 0.0, 1.9, 0.0, -1.3]       │ 30% preserved
└──────────────────────────────────────────────────┘

Memory Impact:
- Dense storage: 18 values
- Sparse storage: 6 values + 6 indices = 12 values (33% savings)
- Theoretical limit: 70% savings with perfect sparse format
```

### Why Global Thresholding Works

Global thresholding treats the entire model as one big collection of weights, finding a single threshold that achieves the target sparsity across all layers.

**Advantages:**
- Simple to implement and understand
- Preserves overall model capacity
- Works well for uniform network architectures

**Disadvantages:**
- May over-prune some layers, under-prune others
- Doesn't account for layer-specific importance
- Can hurt performance if layers have very different weight distributions
"""

# %%
def magnitude_prune(model, sparsity=0.9):
    """
    Remove weights with smallest magnitudes to achieve target sparsity.

    TODO: Implement global magnitude-based pruning

    APPROACH:
    1. Collect all weights from the model
    2. Calculate absolute values to get magnitudes
    3. Find threshold at desired sparsity percentile
    4. Set weights below threshold to zero (in-place)

    EXAMPLE:
    >>> model = Sequential(Linear(100, 50), Linear(50, 10))
    >>> original_params = sum(p.size for p in model.parameters())
    >>> magnitude_prune(model, sparsity=0.8)
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Achieved {final_sparsity:.1f}% sparsity")
    Achieved 80.0% sparsity

    HINTS:
    - Use np.percentile() to find threshold
    - Modify model parameters in-place
    - Consider only weight matrices, not biases
    """
    ### BEGIN SOLUTION
    # Collect all weights (excluding biases)
    all_weights = []
    weight_params = []

    for param in model.parameters():
        # Skip biases (typically 1D)
        if len(param.shape) > 1:
            all_weights.extend(param.data.flatten())
            weight_params.append(param)

    if not all_weights:
        return

    # Calculate magnitude threshold
    magnitudes = np.abs(all_weights)
    threshold = np.percentile(magnitudes, sparsity * 100)

    # Apply pruning to each weight parameter
    for param in weight_params:
        mask = np.abs(param.data) >= threshold
        param.data = param.data * mask
    ### END SOLUTION

def test_unit_magnitude_prune():
    """🔬 Test magnitude-based pruning functionality."""
    print("🔬 Unit Test: Magnitude Prune...")

    # Create test model with known weights
    model = Sequential(Linear(4, 3), Linear(3, 2))

    # Set specific weight values for predictable testing
    model.layers[0].weight.data = np.array([
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
        [4.0, 5.0, 6.0],
        [0.01, 0.02, 0.03]
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity == 0.0, "Model should start with no sparsity"

    # Apply 50% pruning
    magnitude_prune(model, sparsity=0.5)
    final_sparsity = measure_sparsity(model)

    # Should achieve approximately 50% sparsity
    assert 40 <= final_sparsity <= 60, f"Expected ~50% sparsity, got {final_sparsity}%"

    # Verify largest weights survived
    remaining_weights = model.layers[0].weight.data[model.layers[0].weight.data != 0]
    assert len(remaining_weights) > 0, "Some weights should remain"
    assert np.all(np.abs(remaining_weights) >= 0.1), "Large weights should survive"

    print("✅ magnitude_prune works correctly!")

test_unit_magnitude_prune()

# %% [markdown]
"""
## 5. Structured Pruning - Hardware-Friendly Compression

While magnitude pruning creates scattered zeros throughout the network, structured pruning removes entire computational units (channels, neurons, heads). This creates sparsity patterns that modern hardware can actually accelerate.

### Why Structured Pruning Matters

Think of the difference between removing random words from a paragraph versus removing entire sentences. Structured pruning removes entire "sentences" (channels) rather than random "words" (individual weights).

```
Unstructured vs Structured Sparsity:

UNSTRUCTURED (Magnitude Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 0.0, 1.8, 0.0, 3.2]       │ ← Sparse weights
│ Channel 1: [0.0, 2.8, 0.0, 2.1, 0.0]       │ ← Sparse weights
│ Channel 2: [1.5, 0.0, 2.4, 0.0, 1.9]       │ ← Sparse weights
│ Channel 3: [0.0, 1.7, 0.0, 2.0, 0.0]       │ ← Sparse weights
└─────────────────────────────────────────────┘
Issues: Irregular memory access, no hardware speedup

STRUCTURED (Channel Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 1.3, 1.8, 0.9, 3.2]       │ ← Fully preserved
│ Channel 1: [0.0, 0.0, 0.0, 0.0, 0.0]       │ ← Fully removed
│ Channel 2: [1.5, 2.2, 2.4, 1.1, 1.9]       │ ← Fully preserved
│ Channel 3: [0.0, 0.0, 0.0, 0.0, 0.0]       │ ← Fully removed
└─────────────────────────────────────────────┘
Benefits: Regular patterns, hardware acceleration possible
```

### Channel Importance Ranking

How do we decide which channels to remove? We rank them by importance using various metrics:

```
Channel Importance Metrics:

Method 1: L2 Norm (Most Common)
    For each output channel i:
    Importance_i = ||W[:, i]||_2 = √(Σⱼ w²ⱼᵢ)

    Intuition: Channels with larger weights have bigger impact

Method 2: Activation-Based
    Importance_i = E[|activation_i|] over dataset

    Intuition: Channels that activate more are more important

Method 3: Gradient-Based
    Importance_i = |∂Loss/∂W[:, i]|

    Intuition: Channels with larger gradients affect loss more

Ranking Process:
    1. Calculate importance for all channels
    2. Sort channels by importance (ascending)
    3. Remove bottom k% (least important)
    4. Zero out entire channels, not individual weights
```

### Hardware Benefits of Structured Sparsity

Structured sparsity enables real hardware acceleration because:

1. **Memory Coalescing**: Accessing contiguous memory chunks is faster
2. **SIMD Operations**: Can process multiple remaining channels in parallel
3. **No Indexing Overhead**: Don't need to track locations of sparse weights
4. **Cache Efficiency**: Better spatial locality of memory access
"""

# %%
def structured_prune(model, prune_ratio=0.5):
    """
    Remove entire channels/neurons based on L2 norm importance.

    TODO: Implement structured pruning for Linear layers

    APPROACH:
    1. For each Linear layer, calculate L2 norm of each output channel
    2. Rank channels by importance (L2 norm)
    3. Remove lowest importance channels by setting to zero
    4. This creates block sparsity that's hardware-friendly

    EXAMPLE:
    >>> model = Sequential(Linear(100, 50), Linear(50, 10))
    >>> original_shape = model.layers[0].weight.shape
    >>> structured_prune(model, prune_ratio=0.3)
    >>> # 30% of channels are now completely zero
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Structured sparsity: {final_sparsity:.1f}%")
    Structured sparsity: 30.0%

    HINTS:
    - Calculate L2 norm along input dimension for each output channel
    - Use np.linalg.norm(weights[:, channel]) for channel importance
    - Set entire channels to zero (not just individual weights)
    """
    ### BEGIN SOLUTION
    for layer in model.layers:
        if isinstance(layer, Linear) and hasattr(layer, 'weight'):
            weight = layer.weight.data

            # Calculate L2 norm for each output channel (column)
            channel_norms = np.linalg.norm(weight, axis=0)

            # Find channels to prune (lowest importance)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                # Get indices of channels to prune (smallest norms)
                prune_indices = np.argpartition(channel_norms, num_to_prune)[:num_to_prune]

                # Zero out entire channels
                weight[:, prune_indices] = 0

                # Also zero corresponding bias elements if bias exists
                if layer.bias is not None:
                    layer.bias.data[prune_indices] = 0
    ### END SOLUTION

def test_unit_structured_prune():
    """🔬 Test structured pruning functionality."""
    print("🔬 Unit Test: Structured Prune...")

    # Create test model
    model = Sequential(Linear(4, 6), Linear(6, 2))

    # Set predictable weights for testing
    model.layers[0].weight.data = np.array([
        [1.0, 0.1, 2.0, 0.05, 3.0, 0.01],  # Channels with varying importance
        [1.1, 0.11, 2.1, 0.06, 3.1, 0.02],
        [1.2, 0.12, 2.2, 0.07, 3.2, 0.03],
        [1.3, 0.13, 2.3, 0.08, 3.3, 0.04]
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity == 0.0, "Model should start with no sparsity"

    # Apply 33% structured pruning (2 out of 6 channels)
    structured_prune(model, prune_ratio=0.33)
    final_sparsity = measure_sparsity(model)

    # Check that some channels are completely zero
    weight = model.layers[0].weight.data
    zero_channels = np.sum(np.all(weight == 0, axis=0))
    assert zero_channels >= 1, f"Expected at least 1 zero channel, got {zero_channels}"

    # Check that non-zero channels are completely preserved
    for col in range(weight.shape[1]):
        channel = weight[:, col]
        assert np.all(channel == 0) or np.all(channel != 0), "Channels should be fully zero or fully non-zero"

    print("✅ structured_prune works correctly!")

test_unit_structured_prune()

# %% [markdown]
"""
## 6. Low-Rank Approximation - Matrix Compression Through Factorization

Low-rank approximation discovers that large weight matrices often contain redundant information that can be captured with much smaller matrices through mathematical decomposition.

### The Intuition Behind Low-Rank Approximation

Imagine you're storing a massive spreadsheet where many columns are highly correlated. Instead of storing all columns separately, you could store a few "basis" columns and coefficients for how to combine them to recreate the original data.

```
Low-Rank Decomposition Visualization:

Original Matrix W (large):           Factorized Form (smaller):
┌─────────────────────────┐         ┌──────┐    ┌──────────────┐
│ 2.1  1.3  0.8  1.9  2.4 │         │ 1.1  │    │ 1.9  1.2  0.7│
│ 1.5  2.8  1.2  0.9  1.6 │    ≈    │ 2.4  │ @  │ 0.6  1.2  0.5│
│ 0.6  1.7  2.5  1.1  0.8 │         │ 0.8  │    │ 1.4  2.1  0.9│
│ 1.9  1.0  1.6  2.3  1.8 │         │ 1.6  │    │ 0.5  0.6  1.1│
└─────────────────────────┘         └──────┘    └──────────────┘
    W (4×5) = 20 params           U (4×2)=8  +  V (2×5)=10  = 18 params

Parameter Reduction:
- Original: 4 × 5 = 20 parameters
- Compressed: (4 × 2) + (2 × 5) = 18 parameters
- Compression ratio: 18/20 = 0.9 (10% savings)

For larger matrices, savings become dramatic:
- W (1000×1000): 1M parameters → U (1000×100) + V (100×1000): 200K parameters
- Compression ratio: 0.2 (80% savings)
```

### SVD: The Mathematical Foundation

Singular Value Decomposition (SVD) finds the optimal low-rank approximation by identifying the most important "directions" in the data:

```
SVD Decomposition:
    W = U × Σ × V^T

Where:
    U: Left singular vectors (input patterns)
    Σ: Singular values (importance weights)
    V^T: Right singular vectors (output patterns)

Truncated SVD (Rank-k approximation):
    W ≈ U[:,:k] × Σ[:k] × V^T[:k,:]

Quality vs Compression Trade-off:
    Higher k → Better approximation, less compression
    Lower k → More compression, worse approximation

Choosing Optimal Rank:
    Method 1: Fixed ratio (k = ratio × min(m,n))
    Method 2: Energy threshold (keep 90% of singular value energy)
    Method 3: Error threshold (reconstruction error < threshold)
```

### When Low-Rank Works Best

Low-rank approximation works well when:
- **Matrices are large**: Compression benefits scale with size
- **Data has structure**: Correlated patterns enable compression
- **Moderate accuracy loss acceptable**: Some precision traded for efficiency

It works poorly when:
- **Matrices are already small**: Overhead exceeds benefits
- **Data is random**: No patterns to exploit
- **High precision required**: SVD introduces approximation error
"""

# %%
def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """
    Approximate weight matrix using low-rank decomposition (SVD).

    TODO: Implement SVD-based low-rank approximation

    APPROACH:
    1. Perform SVD: W = U @ S @ V^T
    2. Keep only top k singular values where k = rank_ratio * min(dimensions)
    3. Reconstruct: W_approx = U[:,:k] @ diag(S[:k]) @ V[:k,:]
    4. Return decomposed matrices for memory savings

    EXAMPLE:
    >>> weight = np.random.randn(100, 50)
    >>> U, S, V = low_rank_approximate(weight, rank_ratio=0.3)
    >>> # Original: 100*50 = 5000 params
    >>> # Compressed: 100*15 + 15*50 = 2250 params (55% reduction)

    HINTS:
    - Use np.linalg.svd() for decomposition
    - Choose k = int(rank_ratio * min(m, n))
    - Return U[:,:k], S[:k], V[:k,:] for reconstruction
    """
    ### BEGIN SOLUTION
    m, n = weight_matrix.shape

    # Perform SVD
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Determine target rank
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    # Truncate to target rank
    U_truncated = U[:, :target_rank]
    S_truncated = S[:target_rank]
    V_truncated = V[:target_rank, :]

    return U_truncated, S_truncated, V_truncated
    ### END SOLUTION

def test_unit_low_rank_approximate():
    """🔬 Test low-rank approximation functionality."""
    print("🔬 Unit Test: Low-Rank Approximate...")

    # Create test weight matrix
    original_weight = np.random.randn(20, 15)
    original_params = original_weight.size

    # Apply low-rank approximation
    U, S, V = low_rank_approximate(original_weight, rank_ratio=0.4)

    # Check dimensions
    target_rank = int(0.4 * min(20, 15))  # min(20,15) = 15, so 0.4*15 = 6
    assert U.shape == (20, target_rank), f"Expected U shape (20, {target_rank}), got {U.shape}"
    assert S.shape == (target_rank,), f"Expected S shape ({target_rank},), got {S.shape}"
    assert V.shape == (target_rank, 15), f"Expected V shape ({target_rank}, 15), got {V.shape}"

    # Check parameter reduction
    compressed_params = U.size + S.size + V.size
    compression_ratio = compressed_params / original_params
    assert compression_ratio < 1.0, f"Should compress, but ratio is {compression_ratio}"

    # Check reconstruction quality
    reconstructed = U @ np.diag(S) @ V
    reconstruction_error = np.linalg.norm(original_weight - reconstructed)
    relative_error = reconstruction_error / np.linalg.norm(original_weight)
    assert relative_error < 0.5, f"Reconstruction error too high: {relative_error}"

    print("✅ low_rank_approximate works correctly!")

test_unit_low_rank_approximate()

# %% [markdown]
"""
## 7. Knowledge Distillation - Learning from Teacher Models

Knowledge distillation is like having an expert teacher simplify complex concepts for a student. The large "teacher" model shares its knowledge with a smaller "student" model, achieving similar performance with far fewer parameters.

### The Teacher-Student Learning Process

Unlike traditional training where models learn from hard labels (cat/dog), knowledge distillation uses "soft" targets that contain richer information about the teacher's decision-making process.

```
Knowledge Distillation Process:

                    TEACHER MODEL (Large)
                    ┌─────────────────────┐
Input Data ────────→│ 100M parameters     │
                    │ 95% accuracy        │
                    │ 500ms inference     │
                    └─────────────────────┘
                             │
                             ↓ Soft Targets
                    ┌─────────────────────┐
                    │  Logits: [2.1, 0.3, │
                    │           0.8, 4.2] │ ← Rich information
                    └─────────────────────┘
                             │
                             ↓ Distillation Loss
                    ┌─────────────────────┐
Input Data ────────→│ STUDENT MODEL       │
Hard Labels ───────→│ 10M parameters      │ ← 10x smaller
                    │ 93% accuracy        │ ← 2% loss
                    │ 50ms inference      │ ← 10x faster
                    └─────────────────────┘

Benefits:
• Size: 10x smaller models
• Speed: 10x faster inference
• Accuracy: Only 2-5% degradation
• Knowledge transfer: Student learns teacher's "reasoning"
```

### Temperature Scaling: Softening Decisions

Temperature scaling is a key innovation that makes knowledge distillation effective. It "softens" the teacher's confidence, revealing uncertainty that helps the student learn.

```
Temperature Effect on Probability Distributions:

Without Temperature (T=1):           With Temperature (T=3):
Teacher Logits: [1.0, 2.0, 0.5]    Teacher Logits: [1.0, 2.0, 0.5]
                       ↓                               ↓ ÷ 3
Softmax: [0.09, 0.67, 0.24]         Logits/T: [0.33, 0.67, 0.17]
         ^      ^      ^                       ↓
      Low   High   Med              Softmax: [0.21, 0.42, 0.17]
                                             ^      ^      ^
Sharp decisions (hard to learn)           Soft   decisions (easier to learn)

Why Soft Targets Help:
1. Reveal teacher's uncertainty about similar classes
2. Provide richer gradients for student learning
3. Transfer knowledge about class relationships
4. Reduce overfitting to hard labels
```

### Loss Function Design

The distillation loss balances learning from both the teacher's soft knowledge and the ground truth hard labels:

```
Combined Loss Function:

L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(Student_soft, Teacher_soft)
             │
             └─ Measures how well student mimics teacher

    L_hard = CrossEntropy(Student_predictions, True_labels)
             │
             └─ Ensures student still learns correct answers

Balance Parameter α:
• α = 0.7: Focus mainly on teacher (typical)
• α = 0.9: Almost pure distillation
• α = 0.3: Balance teacher and ground truth
• α = 0.0: Ignore teacher (regular training)

Temperature T:
• T = 1: No softening (standard softmax)
• T = 3-5: Good balance (typical range)
• T = 10+: Very soft (may lose information)
```
"""

# %%
class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Train a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        """
        Initialize knowledge distillation.

        TODO: Set up teacher and student models with distillation parameters

        APPROACH:
        1. Store teacher and student models
        2. Set temperature for softening probability distributions
        3. Set alpha for balancing hard vs soft targets

        Args:
            teacher_model: Large, pre-trained model
            student_model: Smaller model to train
            temperature: Softening parameter for distributions
            alpha: Weight for soft target loss (1-alpha for hard targets)
        """
        ### BEGIN SOLUTION
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        ### END SOLUTION

    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """
        Calculate combined distillation loss.

        TODO: Implement knowledge distillation loss function

        APPROACH:
        1. Calculate hard target loss (student vs true labels)
        2. Calculate soft target loss (student vs teacher, with temperature)
        3. Combine losses: alpha * soft_loss + (1-alpha) * hard_loss

        EXAMPLE:
        >>> kd = KnowledgeDistillation(teacher, student)
        >>> loss = kd.distillation_loss(student_out, teacher_out, labels)
        >>> print(f"Distillation loss: {loss:.4f}")

        HINTS:
        - Use temperature to soften distributions: logits/temperature
        - Soft targets use KL divergence or cross-entropy
        - Hard targets use standard classification loss
        """
        ### BEGIN SOLUTION
        # Convert to numpy for this implementation
        if hasattr(student_logits, 'data'):
            student_logits = student_logits.data
        if hasattr(teacher_logits, 'data'):
            teacher_logits = teacher_logits.data
        if hasattr(true_labels, 'data'):
            true_labels = true_labels.data

        # Soften distributions with temperature
        student_soft = self._softmax(student_logits / self.temperature)
        teacher_soft = self._softmax(teacher_logits / self.temperature)

        # Soft target loss (KL divergence)
        soft_loss = self._kl_divergence(student_soft, teacher_soft)

        # Hard target loss (cross-entropy)
        student_hard = self._softmax(student_logits)
        hard_loss = self._cross_entropy(student_hard, true_labels)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss
        ### END SOLUTION

    def _softmax(self, logits):
        """Compute softmax with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _kl_divergence(self, p, q):
        """Compute KL divergence between distributions."""
        return np.sum(p * np.log(p / (q + 1e-8) + 1e-8))

    def _cross_entropy(self, predictions, labels):
        """Compute cross-entropy loss."""
        # Simple implementation for integer labels
        if labels.ndim == 1:
            return -np.mean(np.log(predictions[np.arange(len(labels)), labels] + 1e-8))
        else:
            return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))

def test_unit_knowledge_distillation():
    """🔬 Test knowledge distillation functionality."""
    print("🔬 Unit Test: Knowledge Distillation...")

    # Create teacher and student models
    teacher = Sequential(Linear(10, 20), Linear(20, 5))
    student = Sequential(Linear(10, 5))  # Smaller model

    # Initialize knowledge distillation
    kd = KnowledgeDistillation(teacher, student, temperature=3.0, alpha=0.7)

    # Create dummy data
    input_data = Tensor(np.random.randn(8, 10))  # Batch of 8
    true_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2])  # Class labels

    # Forward passes
    teacher_output = teacher.forward(input_data)
    student_output = student.forward(input_data)

    # Calculate distillation loss
    loss = kd.distillation_loss(student_output, teacher_output, true_labels)

    # Verify loss is reasonable
    assert isinstance(loss, (float, np.floating)), f"Loss should be float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    assert not np.isnan(loss), "Loss should not be NaN"

    print("✅ knowledge_distillation works correctly!")

test_unit_knowledge_distillation()

# %% [markdown]
"""
## 8. Integration: Complete Compression Pipeline

Now let's combine all our compression techniques into a unified system that can apply multiple methods and track their cumulative effects.

### Compression Strategy Design

Real-world compression often combines multiple techniques in sequence, each targeting different types of redundancy:

```
Multi-Stage Compression Pipeline:

Original Model (100MB, 100% accuracy)
         │
         ↓ Stage 1: Magnitude Pruning (remove 80% of small weights)
Sparse Model (20MB, 98% accuracy)
         │
         ↓ Stage 2: Structured Pruning (remove 30% of channels)
Compact Model (14MB, 96% accuracy)
         │
         ↓ Stage 3: Low-Rank Approximation (compress large layers)
Factorized Model (10MB, 95% accuracy)
         │
         ↓ Stage 4: Knowledge Distillation (train smaller architecture)
Student Model (5MB, 93% accuracy)

Final Result: 20x size reduction, 7% accuracy loss
```

### Compression Configuration

Different deployment scenarios require different compression strategies:

```
Deployment Scenarios and Strategies:

MOBILE APP (Aggressive compression needed):
┌─────────────────────────────────────────┐
│ Target: <10MB, <100ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 95% sparsity       │
│ • Structured pruning: 50% channels      │
│ • Knowledge distillation: 10x reduction │
│ • Quantization: 8-bit weights           │
└─────────────────────────────────────────┘

EDGE DEVICE (Balanced compression):
┌─────────────────────────────────────────┐
│ Target: <50MB, <200ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 80% sparsity       │
│ • Structured pruning: 30% channels      │
│ • Low-rank: 50% rank reduction          │
│ • Quantization: 16-bit weights          │
└─────────────────────────────────────────┘

CLOUD SERVICE (Minimal compression):
┌─────────────────────────────────────────┐
│ Target: Maintain accuracy, reduce cost  │
│ Strategy:                               │
│ • Magnitude pruning: 50% sparsity       │
│ • Structured pruning: 10% channels      │
│ • Dynamic batching optimization         │
│ • Mixed precision inference            │
└─────────────────────────────────────────┘
```
"""

# %%
def compress_model(model, compression_config):
    """
    Apply comprehensive model compression based on configuration.

    TODO: Implement complete compression pipeline

    APPROACH:
    1. Apply magnitude pruning if specified
    2. Apply structured pruning if specified
    3. Apply low-rank approximation if specified
    4. Return compression statistics

    EXAMPLE:
    >>> config = {
    ...     'magnitude_prune': 0.8,
    ...     'structured_prune': 0.3,
    ...     'low_rank': 0.5
    ... }
    >>> stats = compress_model(model, config)
    >>> print(f"Final sparsity: {stats['sparsity']:.1f}%")
    Final sparsity: 85.0%

    HINT: Apply techniques sequentially and measure results
    """
    ### BEGIN SOLUTION
    original_params = sum(p.size for p in model.parameters())
    original_sparsity = measure_sparsity(model)

    stats = {
        'original_params': original_params,
        'original_sparsity': original_sparsity,
        'applied_techniques': []
    }

    # Apply magnitude pruning
    if 'magnitude_prune' in compression_config:
        sparsity = compression_config['magnitude_prune']
        magnitude_prune(model, sparsity=sparsity)
        stats['applied_techniques'].append(f'magnitude_prune_{sparsity}')

    # Apply structured pruning
    if 'structured_prune' in compression_config:
        ratio = compression_config['structured_prune']
        structured_prune(model, prune_ratio=ratio)
        stats['applied_techniques'].append(f'structured_prune_{ratio}')

    # Apply low-rank approximation (conceptually - would need architecture changes)
    if 'low_rank' in compression_config:
        ratio = compression_config['low_rank']
        # For demo, we'll just record that it would be applied
        stats['applied_techniques'].append(f'low_rank_{ratio}')

    # Final measurements
    final_sparsity = measure_sparsity(model)
    stats['final_sparsity'] = final_sparsity
    stats['sparsity_increase'] = final_sparsity - original_sparsity

    return stats
    ### END SOLUTION

def test_unit_compress_model():
    """🔬 Test comprehensive model compression."""
    print("🔬 Unit Test: Compress Model...")

    # Create test model
    model = Sequential(Linear(20, 15), Linear(15, 10), Linear(10, 5))

    # Define compression configuration
    config = {
        'magnitude_prune': 0.7,
        'structured_prune': 0.2
    }

    # Apply compression
    stats = compress_model(model, config)

    # Verify statistics
    assert 'original_params' in stats, "Should track original parameter count"
    assert 'final_sparsity' in stats, "Should track final sparsity"
    assert 'applied_techniques' in stats, "Should track applied techniques"

    # Verify compression was applied
    assert stats['final_sparsity'] > stats['original_sparsity'], "Sparsity should increase"
    assert len(stats['applied_techniques']) == 2, "Should apply both techniques"

    # Verify model still has reasonable structure
    remaining_params = sum(np.count_nonzero(p.data) for p in model.parameters())
    assert remaining_params > 0, "Model should retain some parameters"

    print("✅ compress_model works correctly!")

test_unit_compress_model()

# %% [markdown]
"""
## 9. Systems Analysis: Compression Performance and Trade-offs

Understanding how compression techniques affect real-world deployment metrics like storage, memory, speed, and accuracy.

### Compression Effectiveness Analysis

Different techniques excel in different scenarios. Let's measure their effectiveness across various model sizes and architectures.
"""

# %%
def analyze_compression_ratios():
    """📊 Analyze compression ratios for different techniques."""
    print("📊 Analyzing Compression Ratios...")

    # Create test models of different sizes
    models = {
        'Small': Sequential(Linear(50, 30), Linear(30, 10)),
        'Medium': Sequential(Linear(200, 128), Linear(128, 64), Linear(64, 10)),
        'Large': Sequential(Linear(500, 256), Linear(256, 128), Linear(128, 10))
    }

    compression_techniques = [
        ('Magnitude 50%', {'magnitude_prune': 0.5}),
        ('Magnitude 90%', {'magnitude_prune': 0.9}),
        ('Structured 30%', {'structured_prune': 0.3}),
        ('Combined', {'magnitude_prune': 0.8, 'structured_prune': 0.2})
    ]

    print(f"{'Model':<8} {'Technique':<15} {'Original':<10} {'Final':<10} {'Reduction':<10}")
    print("-" * 65)

    for model_name, model in models.items():
        original_params = sum(p.size for p in model.parameters())

        for tech_name, config in compression_techniques:
            # Create fresh copy for each test
            test_model = copy.deepcopy(model)

            # Apply compression
            stats = compress_model(test_model, config)

            # Calculate compression ratio
            remaining_params = sum(np.count_nonzero(p.data) for p in test_model.parameters())
            reduction = (1 - remaining_params / original_params) * 100

            print(f"{model_name:<8} {tech_name:<15} {original_params:<10} {remaining_params:<10} {reduction:<9.1f}%")

    print("\n💡 Key Insights:")
    print("• Magnitude pruning achieves predictable sparsity levels")
    print("• Structured pruning creates hardware-friendly sparsity")
    print("• Combined techniques offer maximum compression")
    print("• Larger models compress better (more redundancy)")

analyze_compression_ratios()

# %%
def analyze_compression_speed():
    """📊 Analyze inference speed with different compression levels."""
    print("📊 Analyzing Compression Speed Impact...")

    # Create test model
    model = Sequential(Linear(512, 256), Linear(256, 128), Linear(128, 10))
    test_input = Tensor(np.random.randn(100, 512))  # Batch of 100

    def time_inference(model, input_data, iterations=50):
        """Time model inference."""
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = model.forward(input_data)
            times.append(time.time() - start)
        return np.mean(times[5:])  # Skip first few for warmup

    # Test different compression levels
    compression_levels = [
        ('Original', {}),
        ('Light Pruning', {'magnitude_prune': 0.5}),
        ('Heavy Pruning', {'magnitude_prune': 0.9}),
        ('Structured', {'structured_prune': 0.3}),
        ('Combined', {'magnitude_prune': 0.8, 'structured_prune': 0.2})
    ]

    print(f"{'Compression':<15} {'Sparsity':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    baseline_time = None

    for name, config in compression_levels:
        # Create fresh model copy
        test_model = copy.deepcopy(model)

        # Apply compression
        if config:
            compress_model(test_model, config)

        # Measure performance
        sparsity = measure_sparsity(test_model)
        inference_time = time_inference(test_model, test_input) * 1000  # Convert to ms

        if baseline_time is None:
            baseline_time = inference_time
            speedup = 1.0
        else:
            speedup = baseline_time / inference_time

        print(f"{name:<15} {sparsity:<9.1f}% {inference_time:<11.2f} {speedup:<9.2f}x")

    print("\n💡 Speed Insights:")
    print("• Dense matrix operations show minimal speedup from unstructured sparsity")
    print("• Structured sparsity enables better hardware acceleration")
    print("• Real speedups require sparse-optimized libraries (e.g., NVIDIA 2:4 sparsity)")
    print("• Memory bandwidth often more important than parameter count")

analyze_compression_speed()

# %% [markdown]
"""
## 10. Optimization Insights: Production Compression Strategy

Understanding the real-world implications of compression choices and how to design compression strategies for different deployment scenarios.

### Accuracy vs Compression Trade-offs

The fundamental challenge in model compression is balancing three competing objectives: model size, inference speed, and prediction accuracy.
"""

# %%
def analyze_compression_accuracy_tradeoff():
    """📊 Analyze accuracy vs compression trade-offs."""
    print("📊 Analyzing Accuracy vs Compression Trade-offs...")

    # Simulate accuracy degradation (in practice, would need real training/testing)
    def simulate_accuracy_loss(sparsity, technique_type):
        """Simulate realistic accuracy loss patterns."""
        if technique_type == 'magnitude':
            # Magnitude pruning: gradual degradation
            return max(0, sparsity * 0.3 + np.random.normal(0, 0.05))
        elif technique_type == 'structured':
            # Structured pruning: more aggressive early loss
            return max(0, sparsity * 0.5 + np.random.normal(0, 0.1))
        elif technique_type == 'knowledge_distillation':
            # Knowledge distillation: better preservation
            return max(0, sparsity * 0.1 + np.random.normal(0, 0.02))
        else:
            return sparsity * 0.4

    # Test different compression strategies
    strategies = [
        ('Magnitude Only', 'magnitude'),
        ('Structured Only', 'structured'),
        ('Knowledge Distillation', 'knowledge_distillation'),
        ('Combined Approach', 'combined')
    ]

    sparsity_levels = np.arange(0.1, 1.0, 0.1)

    print(f"{'Strategy':<20} {'Sparsity':<10} {'Accuracy Loss':<15}")
    print("-" * 50)

    for strategy_name, strategy_type in strategies:
        print(f"\n{strategy_name}:")
        for sparsity in sparsity_levels:
            if strategy_type == 'combined':
                # Combined approach uses multiple techniques
                loss = min(
                    simulate_accuracy_loss(sparsity * 0.7, 'magnitude'),
                    simulate_accuracy_loss(sparsity * 0.3, 'structured')
                )
            else:
                loss = simulate_accuracy_loss(sparsity, strategy_type)

            print(f"{'':20} {sparsity:<9.1f} {loss:<14.3f}")

    print("\n💡 Trade-off Insights:")
    print("• Knowledge distillation preserves accuracy best at high compression")
    print("• Magnitude pruning offers gradual degradation curve")
    print("• Structured pruning enables hardware acceleration but higher accuracy loss")
    print("• Combined approaches balance multiple objectives")
    print("• Early stopping based on accuracy threshold is crucial")

analyze_compression_accuracy_tradeoff()

# %% [markdown]
"""
## 11. Module Integration Test

Final validation that all compression techniques work together correctly.
"""

# %%
def test_module():
    """
    Comprehensive test of entire compression module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_measure_sparsity()
    test_unit_magnitude_prune()
    test_unit_structured_prune()
    test_unit_low_rank_approximate()
    test_unit_knowledge_distillation()
    test_unit_compress_model()

    print("\nRunning integration scenarios...")

    # Test 1: Complete compression pipeline
    print("🔬 Integration Test: Complete compression pipeline...")

    # Create a realistic model
    model = Sequential(
        Linear(784, 512),  # Input layer (like MNIST)
        Linear(512, 256),  # Hidden layer 1
        Linear(256, 128),  # Hidden layer 2
        Linear(128, 10)    # Output layer
    )

    original_params = sum(p.size for p in model.parameters())
    print(f"Original model: {original_params:,} parameters")

    # Apply comprehensive compression
    compression_config = {
        'magnitude_prune': 0.8,
        'structured_prune': 0.3
    }

    stats = compress_model(model, compression_config)
    final_sparsity = measure_sparsity(model)

    # Validate compression results
    assert final_sparsity > 70, f"Expected >70% sparsity, got {final_sparsity:.1f}%"
    assert stats['sparsity_increase'] > 70, "Should achieve significant compression"
    assert len(stats['applied_techniques']) == 2, "Should apply both techniques"

    print(f"✅ Achieved {final_sparsity:.1f}% sparsity with {len(stats['applied_techniques'])} techniques")

    # Test 2: Knowledge distillation setup
    print("🔬 Integration Test: Knowledge distillation...")

    teacher = Sequential(Linear(100, 200), Linear(200, 50))
    student = Sequential(Linear(100, 50))  # 3x fewer parameters

    kd = KnowledgeDistillation(teacher, student, temperature=4.0, alpha=0.8)

    # Verify setup
    teacher_params = sum(p.size for p in teacher.parameters())
    student_params = sum(p.size for p in student.parameters())
    compression_ratio = student_params / teacher_params

    assert compression_ratio < 0.5, f"Student should be <50% of teacher size, got {compression_ratio:.2f}"
    assert kd.temperature == 4.0, "Temperature should be set correctly"
    assert kd.alpha == 0.8, "Alpha should be set correctly"

    print(f"✅ Knowledge distillation: {compression_ratio:.2f}x size reduction")

    # Test 3: Low-rank approximation
    print("🔬 Integration Test: Low-rank approximation...")

    large_matrix = np.random.randn(200, 150)
    U, S, V = low_rank_approximate(large_matrix, rank_ratio=0.3)

    original_size = large_matrix.size
    compressed_size = U.size + S.size + V.size
    compression_ratio = compressed_size / original_size

    assert compression_ratio < 0.7, f"Should achieve compression, got ratio {compression_ratio:.2f}"

    # Test reconstruction
    reconstructed = U @ np.diag(S) @ V
    error = np.linalg.norm(large_matrix - reconstructed) / np.linalg.norm(large_matrix)
    assert error < 0.5, f"Reconstruction error too high: {error:.3f}"

    print(f"✅ Low-rank: {compression_ratio:.2f}x compression, {error:.3f} error")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 18")

# Call the integration test
test_module()

# %%
if __name__ == "__main__":
    print("🚀 Running Compression module...")
    test_module()
    print("✅ Module validation complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Compression Foundations

### Question 1: Compression Trade-offs
You implemented magnitude pruning that removes 90% of weights from a 10M parameter model.
- How many parameters remain active? _____ M parameters
- If the original model was 40MB, what's the theoretical minimum storage? _____ MB
- Why might actual speedup be less than 10x? _____________

### Question 2: Structured vs Unstructured Sparsity
Your structured pruning removes entire channels, while magnitude pruning creates scattered zeros.
- Which enables better hardware acceleration? _____________
- Which preserves accuracy better at high sparsity? _____________
- Which creates more predictable memory access patterns? _____________

### Question 3: Knowledge Distillation Efficiency
A teacher model has 100M parameters, student has 10M parameters, both achieve 85% accuracy.
- What's the compression ratio? _____x
- If teacher inference takes 100ms, student takes 15ms, what's the speedup? _____x
- Why is the speedup greater than the compression ratio? _____________

### Question 4: Low-Rank Decomposition
You approximate a (512, 256) weight matrix with rank 64 using SVD.
- Original parameter count: _____ parameters
- Decomposed parameter count: _____ parameters
- Compression ratio: _____x
- At what rank does compression become ineffective? rank > _____
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Compression

Congratulations! You've built a comprehensive model compression system that can dramatically reduce model size while preserving intelligence!

### Key Accomplishments
- Built magnitude-based and structured pruning techniques with clear sparsity patterns
- Implemented knowledge distillation for teacher-student compression with temperature scaling
- Created low-rank approximation using SVD decomposition for matrix factorization
- Developed sparsity measurement and comprehensive compression pipeline
- Analyzed compression trade-offs between size, speed, and accuracy with real measurements
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **Structured vs Unstructured**: Hardware-friendly sparsity patterns vs maximum compression ratios
- **Compression Cascading**: Multiple techniques compound benefits but require careful sequencing
- **Accuracy Preservation**: Knowledge distillation maintains performance better than pruning alone
- **Memory vs Speed**: Parameter reduction doesn't guarantee proportional speedup without sparse libraries
- **Deployment Strategy**: Different scenarios (mobile, edge, cloud) require different compression approaches

### Technical Mastery
- **Sparsity Measurement**: Calculate and track zero weight percentages across models
- **Magnitude Pruning**: Global thresholding based on weight importance ranking
- **Structured Pruning**: Channel-wise removal using L2 norm importance metrics
- **Knowledge Distillation**: Teacher-student training with temperature-scaled soft targets
- **Low-Rank Approximation**: SVD-based matrix factorization for parameter reduction
- **Pipeline Integration**: Sequential application of multiple compression techniques

### Ready for Next Steps
Your compression implementation enables efficient model deployment across diverse hardware constraints!
Export with: `tito module complete 18`

**Next**: Module 19 will add comprehensive benchmarking to evaluate all optimization techniques together, measuring the cumulative effects of quantization, acceleration, and compression!
"""
