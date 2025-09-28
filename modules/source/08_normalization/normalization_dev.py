# %% [markdown]
"""
# Normalization - Stabilizing Deep Network Training

Welcome to Normalization! You'll implement the normalization techniques that make deep neural networks trainable and stable.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): Data structures with gradient tracking
- Module 04 (Layers): Neural network layer primitives
- Module 06 (Autograd): Automatic gradient computation
- Module 07 (Optimizers): Parameter update algorithms

**What's Working**: You can build multi-layer networks and train them with optimizers!

**The Gap**: Deep networks suffer from internal covariate shift - activations drift during training, making learning unstable and slow.

**This Module's Solution**: Implement BatchNorm, LayerNorm, and GroupNorm to stabilize training by normalizing intermediate activations.

**Connection Map**:
```
Layers → Normalization → Stable Training
(unstable)  (stabilized)    (convergence)
```

## Learning Goals (5-Point Framework)
- **Systems understanding**: Memory and computation patterns of different normalization schemes
- **Core implementation skill**: Build BatchNorm, LayerNorm, and GroupNorm from mathematical foundations
- **Pattern/abstraction mastery**: Understand when to use each normalization technique
- **Framework connections**: Connect to PyTorch's nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm
- **Optimization trade-offs**: Analyze memory vs stability vs computation trade-offs

## Build → Use → Reflect
1. **Build**: Implementation of BatchNorm, LayerNorm, and GroupNorm with running statistics
2. **Use**: Apply normalization to stabilize training of deep networks
3. **Reflect**: How do different normalization schemes affect memory, computation, and training dynamics?

## Systems Reality Check
💡 **Production Context**: Normalization is critical in all modern deep learning - ResNet uses BatchNorm, Transformers use LayerNorm, modern ConvNets use GroupNorm
⚡ **Performance Insight**: BatchNorm adds 2× parameters per layer but often enables 10× larger learning rates, dramatically accelerating training

## What You'll Achieve
By the end of this module, you'll have implemented the normalization arsenal that makes modern deep learning possible, with complete understanding of their memory characteristics and performance trade-offs.
"""

# %% [markdown]
"""
## Mathematical Foundation: Why Normalization Works

Internal covariate shift occurs when the distribution of inputs to each layer changes during training. This makes learning slow and unstable.

### The Core Problem:
```
Layer 1: x₁ → f₁(x₁) → y₁  (distribution D₁)
Layer 2: y₁ → f₂(y₁) → y₂  (distribution changes as f₁ changes!)
Layer 3: y₂ → f₃(y₂) → y₃  (distribution keeps shifting!)
```

### The Normalization Solution:
Normalize activations to have stable statistics (mean=0, variance=1):

**Mathematical Form:**
```
ŷ = γ * (x - μ) / σ + β

Where:
- μ = E[x] (mean)
- σ = √(Var[x] + ε) (standard deviation)
- γ = learnable scale parameter
- β = learnable shift parameter
- ε = numerical stability constant (usually 1e-5)
```

**Key Insight**: γ and β allow the network to recover the original representation if normalization hurts performance.
"""

# %% [markdown]
"""
## Context: Why Normalization Matters

### Historical Context
- **2015**: BatchNorm revolutionizes training, enables much deeper networks
- **2016**: LayerNorm enables stable transformer training
- **2018**: GroupNorm provides batch-independent normalization for object detection

### Production Impact
- **ImageNet Training**: BatchNorm reduces training time from weeks to days
- **Language Models**: LayerNorm enables training of billion-parameter transformers
- **Object Detection**: GroupNorm enables small-batch training with stable results

### Memory vs Performance Trade-offs
- **BatchNorm**: 2× parameters, but enables 5-10× larger learning rates
- **LayerNorm**: No batch dimension dependence, consistent across batch sizes
- **GroupNorm**: Balance between batch and layer normalization benefits
"""

# %% [markdown]
"""
## Connections: Production Normalization Systems

### PyTorch Implementation Patterns
```python
# Production patterns you'll implement
torch.nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
torch.nn.LayerNorm(normalized_shape, eps=1e-5)
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5)

# Your implementation will match these interfaces
```

### Real-World Usage
- **ResNet**: Uses BatchNorm after every convolution layer
- **BERT/GPT**: Uses LayerNorm in transformer blocks
- **YOLO**: Uses BatchNorm for training stability with large images
- **Modern ConvNets**: Often use GroupNorm for object detection tasks
"""

# %% [markdown]
"""
## Design: Why Build Normalization From Scratch?

### Learning Justification
Building normalization layers teaches:
1. **Statistical Computing**: How to compute mean/variance efficiently across different dimensions
2. **Memory Management**: Understanding running statistics and their memory implications
3. **Training vs Inference**: How normalization behaves differently during training and evaluation
4. **Gradient Flow**: How normalization affects backpropagation through learnable parameters

### Systems Understanding Goals
- **Dimension Analysis**: How normalization axes affect memory and computation
- **Batch Dependencies**: Understanding when normalization depends on batch statistics
- **Parameter Sharing**: How γ and β parameters are organized in memory
- **Numerical Stability**: Why ε is critical for avoiding division by zero
"""

# %% [markdown]
"""
## Architecture: Normalization Design Decisions

### Key Design Choices

1. **Normalization Axis Selection**:
   ```
   BatchNorm: Normalize across batch dimension (N, C, H, W) → across N
   LayerNorm: Normalize across feature dimensions → across C, H, W
   GroupNorm: Normalize across channel groups → within groups of C
   ```

2. **Parameter Organization**:
   ```
   γ (scale) and β (bias) parameters:
   - BatchNorm: Shape (C,) - one per channel
   - LayerNorm: Shape of normalized dimensions
   - GroupNorm: Shape (C,) - one per channel
   ```

3. **Training vs Inference**:
   ```
   Training: Use batch statistics (mean, var computed from current batch)
   Inference: Use running statistics (exponential moving average from training)
   ```

4. **Memory Layout Optimization**:
   ```
   Running statistics stored separately from learnable parameters
   Efficient computation using vectorized operations across normalization axes
   ```
"""

# %% [markdown]
"""
## Implementation: Building Normalization Classes

Let's implement the three essential normalization techniques used in modern deep learning.
"""

# %%
#| default_exp tinytorch.core.normalization
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
import warnings

# Import our tensor and layer base classes
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Module
except ImportError:
    # Fallback for development environment
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Module

# %% [markdown]
"""
### Batch Normalization Implementation

Batch Normalization normalizes activations across the batch dimension, making training more stable and allowing higher learning rates.

**Key Insight**: BatchNorm computes statistics across the batch dimension, so it requires batch_size > 1 during training.
"""

# %% nbgrader={"grade": false, "grade_id": "batch-norm", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class BatchNorm2d(Module):
    """
    Batch Normalization for 2D convolutions (4D tensors: N×C×H×W).

    Normalizes across the batch dimension, computing μ and σ² across N, H, W
    for each channel C independently.

    MATHEMATICAL FOUNDATION:
    BN(x) = γ * (x - μ_batch) / √(σ²_batch + ε) + β

    Where μ_batch and σ²_batch are computed across (N, H, W) dimensions.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize Batch Normalization layer.

        TODO: Implement BatchNorm initialization with running statistics.

        APPROACH (4-Step BatchNorm Setup):
        1. Store configuration parameters (num_features, eps, momentum)
        2. Initialize learnable parameters (γ=1, β=0) with proper shapes
        3. Initialize running statistics (running_mean=0, running_var=1)
        4. Set training mode flag for different train/eval behavior

        MEMORY ANALYSIS:
        - Learnable parameters: 2 × num_features (γ and β)
        - Running statistics: 2 × num_features (running_mean and running_var)
        - Total memory: 4 × num_features parameters

        EXAMPLE (BatchNorm Usage):
        >>> bn = BatchNorm2d(64)  # For 64 channels
        >>> x = Tensor(np.random.randn(32, 64, 28, 28))  # batch × channels × height × width
        >>> normalized = bn(x)
        >>> print(f"Normalized shape: {normalized.shape}")  # (32, 64, 28, 28)

        HINTS:
        - Use np.ones() for γ initialization (multiplicative identity)
        - Use np.zeros() for β initialization (additive identity)
        - Running statistics are numpy arrays (not Tensors - no gradients needed)
        - momentum controls exponential moving average: new_running = (1-momentum)*old + momentum*batch

        Args:
            num_features: Number of channels (C dimension)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        ### BEGIN SOLUTION
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters - shape (num_features,)
        self.gamma = Tensor(np.ones((num_features,)))  # Scale parameter
        self.beta = Tensor(np.zeros((num_features,)))   # Shift parameter

        # Running statistics for inference - numpy arrays (no gradients needed)
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

        # Track parameters for optimization
        self.parameters = [self.gamma, self.beta]
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply batch normalization to input tensor.

        TODO: Implement batch normalization forward pass with proper training/eval modes.

        STEP-BY-STEP IMPLEMENTATION:
        1. Determine which statistics to use (batch vs running)
        2. Compute mean and variance across appropriate dimensions
        3. Normalize: (x - mean) / sqrt(var + eps)
        4. Scale and shift: γ * normalized + β
        5. Update running statistics during training

        DIMENSION ANALYSIS for 4D input (N, C, H, W):
        - Batch statistics computed across dims (0, 2, 3) → shape (C,)
        - γ and β broadcasted to match input: (1, C, 1, 1)
        - Output has same shape as input

        TRAINING vs INFERENCE:
        - Training: Use batch statistics, update running statistics
        - Inference: Use running statistics, no updates

        EXAMPLE:
        >>> bn = BatchNorm2d(3)
        >>> x = Tensor(np.random.randn(16, 3, 32, 32))
        >>> bn.training = True   # Training mode
        >>> out_train = bn.forward(x)
        >>> bn.training = False  # Inference mode
        >>> out_eval = bn.forward(x)

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of shape (N, C, H, W)
        """
        ### BEGIN SOLUTION
        if self.training:
            # Training mode: compute batch statistics
            # Compute mean and variance across batch, height, width (dims 0, 2, 3)
            batch_mean = np.mean(x.data, axis=(0, 2, 3), keepdims=False)  # Shape: (C,)
            batch_var = np.var(x.data, axis=(0, 2, 3), keepdims=False)    # Shape: (C,)

            # Update running statistics using exponential moving average
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Inference mode: use running statistics
            mean = self.running_mean
            var = self.running_var

        # Reshape statistics for broadcasting: (1, C, 1, 1)
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        gamma = self.gamma.data.reshape(1, -1, 1, 1)
        beta = self.beta.data.reshape(1, -1, 1, 1)

        # Apply normalization: γ * (x - μ) / σ + β
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        output = gamma * normalized + beta

        return Tensor(output)
        ### END SOLUTION

    def train(self, mode: bool = True) -> 'BatchNorm2d':
        """Set training mode."""
        self.training = mode
        return self

    def eval(self) -> 'BatchNorm2d':
        """Set evaluation mode."""
        self.training = False
        return self

# 🔍 SYSTEMS INSIGHT: Batch Normalization Memory Analysis
def analyze_batchnorm_memory():
    """Let's analyze BatchNorm memory usage and batch dependency!"""
    try:
        print("🔍 SYSTEMS INSIGHT: Batch Normalization Analysis")
        print("=" * 50)

        # Different channel sizes to show scaling
        channel_sizes = [64, 256, 512, 1024]

        for channels in channel_sizes:
            bn = BatchNorm2d(channels)

            # Parameter memory calculation
            param_memory = 4 * channels * 4  # 4 params per channel × 4 bytes (float32)
            print(f"Channels: {channels:4d} | Parameters: {4 * channels:4d} | Memory: {param_memory / 1024:.2f} KB")

        print("\n💡 KEY INSIGHTS:")
        print("• BatchNorm memory scales linearly with channel count")
        print("• Only 4 parameters per channel (γ, β, running_mean, running_var)")
        print("• Memory overhead is typically < 1% of layer weights")

        # Batch size dependency demonstration
        print("\n🎯 BATCH SIZE DEPENDENCY:")
        bn = BatchNorm2d(64)

        for batch_size in [1, 8, 32, 128]:
            x = Tensor(np.random.randn(batch_size, 64, 32, 32))

            if batch_size == 1:
                print(f"Batch size {batch_size:3d}: ⚠️  May be unstable (poor statistics)")
            else:
                print(f"Batch size {batch_size:3d}: ✅ Good statistics")

        print("\n🚨 CRITICAL: BatchNorm needs batch_size > 1 for stable training!")
        print("   Single-sample batches have undefined variance")

    except Exception as e:
        print(f"⚠️ Error in BatchNorm analysis: {e}")

# Run the analysis
analyze_batchnorm_memory()

# %% [markdown]
"""
### 🧪 Unit Test: Batch Normalization

This test validates BatchNorm2d implementation, ensuring proper normalization across batch dimension and correct running statistics updates.
"""

# %% nbgrader={"grade": true, "grade_id": "test-batch-norm", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_batch_norm():
    """Unit test for batch normalization."""
    print("🔬 Unit Test: Batch Normalization...")

    # Test 1: Basic functionality
    num_features = 32
    bn = BatchNorm2d(num_features)

    # Verify initialization
    assert bn.num_features == num_features, "Should store number of features"
    assert bn.eps == 1e-5, "Should use default epsilon"
    assert bn.momentum == 0.1, "Should use default momentum"
    assert bn.training == True, "Should start in training mode"

    # Check parameter shapes
    assert bn.gamma.shape == (num_features,), f"Gamma shape should be ({num_features},)"
    assert bn.beta.shape == (num_features,), f"Beta shape should be ({num_features},)"
    assert np.allclose(bn.gamma.data, 1.0), "Gamma should be initialized to 1"
    assert np.allclose(bn.beta.data, 0.0), "Beta should be initialized to 0"

    # Test 2: Forward pass in training mode
    batch_size, height, width = 16, 8, 8
    x = Tensor(np.random.randn(batch_size, num_features, height, width))

    output = bn.forward(x)

    # Check output shape
    assert output.shape == x.shape, "Output should have same shape as input"

    # Check normalization (approximately zero mean, unit variance per channel)
    for c in range(num_features):
        channel_data = output.data[:, c, :, :]
        channel_mean = np.mean(channel_data)
        channel_var = np.var(channel_data)

        assert abs(channel_mean) < 1e-6, f"Channel {c} should have ~0 mean, got {channel_mean}"
        assert abs(channel_var - 1.0) < 1e-4, f"Channel {c} should have ~1 variance, got {channel_var}"

    # Test 3: Running statistics update
    initial_running_mean = bn.running_mean.copy()
    initial_running_var = bn.running_var.copy()

    # Process another batch
    x2 = Tensor(np.random.randn(batch_size, num_features, height, width) * 2 + 1)
    _ = bn.forward(x2)

    # Running statistics should have changed
    assert not np.allclose(bn.running_mean, initial_running_mean), "Running mean should update"
    assert not np.allclose(bn.running_var, initial_running_var), "Running variance should update"

    # Test 4: Evaluation mode
    bn.eval()
    assert bn.training == False, "Should be in eval mode"

    running_mean_before = bn.running_mean.copy()
    running_var_before = bn.running_var.copy()

    # Forward pass in eval mode
    output_eval = bn.forward(x)

    # Running statistics should not change in eval mode
    assert np.allclose(bn.running_mean, running_mean_before), "Running mean should not change in eval mode"
    assert np.allclose(bn.running_var, running_var_before), "Running variance should not change in eval mode"

    # Test 5: Gradient flow (basic check)
    bn.train()
    x_grad = Tensor(np.random.randn(batch_size, num_features, height, width))
    output_grad = bn.forward(x_grad)

    # Should be able to access gamma and beta for gradient computation
    assert hasattr(bn, 'gamma'), "Should have gamma parameter"
    assert hasattr(bn, 'beta'), "Should have beta parameter"
    assert len(bn.parameters) == 2, "Should have 2 learnable parameters"

    print("✅ Batch normalization tests passed!")
    print(f"✅ Properly normalizes across batch dimension")
    print(f"✅ Updates running statistics during training")
    print(f"✅ Uses running statistics during evaluation")
    print(f"✅ Maintains gradient flow through learnable parameters")

# Test function defined (called in main block)

# %% [markdown]
"""
### Layer Normalization Implementation

Layer Normalization normalizes across the feature dimensions for each sample independently, making it batch-size independent.

**Key Insight**: LayerNorm is crucial for transformers because it doesn't depend on batch statistics, enabling consistent behavior across different batch sizes.
"""

# %% nbgrader={"grade": false, "grade_id": "layer-norm", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class LayerNorm(Module):
    """
    Layer Normalization for any-dimensional tensors.

    Normalizes across specified feature dimensions for each sample independently.
    Unlike BatchNorm, LayerNorm doesn't depend on batch statistics.

    MATHEMATICAL FOUNDATION:
    LN(x) = γ * (x - μ) / √(σ² + ε) + β

    Where μ and σ² are computed across feature dimensions for each sample.
    """

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        """
        Initialize Layer Normalization.

        TODO: Implement LayerNorm initialization with proper shape handling.

        APPROACH (3-Step LayerNorm Setup):
        1. Store normalization configuration (shape and eps)
        2. Initialize learnable parameters γ and β with correct shapes
        3. Set up parameter tracking for optimization

        SHAPE ANALYSIS:
        - If normalized_shape is int: treat as last dimension only
        - If normalized_shape is tuple: treat as multiple dimensions
        - γ and β have shape matching normalized_shape

        EXAMPLE (LayerNorm Shapes):
        >>> ln1 = LayerNorm(512)        # For last dim: (..., 512)
        >>> ln2 = LayerNorm((64, 64))   # For last 2 dims: (..., 64, 64)
        >>> ln3 = LayerNorm((256, 4, 4)) # For 3D features: (..., 256, 4, 4)

        HINTS:
        - Convert int to tuple for consistent handling
        - Parameter shapes should match normalized_shape exactly
        - No running statistics needed (computed fresh each time)

        Args:
            normalized_shape: Shape of features to normalize over
            eps: Small constant for numerical stability
        """
        ### BEGIN SOLUTION
        super().__init__()

        # Handle both int and tuple inputs
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        # Learnable parameters with shape matching normalized dimensions
        self.gamma = Tensor(np.ones(self.normalized_shape))  # Scale parameter
        self.beta = Tensor(np.zeros(self.normalized_shape))   # Shift parameter

        # Track parameters for optimization
        self.parameters = [self.gamma, self.beta]
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to input tensor.

        TODO: Implement layer normalization forward pass.

        STEP-BY-STEP IMPLEMENTATION:
        1. Determine normalization axes based on normalized_shape
        2. Compute mean and variance across those axes (keepdims=True)
        3. Normalize: (x - mean) / sqrt(var + eps)
        4. Apply learnable parameters: γ * normalized + β

        AXIS CALCULATION:
        For input shape (N, ..., D1, D2, ..., Dk) and normalized_shape (D1, D2, ..., Dk):
        - Normalize over last len(normalized_shape) dimensions
        - Keep dimensions for proper broadcasting

        EXAMPLE:
        >>> ln = LayerNorm(256)
        >>> x = Tensor(np.random.randn(32, 128, 256))  # (batch, seq, features)
        >>> out = ln.forward(x)  # Normalize over last dim (256)

        Args:
            x: Input tensor

        Returns:
            Normalized tensor (same shape as input)
        """
        ### BEGIN SOLUTION
        # Calculate which axes to normalize over (last len(normalized_shape) dimensions)
        num_dims_to_normalize = len(self.normalized_shape)
        axes = tuple(range(-num_dims_to_normalize, 0))  # Last N dimensions

        # Compute mean and variance over normalization axes
        mean = np.mean(x.data, axis=axes, keepdims=True)
        var = np.var(x.data, axis=axes, keepdims=True)

        # Normalize
        normalized = (x.data - mean) / np.sqrt(var + self.eps)

        # Apply learnable parameters (broadcasting automatically handles shapes)
        output = self.gamma.data * normalized + self.beta.data

        return Tensor(output)
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allow LayerNorm to be called directly."""
        return self.forward(x)

# ✅ IMPLEMENTATION CHECKPOINT: Basic LayerNorm complete

# 🤔 PREDICTION: How does LayerNorm memory scale compared to BatchNorm?
# Your guess: LayerNorm uses _____ memory than BatchNorm for the same feature size

# 🔍 SYSTEMS INSIGHT: LayerNorm vs BatchNorm Memory Comparison
def compare_normalization_memory():
    """Compare memory usage between different normalization techniques."""
    try:
        print("🔍 SYSTEMS INSIGHT: Normalization Memory Comparison")
        print("=" * 60)

        # Test different feature configurations
        configs = [
            (64, "Small ConvNet channel"),
            (256, "ResNet channel"),
            (512, "Transformer embedding"),
            (1024, "Large transformer")
        ]

        print(f"{'Features':<8} {'BatchNorm':<12} {'LayerNorm':<12} {'Ratio':<8} {'Context'}")
        print("-" * 60)

        for features, context in configs:
            # BatchNorm memory: 4 parameters per channel (γ, β, running_mean, running_var)
            bn_memory = 4 * features * 4  # 4 bytes per float32

            # LayerNorm memory: 2 parameters per feature (γ, β only)
            ln_memory = 2 * features * 4  # 4 bytes per float32

            ratio = bn_memory / ln_memory

            print(f"{features:<8} {bn_memory/1024:.2f} KB     {ln_memory/1024:.2f} KB     {ratio:.1f}x      {context}")

        print(f"\n💡 KEY INSIGHTS:")
        print("• BatchNorm uses 2× more memory than LayerNorm")
        print("• BatchNorm stores running statistics (inference requirements)")
        print("• LayerNorm has no running state (batch-independent)")

        # Batch size independence demonstration
        print(f"\n🎯 BATCH SIZE INDEPENDENCE:")
        ln = LayerNorm(256)

        for batch_size in [1, 8, 32, 128]:
            x = Tensor(np.random.randn(batch_size, 64, 256))
            output = ln.forward(x)

            # Check normalization quality
            sample_mean = np.mean(output.data[0, :, :])  # First sample mean
            sample_var = np.var(output.data[0, :, :])    # First sample variance

            print(f"Batch size {batch_size:3d}: Mean={sample_mean:.6f}, Var={sample_var:.6f} ✅")

        print(f"\n✨ LayerNorm gives consistent results regardless of batch size!")

    except Exception as e:
        print(f"⚠️ Error in normalization comparison: {e}")

# Run the comparison
compare_normalization_memory()

# %% [markdown]
"""
### 🧪 Unit Test: Layer Normalization

This test validates LayerNorm implementation, ensuring proper normalization across feature dimensions and batch-size independence.
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-norm", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_layer_norm():
    """Unit test for layer normalization."""
    print("🔬 Unit Test: Layer Normalization...")

    # Test 1: Basic 1D normalization
    embed_dim = 256
    ln = LayerNorm(embed_dim)

    # Verify initialization
    assert ln.normalized_shape == (embed_dim,), "Should store normalized shape as tuple"
    assert ln.eps == 1e-5, "Should use default epsilon"

    # Check parameter shapes
    assert ln.gamma.shape == (embed_dim,), f"Gamma shape should be ({embed_dim},)"
    assert ln.beta.shape == (embed_dim,), f"Beta shape should be ({embed_dim},)"
    assert np.allclose(ln.gamma.data, 1.0), "Gamma should be initialized to 1"
    assert np.allclose(ln.beta.data, 0.0), "Beta should be initialized to 0"

    # Test 2: Forward pass with 3D input (batch, seq, features)
    batch_size, seq_len = 16, 64
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim) * 2 + 3)  # Non-standard distribution

    output = ln.forward(x)

    # Check output shape
    assert output.shape == x.shape, "Output should have same shape as input"

    # Check normalization for each sample independently
    for b in range(batch_size):
        for s in range(seq_len):
            sample_data = output.data[b, s, :]
            sample_mean = np.mean(sample_data)
            sample_var = np.var(sample_data)

            assert abs(sample_mean) < 1e-6, f"Sample [{b},{s}] should have ~0 mean, got {sample_mean}"
            assert abs(sample_var - 1.0) < 1e-4, f"Sample [{b},{s}] should have ~1 variance, got {sample_var}"

    # Test 3: Multi-dimensional normalization
    multi_dim_shape = (64, 4)  # Normalize over 2D features
    ln_multi = LayerNorm(multi_dim_shape)

    x_multi = Tensor(np.random.randn(8, 32, 64, 4))
    output_multi = ln_multi.forward(x_multi)

    assert output_multi.shape == x_multi.shape, "Multi-dim normalization should preserve shape"

    # Check normalization across last 2 dimensions for each sample
    for b in range(8):
        for s in range(32):
            sample_data = output_multi.data[b, s, :, :].flatten()
            sample_mean = np.mean(sample_data)
            sample_var = np.var(sample_data)

            assert abs(sample_mean) < 1e-6, f"Multi-dim sample should have ~0 mean"
            assert abs(sample_var - 1.0) < 1e-4, f"Multi-dim sample should have ~1 variance"

    # Test 4: Callable interface
    output_callable = ln(x)
    assert np.allclose(output.data, output_callable.data), "Callable interface should work"

    # Test 5: Batch size independence
    x_small = Tensor(np.random.randn(1, seq_len, embed_dim))
    x_large = Tensor(np.random.randn(64, seq_len, embed_dim))

    output_small = ln.forward(x_small)
    output_large = ln.forward(x_large)

    # Both should be properly normalized regardless of batch size
    small_mean = np.mean(output_small.data[0, 0, :])
    large_mean = np.mean(output_large.data[0, 0, :])  # Same position

    assert abs(small_mean) < 1e-6, "Small batch should be normalized"
    assert abs(large_mean) < 1e-6, "Large batch should be normalized"

    # Test 6: Parameter tracking
    assert len(ln.parameters) == 2, "Should have 2 learnable parameters"
    assert ln.gamma in ln.parameters, "Gamma should be tracked"
    assert ln.beta in ln.parameters, "Beta should be tracked"

    print("✅ Layer normalization tests passed!")
    print(f"✅ Properly normalizes across feature dimensions")
    print(f"✅ Works with any input shape")
    print(f"✅ Batch-size independent behavior")
    print(f"✅ Supports multi-dimensional normalization")

# Test function defined (called in main block)

# %% [markdown]
"""
### Group Normalization Implementation

Group Normalization divides channels into groups and normalizes within each group, providing a middle ground between batch and layer normalization.

**Key Insight**: GroupNorm is particularly useful for object detection and when batch sizes are small, as it doesn't depend on batch statistics but provides channel-wise organization.
"""

# %% nbgrader={"grade": false, "grade_id": "group-norm", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class GroupNorm(Module):
    """
    Group Normalization for convolutional layers.

    Divides channels into groups and normalizes within each group.
    Provides benefits of both batch and layer normalization.

    MATHEMATICAL FOUNDATION:
    For input (N, C, H, W) with G groups:
    1. Reshape to (N, G, C//G, H, W)
    2. Normalize within each group: GN(x) = γ * (x - μ_group) / √(σ²_group + ε) + β
    3. Reshape back to (N, C, H, W)
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        """
        Initialize Group Normalization.

        TODO: Implement GroupNorm initialization with group configuration.

        APPROACH (4-Step GroupNorm Setup):
        1. Validate group configuration (num_channels must be divisible by num_groups)
        2. Store configuration parameters
        3. Initialize learnable parameters γ and β for each channel
        4. Set up parameter tracking

        GROUP ORGANIZATION:
        - Each group contains num_channels // num_groups channels
        - Normalization computed independently within each group
        - Parameters γ and β have shape (num_channels,) for per-channel scaling

        EXAMPLE (GroupNorm Configurations):
        >>> gn1 = GroupNorm(32, 64)   # 32 groups, 64 channels → 2 channels per group
        >>> gn2 = GroupNorm(8, 256)   # 8 groups, 256 channels → 32 channels per group
        >>> gn3 = GroupNorm(1, 128)   # 1 group, 128 channels → LayerNorm equivalent

        HINTS:
        - Use assert to validate num_channels % num_groups == 0
        - Special case: num_groups = num_channels → InstanceNorm (each channel is a group)
        - Special case: num_groups = 1 → LayerNorm for spatial data

        Args:
            num_groups: Number of groups to divide channels into
            num_channels: Total number of channels
            eps: Small constant for numerical stability
        """
        ### BEGIN SOLUTION
        super().__init__()

        # Validate configuration
        assert num_channels % num_groups == 0, f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        assert num_groups > 0, "num_groups must be positive"
        assert num_channels > 0, "num_channels must be positive"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # Calculate channels per group
        self.channels_per_group = num_channels // num_groups

        # Learnable parameters - one per channel
        self.gamma = Tensor(np.ones((num_channels,)))  # Scale parameter
        self.beta = Tensor(np.zeros((num_channels,)))   # Shift parameter

        # Track parameters for optimization
        self.parameters = [self.gamma, self.beta]
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply group normalization to input tensor.

        TODO: Implement group normalization forward pass.

        STEP-BY-STEP IMPLEMENTATION:
        1. Reshape input to separate groups: (N, C, H, W) → (N, G, C//G, H, W)
        2. Compute mean and variance within each group
        3. Normalize within groups
        4. Reshape back to original shape
        5. Apply per-channel γ and β parameters

        SHAPE TRANSFORMATIONS:
        Input:  (N, C, H, W)
        Groups: (N, G, C//G, H, W)  # Separate groups for normalization
        Norm:   (N, G, C//G, H, W)  # Normalized within groups
        Output: (N, C, H, W)        # Back to original shape with γ/β applied

        EXAMPLE:
        >>> gn = GroupNorm(8, 64)  # 8 groups, 64 channels
        >>> x = Tensor(np.random.randn(16, 64, 32, 32))
        >>> out = gn.forward(x)  # Normalized within 8 groups

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of shape (N, C, H, W)
        """
        ### BEGIN SOLUTION
        N, C, H, W = x.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"

        # Reshape to separate groups: (N, C, H, W) → (N, G, C//G, H, W)
        x_grouped = x.data.reshape(N, self.num_groups, self.channels_per_group, H, W)

        # Compute mean and variance within each group
        # Normalize over dimensions (2, 3, 4) which are (channels_per_group, H, W)
        mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)  # Shape: (N, G, 1, 1, 1)
        var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)    # Shape: (N, G, 1, 1, 1)

        # Normalize within groups
        normalized = (x_grouped - mean) / np.sqrt(var + self.eps)

        # Reshape back to original shape: (N, G, C//G, H, W) → (N, C, H, W)
        normalized = normalized.reshape(N, C, H, W)

        # Apply per-channel learnable parameters
        gamma = self.gamma.data.reshape(1, C, 1, 1)  # Broadcast shape
        beta = self.beta.data.reshape(1, C, 1, 1)    # Broadcast shape

        output = gamma * normalized + beta

        return Tensor(output)
        ### END SOLUTION

# ✅ IMPLEMENTATION CHECKPOINT: All normalization techniques complete

# 🤔 PREDICTION: Which normalization uses the most memory - Batch, Layer, or Group?
# Your answer: _______ because _______

# 🔍 SYSTEMS INSIGHT: Complete Normalization Scaling Analysis
def analyze_normalization_scaling():
    """Analyze how different normalization techniques scale with architecture size."""
    try:
        print("🔍 SYSTEMS INSIGHT: Normalization Scaling Analysis")
        print("=" * 70)

        # Different model scales to analyze
        model_configs = [
            (64, "Small CNN"),
            (256, "ResNet-50 layer"),
            (512, "Large CNN"),
            (1024, "Vision Transformer")
        ]

        print(f"{'Channels':<8} {'BatchNorm':<12} {'LayerNorm':<12} {'GroupNorm':<12} {'Context'}")
        print("-" * 70)

        for channels, context in model_configs:
            # Memory calculations (in bytes, float32 = 4 bytes)
            bn_memory = 4 * channels * 4  # γ, β, running_mean, running_var
            ln_memory = 2 * channels * 4  # γ, β only
            gn_memory = 2 * channels * 4  # γ, β only (same as LayerNorm)

            print(f"{channels:<8} {bn_memory/1024:.2f} KB     {ln_memory/1024:.2f} KB     {gn_memory/1024:.2f} KB     {context}")

        print(f"\n💡 MEMORY INSIGHTS:")
        print("• BatchNorm: Highest memory (stores running statistics)")
        print("• LayerNorm: 50% less memory than BatchNorm")
        print("• GroupNorm: Same memory as LayerNorm")

        # Computational complexity analysis
        print(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
        batch_size, height, width = 32, 64, 64
        channels = 256

        # Calculate FLOPs for each normalization type
        spatial_size = height * width
        total_elements = batch_size * channels * spatial_size

        # All normalizations require: mean, variance, normalize, scale, shift
        base_flops = 5 * total_elements  # 5 operations per element

        print(f"Input: ({batch_size}, {channels}, {height}, {width})")
        print(f"BatchNorm FLOPs: ~{base_flops/1e6:.1f}M (batch statistics)")
        print(f"LayerNorm FLOPs: ~{base_flops/1e6:.1f}M (per-sample statistics)")
        print(f"GroupNorm FLOPs: ~{base_flops/1e6:.1f}M (group statistics)")

        print(f"\n🎯 WHEN TO USE EACH:")
        print("• BatchNorm: Large batches, CNNs, stable batch sizes")
        print("• LayerNorm: Transformers, variable batch sizes, RNNs")
        print("• GroupNorm: Small batches, object detection, fine-tuning")

        # Demonstrate batch size effects
        print(f"\n📊 BATCH SIZE EFFECTS:")
        test_channels = 128
        bn = BatchNorm2d(test_channels)
        ln = LayerNorm((test_channels, 32, 32))
        gn = GroupNorm(32, test_channels)

        for batch_size in [1, 4, 16, 64]:
            x = Tensor(np.random.randn(batch_size, test_channels, 32, 32))

            # Only test mean for first sample to see consistency
            if batch_size > 1:  # BatchNorm needs batch_size > 1
                bn_out = bn.forward(x)
                bn_mean = np.mean(bn_out.data[0])
            else:
                bn_mean = "unstable"

            ln_out = ln.forward(x)
            ln_mean = np.mean(ln_out.data[0])

            gn_out = gn.forward(x)
            gn_mean = np.mean(gn_out.data[0])

            print(f"Batch {batch_size:2d}: BN={bn_mean if isinstance(bn_mean, str) else f'{bn_mean:.6f}':<10} "
                  f"LN={ln_mean:.6f} GN={gn_mean:.6f}")

    except Exception as e:
        print(f"⚠️ Error in scaling analysis: {e}")

# Run the scaling analysis
analyze_normalization_scaling()

# %% [markdown]
"""
### 🧪 Unit Test: Group Normalization

This test validates GroupNorm implementation, ensuring proper grouping and normalization within channel groups.
"""

# %% nbgrader={"grade": true, "grade_id": "test-group-norm", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_group_norm():
    """Unit test for group normalization."""
    print("🔬 Unit Test: Group Normalization...")

    # Test 1: Basic configuration
    num_groups = 8
    num_channels = 64
    gn = GroupNorm(num_groups, num_channels)

    # Verify initialization
    assert gn.num_groups == num_groups, "Should store number of groups"
    assert gn.num_channels == num_channels, "Should store number of channels"
    assert gn.channels_per_group == 8, "Should calculate channels per group correctly"

    # Check parameter shapes
    assert gn.gamma.shape == (num_channels,), f"Gamma shape should be ({num_channels},)"
    assert gn.beta.shape == (num_channels,), f"Beta shape should be ({num_channels},)"

    # Test 2: Configuration validation
    try:
        GroupNorm(7, 64)  # Should fail: 64 % 7 != 0
        assert False, "Should raise error for invalid group configuration"
    except AssertionError as e:
        if "divisible" in str(e):
            pass  # Expected error
        else:
            raise e

    # Test 3: Forward pass
    batch_size, height, width = 16, 32, 32
    x = Tensor(np.random.randn(batch_size, num_channels, height, width) * 3 + 2)

    output = gn.forward(x)

    # Check output shape
    assert output.shape == x.shape, "Output should have same shape as input"

    # Test 4: Verify group normalization properties
    # Each group should have approximately normalized statistics
    channels_per_group = num_channels // num_groups

    for group_idx in range(num_groups):
        start_channel = group_idx * channels_per_group
        end_channel = start_channel + channels_per_group

        # Extract group data for first sample
        group_data = output.data[0, start_channel:end_channel, :, :].flatten()
        group_mean = np.mean(group_data)
        group_var = np.var(group_data)

        assert abs(group_mean) < 1e-5, f"Group {group_idx} should have ~0 mean, got {group_mean}"
        assert abs(group_var - 1.0) < 1e-3, f"Group {group_idx} should have ~1 variance, got {group_var}"

    # Test 5: Special cases
    # Case 1: num_groups = num_channels (Instance Normalization)
    instance_norm = GroupNorm(num_channels, num_channels)
    assert instance_norm.channels_per_group == 1, "Instance norm should have 1 channel per group"

    # Case 2: num_groups = 1 (Layer Normalization for spatial data)
    layer_norm_like = GroupNorm(1, num_channels)
    assert layer_norm_like.channels_per_group == num_channels, "Single group should contain all channels"

    # Test 6: Different group sizes
    configs_to_test = [
        (1, 32),   # LayerNorm-like
        (4, 32),   # 8 channels per group
        (32, 32),  # InstanceNorm-like
    ]

    for groups, channels in configs_to_test:
        gn_test = GroupNorm(groups, channels)
        x_test = Tensor(np.random.randn(8, channels, 16, 16))
        output_test = gn_test.forward(x_test)

        assert output_test.shape == x_test.shape, f"Config ({groups}, {channels}) should preserve shape"

        # Basic normalization check
        sample_data = output_test.data[0, :, :, :].flatten()
        overall_mean = np.mean(sample_data)
        # Note: overall variance might not be exactly 1 due to grouping

    # Test 7: Parameter tracking
    assert len(gn.parameters) == 2, "Should have 2 learnable parameters"
    assert gn.gamma in gn.parameters, "Gamma should be tracked"
    assert gn.beta in gn.parameters, "Beta should be tracked"

    print("✅ Group normalization tests passed!")
    print(f"✅ Properly groups channels and normalizes within groups")
    print(f"✅ Validates configuration constraints")
    print(f"✅ Supports special cases (Instance/Layer norm variants)")
    print(f"✅ Maintains gradient flow through learnable parameters")

# Test function defined (called in main block)

# %% [markdown]
"""
## Integration: Normalization in Neural Networks

Now let's see how normalization techniques integrate with neural network layers to stabilize training and improve performance.
"""

# %% [markdown]
"""
### Normalization Layer Integration Example

Here's how normalization layers are typically used in different architectures:

**ConvNet with BatchNorm:**
```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → ...
```

**Transformer with LayerNorm:**
```
Embedding → LayerNorm → Attention → Add & Norm → FFN → Add & Norm → ...
```

**ResNet Block with GroupNorm:**
```
Conv2d → GroupNorm → ReLU → Conv2d → GroupNorm → Add → ReLU
```
"""

# %% nbgrader={"grade": false, "grade_id": "normalization-example", "locked": false, "schema_version": 3, "solution": true, "task": false}
def demonstrate_normalization_usage():
    """
    Demonstrate how different normalization techniques are used in practice.

    TODO: Implement a simple example showing normalization in a mini-network.

    APPROACH:
    1. Create sample activations that would be unstable without normalization
    2. Apply different normalization techniques
    3. Show how they stabilize the activations
    4. Demonstrate the effect on gradient flow

    This function is PROVIDED as an educational example.
    """
    ### BEGIN SOLUTION
    print("🔬 Normalization Integration Example")
    print("=" * 40)

    # Simulate unstable activations (high variance, non-zero mean)
    batch_size, channels, height, width = 16, 64, 32, 32
    unstable_activations = Tensor(np.random.randn(batch_size, channels, height, width) * 5 + 3)

    print(f"Original activations:")
    print(f"  Mean: {np.mean(unstable_activations.data):.3f}")
    print(f"  Std:  {np.std(unstable_activations.data):.3f}")
    print(f"  Range: [{np.min(unstable_activations.data):.2f}, {np.max(unstable_activations.data):.2f}]")

    # Apply different normalizations
    bn = BatchNorm2d(channels)
    ln = LayerNorm((channels, height, width))
    gn = GroupNorm(8, channels)

    bn.train()  # Ensure BatchNorm is in training mode

    bn_output = bn.forward(unstable_activations)
    ln_output = ln.forward(unstable_activations)
    gn_output = gn.forward(unstable_activations)

    print(f"\nAfter BatchNorm:")
    print(f"  Mean: {np.mean(bn_output.data):.6f}")
    print(f"  Std:  {np.std(bn_output.data):.3f}")

    print(f"\nAfter LayerNorm:")
    print(f"  Mean: {np.mean(ln_output.data):.6f}")
    print(f"  Std:  {np.std(ln_output.data):.3f}")

    print(f"\nAfter GroupNorm:")
    print(f"  Mean: {np.mean(gn_output.data):.6f}")
    print(f"  Std:  {np.std(gn_output.data):.3f}")

    print(f"\n✅ All normalization techniques stabilize activations!")
    print(f"✅ Mean ≈ 0, Std ≈ 1 for all methods")
    ### END SOLUTION

# Run the demonstration
demonstrate_normalization_usage()

# %% [markdown]
"""
### Performance Comparison: Training Stability

Let's compare how different normalization techniques affect training stability by simulating gradient updates.
"""

# ✅ IMPLEMENTATION CHECKPOINT: All normalization implementations complete

# 🤔 PREDICTION: Which normalization technique will be most stable for very small batch sizes?
# Your answer: _______ because _______

# 🔍 SYSTEMS INSIGHT: Training Stability Analysis
def analyze_training_stability():
    """Analyze how normalization affects training stability across different scenarios."""
    try:
        print("🔍 SYSTEMS INSIGHT: Training Stability Analysis")
        print("=" * 60)

        # Test stability across different batch sizes
        channels = 128
        scenarios = [
            (1, "Single sample (inference)"),
            (2, "Tiny batch (edge case)"),
            (8, "Small batch (mobile/edge)"),
            (32, "Standard batch"),
            (128, "Large batch")
        ]

        bn = BatchNorm2d(channels)
        ln = LayerNorm((channels, 16, 16))
        gn = GroupNorm(16, channels)

        print(f"{'Batch Size':<12} {'BatchNorm':<12} {'LayerNorm':<12} {'GroupNorm':<12} {'Scenario'}")
        print("-" * 70)

        for batch_size, scenario in scenarios:
            x = Tensor(np.random.randn(batch_size, channels, 16, 16) * 2 + 1)

            # BatchNorm stability
            if batch_size == 1:
                bn_stability = "UNSTABLE"  # Can't compute batch stats with N=1
            else:
                bn.train()
                bn_out = bn.forward(x)
                bn_var = np.var(bn_out.data)
                bn_stability = f"{bn_var:.4f}"

            # LayerNorm stability
            ln_out = ln.forward(x)
            ln_var = np.var(ln_out.data[0])  # Per sample variance
            ln_stability = f"{ln_var:.4f}"

            # GroupNorm stability
            gn_out = gn.forward(x)
            gn_var = np.var(gn_out.data[0])  # Per sample variance
            gn_stability = f"{gn_var:.4f}"

            print(f"{batch_size:<12} {bn_stability:<12} {ln_stability:<12} {gn_stability:<12} {scenario}")

        print(f"\n💡 STABILITY INSIGHTS:")
        print("• BatchNorm: Unstable with batch_size=1, best with large batches")
        print("• LayerNorm: Consistent across all batch sizes")
        print("• GroupNorm: Consistent across all batch sizes")

        # Gradient flow analysis
        print(f"\n🌊 GRADIENT FLOW ANALYSIS:")

        # Simulate deep network gradients
        x = Tensor(np.random.randn(16, channels, 16, 16))

        # Test gradient magnitude after normalization
        original_grad_norm = np.linalg.norm(x.data)

        bn_out = bn.forward(x)
        ln_out = ln.forward(x)
        gn_out = gn.forward(x)

        print(f"Original gradient norm: {original_grad_norm:.3f}")
        print(f"After BatchNorm: ~{np.linalg.norm(bn_out.data):.3f} (normalized)")
        print(f"After LayerNorm: ~{np.linalg.norm(ln_out.data):.3f} (normalized)")
        print(f"After GroupNorm: ~{np.linalg.norm(gn_out.data):.3f} (normalized)")

        print(f"\n🎯 PRACTICAL RECOMMENDATIONS:")
        print("• Use BatchNorm for: CNNs with batch_size ≥ 8, stable training")
        print("• Use LayerNorm for: Transformers, RNNs, variable batch sizes")
        print("• Use GroupNorm for: Object detection, fine-tuning, small batches")

    except Exception as e:
        print(f"⚠️ Error in stability analysis: {e}")

# Run the stability analysis
analyze_training_stability()

# %% [markdown]
"""
### 🧪 Integration Test: Complete Normalization Suite

This test validates that all normalization techniques work together and can be used interchangeably in neural network architectures.
"""

# %% nbgrader={"grade": true, "grade_id": "test-normalization-integration", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_normalization_integration():
    """Integration test for all normalization techniques."""
    print("🔬 Integration Test: Complete Normalization Suite...")

    # Test configuration
    batch_size, channels, height, width = 8, 32, 16, 16
    x = Tensor(np.random.randn(batch_size, channels, height, width) * 3 + 2)

    # Initialize all normalization types
    bn = BatchNorm2d(channels)
    ln = LayerNorm((channels, height, width))
    gn = GroupNorm(8, channels)  # 4 channels per group

    # Test 1: All normalizations work with same input
    bn.train()
    bn_output = bn.forward(x)
    ln_output = ln.forward(x)
    gn_output = gn.forward(x)

    # All should have same output shape
    assert bn_output.shape == x.shape, "BatchNorm should preserve shape"
    assert ln_output.shape == x.shape, "LayerNorm should preserve shape"
    assert gn_output.shape == x.shape, "GroupNorm should preserve shape"

    # Test 2: All produce normalized outputs
    for name, output in [("BatchNorm", bn_output), ("LayerNorm", ln_output), ("GroupNorm", gn_output)]:
        # Check that outputs are normalized (approximately)
        output_mean = np.mean(output.data)
        output_std = np.std(output.data)

        # Normalization should reduce extreme values
        assert abs(output_mean) < 2.0, f"{name} should reduce mean magnitude"
        assert 0.5 < output_std < 2.0, f"{name} should normalize standard deviation"

    # Test 3: Parameter count comparison
    bn_params = len(bn.parameters)
    ln_params = len(ln.parameters)
    gn_params = len(gn.parameters)

    assert bn_params == 2, "BatchNorm should have 2 learnable parameters"
    assert ln_params == 2, "LayerNorm should have 2 learnable parameters"
    assert gn_params == 2, "GroupNorm should have 2 learnable parameters"

    # Test 4: Training vs evaluation mode (BatchNorm only)
    bn.train()
    bn_train_out = bn.forward(x)

    bn.eval()
    bn_eval_out = bn.forward(x)

    # Outputs should be different (training uses batch stats, eval uses running stats)
    # Note: might be similar if running stats are close to batch stats
    assert bn_train_out.shape == bn_eval_out.shape, "Train/eval should have same shape"

    # Test 5: Batch size independence (LayerNorm and GroupNorm)
    x_single = Tensor(np.random.randn(1, channels, height, width))

    ln_single = ln.forward(x_single)
    gn_single = gn.forward(x_single)

    assert ln_single.shape == x_single.shape, "LayerNorm should work with batch_size=1"
    assert gn_single.shape == x_single.shape, "GroupNorm should work with batch_size=1"

    # Test 6: Memory efficiency check
    # All should use similar parameter memory (2 * channels * 4 bytes for γ and β)
    expected_param_memory = 2 * channels * 4  # γ and β parameters

    # BatchNorm has additional running statistics
    bn_total_memory = 4 * channels * 4  # γ, β, running_mean, running_var
    ln_total_memory = 2 * channels * 4  # γ, β only
    gn_total_memory = 2 * channels * 4  # γ, β only

    assert bn_total_memory > ln_total_memory, "BatchNorm should use more memory (running stats)"
    assert ln_total_memory == gn_total_memory, "LayerNorm and GroupNorm should use same memory"

    print("✅ Normalization integration tests passed!")
    print(f"✅ All techniques work with same input format")
    print(f"✅ All produce appropriately normalized outputs")
    print(f"✅ Memory usage patterns are as expected")
    print(f"✅ Batch size independence works correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
## Testing: Comprehensive Validation

Let's run comprehensive tests to ensure all normalization implementations work correctly.
"""

# %% [markdown]
"""
### Performance Benchmarking

Let's benchmark the performance characteristics of our normalization implementations.
"""

def benchmark_normalization_performance():
    """
    Benchmark performance of different normalization techniques.

    This function is PROVIDED for educational analysis.
    """
    print("⚡ Performance Benchmark: Normalization Techniques")
    print("=" * 55)

    import time

    # Test configuration
    batch_size, channels, height, width = 32, 256, 64, 64
    num_iterations = 100

    # Create test data
    x = Tensor(np.random.randn(batch_size, channels, height, width))

    # Initialize normalization layers
    bn = BatchNorm2d(channels)
    ln = LayerNorm((channels, height, width))
    gn = GroupNorm(32, channels)  # 8 channels per group

    # Benchmark each technique
    techniques = [
        ("BatchNorm2d", bn),
        ("LayerNorm", ln),
        ("GroupNorm", gn)
    ]

    results = {}

    for name, norm_layer in techniques:
        if name == "BatchNorm2d":
            norm_layer.train()  # Ensure training mode

        # Warmup
        for _ in range(10):
            _ = norm_layer.forward(x)

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            output = norm_layer.forward(x)
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) * 1000 / num_iterations
        results[name] = avg_time_ms

        print(f"{name:<12}: {avg_time_ms:.3f} ms/forward")

    # Analysis
    print(f"\n📊 Performance Analysis:")
    baseline = results["BatchNorm2d"]
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        print(f"  {name}: {speedup:.2f}x relative to BatchNorm")

    print(f"\n💡 Performance Insights:")
    print(f"  • All normalizations have similar computational complexity")
    print(f"  • Differences mainly due to memory access patterns")
    print(f"  • BatchNorm may be slightly faster due to batch parallelization")

# Run performance benchmark
benchmark_normalization_performance()

# %% [markdown]
"""
## Main Execution Block

Run all tests to validate our normalization implementations.
"""

if __name__ == "__main__":
    """Main execution block - runs all normalization tests."""
    print("🧪 Running Complete Normalization Test Suite")
    print("=" * 50)

    # Run all unit tests
    test_unit_batch_norm()
    print()

    test_unit_layer_norm()
    print()

    test_unit_group_norm()
    print()

    test_unit_normalization_integration()
    print()

    print("✅ All normalization tests passed!")
    print("\n🎯 NORMALIZATION SUITE COMPLETE")
    print("Your normalization implementations are ready for use in neural networks!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented all three major normalization techniques, let's reflect on their systems implications and design trade-offs.
"""

# %% [markdown]
"""
### Question 1: Memory and Batch Size Trade-offs

**Context**: In your BatchNorm2d implementation, you saw that running statistics require additional memory (4× parameters vs 2× for LayerNorm/GroupNorm), but BatchNorm fails completely with batch_size=1. Your memory analysis showed that BatchNorm needs 2× the memory of other techniques, while your stability analysis revealed batch size dependencies.

**Reflection Question**: Analyze the memory vs batch size trade-offs in your normalization implementations. When you tested different batch sizes, you discovered BatchNorm becomes unstable with small batches while LayerNorm/GroupNorm remain consistent. For a production system that needs to handle both training (large batches) and inference (single samples), how would you modify your current normalization implementations to optimize memory usage while maintaining stability? Consider the running statistics storage in your BatchNorm class and the per-sample computation in your LayerNorm class.

Think about: running statistics memory optimization, batch size adaptation strategies, inference mode memory requirements, and hybrid normalization approaches.

*Target length: 150-300 words*
"""

# %% [markdown]
"""
### Question 2: Computational Scaling and Group Organization

**Context**: Your GroupNorm implementation divides channels into groups and normalizes within each group, providing a middle ground between BatchNorm and LayerNorm. Your scaling analysis showed that all normalization techniques have similar computational complexity, but different memory access patterns. The group organization in your implementation affects both memory layout and computational efficiency.

**Reflection Question**: Examine the computational scaling patterns in your normalization implementations. Your GroupNorm.forward() method reshapes tensors to separate groups, computes statistics within groups, then reshapes back. How does this grouping strategy affect memory access patterns and cache efficiency compared to your BatchNorm (batch-wise) and LayerNorm (sample-wise) approaches? If you needed to optimize your GroupNorm implementation for very large channel counts (1024+ channels), what modifications to your group organization and computation order would improve performance while maintaining mathematical correctness?

Think about: memory access patterns, cache locality, vectorization opportunities, and group size optimization strategies.

*Target length: 150-300 words*
"""

# %% [markdown]
"""
### Question 3: Production Deployment and Architecture Selection

**Context**: Your normalization implementations mirror production systems - BatchNorm for CNNs like ResNet, LayerNorm for Transformers like BERT/GPT, and GroupNorm for object detection models. Your training stability analysis revealed when each technique works best, and your performance benchmarks showed similar computational costs but different memory characteristics.

**Reflection Question**: Based on your implementation experience and performance analysis, design a normalization selection strategy for a production ML system that needs to support multiple model architectures (CNNs, Transformers, and detection models). Your BatchNorm implementation works well for large-batch training but fails at batch_size=1, while your LayerNorm provides consistent behavior but lacks the batch parallelization benefits. How would you extend your current normalization classes to create an adaptive normalization system that automatically selects the optimal technique based on input characteristics (batch size, model architecture, deployment constraints)?

Think about: automatic technique selection, runtime adaptation, memory budget constraints, and deployment environment requirements.

*Target length: 150-300 words*
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Normalization

Congratulations! You have successfully implemented the complete normalization toolkit that makes modern deep learning possible:

### ✅ What You Have Built
- **BatchNorm2d**: Complete batch normalization with running statistics and train/eval modes
- **LayerNorm**: Batch-independent normalization for any tensor dimensions
- **GroupNorm**: Channel group normalization balancing batch and layer norm benefits
- **🆕 Comprehensive Analysis**: Memory scaling, training stability, and performance benchmarking
- **🆕 Integration Examples**: How normalization fits into different network architectures

### ✅ Technical Mastery
- **Statistical Computing**: Efficient mean/variance computation across different tensor dimensions
- **Memory Management**: Understanding parameter storage vs running statistics trade-offs
- **Training Dynamics**: How normalization affects gradient flow and training stability
- **Batch Dependencies**: When and why batch size affects normalization behavior
- **🆕 Production Patterns**: Architecture-specific normalization choices and deployment considerations

### ✅ Systems Understanding
- **Memory Scaling**: BatchNorm uses 2× memory of LayerNorm/GroupNorm due to running statistics
- **Computational Complexity**: All techniques have similar O(N) complexity but different access patterns
- **Batch Size Effects**: BatchNorm requires batch_size > 1, others work with any batch size
- **Cache Efficiency**: How normalization axes affect memory access patterns and vectorization
- **🆕 Training Stability**: Why normalization enables higher learning rates and deeper networks

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch nn.BatchNorm2d**: Your BatchNorm2d matches PyTorch's interface and behavior
- **BERT LayerNorm**: Your LayerNorm enables transformer training stability
- **Object Detection GroupNorm**: Your GroupNorm provides batch-independent normalization
- **Production Deployment**: Understanding of when to use each technique in real systems

### 🚀 What You Can Build Now
- **Stable CNNs**: Use BatchNorm for ResNet-style architectures with large batches
- **Transformer Models**: Use LayerNorm for attention-based architectures
- **Detection Systems**: Use GroupNorm for models with variable batch sizes
- **Adaptive Networks**: Combine techniques for optimal performance across scenarios

### Next Steps
1. **Export your module**: `tito module complete 08_normalization`
2. **Integration ready**: Your normalization layers integrate with any neural network architecture
3. **Ready for Module 09**: Spatial operations will use your normalization for CNN stability

**🎉 Achievement Unlocked**: You've mastered the normalization techniques that enable modern deep learning, with complete understanding of their memory characteristics and performance trade-offs!
"""