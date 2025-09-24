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
# Loss Functions - Essential Training Objectives for Neural Networks

Welcome to the Loss Functions module! You'll implement the essential loss functions that define learning objectives and enable neural networks to learn from data through gradient-based optimization.

## Learning Goals
- Systems understanding: How loss functions define learning objectives and drive gradient-based optimization
- Core implementation skill: Build MSE, CrossEntropy, and BinaryCrossEntropy with proper numerical stability
- Pattern recognition: Understand how different loss functions shape learning dynamics and convergence behavior
- Framework connection: See how your loss implementations mirror PyTorch's loss functions and autograd integration
- Performance insight: Learn why numerically stable loss computation affects training reliability and convergence speed

## Build → Use → Reflect
1. **Build**: Complete loss function implementations with numerical stability and gradient support
2. **Use**: Apply loss functions to regression and classification problems with real neural networks
3. **Reflect**: Why do different loss functions lead to different learning behaviors, and when does numerical stability matter?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how loss functions translate learning problems into optimization objectives
- Practical capability to implement production-quality loss functions with proper numerical stability
- Systems insight into why loss function design affects training stability and convergence characteristics
- Performance consideration of how numerical precision in loss computation affects training reliability
- Connection to production ML systems and how frameworks implement robust loss computation

## Systems Reality Check
💡 **Production Context**: PyTorch's loss functions use numerically stable implementations and automatic mixed precision to handle extreme gradients and values
⚡ **Performance Note**: Numerically unstable loss functions can cause training to fail catastrophically - proper implementation is critical for reliable ML systems
"""

# %% nbgrader={"grade": false, "grade_id": "losses-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.losses

#| export
import numpy as np
import sys
import os

# Import our building blocks - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    # Note: For now, we'll use simplified implementations without full autograd
    # In a complete system, these would integrate with the autograd Variable system
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor

# %% nbgrader={"grade": false, "grade_id": "losses-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Loss Functions Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build loss functions for neural network training!")

# %% [markdown]
"""
## Where This Code Lives in the Final Package

**Learning Side:** You work in modules/05_losses/losses_dev.py  
**Building Side:** Code exports to tinytorch.core.losses

```python
# Final package structure:
from tinytorch.core.losses import MeanSquaredError, CrossEntropyLoss, BinaryCrossEntropyLoss  # All loss functions!
from tinytorch.core.tensor import Tensor  # The foundation
from tinytorch.core.layers import Linear, Sequential  # Network components
```

**Why this matters:**
- **Learning:** Focused module for understanding loss functions and training objectives
- **Production:** Proper organization like PyTorch's torch.nn with all loss functions together
- **Consistency:** All loss functions live together in core.losses for easy access
- **Integration:** Works seamlessly with tensors and neural networks for complete training systems
"""

# %% [markdown]
"""
# Understanding Loss Functions in Neural Networks

## What are Loss Functions?

Loss functions (also called cost functions or objective functions) quantify how far your model's predictions are from the true targets. They provide:

🎯 **Learning Objectives**: Define what "good" performance means for your specific problem  
📈 **Gradient Signal**: Provide gradients that guide parameter updates during training  
🔍 **Progress Measurement**: Enable monitoring training progress and convergence  
⚖️ **Trade-off Control**: Balance different aspects of model performance (accuracy vs regularization)  

## Why Loss Functions Matter for ML Systems

### The Learning Loop
```
1. Forward Pass: Input → Network → Predictions
2. Loss Computation: Loss = loss_function(predictions, targets) 
3. Backward Pass: Compute gradients of loss w.r.t. parameters
4. Parameter Update: parameters -= learning_rate * gradients
5. Repeat until convergence
```

### Different Problems Need Different Loss Functions

🔢 **Regression Problems**: Mean Squared Error (MSE)
- Predicting continuous values (house prices, temperatures, stock prices)
- Penalizes large errors more than small errors (quadratic penalty)

🏷️ **Multi-Class Classification**: Cross-Entropy Loss
- Predicting one class from many options (image classification, text categorization)
- Works with probability distributions over classes

⚪ **Binary Classification**: Binary Cross-Entropy Loss
- Predicting yes/no, positive/negative (spam detection, medical diagnosis)
- Optimized for two-class problems

Let's implement these essential loss functions!
"""

# %% [markdown]
"""
# Mean Squared Error - Foundation for Regression

MSE measures the average squared difference between predictions and targets. It's the most fundamental loss function for regression problems.

## Why MSE Matters

📊 **Regression Standard**: The go-to loss function for predicting continuous values  
🎯 **Quadratic Penalty**: Penalizes large errors more than small errors  
📈 **Smooth Gradients**: Provides smooth gradients for stable optimization  
🔢 **Interpretable**: Loss value has same units as your target variable (squared)  

## Mathematical Foundation

For batch of predictions and targets:
```
MSE = (1/n) × Σ(y_pred - y_true)²
```

## Learning Objectives
By implementing MSE, you'll understand:
- How regression loss functions quantify prediction quality
- Why squared error creates smooth gradients for optimization
- How batch processing enables efficient training on multiple samples
- The connection between mathematical loss functions and practical ML training
"""

# %% nbgrader={"grade": false, "grade_id": "mse-loss-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MeanSquaredError:
    """
    Mean Squared Error Loss for Regression Problems
    
    Computes the average squared difference between predictions and targets:
    MSE = (1/n) × Σ(y_pred - y_true)²
    
    Features:
    - Numerically stable computation
    - Efficient batch processing
    - Clean gradient properties for optimization
    - Compatible with tensor operations
    
    Example Usage:
        mse = MeanSquaredError()
        loss = mse(predictions, targets)  # Returns scalar loss value
    """
    
    def __init__(self):
        """Initialize MSE loss function."""
        pass
    
    def __call__(self, y_pred, y_true):
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            y_pred: Model predictions (Tensor, shape: [batch_size, ...])
            y_true: True targets (Tensor, shape: [batch_size, ...])
            
        Returns:
            Tensor with scalar loss value
        """
        # Convert to tensors if needed
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Compute mean squared error
        diff = y_pred.data - y_true.data
        squared_diff = diff * diff
        mean_loss = np.mean(squared_diff)
        
        return Tensor(mean_loss)
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
## Testing Mean Squared Error

Let's verify our MSE implementation works correctly with various test cases.
"""

# %% nbgrader={"grade": true, "grade_id": "test-mse-loss", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
def test_mse_loss():
    """Test MSE loss implementation."""
    print("🧪 Testing Mean Squared Error Loss...")
    
    mse = MeanSquaredError()
    
    # Test case 1: Perfect predictions (loss should be 0)
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = mse(y_pred, y_true)
    assert abs(loss.data) < 1e-6, f"Perfect predictions should have loss ≈ 0, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test case 2: Known loss computation
    y_pred = Tensor([[1.0, 2.0]])
    y_true = Tensor([[0.0, 1.0]])
    loss = mse(y_pred, y_true)
    expected = 1.0  # [(1-0)² + (2-1)²] / 2 = [1 + 1] / 2 = 1.0
    assert abs(loss.data - expected) < 1e-6, f"Expected loss {expected}, got {loss.data}"
    print("✅ Known loss computation test passed")
    
    # Test case 3: Batch processing
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.5, 2.5], [2.5, 3.5]])
    loss = mse(y_pred, y_true)
    expected = 0.25  # All squared differences are 0.25
    assert abs(loss.data - expected) < 1e-6, f"Expected batch loss {expected}, got {loss.data}"
    print("✅ Batch processing test passed")
    
    # Test case 4: Single value
    y_pred = Tensor([5.0])
    y_true = Tensor([3.0])
    loss = mse(y_pred, y_true)
    expected = 4.0  # (5-3)² = 4
    assert abs(loss.data - expected) < 1e-6, f"Expected single value loss {expected}, got {loss.data}"
    print("✅ Single value test passed")
    
    print("🎉 All MSE loss tests passed!")

test_mse_loss()

# %% [markdown]
"""
# Cross-Entropy Loss - Foundation for Multi-Class Classification

Cross-Entropy Loss measures the difference between predicted probability distributions and true class labels. It's the standard loss function for multi-class classification problems.

## Why Cross-Entropy Matters

🎯 **Classification Standard**: The go-to loss function for multi-class problems  
📊 **Probability Interpretation**: Works naturally with softmax probability outputs  
🔄 **Information Theory**: Measures information distance between distributions  
⚖️ **Class Balance**: Handles multiple classes in a principled way  

## Mathematical Foundation

For predictions and class indices:
```
CrossEntropy = -Σ y_true × log(softmax(y_pred))
```

With softmax normalization:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

## Learning Objectives
By implementing Cross-Entropy, you'll understand:
- How classification losses work with probability distributions
- Why softmax normalization is essential for multi-class problems
- The importance of numerical stability in log computations
- How cross-entropy encourages confident, correct predictions
"""

# %% nbgrader={"grade": false, "grade_id": "crossentropy-loss-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class CrossEntropyLoss:
    """
    Cross-Entropy Loss for Multi-Class Classification Problems
    
    Computes the cross-entropy between predicted probability distributions
    and true class labels with numerically stable implementation.
    
    Features:
    - Numerically stable softmax computation
    - Support for both class indices and one-hot encoding
    - Efficient batch processing
    - Automatic handling of edge cases
    
    Example Usage:
        ce_loss = CrossEntropyLoss()
        loss = ce_loss(logits, class_indices)  # Returns scalar loss value
    """
    
    def __init__(self):
        """Initialize CrossEntropy loss function."""
        pass
    
    def __call__(self, y_pred, y_true):
        """
        Compute CrossEntropy loss between predictions and targets.
        
        Args:
            y_pred: Model predictions/logits (Tensor, shape: [batch_size, num_classes])
            y_true: True class indices (Tensor, shape: [batch_size]) or one-hot encoding
            
        Returns:
            Tensor with scalar loss value
        """
        # Convert to tensors if needed
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Get data arrays
        pred_data = y_pred.data
        true_data = y_true.data
        
        # Handle both 1D and 2D prediction arrays
        if pred_data.ndim == 1:
            pred_data = pred_data.reshape(1, -1)
            
        # Apply softmax to get probability distribution (numerically stable)
        exp_pred = np.exp(pred_data - np.max(pred_data, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        softmax_pred = np.clip(softmax_pred, epsilon, 1.0 - epsilon)
        
        # Handle class indices vs one-hot encoding
        if len(true_data.shape) == 1:
            # y_true contains class indices
            batch_size = true_data.shape[0]
            log_probs = np.log(softmax_pred[np.arange(batch_size), true_data.astype(int)])
            loss_value = -np.mean(log_probs)
        else:
            # y_true is one-hot encoded
            log_probs = np.log(softmax_pred)
            loss_value = -np.mean(np.sum(true_data * log_probs, axis=1))
        
        return Tensor(loss_value)
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
## Testing Cross-Entropy Loss

Let's verify our Cross-Entropy implementation handles various classification scenarios.
"""

# %% nbgrader={"grade": true, "grade_id": "test-crossentropy-loss", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
def test_crossentropy_loss():
    """Test CrossEntropy loss implementation."""
    print("🧪 Testing Cross-Entropy Loss...")
    
    ce = CrossEntropyLoss()
    
    # Test case 1: Perfect predictions
    y_pred = Tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])  # Very confident correct predictions
    y_true = Tensor([0, 1])  # Class indices
    loss = ce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test case 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Uniform after softmax
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    expected_random = -np.log(1.0/3.0)  # log(1/num_classes) for uniform distribution
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ≈ {expected_random}, got {loss.data}"
    print("✅ Random predictions test passed")
    
    # Test case 3: Binary classification
    y_pred = Tensor([[2.0, 1.0], [1.0, 2.0]])
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Binary classification loss should be reasonable, got {loss.data}"
    print("✅ Binary classification test passed")
    
    # Test case 4: One-hot encoded labels
    y_pred = Tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y_true = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # One-hot encoded
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"One-hot encoded loss should be reasonable, got {loss.data}"
    print("✅ One-hot encoded labels test passed")
    
    print("🎉 All Cross-Entropy loss tests passed!")

test_crossentropy_loss()

# %% [markdown]
"""
# Binary Cross-Entropy Loss - Optimized for Binary Classification

Binary Cross-Entropy Loss is specifically designed for binary classification problems. While you could use regular Cross-Entropy with 2 classes, BCE is more efficient and numerically stable for binary problems.

## Why Binary Cross-Entropy Matters

⚪ **Binary Optimization**: Specifically designed for two-class problems  
🔢 **Efficiency**: More efficient than multi-class cross-entropy for binary cases  
🎯 **Stability**: Better numerical stability with sigmoid outputs  
📈 **Standard Practice**: Industry standard for binary classification  

## Mathematical Foundation

For binary predictions and labels:
```
BCE = -y_true × log(σ(y_pred)) - (1-y_true) × log(1-σ(y_pred))
```

Where σ(x) is the sigmoid function:
```
σ(x) = 1 / (1 + exp(-x))
```

## Learning Objectives
By implementing Binary Cross-Entropy, you'll understand:
- How binary classification differs from multi-class problems
- Why sigmoid activation is natural for binary problems
- The importance of numerical stability in sigmoid + log computations
- How BCE loss shapes binary decision boundaries
"""

# %% nbgrader={"grade": false, "grade_id": "binary-crossentropy-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss for Binary Classification Problems
    
    Computes binary cross-entropy between predictions and binary labels
    with numerically stable sigmoid + BCE implementation.
    
    Features:
    - Numerically stable computation from logits
    - Efficient batch processing
    - Automatic sigmoid application
    - Robust to extreme input values
    
    Example Usage:
        bce_loss = BinaryCrossEntropyLoss()
        loss = bce_loss(logits, binary_labels)  # Returns scalar loss value
    """
    
    def __init__(self):
        """Initialize Binary CrossEntropy loss function."""
        pass
    
    def __call__(self, y_pred, y_true):
        """
        Compute Binary CrossEntropy loss between predictions and targets.
        
        Args:
            y_pred: Model predictions/logits (Tensor, shape: [batch_size, 1] or [batch_size])
            y_true: True binary labels (Tensor, shape: [batch_size, 1] or [batch_size])
            
        Returns:
            Tensor with scalar loss value
        """
        # Convert to tensors if needed
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Get flat arrays for computation
        logits = y_pred.data.flatten()
        labels = y_true.data.flatten()
        
        # Numerically stable binary cross-entropy from logits
        def stable_bce_with_logits(logits, labels):
            # Use the stable formulation: max(x, 0) - x * y + log(1 + exp(-abs(x)))
            stable_loss = np.maximum(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
            return stable_loss
        
        # Compute loss for each sample
        losses = stable_bce_with_logits(logits, labels)
        mean_loss = np.mean(losses)
        
        return Tensor(mean_loss)
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
## Testing Binary Cross-Entropy Loss

Let's verify our Binary Cross-Entropy implementation handles binary classification correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-binary-crossentropy", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
def test_binary_crossentropy_loss():
    """Test Binary CrossEntropy loss implementation."""
    print("🧪 Testing Binary Cross-Entropy Loss...")
    
    bce = BinaryCrossEntropyLoss()
    
    # Test case 1: Perfect predictions
    y_pred = Tensor([[10.0], [-10.0]])  # Very confident correct predictions
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test case 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0], [0.0]])  # 0.5 probability after sigmoid
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    expected_random = -np.log(0.5)  # log(0.5) for random guessing
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ≈ {expected_random}, got {loss.data}"
    print("✅ Random predictions test passed")
    
    # Test case 3: Batch processing
    y_pred = Tensor([[1.0], [2.0], [-1.0]])
    y_true = Tensor([[1.0], [1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Batch processing loss should be reasonable, got {loss.data}"
    print("✅ Batch processing test passed")
    
    # Test case 4: Extreme values (test numerical stability)
    y_pred = Tensor([[100.0], [-100.0]])  # Extreme logits
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert not np.isnan(loss.data) and not np.isinf(loss.data), f"Extreme values should not cause NaN/Inf, got {loss.data}"
    assert loss.data < 1.0, f"Extreme correct predictions should have low loss, got {loss.data}"
    print("✅ Extreme values test passed")
    
    print("🎉 All Binary Cross-Entropy loss tests passed!")

test_binary_crossentropy_loss()

# %% [markdown]
"""
# Loss Function Comparison and Usage Guide

## When to Use Each Loss Function

### Mean Squared Error (MSE)
**Best for:** Regression problems where you predict continuous values
- **Examples:** Predicting house prices, temperature, stock values, ages
- **Characteristics:** Penalizes large errors more than small ones
- **Output:** Any real number
- **Activation:** Usually none (linear output)

### Cross-Entropy Loss
**Best for:** Multi-class classification (3+ classes)
- **Examples:** Image classification (cats/dogs/birds), text categorization, medical diagnosis
- **Characteristics:** Works with probability distributions over classes
- **Output:** Class probabilities (sums to 1)
- **Activation:** Softmax

### Binary Cross-Entropy Loss
**Best for:** Binary classification (2 classes)
- **Examples:** Spam detection, medical positive/negative, fraud detection
- **Characteristics:** Optimized for binary decisions
- **Output:** Single probability (0 to 1)
- **Activation:** Sigmoid

## Numerical Stability Considerations

All our implementations include numerical stability features:

🔢 **MSE**: Straightforward - no special numerical concerns  
📊 **Cross-Entropy**: Uses log-sum-exp trick, clips probabilities, handles edge cases  
⚪ **Binary CE**: Uses stable logits formulation, clips extreme values  

## Integration with Neural Networks

```python
# Example usage in training loop
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
    Softmax()
])

# Choose appropriate loss for your problem
loss_fn = CrossEntropyLoss()  # For 10-class classification

# In training loop
predictions = model(inputs)
loss = loss_fn(predictions, targets)
# loss.backward()  # Would trigger gradient computation (when autograd is integrated)
```
"""

# %% [markdown]
"""
# Comprehensive Loss Function Testing

Let's verify all our loss functions work correctly together and can be used interchangeably.
"""

# %% nbgrader={"grade": false, "grade_id": "comprehensive-loss-tests", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_all_loss_functions():
    """Test all loss functions work correctly together."""
    print("🔬 Comprehensive Loss Function Testing")
    print("=" * 45)
    
    # Test 1: All losses can be instantiated
    print("\\n1. Loss Function Instantiation:")
    mse = MeanSquaredError()
    ce = CrossEntropyLoss()
    bce = BinaryCrossEntropyLoss()
    print("   ✅ All loss functions created successfully")
    
    # Test 2: Loss functions return appropriate types
    print("\\n2. Return Type Verification:")
    
    # MSE test
    pred = Tensor([[1.0, 2.0]])
    target = Tensor([[1.0, 2.0]])
    loss = mse(pred, target)
    assert isinstance(loss, Tensor), "MSE should return Tensor"
    assert loss.data.shape == (), "MSE should return scalar"
    
    # Cross-entropy test
    pred = Tensor([[1.0, 2.0], [2.0, 1.0]])
    target = Tensor([1, 0])
    loss = ce(pred, target)
    assert isinstance(loss, Tensor), "CrossEntropy should return Tensor"
    assert loss.data.shape == (), "CrossEntropy should return scalar"
    
    # Binary cross-entropy test
    pred = Tensor([[1.0], [-1.0]])
    target = Tensor([[1.0], [0.0]])
    loss = bce(pred, target)
    assert isinstance(loss, Tensor), "Binary CrossEntropy should return Tensor"
    assert loss.data.shape == (), "Binary CrossEntropy should return scalar"
    
    print("   ✅ All loss functions return correct types")
    
    # Test 3: Loss values are reasonable
    print("\\n3. Loss Value Sanity Checks:")
    
    # All losses should be non-negative
    assert mse.forward(Tensor([1.0]), Tensor([2.0])).data >= 0, "MSE should be non-negative"
    assert ce.forward(Tensor([[1.0, 0.0]]), Tensor([0])).data >= 0, "CrossEntropy should be non-negative"
    assert bce.forward(Tensor([1.0]), Tensor([1.0])).data >= 0, "Binary CrossEntropy should be non-negative"
    
    print("   ✅ All loss functions produce reasonable values")
    
    # Test 4: Perfect predictions give low loss
    print("\\n4. Perfect Prediction Tests:")
    
    perfect_mse = mse(Tensor([5.0]), Tensor([5.0]))
    perfect_ce = ce(Tensor([[10.0, 0.0]]), Tensor([0]))
    perfect_bce = bce(Tensor([10.0]), Tensor([1.0]))
    
    assert perfect_mse.data < 1e-10, f"Perfect MSE should be ~0, got {perfect_mse.data}"
    assert perfect_ce.data < 0.1, f"Perfect CE should be low, got {perfect_ce.data}"
    assert perfect_bce.data < 0.1, f"Perfect BCE should be low, got {perfect_bce.data}"
    
    print("   ✅ Perfect predictions produce low loss")
    
    print("\\n🎉 All comprehensive tests passed!")
    print("   • Loss functions instantiate correctly")
    print("   • Return types are consistent (Tensor scalars)")
    print("   • Loss values are mathematically sound")
    print("   • Perfect predictions are handled correctly")
    print("   • Ready for integration with neural network training!")

test_all_loss_functions()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've implemented all the core loss functions, let's think about their implications for ML systems:
"""

# %% nbgrader={"grade": false, "grade_id": "question-1", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 1: Loss Function Selection and System Performance
"""
🤔 **Question 1: Loss Function Selection Impact**

You're building a production recommendation system that needs to predict user ratings (1-5 stars) for movies.

You have three options:
A) Treat as regression: Use MSE loss with continuous outputs (1.0-5.0)
B) Treat as classification: Use CrossEntropy loss with 5 classes
C) Use a custom loss that penalizes being off by multiple stars more heavily

Analyze each approach considering:
- Training speed and convergence behavior
- Model interpretability and debugging
- Production inference speed
- How well each loss function matches the business objective
- Edge case handling (what happens with ratings like 3.7?)

Which approach would you choose and why? Consider both technical and business factors.
"""

# %% nbgrader={"grade": false, "grade_id": "question-2", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 2: Numerical Stability in Production
"""
🤔 **Question 2: Numerical Stability Analysis**

Your cross-entropy loss function works perfectly in development, but in production you start seeing NaN losses that crash training.

Investigate the numerical stability issues:
1. What specific computations in cross-entropy can produce NaN or infinity values?
2. How do our implementations handle these edge cases?
3. What would happen if you removed the epsilon clipping in softmax computation?
4. How would you debug this in a production system with millions of training examples?

Research areas to consider:
- Floating point precision and representation limits
- Log of very small numbers and exp of very large numbers  
- Batch processing effects on numerical stability
- How PyTorch handles these same numerical challenges
"""

# %% nbgrader={"grade": false, "grade_id": "question-3", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Question 3: Loss Function Innovation
"""
🤔 **Question 3: Custom Loss Functions for Real Problems**

Standard loss functions don't always match real-world objectives. Consider these scenarios:

**Scenario A**: Medical diagnosis where false negatives are 10x more costly than false positives
**Scenario B**: Search ranking where being wrong about the top result is much worse than being wrong about result #50
**Scenario C**: Financial trading where large losses should be penalized exponentially more than small losses

For each scenario:
1. Why would standard loss functions (MSE, CrossEntropy, BCE) be suboptimal?
2. How would you modify the loss function to better match the business objective?
3. What are the implementation challenges of custom loss functions?
4. How would you validate that your custom loss actually improves business outcomes?

Design principles to consider:
- Asymmetric penalties for different types of errors
- Position-aware losses for ranking problems
- Risk-adjusted losses for financial applications
- How custom losses affect gradient flow and training dynamics
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Loss Functions - Learning Objectives Made Concrete

## 🎯 What You've Accomplished

You've successfully implemented the complete foundation for neural network training objectives:

### ✅ **Complete Loss Function Library**
- **Mean Squared Error**: Robust regression loss with smooth gradients for continuous value prediction
- **Cross-Entropy Loss**: Multi-class classification loss with numerically stable softmax integration
- **Binary Cross-Entropy Loss**: Optimized binary classification loss with stable sigmoid computation
- **Numerical Stability**: All implementations handle edge cases and extreme values gracefully

### ✅ **Systems Understanding**
- **Training Objectives**: How loss functions translate business problems into mathematical optimization objectives
- **Numerical Stability**: Why proper implementation prevents catastrophic training failures in production
- **Performance Characteristics**: Understanding computational complexity and batch processing efficiency
- **Problem Matching**: When to use each loss function based on problem structure and data characteristics

### ✅ **ML Engineering Skills**
- **Production-Ready Implementation**: Robust loss functions that handle real-world data edge cases
- **Batch Processing**: Efficient computation across multiple samples for scalable training
- **Error Handling**: Proper numerical stability measures for reliable production deployment
- **Integration Ready**: Clean interfaces that work seamlessly with neural network training loops

## 🔗 **Connection to Production ML Systems**

Your implementations mirror the essential patterns used in:
- **PyTorch's loss functions**: Same mathematical formulations with production-grade numerical stability
- **TensorFlow's losses**: Identical computational patterns and stability measures
- **Production ML pipelines**: The same loss functions that power real ML systems at scale
- **Research frameworks**: Foundation for experimenting with custom loss functions and training objectives

## 🚀 **What's Next**

With solid loss function implementations, you're ready to:
- **Build complete training loops** that optimize neural networks on real data
- **Implement optimizers** that use loss gradients to update model parameters
- **Create training infrastructure** with proper monitoring and convergence detection
- **Experiment with custom losses** for specialized business objectives and research problems

## 💡 **Key Systems Insights**

1. **Loss functions are the interface between business objectives and mathematical optimization** - they translate "what we want" into "what the computer can optimize"
2. **Numerical stability is not optional in production** - unstable loss computation causes catastrophic training failures
3. **Different problem types require different loss functions** - the choice affects both convergence speed and final model behavior
4. **Batch processing efficiency determines training speed** - loss computation must scale to handle large datasets efficiently

You now understand how to implement the mathematical foundation that enables neural networks to learn from data and solve real-world problems!
"""

# %% nbgrader={"grade": false, "grade_id": "final-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    print("🔥 TinyTorch Loss Functions Module - Complete Demo")
    print("=" * 55)
    
    # Test all core implementations
    print("\\n🧪 Testing All Loss Functions:")
    test_mse_loss()
    test_crossentropy_loss()
    test_binary_crossentropy_loss()
    test_all_loss_functions()
    
    print("\\n" + "="*60)
    print("📊 Loss Function Usage Examples")
    print("=" * 35)
    
    # Example 1: Regression with MSE
    print("\\n1. Regression Example (Predicting House Prices):")
    mse = MeanSquaredError()
    house_predictions = Tensor([[250000, 180000, 320000]])  # Predicted prices
    house_actual = Tensor([[240000, 175000, 315000]])       # Actual prices
    regression_loss = mse(house_predictions, house_actual)
    print(f"   House price prediction loss: ${regression_loss.data:,.0f}² average error")
    
    # Example 2: Multi-class classification with CrossEntropy
    print("\\n2. Multi-Class Classification Example (Image Recognition):")
    ce = CrossEntropyLoss()
    image_logits = Tensor([[2.1, 0.5, -0.3, 1.8, 0.1],      # Model outputs for 5 classes
                          [-0.2, 3.1, 0.8, -1.0, 0.4]])      # (cat, dog, bird, fish, rabbit)
    true_classes = Tensor([0, 1])  # First image = cat, second = dog
    classification_loss = ce(image_logits, true_classes)
    print(f"   Image classification loss: {classification_loss.data:.4f}")
    
    # Example 3: Binary classification with BCE
    print("\\n3. Binary Classification Example (Spam Detection):")
    bce = BinaryCrossEntropyLoss()
    spam_logits = Tensor([[1.2], [-0.8], [2.1], [-1.5]])  # Spam prediction logits
    spam_labels = Tensor([[1.0], [0.0], [1.0], [0.0]])     # 1=spam, 0=not spam
    spam_loss = bce(spam_logits, spam_labels)
    print(f"   Spam detection loss: {spam_loss.data:.4f}")
    
    print("\\n" + "="*60)
    print("🎯 Loss Function Characteristics")
    print("=" * 35)
    
    # Compare perfect vs imperfect predictions
    print("\\n📊 Perfect vs Random Predictions:")
    
    # Perfect predictions
    perfect_mse = mse(Tensor([5.0]), Tensor([5.0]))
    perfect_ce = ce(Tensor([[10.0, 0.0, 0.0]]), Tensor([0]))
    perfect_bce = bce(Tensor([10.0]), Tensor([1.0]))
    
    print(f"   Perfect MSE loss: {perfect_mse.data:.6f}")
    print(f"   Perfect CE loss:  {perfect_ce.data:.6f}")
    print(f"   Perfect BCE loss: {perfect_bce.data:.6f}")
    
    # Random predictions
    random_mse = mse(Tensor([3.0]), Tensor([5.0]))  # Off by 2
    random_ce = ce(Tensor([[0.0, 0.0, 0.0]]), Tensor([0]))  # Uniform distribution
    random_bce = bce(Tensor([0.0]), Tensor([1.0]))  # 50% confidence
    
    print(f"   Random MSE loss:  {random_mse.data:.6f}")
    print(f"   Random CE loss:   {random_ce.data:.6f}")
    print(f"   Random BCE loss:  {random_bce.data:.6f}")
    
    print("\\n🎉 Complete loss function foundation ready!")
    print("   ✅ MSE for regression problems")
    print("   ✅ CrossEntropy for multi-class classification")
    print("   ✅ Binary CrossEntropy for binary classification")
    print("   ✅ Numerically stable implementations")
    print("   ✅ Production-ready batch processing")
    print("   ✅ Ready for neural network training!")