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
# Loss Functions - Learning Objectives Made Mathematical

Welcome to Loss Functions! You'll implement the critical bridge between model predictions and learning objectives that makes neural network training possible.

## LINK Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): Data structures for predictions and targets
- Module 03 (Activations): Nonlinear transformations for model outputs
- Module 04 (Layers): Complete neural network layers that produce predictions

**What's Working**: You can build networks that transform inputs into predictions!

**The Gap**: Predictions aren't learning objectives - you need to measure how "wrong" predictions are and provide gradient signals for improvement.

**This Module's Solution**: Implement MSE, CrossEntropy, and BinaryCrossEntropy loss functions with numerical stability.

**Connection Map**:
```
Layers -> Loss Functions -> Gradients
(predictions)  (objectives)   (learning signals)
```

## Learning Objectives

By completing this module, you will:

1. **Understand loss functions** - Learn how to measure the quality of model predictions
2. **Implement MSE Loss** - Build loss functions for regression problems
3. **Implement CrossEntropy Loss** - Create loss functions for classification tasks
4. **Handle numerical stability** - Deal with edge cases and extreme values safely
5. **Enable learning** - Provide the feedback signal that allows networks to improve

## Build -> Use -> Reflect
1. **Build**: MSE, CrossEntropy, and BinaryCrossEntropy loss functions with proper error handling
2. **Use**: Apply different loss functions to real prediction problems and compare results
3. **Reflect**: Understand when to use each loss function and why numerical stability matters

## What You'll Achieve
- **Mathematical understanding**: How loss functions quantify prediction quality
- **Implementation skills**: Building robust loss functions with error handling
- **Problem matching**: Choosing the right loss function for different ML tasks
- **Numerical awareness**: Understanding and preventing common computational issues
- **Training foundation**: Enabling the learning process that makes neural networks work
"""

# %% nbgrader={"grade": false, "grade_id": "losses-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.losses

#| export
import numpy as np
import sys
import os

# Import our building blocks - Tensor first, autograd operations if available
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# Pure tensor evolution approach:
# - Loss functions use basic Tensor operations directly
# - Module 05 will add gradient tracking via decorator pattern
# - Clean separation of concerns enables focused learning

# %% nbgrader={"grade": false, "grade_id": "losses-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("FIRE TinyTorch Loss Functions Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build loss functions for neural network training!")

# %% [markdown]
"""
## Where This Code Lives in the Final Package

**Learning Side:** You work in modules/04_losses/losses_dev.py  
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

Loss functions are the mathematical bridge between what your model predicts and what you want it to learn. They quantify the "distance" between predictions and reality.

```
Business Goal: "Predict house prices accurately"
            v
Mathematical Loss: MSE = (predicted_price - actual_price)²
            v  
Optimization Signal: gradient = 2 * (predicted - actual)
            v
Learning Update: parameter -= learning_rate * gradient
```

## The Learning Ecosystem

Loss functions provide four critical capabilities:

TARGET **Learning Objectives**: Define what "good" performance means mathematically  
PROGRESS **Gradient Signal**: Provide directional improvement information for parameters  
MAGNIFY **Progress Measurement**: Enable monitoring training progress and convergence detection  
⚖️ **Trade-off Control**: Balance different aspects of model performance and regularization  

## Visual Understanding: Loss Function Landscape

```
Loss Function Behavior:
           MSE Loss                    CrossEntropy Loss
    High |    /\\                High |     /\\
         |   /  \\                    |    /  \\
         |  /    \\                   |   /    \\
         | /      \\                  |  /      \\
     Low |/        \\             Low | /        \\
         +--------------         +--------------
         Wrong  Right              Wrong  Right
         
   • Smooth gradients          • Steep near wrong predictions
   • Quadratic penalty         • Gentle near correct predictions
   • Good for regression       • Good for classification
```

Different loss functions create different optimization landscapes that affect how your model learns!
"""

# %% [markdown]
"""
# Mean Squared Error - Foundation for Regression

MSE is the cornerstone loss function for regression problems. It measures prediction quality by penalizing large errors more than small ones.

## Visual Understanding: MSE Behavior

```
MSE Loss Visualization:

    Loss |     /\\
       4 |    /  \\        • Error = 2 -> Loss = 4
       3 |   /    \\       • Error = 1 -> Loss = 1
       2 |  /      \\      • Error = 0 -> Loss = 0
       1 | /        \\     • Quadratic penalty!
       0 |/__________\\____
         -2  -1   0   1   2
              Error
              
Gradient Flow:
    dLoss/dprediction = 2 * (predicted - actual)
    
    Large errors -> Large gradients -> Big updates
    Small errors -> Small gradients -> Fine tuning
```

## Mathematical Foundation

For batch of predictions and targets:
```
MSE = (1/n) * Sum(y_pred - y_true)²

Gradient: dMSE/dy_pred = (2/n) * (y_pred - y_true)
```

## Learning Objectives
By implementing MSE, you'll understand:
- How regression loss functions translate continuous prediction errors into optimization signals
- Why squared error creates smooth, well-behaved gradients for stable optimization
- How batch processing enables efficient training on multiple samples simultaneously
- The connection between mathematical loss formulations and practical ML training dynamics
"""

# %% nbgrader={"grade": false, "grade_id": "mse-concept-question", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Computational Question: MSE Properties**

Before implementing, let's understand MSE behavior:

1. If you predict house price as $300k but actual is $250k, what's the MSE?
2. If you predict $310k but actual is $250k, what's the MSE? 
3. Which error gets penalized more heavily and why?
4. How does this relate to the quadratic penalty we visualized?

This understanding will guide your implementation approach.
"""

# %% nbgrader={"grade": false, "grade_id": "mse-loss-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MeanSquaredError:
    """
    Mean Squared Error Loss for Regression Problems
    
    Computes the average squared difference between predictions and targets:
    MSE = (1/n) * Sum(y_pred - y_true)²
    
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
            
        TODO: Implement MSE computation with proper tensor handling.
        
        APPROACH:
        1. Convert inputs to tensors for consistent processing
        2. Compute element-wise prediction errors (differences)
        3. Square the errors to create quadratic penalty
        4. Take mean across all elements for final loss
        
        EXAMPLE:
        >>> mse = MeanSquaredError()
        >>> pred = Tensor([[1.0, 2.0]])
        >>> true = Tensor([[1.5, 1.5]])
        >>> loss = mse(pred, true)
        >>> print(loss.data)
        0.25  # [(1.0-1.5)² + (2.0-1.5)²] / 2 = [0.25 + 0.25] / 2
        
        HINTS:
        - Use np.mean() for efficient batch averaging
        - Element-wise operations work naturally with tensor.data
        - Return result wrapped in Tensor for consistent interface
        """
        ### BEGIN SOLUTION
        # Step 1: Ensure we have tensor inputs for consistent processing
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # Step 2: Compute mean squared error with element-wise operations
        prediction_errors = y_pred.data - y_true.data  # Element-wise difference
        squared_errors = prediction_errors * prediction_errors  # Element-wise squaring
        mean_loss = np.mean(squared_errors)  # Average across all elements
        
        return Tensor(mean_loss)
        ### END SOLUTION
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# MAGNIFY SYSTEMS INSIGHT: Gradient Landscape Visualization
def visualize_loss_landscapes():
    """Visualize how different loss functions create different optimization landscapes."""
    print("MAGNIFY Loss Function Landscape Visualization")
    print("=" * 45)

    try:
        import numpy as np

        # Create prediction space for visualization
        prediction_range = np.linspace(-3, 3, 100)
        true_value = 0.0  # Target value

        print("\nPROGRESS Loss Landscape Comparison:")
        print("   How loss changes as predictions move away from target")

        # Calculate loss landscapes
        mse = MeanSquaredError()
        _ = CrossEntropyLoss()  # Not used in this comparison
        bce = BinaryCrossEntropyLoss()

        # MSE landscape (regression)
        mse_losses = []
        for pred in prediction_range:
            loss = mse(Tensor([pred]), Tensor([true_value]))
            mse_losses.append(loss.data)

        # Binary CE landscape (classification)
        bce_losses = []
        for pred in prediction_range:
            loss = bce(Tensor([pred]), Tensor([1.0]))  # Target: positive class
            bce_losses.append(loss.data)

        # Find key gradient characteristics
        mse_gradient_at_zero = 2 * (0 - true_value)  # MSE gradient formula
        mse_gradient_at_one = 2 * (1 - true_value)

        print(f"\nTARGET Gradient Behavior Analysis:")
        print(f"   MSE gradient at prediction=0: {mse_gradient_at_zero:.3f}")
        print(f"   MSE gradient at prediction=1: {mse_gradient_at_one:.3f}")
        print(f"   MSE provides linear gradient growth")

        # Binary CE gradient analysis
        sigmoid_at_zero = 1 / (1 + np.exp(-0))  # = 0.5
        bce_grad_at_zero = sigmoid_at_zero - 1.0  # = -0.5
        sigmoid_at_one = 1 / (1 + np.exp(-1))    # ~= 0.73
        bce_grad_at_one = sigmoid_at_one - 1.0   # ~= -0.27

        print(f"   BCE gradient at logit=0: {bce_grad_at_zero:.3f}")
        print(f"   BCE gradient at logit=1: {bce_grad_at_one:.3f}")
        print(f"   BCE provides adaptive gradient magnitude")

        # Visualize ASCII loss curves
        print(f"\n📊 Loss Function Shapes (ASCII visualization):")
        print(f"   Prediction range: {prediction_range[0]:.1f} to {prediction_range[-1]:.1f}")

        # Sample key points for visualization
        sample_points = [-2, -1, 0, 1, 2]
        print(f"\n   {'Prediction':>10} {'MSE Loss':>10} {'BCE Loss':>10} {'Gradient Type':>15}")
        print(f"   {'-'*10} {'-'*10} {'-'*10} {'-'*15}")

        for point in sample_points:
            mse_loss = mse(Tensor([point]), Tensor([0.0]))
            bce_loss = bce(Tensor([point]), Tensor([1.0]))

            # Characterize gradient steepness
            if abs(point) < 0.5:
                grad_type = "Gentle"
            elif abs(point) < 1.5:
                grad_type = "Moderate"
            else:
                grad_type = "Steep"

            print(f"   {point:>10.1f} {mse_loss.data:>10.3f} {bce_loss.data:>10.3f} {grad_type:>15}")

        # Optimization implications
        print(f"\nROCKET Optimization Implications:")
        print(f"   MSE (Regression):")
        print(f"     • Quadratic penalty grows smoothly")
        print(f"     • Large errors -> large gradients (aggressive correction)")
        print(f"     • Small errors -> small gradients (fine-tuning)")
        print(f"     • Symmetric around target value")

        print(f"   Binary CrossEntropy (Classification):")
        print(f"     • Logarithmic penalty creates adaptive gradients")
        print(f"     • Wrong confident predictions -> steep gradients")
        print(f"     • Right confident predictions -> gentle gradients")
        print(f"     • Asymmetric penalty structure encourages confidence")

        # TIP WHY THIS MATTERS: Different loss landscapes create different
        # optimization dynamics. MSE's smooth quadratic surface enables
        # stable gradient descent, while CrossEntropy's adaptive gradients
        # help classification models learn faster from confident mistakes.

    except Exception as e:
        print(f"WARNING️ Visualization error: {e}")
        print("Ensure loss functions are implemented for landscape analysis")

# MAGNIFY SYSTEMS INSIGHT: MSE Computational Analysis
def analyze_mse_properties():
    """Analyze MSE loss characteristics for systems understanding."""
    print("MAGNIFY MSE Loss Analysis - Understanding the Math")
    print("=" * 45)
    
    try:
        mse = MeanSquaredError()
        
        # Error magnitude vs loss relationship
        print("\n📊 Error Magnitude vs Loss (Quadratic Penalty):")
        errors = [0.1, 0.5, 1.0, 2.0, 5.0]
        for error in errors:
            pred = Tensor([error])
            true = Tensor([0.0])
            loss = mse(pred, true)
            print(f"   Error: {error:4.1f} -> Loss: {loss.data:8.3f} (* {loss.data/(error**2):5.1f} baseline)")
        
        # Batch vs individual processing
        print(f"\nSPEED Batch Processing Efficiency:")
        single_losses = []
        for _ in range(100):
            pred = Tensor([np.random.randn()])
            true = Tensor([np.random.randn()])
            loss = mse(pred, true)
            single_losses.append(loss.data)
        
        # Batch version
        batch_pred = Tensor(np.random.randn(100))
        batch_true = Tensor(np.random.randn(100))
        batch_loss = mse(batch_pred, batch_true)
        
        individual_mean = np.mean(single_losses)
        print(f"   Individual losses mean: {individual_mean:.6f}")
        print(f"   Batch loss:            {batch_loss.data:.6f}")
        print(f"   Difference:            {abs(individual_mean - batch_loss.data):.8f}")
        
        # Memory efficiency analysis
        import sys
        small_tensor = Tensor([1.0])
        large_tensor = Tensor(np.random.randn(1000))
        
        print(f"\n💾 Memory Efficiency:")
        print(f"   Small loss memory: {sys.getsizeof(small_tensor.data)} bytes")
        print(f"   Large loss memory: {sys.getsizeof(large_tensor.data)} bytes")
        print(f"   MSE memory is independent of input size!")
        
        # TIP WHY THIS MATTERS: MSE provides stable, well-behaved gradients
        # that are proportional to error magnitude, making optimization smooth.
        # The quadratic penalty means large errors dominate learning initially,
        # then fine-tuning happens as errors get smaller.
        
    except Exception as e:
        print(f"WARNING️ Analysis error: {e}")
        print("Ensure MSE implementation is complete before running analysis")

# %% [markdown]
"""
### TEST Unit Test: MSE Loss Computation
This test validates `MeanSquaredError.__call__`, ensuring correct MSE computation with various input types and batch sizes.
"""

# %% nbgrader={"grade": true, "grade_id": "test-mse-loss", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
def test_unit_mse_loss():
    """Test MSE loss implementation."""
    print("TEST Testing Mean Squared Error Loss...")
    
    mse = MeanSquaredError()
    
    # Test case 1: Perfect predictions (loss should be 0)
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = mse(y_pred, y_true)
    assert abs(loss.data) < 1e-6, f"Perfect predictions should have loss ~= 0, got {loss.data}"
    print("PASS Perfect predictions test passed")
    
    # Test case 2: Known loss computation
    y_pred = Tensor([[1.0, 2.0]])
    y_true = Tensor([[0.0, 1.0]])
    loss = mse(y_pred, y_true)
    expected = 1.0  # [(1-0)² + (2-1)²] / 2 = [1 + 1] / 2 = 1.0
    assert abs(loss.data - expected) < 1e-6, f"Expected loss {expected}, got {loss.data}"
    print("PASS Known loss computation test passed")
    
    # Test case 3: Batch processing
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.5, 2.5], [2.5, 3.5]])
    loss = mse(y_pred, y_true)
    expected = 0.25  # All squared differences are 0.25
    assert abs(loss.data - expected) < 1e-6, f"Expected batch loss {expected}, got {loss.data}"
    print("PASS Batch processing test passed")
    
    # Test case 4: Single value
    y_pred = Tensor([5.0])
    y_true = Tensor([3.0])
    loss = mse(y_pred, y_true)
    expected = 4.0  # (5-3)² = 4
    assert abs(loss.data - expected) < 1e-6, f"Expected single value loss {expected}, got {loss.data}"
    print("PASS Single value test passed")
    
    print("CELEBRATE MSE loss tests passed! Understanding regression objectives.")

test_unit_mse_loss()

# %% [markdown]
"""
# Cross-Entropy Loss - Foundation for Multi-Class Classification

Cross-Entropy Loss measures the "information distance" between predicted probability distributions and true class labels. It's the gold standard for classification problems.

## Visual Understanding: Cross-Entropy Behavior

```
Cross-Entropy Loss for 3-Class Problem:

Class Probabilities after Softmax:
    Input: [2.0, 1.0, 0.1]    ->    Probabilities: [0.66, 0.24, 0.10]
    True:  Class 0 (index 0)   ->    Target:       [1.0,  0.0,  0.0]
    
Loss Computation:
    CE = -log(probability_of_correct_class)
    CE = -log(0.66) = 0.415
    
Intuition:
    High confidence + Correct -> Low loss
    High confidence + Wrong   -> High loss  
    Low confidence  + Any     -> Medium loss

Gradient Behavior:
    Wrong predictions -> Steep gradients -> Big corrections
    Right predictions -> Gentle gradients -> Fine tuning
```

## Numerical Stability Challenge

```
The Numerical Stability Problem:
    
    Raw logits: [50.0, 49.0, 48.0]
    Naive softmax: exp(50)/[exp(50)+exp(49)+exp(48)]
    Problem: exp(50) ~= 5*10²¹ -> Overflow!
    
Our Solution (Log-Sum-Exp Trick):
    1. max_val = max(logits) = 50.0
    2. stable_logits = [0.0, -1.0, -2.0]  # Subtract max
    3. exp([0.0, -1.0, -2.0]) = [1.0, 0.37, 0.14]
    4. Safe softmax: [0.67, 0.25, 0.09]
```

## Mathematical Foundation

For predictions and class indices:
```
CrossEntropy = -Sum y_true * log(softmax(y_pred))

Softmax: softmax(x_i) = exp(x_i) / Sum exp(x_j)
Stable: softmax(x_i) = exp(x_i - max(x)) / Sum exp(x_j - max(x))
```

## Learning Objectives
By implementing Cross-Entropy, you'll understand:
- How classification losses work with probability distributions and information theory
- Why softmax normalization creates proper probability distributions for multi-class problems
- The critical importance of numerical stability in exponential and logarithmic computations
- How cross-entropy naturally encourages confident, correct predictions through its gradient structure
"""

# %% nbgrader={"grade": false, "grade_id": "crossentropy-concept-question", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Computational Question: CrossEntropy Stability**

Consider numerical stability in cross-entropy:

1. What happens if you compute exp(100) directly?
2. Why does subtracting the maximum value prevent overflow?
3. What happens if log(0) occurs during loss computation?
4. How does epsilon clipping prevent this issue?

Understanding these edge cases is crucial for reliable implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "crossentropy-loss-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class CrossEntropyLoss:
    """
    Cross-Entropy Loss for Multi-Class Classification Problems
    
    Computes the cross-entropy between predicted probability distributions
    and true class labels with numerically stable implementation.
    
    Features:
    - Numerically stable softmax computation using log-sum-exp trick
    - Support for both class indices and one-hot encoding
    - Efficient batch processing with proper broadcasting
    - Automatic handling of edge cases and extreme values
    
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
            
        TODO: Implement CrossEntropy with numerically stable softmax computation.
        
        APPROACH:
        1. Convert inputs to tensors and handle single samples
        2. Apply log-sum-exp trick for numerically stable softmax
        3. Clip probabilities to prevent log(0) issues
        4. Compute cross-entropy based on target format (indices vs one-hot)
        
        EXAMPLE:
        >>> ce = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.0]])  # Raw model outputs
        >>> targets = Tensor([0])  # Class 0 is correct
        >>> loss = ce(logits, targets)
        >>> print(loss.data)
        0.407  # -log(softmax([2.0, 1.0, 0.0])[0])
        
        HINTS:
        - Use np.max(axis=1, keepdims=True) for stable max computation
        - Use np.clip(probabilities, 1e-15, 1.0-1e-15) to prevent log(0)
        - Handle both index format [0,1,2] and one-hot format [[1,0,0], [0,1,0]]
        - Use advanced indexing: probs[np.arange(batch_size), class_indices]
        """
        ### BEGIN SOLUTION
        # Step 1: Ensure we have tensor inputs for consistent processing
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)  # Convert predictions to tensor format
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)  # Convert targets to tensor format
        
        # Step 1: Extract numpy arrays for computation
        prediction_logits = y_pred.data  # Raw model outputs (pre-softmax)
        target_labels = y_true.data      # True class indices or one-hot vectors
        
        # Step 2: Handle both single predictions and batches consistently
        if prediction_logits.ndim == 1:
            prediction_logits = prediction_logits.reshape(1, -1)  # Convert to batch format [1, num_classes]
            
        # Step 3: Apply numerically stable softmax transformation
        # Subtract max to prevent overflow: exp(x-max) is equivalent but stable
        max_logits = np.max(prediction_logits, axis=1, keepdims=True)
        exp_pred = np.exp(prediction_logits - max_logits)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Step 4: Prevent numerical instability in log computation
        epsilon = 1e-15  # Small value to prevent log(0) -> -inf and log(1) -> 0 issues
        softmax_pred = np.clip(softmax_pred, epsilon, 1.0 - epsilon)
        
        # Step 5: Compute cross-entropy loss based on target format
        if len(target_labels.shape) == 1:
            # Format A: y_true contains class indices [0, 1, 2, ...]
            batch_size = target_labels.shape[0]
            # Extract probabilities for correct classes using advanced indexing
            correct_class_probs = softmax_pred[np.arange(batch_size), target_labels.astype(int)]
            log_probs = np.log(correct_class_probs)
            loss_value = -np.mean(log_probs)  # Negative log-likelihood
        else:
            # Format B: y_true is one-hot encoded [[1,0,0], [0,1,0], ...]
            log_probs = np.log(softmax_pred)
            # Multiply one-hot targets with log probabilities, sum across classes
            weighted_log_probs = target_labels * log_probs
            loss_value = -np.mean(np.sum(weighted_log_probs, axis=1))
        
        return Tensor(loss_value)
        ### END SOLUTION
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# MAGNIFY SYSTEMS INSIGHT: CrossEntropy Stability Analysis
def analyze_crossentropy_stability():
    """Analyze numerical stability in cross-entropy computation."""
    print("MAGNIFY CrossEntropy Stability Analysis")
    print("=" * 40)
    
    try:
        ce = CrossEntropyLoss()
        
        # Test numerical stability with extreme values
        print("\nSPEED Numerical Stability Testing:")
        
        # Extreme logits that would overflow in naive implementation
        extreme_logits = Tensor([[100.0, 99.0, 98.0]])
        safe_labels = Tensor([0])
        
        loss = ce(extreme_logits, safe_labels)
        print(f"   Extreme logits [100, 99, 98]: Loss = {loss.data:.6f}")
        print(f"   No overflow or NaN: {not np.isnan(loss.data) and not np.isinf(loss.data)}")
        
        # Test epsilon clipping effectiveness
        print(f"\n🛡️ Epsilon Clipping Protection:")
        very_confident = Tensor([[10.0, -10.0, -10.0]])  # Very confident about class 0
        confident_labels = Tensor([0])
        
        loss = ce(very_confident, confident_labels)
        print(f"   Very confident correct prediction: Loss = {loss.data:.6f}")
        print(f"   Should be near 0: {loss.data < 0.01}")
        
        # Compare different confidence levels
        print(f"\n📊 Confidence vs Loss Relationship:")
        confidence_levels = [
            ("Low confidence", [[0.1, 0.0, -0.1]]),
            ("Medium confidence", [[1.0, 0.0, -1.0]]),
            ("High confidence", [[5.0, 0.0, -5.0]]),
            ("Very high", [[10.0, 0.0, -10.0]])
        ]
        
        for name, logits in confidence_levels:
            test_logits = Tensor(logits)
            test_loss = ce(test_logits, Tensor([0]))
            print(f"   {name:15}: Loss = {test_loss.data:.6f}")
        
        # Memory efficiency for large vocabularies
        print(f"\n💾 Memory Scaling Analysis:")
        small_vocab = Tensor(np.random.randn(32, 100))    # 100 classes
        large_vocab = Tensor(np.random.randn(32, 10000))  # 10k classes
        
        import sys
        small_memory = sys.getsizeof(small_vocab.data)
        large_memory = sys.getsizeof(large_vocab.data)
        
        print(f"   Small vocab (100 classes): {small_memory / 1024:.1f} KB")
        print(f"   Large vocab (10k classes): {large_memory / 1024:.1f} KB")
        print(f"   Memory scales O(batch_size * num_classes)")
        
        # TIP WHY THIS MATTERS: CrossEntropy memory scales with vocabulary size.
        # This is why large language models use techniques like hierarchical softmax
        # or sampling-based training to handle vocabularies with 50k+ tokens.
        
    except Exception as e:
        print(f"WARNING️ Analysis error: {e}")
        print("Ensure CrossEntropy implementation is complete")

# %% [markdown]
"""
### TEST Unit Test: Cross-Entropy Loss Computation
This test validates `CrossEntropyLoss.__call__`, ensuring correct cross-entropy computation with numerically stable softmax.
"""

# %% nbgrader={"grade": true, "grade_id": "test-crossentropy-loss", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
def test_unit_crossentropy_loss():
    """Test CrossEntropy loss implementation."""
    print("TEST Testing Cross-Entropy Loss...")
    
    ce = CrossEntropyLoss()
    
    # Test case 1: Perfect predictions
    y_pred = Tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])  # Very confident correct predictions
    y_true = Tensor([0, 1])  # Class indices
    loss = ce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("PASS Perfect predictions test passed")
    
    # Test case 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Uniform after softmax
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    expected_random = -np.log(1.0/3.0)  # log(1/num_classes) for uniform distribution
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ~= {expected_random}, got {loss.data}"
    print("PASS Random predictions test passed")
    
    # Test case 3: Binary classification
    y_pred = Tensor([[2.0, 1.0], [1.0, 2.0]])
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Binary classification loss should be reasonable, got {loss.data}"
    print("PASS Binary classification test passed")
    
    # Test case 4: One-hot encoded labels
    y_pred = Tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y_true = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # One-hot encoded
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"One-hot encoded loss should be reasonable, got {loss.data}"
    print("PASS One-hot encoded labels test passed")
    
    print("CELEBRATE Cross-Entropy loss tests passed! Understanding classification objectives.")

test_unit_crossentropy_loss()

# %% [markdown]
"""
# Binary Cross-Entropy Loss - Optimized for Binary Classification

Binary Cross-Entropy Loss is the specialized, efficient version of cross-entropy for binary (two-class) problems. It's more stable and faster than using regular cross-entropy with 2 classes.

## Visual Understanding: Binary Cross-Entropy

```
Binary Classification Landscape:

Sigmoid Activation:
    Raw Logit -> Sigmoid -> Probability -> Loss
    -5.0     -> 0.007   -> 0.007       -> High loss (if true=1)
     0.0     -> 0.500   -> 0.500       -> Medium loss
    +5.0     -> 0.993   -> 0.993       -> Low loss (if true=1)

Loss Behavior:
    BCE = -[y*log(p) + (1-y)*log(1-p)]
    
    For y=1 (positive class):
        p=0.9 -> -log(0.9) = 0.105  (low loss)
        p=0.1 -> -log(0.1) = 2.303  (high loss)
    
    For y=0 (negative class):
        p=0.1 -> -log(0.9) = 0.105  (low loss)  
        p=0.9 -> -log(0.1) = 2.303  (high loss)
```

## Numerical Stability Solution

```
The Binary Cross-Entropy Stability Problem:
    
    BCE = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    
    Where σ(x) = 1/(1+exp(-x))
    
    Problems:
    - Large positive x: exp(-x) -> 0, then log(1) -> 0 (loss of precision)
    - Large negative x: σ(x) -> 0, then log(0) -> -inf
    
Our Stable Solution:
    BCE = max(x,0) - x*y + log(1 + exp(-|x|))
    
    Why this works:
    - max(x,0) handles positive values
    - -x*y is the "cross" term  
    - log(1+exp(-|x|)) is always stable (exp<=1)
```

## Mathematical Foundation

For binary predictions and labels:
```
BCE = -y * log(σ(x)) - (1-y) * log(1-σ(x))

Stable form: BCE = max(x,0) - x*y + log(1 + exp(-|x|))
```

## Learning Objectives
By implementing Binary Cross-Entropy, you'll understand:
- How binary classification creates simpler optimization landscapes than multi-class problems
- Why sigmoid activation naturally pairs with binary cross-entropy loss through its gradient structure
- The critical importance of numerically stable formulations for reliable production training
- How specialized binary losses achieve better efficiency and stability than general solutions
"""

# %% nbgrader={"grade": false, "grade_id": "binary-crossentropy-concept", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Computational Question: Binary Stability**

Consider the stable BCE formulation:

1. Why does max(x,0) - x*y + log(1+exp(-|x|)) work?
2. What happens when x=100? (trace through the computation)
3. What happens when x=-100? (trace through the computation)
4. How does this prevent both overflow and underflow?

This mathematical insight is crucial for production systems.
"""

# %% nbgrader={"grade": false, "grade_id": "binary-crossentropy-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss for Binary Classification Problems
    
    Computes binary cross-entropy between predictions and binary labels
    with numerically stable sigmoid + BCE implementation.
    
    Features:
    - Numerically stable computation from logits using stable BCE formula
    - Efficient batch processing with vectorized operations
    - Automatic sigmoid application through stable formulation
    - Robust to extreme input values without overflow/underflow
    
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
            
        TODO: Implement stable binary cross-entropy using the logits formulation.
        
        APPROACH:
        1. Convert inputs to tensors and flatten for consistent processing
        2. Use stable BCE formula: max(x,0) - x*y + log(1+exp(-|x|))
        3. Apply this formula element-wise across the batch
        4. Return mean loss across all samples
        
        EXAMPLE:
        >>> bce = BinaryCrossEntropyLoss()
        >>> logits = Tensor([[2.0], [-1.0]])  # Raw outputs
        >>> labels = Tensor([[1.0], [0.0]])   # Binary targets
        >>> loss = bce(logits, labels)
        >>> print(loss.data)
        0.693  # Stable computation of binary cross-entropy
        
        HINTS:
        - Use np.maximum(logits, 0) for the max(x,0) term
        - Use np.abs(logits) to ensure exp argument is <= 0
        - The formula naturally handles both positive and negative logits
        - Return np.mean() for batch averaging
        """
        ### BEGIN SOLUTION
        # Step 1: Ensure we have tensor inputs for consistent processing
        if not isinstance(y_pred, Tensor):
            y_pred = Tensor(y_pred)  # Convert predictions to tensor format
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)  # Convert targets to tensor format
        
        # Get flat arrays for computation
        logits = y_pred.data.flatten()
        labels = y_true.data.flatten()
        
        # Step 1: Define numerically stable binary cross-entropy computation
        def stable_bce_with_logits(logits, labels):
            """
            Numerically stable BCE using the logits formulation:
            BCE(logits, y) = max(logits, 0) - logits * y + log(1 + exp(-|logits|))
            
            This formulation prevents:
            - exp(large_positive_logit) -> overflow
            - log(very_small_sigmoid) -> -inf
            
            Mathematical equivalence:
            - For positive logits: x - x*y + log(1 + exp(-x))
            - For negative logits: -x*y + log(1 + exp(x))
            """
            # Step 1a: Handle positive logits to prevent exp(large_positive) overflow
            positive_part = np.maximum(logits, 0)
            
            # Step 1b: Subtract logit-label product (the "cross" in cross-entropy)
            cross_term = logits * labels
            
            # Step 1c: Add log(1 + exp(-|logits|)) for numerical stability
            # Using abs(logits) ensures the exponent is always negative or zero
            stability_term = np.log(1 + np.exp(-np.abs(logits)))
            
            return positive_part - cross_term + stability_term
        
        # Step 2: Apply stable BCE computation across the batch
        individual_losses = stable_bce_with_logits(logits, labels)
        mean_loss = np.mean(individual_losses)  # Average loss across batch
        
        return Tensor(mean_loss)
        ### END SOLUTION
    
    def forward(self, y_pred, y_true):
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# MAGNIFY SYSTEMS INSIGHT: Binary CrossEntropy Efficiency Analysis
def analyze_binary_crossentropy_efficiency():
    """Analyze binary cross-entropy computational efficiency."""
    print("MAGNIFY Binary CrossEntropy Efficiency Analysis")
    print("=" * 45)
    
    try:
        bce = BinaryCrossEntropyLoss()
        ce = CrossEntropyLoss()  # For comparison
        
        # Compare binary-specific vs general cross-entropy
        print("\nSPEED Binary vs Multi-Class Efficiency:")
        
        # Binary problem solved two ways
        binary_logits = Tensor([[1.5], [-0.8], [2.1]])
        binary_labels = Tensor([[1.0], [0.0], [1.0]])
        
        # Method 1: Binary CrossEntropy
        binary_loss = bce(binary_logits, binary_labels)
        
        # Method 2: 2-class CrossEntropy (equivalent but less efficient)
        multiclass_logits = Tensor([[1.5, 0.0], [-0.8, 0.0], [2.1, 0.0]])
        multiclass_labels = Tensor([0, 1, 0])  # Convert to class indices
        multiclass_loss = ce(multiclass_logits, multiclass_labels)
        
        print(f"   Binary CE Loss:     {binary_loss.data:.6f}")
        print(f"   2-Class CE Loss:    {multiclass_loss.data:.6f}")
        print(f"   Difference:         {abs(binary_loss.data - multiclass_loss.data):.8f}")
        
        # Memory efficiency comparison
        print(f"\n💾 Memory Efficiency Comparison:")
        
        batch_size = 1000
        binary_memory = batch_size * 1 * 8  # 1 value per sample, 8 bytes per float64
        multiclass_memory = batch_size * 2 * 8  # 2 classes, 8 bytes per float64
        
        print(f"   Binary approach:    {binary_memory / 1024:.1f} KB")
        print(f"   Multi-class (2):    {multiclass_memory / 1024:.1f} KB")
        print(f"   Binary is {multiclass_memory/binary_memory:.1f}* more memory efficient")
        
        # Stability test with extreme values
        print(f"\n🛡️ Extreme Value Stability:")
        extreme_tests = [
            ("Large positive", [[100.0]], [[1.0]]),
            ("Large negative", [[-100.0]], [[0.0]]),
            ("Mixed extreme", [[100.0], [-100.0]], [[1.0], [0.0]])
        ]
        
        for name, logits, labels in extreme_tests:
            test_logits = Tensor(logits)
            test_labels = Tensor(labels)
            loss = bce(test_logits, test_labels)
            is_stable = not (np.isnan(loss.data) or np.isinf(loss.data))
            print(f"   {name:15}: Loss = {loss.data:.6f}, Stable = {is_stable}")
        
        # TIP WHY THIS MATTERS: Binary CrossEntropy is 2* more memory efficient
        # than regular CrossEntropy for binary problems, and provides better
        # numerical stability through its specialized formulation.
        
    except Exception as e:
        print(f"WARNING️ Analysis error: {e}")
        print("Ensure BinaryCrossEntropy implementation is complete")

# %% [markdown]
"""
### TEST Unit Test: Binary Cross-Entropy Loss
This test validates `BinaryCrossEntropyLoss.__call__`, ensuring stable binary cross-entropy computation with extreme values.
"""

# %% nbgrader={"grade": true, "grade_id": "test-binary-crossentropy", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
def test_unit_binary_crossentropy_loss():
    """Test Binary CrossEntropy loss implementation."""
    print("TEST Testing Binary Cross-Entropy Loss...")
    
    bce = BinaryCrossEntropyLoss()
    
    # Test case 1: Perfect predictions
    y_pred = Tensor([[10.0], [-10.0]])  # Very confident correct predictions
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("PASS Perfect predictions test passed")
    
    # Test case 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0], [0.0]])  # 0.5 probability after sigmoid
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    expected_random = -np.log(0.5)  # log(0.5) for random guessing
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ~= {expected_random}, got {loss.data}"
    print("PASS Random predictions test passed")
    
    # Test case 3: Batch processing
    y_pred = Tensor([[1.0], [2.0], [-1.0]])
    y_true = Tensor([[1.0], [1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Batch processing loss should be reasonable, got {loss.data}"
    print("PASS Batch processing test passed")
    
    # Test case 4: Extreme values (test numerical stability)
    y_pred = Tensor([[100.0], [-100.0]])  # Extreme logits
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert not np.isnan(loss.data) and not np.isinf(loss.data), f"Extreme values should not cause NaN/Inf, got {loss.data}"
    assert loss.data < 1.0, f"Extreme correct predictions should have low loss, got {loss.data}"
    print("PASS Extreme values test passed")
    
    print("CELEBRATE Binary Cross-Entropy loss tests passed! Understanding binary objectives.")

test_unit_binary_crossentropy_loss()

# %% [markdown]
"""
# Custom Loss Functions - Aligning with Business Objectives

Beyond standard loss functions, production ML systems often need custom losses that align with specific business objectives and domain constraints.

## Business-Aligned Loss Design Patterns

### Asymmetric Loss Functions
When false positives and false negatives have different costs:

```python
# Medical diagnosis: False negatives (missing disease) cost 10* more
class AsymmetricBinaryCrossEntropy(BinaryCrossEntropyLoss):
    def __init__(self, false_negative_weight=10.0):
        super().__init__()
        self.fn_weight = false_negative_weight

    def __call__(self, y_pred, y_true):
        # Standard BCE
        base_loss = super().__call__(y_pred, y_true)

        # Weight false negatives more heavily
        # When y_true=1 and y_pred is low, increase penalty
        sigmoid_pred = 1 / (1 + np.exp(-y_pred.data))
        fn_penalty = y_true.data * (1 - sigmoid_pred) * self.fn_weight

        weighted_loss = base_loss.data + np.mean(fn_penalty)
        return Tensor(weighted_loss)
```

### Focal Loss for Imbalanced Data
Addresses class imbalance by focusing on hard examples:

```python
class FocalLoss(CrossEntropyLoss):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class balance weight
        self.gamma = gamma  # Focusing parameter

    def __call__(self, y_pred, y_true):
        # Get standard cross-entropy
        ce_loss = super().__call__(y_pred, y_true)

        # Calculate softmax probabilities
        max_logits = np.max(y_pred.data, axis=1, keepdims=True)
        stable_logits = y_pred.data - max_logits
        exp_logits = np.exp(stable_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get probability of correct class
        batch_size = y_true.data.shape[0]
        correct_probs = softmax_probs[np.arange(batch_size), y_true.data.astype(int)]

        # Apply focal loss formula: -α(1-p)^γ log(p)
        focal_weight = self.alpha * ((1 - correct_probs) ** self.gamma)
        focal_loss = focal_weight * ce_loss.data

        return Tensor(np.mean(focal_loss))
```
"""

# %% [markdown]
"""
### Ranking-Aware Loss
For problems where order matters (search, recommendations):
"""

# %% nbgrader={"grade": false, "grade_id": "ranking-loss", "solution": true}
class RankingAwareLoss:
    def __init__(self, position_weights=None):
        # Higher weights for top positions
        self.position_weights = position_weights or [10.0, 5.0, 2.0, 1.0, 0.5]

    def __call__(self, predictions, targets, positions):
        """predictions: relevance scores, targets: true relevance, positions: result positions"""
        # Not using MeanSquaredError() - computing directly

        # Weight errors by position importance
        weighted_errors = []
        for pred, target, pos in zip(predictions.data, targets.data, positions.data):
            pos_weight = self.position_weights[min(int(pos), len(self.position_weights)-1)]
            error = ((pred - target) ** 2) * pos_weight
            weighted_errors.append(error)

        return Tensor(np.mean(weighted_errors))

# %% [markdown]
"""
## Advanced Custom Loss Patterns

### Multi-Task Learning Loss
Combining multiple objectives with learned weights:
"""

# %% nbgrader={"grade": false, "grade_id": "multitask-loss", "solution": true}
class MultiTaskLoss:
    def __init__(self, num_tasks=3):
        # Learnable loss weights (log-variance parameterization for stability)
        self.log_vars = [0.0] * num_tasks

    def __call__(self, predictions_list, targets_list):
        """predictions_list: [task1_preds, task2_preds, ...]"""
        total_loss = 0

        for i, (preds, targets) in enumerate(zip(predictions_list, targets_list)):
            # Choose appropriate loss for each task
            if i == 0:  # Regression task
                task_loss = MeanSquaredError()(preds, targets)
            else:  # Classification tasks
                task_loss = CrossEntropyLoss()(preds, targets)

            # Uncertainty-weighted combination
            precision = np.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss.data + self.log_vars[i]
            total_loss += weighted_loss

        return Tensor(total_loss)

# %% [markdown]
"""
### Contrastive Loss for Similarity Learning
For learning embeddings and similarity:
"""

# %% nbgrader={"grade": false, "grade_id": "contrastive-loss", "solution": true}
class ContrastiveLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, embeddings1, embeddings2, labels):
        """labels: 1 for similar pairs, 0 for dissimilar"""
        # Euclidean distance between embeddings
        distances = np.sqrt(np.sum((embeddings1.data - embeddings2.data) ** 2, axis=1))

        # Contrastive loss formula
        positive_loss = labels.data * (distances ** 2)
        negative_loss = (1 - labels.data) * np.maximum(0, self.margin - distances) ** 2

        total_loss = 0.5 * (positive_loss + negative_loss)
        return Tensor(np.mean(total_loss))

# %% [markdown]
"""
## Custom Loss Implementation Guidelines

### Numerical Stability Considerations
"""

# %% nbgrader={"grade": false, "grade_id": "stable-loss", "solution": true}
# Always include stability measures in custom losses
class StableCustomLoss:
    def __call__(self, predictions, targets):
        # 1. Input validation
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)

        # 2. Handle edge cases
        # predictions_clipped would be used here for actual computation
        # predictions_clipped = np.clip(predictions.data, -100, 100)  # Prevent overflow

        # 3. Use numerically stable formulations
        # Avoid: exp(large_number), log(small_number)
        # Use: log-sum-exp trick, epsilon clipping

        # 4. Compute loss (example - actual implementation depends on loss type)
        computed_loss = np.mean((predictions.data - targets.data) ** 2)

        # 5. Return tensor for consistency
        return Tensor(computed_loss)

# %% [markdown]
"""
### Gradient-Friendly Design
```python
# Ensure gradients flow properly
class GradientFriendlyLoss:
    def __call__(self, predictions, targets):
        # Avoid operations that create zero gradients:
        # - Hard thresholding: use soft approximations
        # - Discrete operations: use continuous relaxations
        # - Large plateaus: ensure non-zero gradients everywhere

        # Good: Smooth, differentiable operations
        smooth_loss = self.smooth_l1_loss(predictions, targets)
        return smooth_loss

    def smooth_l1_loss(self, pred, target, beta=1.0):
        \"\"\"Smooth L1 loss - less sensitive to outliers than MSE\"\"\"
        diff = np.abs(pred.data - target.data)
        loss = np.where(diff < beta,
                       0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
        return Tensor(np.mean(loss))
```
"""

# %% [markdown]
"""
# Loss Function Application Guide and Comparison

## When to Use Each Loss Function

Understanding which loss function to use is critical for successful ML projects:

### Mean Squared Error (MSE) - Regression Problems
```
Use when: Predicting continuous values
Examples: House prices, temperature, stock values, ages
Output: Any real number
Activation: Usually none (linear output)
Penalty: Quadratic (large errors >> small errors)

Model Architecture:
Input -> Hidden Layers -> Linear Output -> MSE Loss
```

### Cross-Entropy Loss - Multi-Class Classification  
```
Use when: Choosing one class from 3+ options
Examples: Image classification, text categorization, medical diagnosis
Output: Probability distribution (sums to 1)
Activation: Softmax
Penalty: Logarithmic (encouraging confident correct predictions)

Model Architecture:
Input -> Hidden Layers -> Softmax -> CrossEntropy Loss
```

### Binary Cross-Entropy Loss - Binary Classification
```
Use when: Binary decisions (yes/no, positive/negative)
Examples: Spam detection, fraud detection, medical screening
Output: Single probability (0 to 1)
Activation: Sigmoid
Penalty: Asymmetric (confident wrong predictions heavily penalized)

Model Architecture:
Input -> Hidden Layers -> Sigmoid -> Binary CrossEntropy Loss
```

## Performance and Stability Comparison

```
Computational Characteristics:
                      MSE    CrossEntropy    Binary CE
Time Complexity:     O(n)      O(n*c)        O(n)
Memory Complexity:   O(1)      O(n*c)        O(n)
Numerical Stability: High      Medium        High
Convergence Speed:   Fast      Medium        Fast

Where: n = batch size, c = number of classes
```

## Integration with Neural Networks

```python
# Example training setup for different problem types:

# Regression Problem (House Price Prediction)
regression_model = Sequential([
    Linear(10, 64),   # Input features -> Hidden
    ReLU(),
    Linear(64, 1),    # Hidden -> Single output
    # No activation - linear output for regression
])
loss_fn = MeanSquaredError()

# Multi-Class Classification (Image Recognition)
classification_model = Sequential([
    Linear(784, 128), # Flattened image -> Hidden
    ReLU(),
    Linear(128, 10),  # Hidden -> 10 classes
    Softmax()         # Convert to probabilities
])
loss_fn = CrossEntropyLoss()

# Binary Classification (Spam Detection)
binary_model = Sequential([
    Linear(100, 64),  # Text features -> Hidden
    ReLU(),
    Linear(64, 1),    # Hidden -> Single output
    Sigmoid()         # Convert to probability
])
loss_fn = BinaryCrossEntropyLoss()

# Training loop pattern (same for all):
for batch in dataloader:
    predictions = model(batch.inputs)
    loss = loss_fn(predictions, batch.targets)
    # loss.backward()  # Compute gradients (when autograd is available)
    # optimizer.step() # Update parameters
```
"""

# %% [markdown]
"""
### TEST Comprehensive Integration Test
This test validates all loss functions work together correctly and can be used interchangeably in production systems.
"""

# %% nbgrader={"grade": false, "grade_id": "comprehensive-loss-tests", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_unit_comprehensive_loss_integration():
    """Test all loss functions work correctly together."""
    print("🔬 Comprehensive Loss Function Integration Testing")
    print("=" * 55)
    
    # Test 1: All losses can be instantiated
    print("\n1. Loss Function Instantiation:")
    mse = MeanSquaredError()
    ce = CrossEntropyLoss()
    bce = BinaryCrossEntropyLoss()
    print("   PASS All loss functions created successfully")
    
    # Test 2: Loss functions return appropriate types
    print("\n2. Return Type Verification:")
    
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
    
    print("   PASS All loss functions return correct types")
    
    # Test 3: Loss values are reasonable
    print("\n3. Loss Value Sanity Checks:")
    
    # All losses should be non-negative
    assert mse.forward(Tensor([1.0]), Tensor([2.0])).data >= 0, "MSE should be non-negative"
    assert ce.forward(Tensor([[1.0, 0.0]]), Tensor([0])).data >= 0, "CrossEntropy should be non-negative"
    assert bce.forward(Tensor([1.0]), Tensor([1.0])).data >= 0, "Binary CrossEntropy should be non-negative"
    
    print("   PASS All loss functions produce reasonable values")
    
    # Test 4: Perfect predictions give low loss
    print("\n4. Perfect Prediction Tests:")
    
    perfect_mse = mse(Tensor([5.0]), Tensor([5.0]))
    perfect_ce = ce(Tensor([[10.0, 0.0]]), Tensor([0]))
    perfect_bce = bce(Tensor([10.0]), Tensor([1.0]))
    
    assert perfect_mse.data < 1e-10, f"Perfect MSE should be ~0, got {perfect_mse.data}"
    assert perfect_ce.data < 0.1, f"Perfect CE should be low, got {perfect_ce.data}"
    assert perfect_bce.data < 0.1, f"Perfect BCE should be low, got {perfect_bce.data}"
    
    print("   PASS Perfect predictions produce low loss")
    
    print("\nCELEBRATE All comprehensive integration tests passed!")
    print("   • Loss functions instantiate correctly")
    print("   • Return types are consistent (Tensor scalars)")
    print("   • Loss values are mathematically sound")
    print("   • Perfect predictions are handled correctly")
    print("   • Ready for integration with neural network training!")

test_unit_comprehensive_loss_integration()

# %% [markdown]
"""
# Systems Analysis: Loss Function Performance and Engineering

Let's analyze loss functions from an ML systems engineering perspective, focusing on performance, memory usage, and production implications.

## Computational Complexity Deep Dive

```
Algorithmic Analysis by Loss Type:

MSE (Mean Squared Error):
    Time: O(n) - linear in number of predictions
    Space: O(1) - constant additional memory
    Operations: n subtractions + n multiplications + 1 mean
    Bottleneck: Memory bandwidth (simple arithmetic operations)
    
CrossEntropy (Multi-Class):
    Time: O(n*c) - linear in samples * classes  
    Space: O(n*c) - store full probability distributions
    Operations: n*c exp + n*c divisions + n*c logs + reductions
    Bottleneck: Exponential computations and memory bandwidth
    
Binary CrossEntropy:
    Time: O(n) - linear in number of samples
    Space: O(n) - store one probability per sample
    Operations: n max + n multiplications + n exp + n logs
    Bottleneck: Transcendental functions (exp, log)
```

## Memory Scaling Analysis

Understanding memory requirements is crucial for large-scale training:

```
Memory Requirements by Problem Scale:

Small Problem (1K samples, 100 classes):
    MSE:         8 KB (1K samples * 8 bytes)
    CrossEntropy: 800 KB (1K * 100 * 8 bytes)
    Binary CE:   16 KB (1K * 2 * 8 bytes)

Large Problem (100K samples, 10K classes):
    MSE:         800 KB (independent of classes!)
    CrossEntropy: 8 GB (memory bottleneck)
    Binary CE:   1.6 MB (scales with samples only)

Production Scale (1M samples, 50K vocab):
    MSE:         8 MB
    CrossEntropy: 400 GB (requires distributed memory)
    Binary CE:   16 MB
```

## Numerical Stability Engineering Analysis

Production systems must handle edge cases robustly:

```
Stability Challenges and Solutions:

CrossEntropy Stability Issues:
    Problem: exp(large_logit) -> overflow -> NaN gradients
    Solution: log-sum-exp trick with max subtraction
    
    Problem: log(very_small_prob) -> -inf -> training collapse
    Solution: epsilon clipping (1e-15 to 1-1e-15)
    
Binary CrossEntropy Stability Issues:
    Problem: sigmoid(large_positive) -> 1.0 -> log(0) issues
    Solution: stable logits formulation bypasses sigmoid
    
    Problem: exp(large_negative) in naive implementation
    Solution: max(x,0) - x*y + log(1+exp(-|x|)) formulation
```
"""

# %% [markdown]
"""
## Production Performance Benchmarks

Real-world performance characteristics matter for deployment:

```
Inference Throughput (measured on modern hardware):
    MSE:              ~100M predictions/second
    CrossEntropy:     ~10M predictions/second  
    Binary CrossEntropy: ~80M predictions/second

Training Memory Bandwidth Requirements:
    MSE:         ~800 MB/s (lightweight computation)
    CrossEntropy: ~80 GB/s (10* higher due to softmax!)
    Binary CE:   ~1.6 GB/s (moderate requirements)

Gradient Computation Overhead:
    MSE:         1.1* forward pass time (simple derivatives)
    CrossEntropy: 1.5* forward pass time (softmax gradients)
    Binary CE:   1.2* forward pass time (sigmoid gradients)
```

## Framework Integration and Production Patterns

Understanding how production systems implement these concepts:

```
PyTorch Implementation Patterns:
    torch.nn.MSELoss() - Direct implementation, minimal overhead
    torch.nn.CrossEntropyLoss() - Fused softmax+CE for efficiency
    torch.nn.BCEWithLogitsLoss() - Stable logits formulation
    
TensorFlow Implementation Patterns:
    tf.keras.losses.MeanSquaredError() - Vectorized operations
    tf.keras.losses.SparseCategoricalCrossentropy() - Memory efficient
    tf.keras.losses.BinaryCrossentropy() - From logits option
    
Production Optimizations:
    - Mixed precision (FP16) for memory efficiency
    - Gradient accumulation for large batch simulation
    - Loss scaling to prevent underflow in mixed precision
    - Checkpointing to trade memory for computation
```

## Edge Device and Deployment Considerations

Loss function choice affects deployment feasibility:

```
Edge Device Constraints:
    Memory-limited (phones, IoT): Prefer Binary CE > MSE > CrossEntropy
    CPU-only inference: MSE has best compute efficiency
    Real-time requirements: Binary classification most predictable
    
Distributed Training Challenges:
    CrossEntropy: Requires all-reduce across all classes (expensive!)
    Gradient accumulation: MSE linear, CrossEntropy non-linear dependencies
    Mixed precision: Different overflow handling per loss type
    
Monitoring and Debugging:
    MSE divergence: Explodes quadratically (easy to detect)
    CrossEntropy divergence: More gradual degradation  
    BCE monitoring: Natural bounded behavior aids debugging
```
"""

# MAGNIFY SYSTEMS INSIGHT: Performance Profiling Analysis
def analyze_loss_performance_characteristics():
    """Comprehensive performance analysis of all loss functions."""
    print("MAGNIFY Loss Function Performance Analysis")
    print("=" * 45)
    
    try:
        import time
        
        # Initialize loss functions
        mse = MeanSquaredError()
        ce = CrossEntropyLoss()
        bce = BinaryCrossEntropyLoss()
        
        print("\nSPEED Computational Complexity Measurement:")
        
        # Test different batch sizes to see scaling behavior
        batch_sizes = [100, 1000, 10000]
        
        for batch_size in batch_sizes:
            print(f"\n   Batch size: {batch_size:,}")
            
            # MSE timing
            mse_pred = Tensor(np.random.randn(batch_size, 10))
            mse_true = Tensor(np.random.randn(batch_size, 10))
            
            start = time.perf_counter()
            for _ in range(100):  # Average over multiple runs
                _ = mse(mse_pred, mse_true)
            mse_time = (time.perf_counter() - start) / 100
            
            # CrossEntropy timing
            ce_pred = Tensor(np.random.randn(batch_size, 100))  # 100 classes
            ce_true = Tensor(np.random.randint(0, 100, batch_size))
            
            start = time.perf_counter()
            for _ in range(100):
                _ = ce(ce_pred, ce_true)
            ce_time = (time.perf_counter() - start) / 100
            
            # Binary CrossEntropy timing
            bce_pred = Tensor(np.random.randn(batch_size, 1))
            bce_true = Tensor(np.random.randint(0, 2, (batch_size, 1)).astype(float))
            
            start = time.perf_counter()
            for _ in range(100):
                _ = bce(bce_pred, bce_true)
            bce_time = (time.perf_counter() - start) / 100
            
            print(f"      MSE:         {mse_time*1000:8.3f} ms")
            print(f"      CrossEntropy: {ce_time*1000:8.3f} ms")
            print(f"      Binary CE:    {bce_time*1000:8.3f} ms")
            print(f"      CE/MSE ratio: {ce_time/mse_time:8.1f}x")
        
        print("\n💾 Memory Efficiency Analysis:")
        
        # Compare memory usage for different problem sizes
        problem_configs = [
            ("Small (1K samples, 10 classes)", 1000, 10),
            ("Medium (10K samples, 100 classes)", 10000, 100),
            ("Large (100K samples, 1K classes)", 100000, 1000)
        ]
        
        for name, samples, classes in problem_configs:
            print(f"\n   {name}:")
            
            # Memory calculations (bytes)
            mse_memory = samples * 8  # One value per sample
            ce_memory = samples * classes * 8  # Full probability distribution
            bce_memory = samples * 8  # One probability per sample
            
            print(f"      MSE memory:    {mse_memory / 1024 / 1024:8.1f} MB")
            print(f"      CE memory:     {ce_memory / 1024 / 1024:8.1f} MB") 
            print(f"      BCE memory:    {bce_memory / 1024 / 1024:8.1f} MB")
            print(f"      CE overhead:   {ce_memory/mse_memory:8.1f}x")
        
        # TIP WHY THIS MATTERS: These performance characteristics determine
        # which loss functions are feasible for different deployment scenarios.
        # CrossEntropy's O(n*c) memory scaling makes it prohibitive for 
        # large vocabularies without specialized techniques.
        
    except Exception as e:
        print(f"WARNING️ Performance analysis error: {e}")
        print("Performance analysis requires complete implementations")

# MAGNIFY SYSTEMS INSIGHT: Numerical Stability Deep Analysis
def analyze_numerical_stability_edge_cases():
    """Deep analysis of numerical stability across all loss functions."""
    print("MAGNIFY Numerical Stability Edge Case Analysis")
    print("=" * 50)
    
    try:
        mse = MeanSquaredError()
        ce = CrossEntropyLoss()
        bce = BinaryCrossEntropyLoss()
        
        print("\n🛡️ Extreme Value Stability Testing:")
        
        # Test extreme values that could cause numerical issues
        extreme_tests = [
            ("Huge positive", 1e10),
            ("Huge negative", -1e10),
            ("Tiny positive", 1e-10),
            ("NaN input", float('nan')),
            ("Infinity", float('inf')),
            ("Negative infinity", float('-inf'))
        ]
        
        for name, value in extreme_tests:
            print(f"\n   Testing {name} ({value}):")
            
            # MSE stability
            try:
                mse_loss = mse(Tensor([value]), Tensor([0.0]))
                mse_stable = not (np.isnan(mse_loss.data) or np.isinf(mse_loss.data))
                print(f"      MSE stable:    {mse_stable} (loss: {mse_loss.data:.3e})")
            except:
                print(f"      MSE stable:    False (exception)")
            
            # CrossEntropy stability  
            try:
                ce_loss = ce(Tensor([[value, 0.0, 0.0]]), Tensor([0]))
                ce_stable = not (np.isnan(ce_loss.data) or np.isinf(ce_loss.data))
                print(f"      CE stable:     {ce_stable} (loss: {ce_loss.data:.3e})")
            except:
                print(f"      CE stable:     False (exception)")
            
            # Binary CrossEntropy stability
            try:
                bce_loss = bce(Tensor([value]), Tensor([1.0]))
                bce_stable = not (np.isnan(bce_loss.data) or np.isinf(bce_loss.data))
                print(f"      BCE stable:    {bce_stable} (loss: {bce_loss.data:.3e})")
            except:
                print(f"      BCE stable:    False (exception)")
        
        print("\n🔬 Gradient Behavior Analysis:")
        
        # Analyze gradient magnitudes for different prediction qualities
        confidence_levels = [
            ("Very wrong", [[-5.0, 5.0, 0.0]], [0]),  # Predict class 1, actual class 0
            ("Slightly wrong", [[-0.5, 0.5, 0.0]], [0]),
            ("Uncertain", [[0.0, 0.0, 0.0]], [0]), 
            ("Slightly right", [[0.5, -0.5, 0.0]], [0]),
            ("Very right", [[5.0, -5.0, 0.0]], [0])
        ]
        
        print("      Prediction Quality -> CrossEntropy Loss:")
        for name, logits, labels in confidence_levels:
            loss = ce(Tensor(logits), Tensor(labels))
            print(f"      {name:15}: {loss.data:8.4f}")
        
        # TIP WHY THIS MATTERS: Understanding how loss functions behave
        # at extremes helps debug training failures and choose appropriate
        # loss scaling and clipping strategies for production systems.
        
    except Exception as e:
        print(f"WARNING️ Stability analysis error: {e}")
        print("Stability analysis requires complete implementations")

# MAGNIFY SYSTEMS INSIGHT: Mixed Precision Training Analysis
def analyze_mixed_precision_considerations():
    """Analyze loss function behavior with FP16 mixed precision training."""
    print("MAGNIFY Mixed Precision Training Analysis")
    print("=" * 40)

    try:
        print("\nSPEED FP16 Numerical Range Analysis:")
        print("   FP16 range: ~±65,504 (much smaller than FP32's ~±3.4*10³⁸)")

        # Simulate FP16 range limitations
        fp16_max = 65504.0
        fp16_min_normal = 2**-14  # Smallest normal FP16 number ~= 6.1*10⁻⁵

        print(f"   FP16 maximum: ±{fp16_max:,.0f}")
        print(f"   FP16 min normal: {fp16_min_normal:.2e}")
        print(f"   Risk: Gradients/losses exceeding range -> infinity/NaN")

        mse = MeanSquaredError()
        # ce = CrossEntropyLoss()  # Not used in this test
        # bce = BinaryCrossEntropyLoss()  # Not used in this test

        print(f"\nTARGET Loss Function Mixed Precision Compatibility:")

        # Test cases that might overflow in FP16
        test_cases = [
            ("Small values", 1.0, 1.1),
            ("Medium values", 100.0, 110.0),
            ("Large values", 1000.0, 1100.0),
            ("FP16 edge", 200.0, 250.0)  # Could cause issues when squared
        ]

        print(f"\n   {'Test Case':>15} {'MSE Loss':>12} {'FP16 Safe?':>12}")
        print(f"   {'-'*15} {'-'*12} {'-'*12}")

        for name, pred, true in test_cases:
            mse_loss = mse(Tensor([pred]), Tensor([true]))
            squared_error = (pred - true) ** 2
            fp16_safe = squared_error < fp16_max

            print(f"   {name:>15} {mse_loss.data:>12.1f} {'PASS' if fp16_safe else 'FAIL':>12}")

        print(f"\n🛡️ Mixed Precision Loss Scaling Strategy:")

        # Demonstrate loss scaling concept
        loss_scales = [1.0, 128.0, 1024.0, 8192.0]
        base_loss = 0.01  # Small loss that might underflow

        print(f"   {'Scale Factor':>12} {'Scaled Loss':>12} {'FP16 Precision':>15}")
        print(f"   {'-'*12} {'-'*12} {'-'*15}")

        for scale in loss_scales:
            scaled_loss = base_loss * scale

            # Check if loss is representable in FP16
            if scaled_loss > fp16_min_normal and scaled_loss < fp16_max:
                precision = "Good"
            elif scaled_loss <= fp16_min_normal:
                precision = "Underflow risk"
            else:
                precision = "Overflow risk"

            print(f"   {scale:>12.0f} {scaled_loss:>12.3f} {precision:>15}")

        print(f"\n⚖️ Loss Function Mixed Precision Recommendations:")

        recommendations = [
            ("MSE", "Monitor for gradient explosion in high-dynamic-range problems", "Medium risk"),
            ("CrossEntropy", "Use FP32 for softmax computation, FP16 for storage", "High risk"),
            ("Binary CE", "Stable formulation handles FP16 well with proper scaling", "Low risk")
        ]

        for loss_type, recommendation, risk in recommendations:
            print(f"   {loss_type:>12}: {recommendation} ({risk})")

        print(f"\n🔧 Implementation Best Practices for Mixed Precision:")

        best_practices = [
            "1. Use automatic mixed precision (AMP) libraries that handle scaling",
            "2. Keep loss computation in FP32, only cast inputs to FP16",
            "3. Monitor for overflow/underflow during training",
            "4. Use gradient clipping to prevent extreme gradients",
            "5. Scale losses up during forward pass, scale gradients down during backward"
        ]

        for practice in best_practices:
            print(f"      {practice}")

        # Example mixed precision training pattern
        print(f"\n💻 Mixed Precision Training Pattern:")
        print(f"   ```python")
        print(f"   # Forward pass in FP16")
        print(f"   with autocast():")
        print(f"       predictions = model(inputs.half())  # FP16 inputs")
        print(f"       loss = loss_fn(predictions, targets)  # Loss computed in FP32")
        print(f"   ")
        print(f"   # Scale loss to prevent underflow")
        print(f"   scaled_loss = loss * scale_factor")
        print(f"   scaled_loss.backward()")
        print(f"   ")
        print(f"   # Unscale gradients before optimizer step")
        print(f"   scaler.step(optimizer)  # Automatically unscales gradients")
        print(f"   ```")

        # TIP WHY THIS MATTERS: Mixed precision training can provide 1.5-2* speedup
        # and 50% memory reduction, but loss functions must be carefully implemented
        # to handle the reduced numerical precision without losing training stability.

    except Exception as e:
        print(f"WARNING️ Mixed precision analysis error: {e}")
        print("Mixed precision analysis requires complete loss implementations")

# MAGNIFY SYSTEMS INSIGHT: Production Deployment Analysis
def analyze_production_deployment_patterns():
    """Analyze how loss functions affect production ML system design."""
    print("MAGNIFY Production Deployment Pattern Analysis")
    print("=" * 50)
    
    try:
        print("\nROCKET Deployment Scenario Analysis:")
        
        # Different deployment scenarios with constraints
        scenarios = [
            {
                "name": "Mobile App (Spam Detection)",
                "constraints": "Memory < 50MB, Latency < 100ms",
                "problem": "Binary classification",
                "recommendation": "Binary CrossEntropy",
                "reasoning": "Minimal memory, fast inference, stable numerics"
            },
            {
                "name": "Cloud API (Image Classification)", 
                "constraints": "Throughput > 1000 QPS, Cost optimization",
                "problem": "1000-class classification",
                "recommendation": "CrossEntropy with mixed precision",
                "reasoning": "Can handle memory cost, needs throughput"
            },
            {
                "name": "Edge IoT (Temperature Prediction)",
                "constraints": "Memory < 1MB, Power < 1W",
                "problem": "Regression",
                "recommendation": "MSE with quantization",
                "reasoning": "Minimal compute, no transcendental functions"
            },
            {
                "name": "Large Language Model Training",
                "constraints": "50K vocabulary, Multi-GPU",
                "problem": "Next token prediction",
                "recommendation": "Hierarchical Softmax or Sampling",
                "reasoning": "Standard CrossEntropy too memory intensive"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n   📱 {scenario['name']}:")
            print(f"      Constraints:     {scenario['constraints']}")
            print(f"      Problem Type:    {scenario['problem']}")
            print(f"      Best Loss:       {scenario['recommendation']}")
            print(f"      Why:             {scenario['reasoning']}")
        
        print("\n⚖️ Production Trade-off Analysis:")
        
        trade_offs = [
            ("Memory Efficiency", "MSE > Binary CE >> CrossEntropy"),
            ("Computational Speed", "MSE > Binary CE > CrossEntropy"),
            ("Numerical Stability", "MSE ~= Binary CE > CrossEntropy"), 
            ("Implementation Complexity", "MSE > CrossEntropy > Binary CE"),
            ("Gradient Quality", "CrossEntropy > Binary CE > MSE"),
            ("Debug-ability", "MSE > Binary CE > CrossEntropy")
        ]
        
        for criterion, ranking in trade_offs:
            print(f"      {criterion:20}: {ranking}")
        
        print("\n🔧 Framework Integration Patterns:")
        
        frameworks = [
            ("PyTorch", "nn.MSELoss(), nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()"),
            ("TensorFlow", "keras.losses.MSE, SparseCategoricalCrossentropy, BinaryCrossentropy"),
            ("JAX", "optax.l2_loss, optax.softmax_cross_entropy, optax.sigmoid_binary_cross_entropy"),
            ("Production", "Custom implementations with monitoring and fallbacks")
        ]
        
        for framework, losses in frameworks:
            print(f"      {framework:12}: {losses}")
        
        # TIP WHY THIS MATTERS: Loss function choice affects every aspect
        # of ML system design - from memory requirements to latency to
        # debugging complexity. Understanding these trade-offs enables
        # informed architectural decisions for production systems.
        
    except Exception as e:
        print(f"WARNING️ Deployment analysis error: {e}")

# %% [markdown]
"""
## THINK ML Systems Thinking: Interactive Questions

Now that you've implemented all core loss functions and analyzed their systems characteristics, let's explore their implications for real ML systems:
"""

# %% nbgrader={"grade": false, "grade_id": "question-1-loss-selection", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Question 1: Loss Function Selection for Production Systems**

You're building a production recommendation system that predicts user ratings (1-5 stars) for movies.

Your team proposes three approaches:
A) Regression approach: Use MSE loss with continuous outputs (1.0-5.0)
B) Classification approach: Use CrossEntropy loss with 5 distinct classes  
C) Ordinal approach: Use a custom loss that penalizes being off by multiple stars more heavily

Analyze each approach considering your implementations:

**Technical Analysis:**
- How does the memory scaling of CrossEntropy (O(batch_size * num_classes)) affect this 5-class problem?
- What are the computational complexity differences between MSE's O(n) and CrossEntropy's O(n*c) for c=5?
- How do the gradient behaviors differ? (MSE's quadratic vs CrossEntropy's logarithmic penalties)

**Systems Implications:**
- Which approach would be most memory efficient for large batch training?
- How does numerical stability differ when handling edge cases (ratings at boundaries)?
- Which approach would have the most predictable inference latency?

**Business Alignment:**
- How well does each loss function's penalty structure match the business objective?
- What happens with fractional ratings like 3.7? How would each approach handle this?
- Which approach would be easiest to monitor and debug in production?

Recommend an approach with justification based on your implementation experience.
"""

# %% nbgrader={"grade": false, "grade_id": "question-2-numerical-stability", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Question 2: Debugging Numerical Stability in Production**

Your cross-entropy loss function works perfectly in development, but in production you start seeing NaN losses that crash training after several hours.

**Root Cause Analysis:**
Based on your implementation of the log-sum-exp trick and epsilon clipping:
1. What specific numerical computations in cross-entropy can produce NaN values?
2. Walk through how your `max_logits = np.max(prediction_logits, axis=1, keepdims=True)` prevents overflow
3. Explain why `np.clip(softmax_pred, epsilon, 1.0 - epsilon)` prevents underflow
4. What would happen if you removed epsilon clipping? Trace through the computation.

**Production Debugging:**
Given millions of training examples, how would you:
1. Identify which specific inputs trigger the numerical instability?
2. Modify your CrossEntropy implementation to add monitoring without affecting performance?
3. Design fallback behavior when numerical issues are detected?
4. Validate that your fixes don't change the mathematical behavior for normal inputs?

**Comparison Analysis:**
- How does your stable Binary CrossEntropy formulation `max(x,0) - x*y + log(1 + exp(-|x|))` prevent similar issues?
- Why is MSE generally more numerically stable than CrossEntropy?
- How would you modify loss functions for mixed precision (FP16) training where numerical ranges are more limited?

Research how PyTorch and TensorFlow handle these same challenges in their loss implementations.
"""

# %% nbgrader={"grade": false, "grade_id": "question-3-custom-loss-design", "locked": false, "schema_version": 3, "solution": false, "task": false}
"""
THINK **Question 3: Implementing and Optimizing Custom Loss Functions**

You've seen examples of custom loss functions for business objectives. Now analyze implementation and optimization challenges:

**Scenario Analysis:**
Choose one custom loss from the examples (Asymmetric BCE, Focal Loss, Ranking-Aware, Multi-Task, or Contrastive) and analyze:

**Implementation Deep Dive:**
1. Trace through the numerical computation step-by-step for your chosen custom loss
2. Identify potential numerical stability issues compared to standard loss functions
3. How does the computational complexity compare to MSE/CrossEntropy/Binary CE?
4. What additional memory overhead does the custom formulation introduce?

**Gradient Flow Analysis:**
5. How do the custom weighting schemes affect gradient magnitudes during backpropagation?
6. What happens to gradient flow when the custom weights become extreme (very large or very small)?
7. How would you detect and handle gradient explosion or vanishing in your custom loss?
8. Design gradient clipping strategies specific to your chosen custom loss function

**Production Integration Challenges:**
9. How would you implement your custom loss to work with mixed precision training (FP16)?
10. What logging and monitoring would you add to track custom loss behavior in production?
11. How would you A/B test a custom loss against standard losses without affecting user experience?
12. Design a rollback strategy if the custom loss causes training instability

**Performance Optimization:**
13. Identify computational bottlenecks in your chosen custom loss implementation
14. How could you vectorize operations to improve batch processing efficiency?
15. What caching strategies could reduce redundant computations?
16. How would you benchmark training speed impact compared to standard losses?

**Business Validation Framework:**
17. Design metrics to validate that your custom loss actually improves business objectives
18. How would you separate loss function improvements from other training improvements?
19. What offline evaluation would you perform before deploying the custom loss?
20. How would you monitor for unexpected business metric changes after deployment?

Implement one optimization for your chosen custom loss and explain how it addresses a specific production challenge.
"""

# %% [markdown]
"""
## TARGET MODULE SUMMARY: Loss Functions - Learning Objectives Made Mathematical

Congratulations! You've successfully implemented the complete foundation for neural network training objectives:

### What You've Accomplished
PASS **Complete Loss Function Library**: MSE for regression, CrossEntropy for multi-class classification, and Binary CrossEntropy for binary classification with production-grade numerical stability
PASS **Systems Engineering Understanding**: Deep comprehension of computational complexity, memory scaling, and numerical stability requirements for reliable ML systems
PASS **Mathematical Implementation Mastery**: Built loss functions from mathematical foundations through stable computational formulations to working code
PASS **Production Readiness Knowledge**: Understanding of how loss function choice affects training speed, memory usage, and deployment feasibility
PASS **Framework Integration Insight**: Clear connection between your implementations and how PyTorch/TensorFlow solve the same problems

### Key Learning Outcomes
- **Loss Function Theory**: How mathematical loss functions translate business objectives into optimization targets that neural networks can learn from
- **Numerical Stability Engineering**: Critical importance of stable implementations that prevent catastrophic training failures in production systems
- **Systems Performance Analysis**: Understanding of computational complexity, memory scaling, and performance trade-offs that affect production deployment
- **Production ML Patterns**: Knowledge of how loss function choice affects system architecture, monitoring requirements, and debugging complexity

### Mathematical Foundations Mastered  
- **MSE computation**: `(1/n) * Sum(y_pred - y_true)²` with smooth quadratic gradients for regression optimization
- **CrossEntropy with stable softmax**: Log-sum-exp trick and epsilon clipping for numerically robust classification
- **Binary CrossEntropy stability**: `max(x,0) - x*y + log(1 + exp(-|x|))` formulation preventing overflow/underflow issues
- **Gradient behavior understanding**: How different loss functions create different optimization landscapes and learning dynamics

### Professional Skills Developed
- **Production-quality implementation**: Robust numerical stability measures that prevent training failures with real-world data
- **Performance optimization**: Understanding of computational and memory complexity that affects scalability and deployment
- **Systems debugging**: Knowledge of how to identify and fix numerical stability issues in production ML systems
- **Framework integration**: Clear understanding of how your implementations connect to professional ML development workflows

### Ready for Advanced Applications
Your loss function implementations now enable:
- **Complete training loops** that optimize neural networks on real datasets with proper convergence monitoring
- **Custom loss functions** that align with specific business objectives and domain requirements
- **Production deployment** with confidence in numerical stability and performance characteristics
- **Advanced optimization** techniques that build on solid loss function foundations

### Connection to Real ML Systems
Your implementations mirror the essential patterns used in:
- **PyTorch's loss functions**: Same mathematical formulations with identical numerical stability measures
- **TensorFlow's losses**: Equivalent computational patterns and production-grade error handling
- **Production ML pipelines**: The exact loss functions that power real ML systems at companies like Google, Meta, and OpenAI
- **Research frameworks**: Foundation for experimenting with novel loss functions and training objectives

### Next Steps
With solid loss function implementations, you're ready to:
1. **Export your module**: `tito module complete 04_losses`
2. **Validate integration**: `tito test --module losses`
3. **Explore autograd integration**: See how loss functions connect with automatic differentiation
4. **Ready for Module 06**: Build automatic gradient computation that makes loss-based learning possible!

**Your achievement**: You've built the mathematical foundation that transforms predictions into learning signals - the critical bridge between model outputs and optimization objectives that makes neural network training possible!
"""

# %% nbgrader={"grade": false, "grade_id": "final-demo", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    print("FIRE TinyTorch Loss Functions Module - Complete Demo")
    print("=" * 55)
    
    # Test all core implementations
    print("\nTEST Testing All Loss Functions:")
    test_unit_mse_loss()
    test_unit_crossentropy_loss()
    test_unit_binary_crossentropy_loss()
    test_unit_comprehensive_loss_integration()
    
    # Run systems analysis functions
    print("\n" + "="*60)
    print("MAGNIFY Systems Analysis Functions")
    print("=" * 30)

    visualize_loss_landscapes()
    analyze_mse_properties()
    analyze_crossentropy_stability()
    analyze_binary_crossentropy_efficiency()
    analyze_mixed_precision_considerations()
    analyze_loss_performance_characteristics()
    analyze_numerical_stability_edge_cases()
    analyze_production_deployment_patterns()
    
    print("\n" + "="*60)
    print("📊 Loss Function Usage Examples")
    print("=" * 35)
    
    # Example 1: Regression with MSE
    print("\n1. Regression Example (Predicting House Prices):")
    mse = MeanSquaredError()
    house_predictions = Tensor([[250000, 180000, 320000]])  # Predicted prices
    house_actual = Tensor([[240000, 175000, 315000]])       # Actual prices
    regression_loss = mse(house_predictions, house_actual)
    print(f"   House price prediction loss: ${regression_loss.data:,.0f}² average error")
    
    # Example 2: Multi-class classification with CrossEntropy
    print("\n2. Multi-Class Classification Example (Image Recognition):")
    ce = CrossEntropyLoss()
    image_logits = Tensor([[2.1, 0.5, -0.3, 1.8, 0.1],      # Model outputs for 5 classes
                          [-0.2, 3.1, 0.8, -1.0, 0.4]])      # (cat, dog, bird, fish, rabbit)
    true_classes = Tensor([0, 1])  # First image = cat, second = dog
    classification_loss = ce(image_logits, true_classes)
    print(f"   Image classification loss: {classification_loss.data:.4f}")
    
    # Example 3: Binary classification with BCE
    print("\n3. Binary Classification Example (Spam Detection):")
    bce = BinaryCrossEntropyLoss()
    spam_logits = Tensor([[1.2], [-0.8], [2.1], [-1.5]])  # Spam prediction logits
    spam_labels = Tensor([[1.0], [0.0], [1.0], [0.0]])     # 1=spam, 0=not spam
    spam_loss = bce(spam_logits, spam_labels)
    print(f"   Spam detection loss: {spam_loss.data:.4f}")
    
    print("\n" + "="*60)
    print("TARGET Loss Function Characteristics")
    print("=" * 35)
    
    # Compare perfect vs imperfect predictions
    print("\n📊 Perfect vs Random Predictions:")
    
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
    
    print("\nCELEBRATE Complete loss function foundation ready!")
    print("   PASS MSE for regression problems")
    print("   PASS CrossEntropy for multi-class classification")
    print("   PASS Binary CrossEntropy for binary classification")
    print("   PASS Numerically stable implementations")
    print("   PASS Production-ready batch processing")
    print("   PASS Systems analysis and performance insights")
    print("   PASS Ready for neural network training!")

# %% [markdown]
"""
## CRITICAL FIX: Autograd-Integrated Loss Functions

The above implementations use basic Tensor operations without gradient tracking.
For neural network training, we need loss functions that integrate with the autograd system
to enable proper backpropagation through the computational graph.
"""

# %% nbgrader={"grade": false, "grade_id": "autograd-losses", "solution": true}
#| export
class MSELoss:
    """
    Mean Squared Error Loss - Works with both Tensors and Variables

    Initially works with basic Tensors (modules 01-04).
    Automatically upgrades to use Variables when autograd is available (module 05+).
    This staged approach allows testing loss functions before learning automatic differentiation.
    """

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def __call__(self, predictions, targets):
        """
        Compute MSE loss.

        Args:
            predictions: Model predictions (Tensor/Variable)
            targets: True targets (Tensor/Variable)

        Returns:
            Scalar loss value (Tensor initially, Variable after autograd)
        """
        # Clean Tensor Evolution Pattern:
        # - Modules 01-04: Use basic Tensor operations
        # - Module 05+: Same operations become autograd-capable automatically

        # Ensure inputs are Tensors
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Compute MSE using clean Tensor operations
        diff = predictions - targets  # Uses Tensor.__sub__
        squared_diff = diff * diff      # Uses Tensor.__mul__

        # Use numpy for mean calculation (will be enhanced in autograd)
        # Access the underlying numpy data for aggregation
        mean_loss = Tensor(np.mean(squared_diff.data))

        return mean_loss

#| export
class CrossEntropyLoss:
    """
    Cross-Entropy Loss - Works with both Tensors and Variables

    Initially works with basic Tensors (modules 01-04).
    Automatically upgrades to use Variables when autograd is available (module 05+).
    This staged approach allows testing loss functions before learning automatic differentiation.
    """

    def __init__(self):
        """Initialize CrossEntropy loss function."""
        self.epsilon = 1e-7  # For numerical stability

    def __call__(self, predictions, targets):
        """
        Compute cross-entropy loss.

        Args:
            predictions: Model predictions/logits (Tensor/Variable)
            targets: True class indices (Tensor/Variable or numpy array)

        Returns:
            Scalar loss value (Tensor initially, Variable after autograd)
        """
        # Clean Tensor Evolution Pattern: Extract data cleanly
        # Ensure inputs are Tensors and get their data
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        pred_data = predictions.data
        target_data = targets.data

        # Apply softmax to predictions (numerically stable)
        exp_pred = np.exp(pred_data - np.max(pred_data, axis=-1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

        # Clip for numerical stability
        softmax_pred = np.clip(softmax_pred, self.epsilon, 1 - self.epsilon)

        # Compute cross-entropy loss
        if len(target_data.shape) == 1 or target_data.shape[-1] == 1:
            # Integer labels
            batch_size = pred_data.shape[0]
            loss = 0
            for i in range(batch_size):
                label = int(target_data[i])
                loss -= np.log(softmax_pred[i, label])
            loss /= batch_size
        else:
            # One-hot labels
            loss = -np.mean(np.sum(target_data * np.log(softmax_pred), axis=-1))

        # Pure tensor evolution - gradient tracking will be added via decorator in Module 05
        return Tensor(loss)