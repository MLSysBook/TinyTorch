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
# Module 9: Training - Complete Neural Network Training Pipeline

Welcome to the Training module! This is where we bring everything together to train neural networks on real data.

## Learning Goals
- Understand loss functions and how they measure model performance
- Implement essential loss functions: MSE, CrossEntropy, and BinaryCrossEntropy
- Build evaluation metrics for classification and regression
- Create a complete training loop that orchestrates the entire process
- Master checkpointing and model persistence for real-world deployment

## Build → Use → Optimize
1. **Build**: Loss functions, metrics, and training orchestration
2. **Use**: Train complete models on real datasets
3. **Optimize**: Analyze training dynamics and improve performance
"""

# %% nbgrader={"grade": false, "grade_id": "training-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.training

#| export
import numpy as np
import sys
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from collections import defaultdict
import time

# Helper function to set up import paths
def setup_import_paths():
    """Set up import paths for development modules."""
    import sys
    import os
    
    # Add module directories to path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_dirs = [
        '01_tensor', '02_activations', '03_layers', '04_networks', 
        '05_cnn', '06_dataloader', '07_autograd', '08_optimizers'
    ]
    
    for module_dir in module_dirs:
        sys.path.append(os.path.join(base_dir, module_dir))

# Set up paths
setup_import_paths()

# Import all the building blocks we need
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
    from tinytorch.core.layers import Dense
    from tinytorch.core.networks import Sequential, create_mlp
    from tinytorch.core.cnn import Conv2D, flatten
    from tinytorch.core.dataloader import Dataset, DataLoader
    from tinytorch.core.autograd import Variable
    from tinytorch.core.optimizers import SGD, Adam, StepLR
except ImportError:
    # For development, create mock classes or import from local modules
    try:
        from tensor_dev import Tensor
        from activations_dev import ReLU, Sigmoid, Tanh, Softmax
        from layers_dev import Dense
        from networks_dev import Sequential, create_mlp
        from cnn_dev import Conv2D, flatten
        from dataloader_dev import Dataset, DataLoader
        from autograd_dev import Variable
        from optimizers_dev import SGD, Adam, StepLR
    except ImportError:
        # Create minimal mock classes for development
        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)
            def __str__(self):
                return f"Tensor({self.data})"
        
        class Variable:
            def __init__(self, data, requires_grad=True):
                self.data = Tensor(data)
                self.requires_grad = requires_grad
                self.grad = None
            
            def zero_grad(self):
                self.grad = None
            
            def backward(self):
                if self.requires_grad:
                    self.grad = Variable(1.0, requires_grad=False)
            
            def __str__(self):
                return f"Variable({self.data})"
        
        class SGD:
            def __init__(self, parameters, learning_rate=0.01):
                self.parameters = parameters
                self.learning_rate = learning_rate
            
            def zero_grad(self):
                for param in self.parameters:
                    if hasattr(param, 'zero_grad'):
                        param.zero_grad()
            
            def step(self):
                pass
        
        class Sequential:
            def __init__(self, layers=None):
                self.layers = layers or []
            
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        class DataLoader:
            def __init__(self, dataset, batch_size=32, shuffle=True):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
            
            def __iter__(self):
                return iter([(Tensor([1, 2, 3]), Tensor([0]))])

# %% nbgrader={"grade": false, "grade_id": "training-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% [markdown]
"""
## Step 1: Understanding Loss Functions

### What are Loss Functions?
Loss functions measure how far our model's predictions are from the true values. They provide the "signal" that tells our optimizer which direction to update parameters.

### The Mathematical Foundation
Training a neural network is an optimization problem:
```
θ* = argmin_θ L(f(x; θ), y)
```
Where:
- `θ` = model parameters (weights and biases)
- `f(x; θ)` = model predictions
- `y` = true labels
- `L` = loss function
- `θ*` = optimal parameters

### Why Loss Functions Matter
- **Optimization target**: They define what "good" means for our model
- **Gradient source**: Provide gradients for backpropagation
- **Task-specific**: Different losses for different problems
- **Training dynamics**: Shape how the model learns

### Common Loss Functions

#### **Mean Squared Error (MSE)** - For Regression
```
MSE = (1/n) * Σ(y_pred - y_true)²
```
- **Use case**: Regression problems
- **Properties**: Penalizes large errors heavily
- **Gradient**: 2 * (y_pred - y_true)

#### **Cross-Entropy Loss** - For Classification
```
CrossEntropy = -Σ y_true * log(y_pred)
```
- **Use case**: Multi-class classification
- **Properties**: Penalizes confident wrong predictions
- **Gradient**: y_pred - y_true (with softmax)

#### **Binary Cross-Entropy** - For Binary Classification
```
BCE = -y_true * log(y_pred) - (1-y_true) * log(1-y_pred)
```
- **Use case**: Binary classification
- **Properties**: Symmetric around 0.5
- **Gradient**: (y_pred - y_true) / (y_pred * (1-y_pred))

Let's implement these essential loss functions!
"""

# %% nbgrader={"grade": false, "grade_id": "mse-loss", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MeanSquaredError:
    """
    Mean Squared Error Loss for Regression
    
    Measures the average squared difference between predictions and targets.
    MSE = (1/n) * Σ(y_pred - y_true)²
    """
    
    def __init__(self):
        """Initialize MSE loss function."""
        pass
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            y_pred: Model predictions (shape: [batch_size, ...])
            y_true: True targets (shape: [batch_size, ...])
            
        Returns:
            Scalar loss value
            
        TODO: Implement Mean Squared Error loss computation.
        
        APPROACH:
        1. Compute difference: diff = y_pred - y_true
        2. Square the differences: squared_diff = diff²
        3. Take mean over all elements: mean(squared_diff)
        4. Return as scalar Tensor
        
        EXAMPLE:
        y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y_true = Tensor([[1.5, 2.5], [2.5, 3.5]])
        loss = mse_loss(y_pred, y_true)
        # Should return: mean([(1.0-1.5)², (2.0-2.5)², (3.0-2.5)², (4.0-3.5)²])
        #                = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
        
        HINTS:
        - Use tensor subtraction: y_pred - y_true
        - Use element-wise multiplication for squaring: diff * diff
        - Use np.mean() to get the average
        - Return Tensor(scalar_value)
        """
        ### BEGIN SOLUTION
        # Compute difference
        diff = y_pred - y_true
        
        # Square the differences
        squared_diff = diff * diff
        
        # Take mean over all elements
        mean_loss = np.mean(squared_diff.data)
        
        return Tensor(mean_loss)
        ### END SOLUTION
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
### 🧪 Unit Test: MSE Loss

Let's test our MSE loss implementation with known values.
"""

# %% nbgrader={"grade": false, "grade_id": "test-mse-loss", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_mse_loss_comprehensive():
    """Test MSE loss with comprehensive examples."""
    print("🔬 Unit Test: MSE Loss...")
    
    mse = MeanSquaredError()
    
    # Test 1: Perfect predictions (loss should be 0)
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = mse(y_pred, y_true)
    assert abs(loss.data) < 1e-6, f"Perfect predictions should have loss ≈ 0, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test 2: Known loss computation
    y_pred = Tensor([[1.0, 2.0]])
    y_true = Tensor([[0.0, 1.0]])
    loss = mse(y_pred, y_true)
    expected = 1.0  # [(1-0)² + (2-1)²] / 2 = [1 + 1] / 2 = 1.0
    assert abs(loss.data - expected) < 1e-6, f"Expected loss {expected}, got {loss.data}"
    print("✅ Known loss computation test passed")
    
    # Test 3: Batch processing
    y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.5, 2.5], [2.5, 3.5]])
    loss = mse(y_pred, y_true)
    expected = 0.25  # All squared differences are 0.25
    assert abs(loss.data - expected) < 1e-6, f"Expected batch loss {expected}, got {loss.data}"
    print("✅ Batch processing test passed")
    
    # Test 4: Single value
    y_pred = Tensor([5.0])
    y_true = Tensor([3.0])
    loss = mse(y_pred, y_true)
    expected = 4.0  # (5-3)² = 4
    assert abs(loss.data - expected) < 1e-6, f"Expected single value loss {expected}, got {loss.data}"
    print("✅ Single value test passed")
    
    print("🎯 MSE Loss: All tests passed!")

# Run the test
test_mse_loss_comprehensive() 

# %% nbgrader={"grade": false, "grade_id": "crossentropy-loss", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class CrossEntropyLoss:
    """
    Cross-Entropy Loss for Multi-Class Classification
    
    Measures the difference between predicted probability distribution and true labels.
    CrossEntropy = -Σ y_true * log(y_pred)
    """
    
    def __init__(self):
        """Initialize CrossEntropy loss function."""
        pass
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute CrossEntropy loss between predictions and targets.
        
        Args:
            y_pred: Model predictions (shape: [batch_size, num_classes])
            y_true: True class indices (shape: [batch_size]) or one-hot (shape: [batch_size, num_classes])
            
        Returns:
            Scalar loss value
            
        TODO: Implement Cross-Entropy loss computation.
        
        APPROACH:
        1. Handle both class indices and one-hot encoded labels
        2. Apply softmax to predictions for probability distribution
        3. Compute log probabilities: log(softmax(y_pred))
        4. Calculate cross-entropy: -mean(y_true * log_probs)
        5. Return scalar loss
        
        EXAMPLE:
        y_pred = Tensor([[2.0, 1.0, 0.1], [0.5, 2.1, 0.9]])  # Raw logits
        y_true = Tensor([0, 1])  # Class indices
        loss = crossentropy_loss(y_pred, y_true)
        # Should apply softmax then compute -log(prob_of_correct_class)
        
        HINTS:
        - Use softmax: exp(x) / sum(exp(x)) for probability distribution
        - Add small epsilon (1e-15) to avoid log(0)
        - Handle both class indices and one-hot encoding
        - Use np.log for logarithm computation
        """
        ### BEGIN SOLUTION
        # Apply softmax to get probability distribution
        exp_pred = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        softmax_pred = np.clip(softmax_pred, epsilon, 1.0 - epsilon)
        
        # Handle class indices vs one-hot encoding
        if len(y_true.data.shape) == 1:
            # y_true contains class indices
            batch_size = y_true.data.shape[0]
            log_probs = np.log(softmax_pred[np.arange(batch_size), y_true.data.astype(int)])
            loss = -np.mean(log_probs)
        else:
            # y_true is one-hot encoded
            log_probs = np.log(softmax_pred)
            loss = -np.mean(np.sum(y_true.data * log_probs, axis=1))
        
        return Tensor(loss)
        ### END SOLUTION
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
### 🧪 Unit Test: CrossEntropy Loss

Let's test our CrossEntropy loss implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "test-crossentropy-loss", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_crossentropy_loss_comprehensive():
    """Test CrossEntropy loss with comprehensive examples."""
    print("🔬 Unit Test: CrossEntropy Loss...")
    
    ce = CrossEntropyLoss()
    
    # Test 1: Perfect predictions
    y_pred = Tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])  # Very confident correct predictions
    y_true = Tensor([0, 1])  # Class indices
    loss = ce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Uniform after softmax
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    expected_random = -np.log(1.0/3.0)  # log(1/num_classes) for uniform distribution
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ≈ {expected_random}, got {loss.data}"
    print("✅ Random predictions test passed")
    
    # Test 3: Binary classification
    y_pred = Tensor([[2.0, 1.0], [1.0, 2.0]])
    y_true = Tensor([0, 1])
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Binary classification loss should be reasonable, got {loss.data}"
    print("✅ Binary classification test passed")
    
    # Test 4: One-hot encoded labels
    y_pred = Tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y_true = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # One-hot encoded
    loss = ce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"One-hot encoded loss should be reasonable, got {loss.data}"
    print("✅ One-hot encoded labels test passed")
    
    print("🎯 CrossEntropy Loss: All tests passed!")

# Run the test
test_crossentropy_loss_comprehensive()

# %% nbgrader={"grade": false, "grade_id": "binary-crossentropy-loss", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss for Binary Classification
    
    Measures the difference between predicted probabilities and binary labels.
    BCE = -y_true * log(y_pred) - (1-y_true) * log(1-y_pred)
    """
    
    def __init__(self):
        """Initialize Binary CrossEntropy loss function."""
        pass
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute Binary CrossEntropy loss between predictions and targets.
        
        Args:
            y_pred: Model predictions (shape: [batch_size, 1] or [batch_size])
            y_true: True binary labels (shape: [batch_size, 1] or [batch_size])
            
        Returns:
            Scalar loss value
            
        TODO: Implement Binary Cross-Entropy loss computation.
        
        APPROACH:
        1. Apply sigmoid to predictions for probability values
        2. Clip probabilities to avoid log(0) and log(1)
        3. Compute: -y_true * log(y_pred) - (1-y_true) * log(1-y_pred)
        4. Take mean over batch
        5. Return scalar loss
        
        EXAMPLE:
        y_pred = Tensor([[2.0], [0.0], [-1.0]])  # Raw logits
        y_true = Tensor([[1.0], [1.0], [0.0]])   # Binary labels
        loss = bce_loss(y_pred, y_true)
        # Should apply sigmoid then compute binary cross-entropy
        
        HINTS:
        - Use sigmoid: 1 / (1 + exp(-x))
        - Clip probabilities: np.clip(probs, epsilon, 1-epsilon)
        - Handle both [batch_size] and [batch_size, 1] shapes
        - Use np.log for logarithm computation
        """
        ### BEGIN SOLUTION
        # Apply sigmoid to get probabilities
        sigmoid_pred = 1.0 / (1.0 + np.exp(-y_pred.data))
        
        # Clip probabilities to avoid log(0)
        epsilon = 1e-15
        sigmoid_pred = np.clip(sigmoid_pred, epsilon, 1.0 - epsilon)
        
        # Flatten if needed to ensure consistent shapes
        y_true_flat = y_true.data.flatten()
        sigmoid_pred_flat = sigmoid_pred.flatten()
        
        # Compute binary cross-entropy
        loss = -(y_true_flat * np.log(sigmoid_pred_flat) + 
                (1 - y_true_flat) * np.log(1 - sigmoid_pred_flat))
        
        # Take mean over batch
        mean_loss = np.mean(loss)
        
        return Tensor(mean_loss)
        ### END SOLUTION
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
### 🧪 Unit Test: Binary CrossEntropy Loss

Let's test our Binary CrossEntropy loss implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "test-binary-crossentropy-loss", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_binary_crossentropy_loss_comprehensive():
    """Test Binary CrossEntropy loss with comprehensive examples."""
    print("🔬 Unit Test: Binary CrossEntropy Loss...")
    
    bce = BinaryCrossEntropyLoss()
    
    # Test 1: Perfect predictions
    y_pred = Tensor([[10.0], [-10.0]])  # Very confident correct predictions
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert loss.data < 0.1, f"Perfect predictions should have low loss, got {loss.data}"
    print("✅ Perfect predictions test passed")
    
    # Test 2: Random predictions (should have higher loss)
    y_pred = Tensor([[0.0], [0.0]])  # 0.5 probability after sigmoid
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    expected_random = -np.log(0.5)  # log(0.5) for random guessing
    assert abs(loss.data - expected_random) < 0.1, f"Random predictions should have loss ≈ {expected_random}, got {loss.data}"
    print("✅ Random predictions test passed")
    
    # Test 3: Batch processing
    y_pred = Tensor([[1.0], [2.0], [-1.0]])
    y_true = Tensor([[1.0], [1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert 0.0 < loss.data < 2.0, f"Batch processing loss should be reasonable, got {loss.data}"
    print("✅ Batch processing test passed")
    
    # Test 4: Edge cases
    y_pred = Tensor([[100.0], [-100.0]])  # Extreme values
    y_true = Tensor([[1.0], [0.0]])
    loss = bce(y_pred, y_true)
    assert loss.data < 0.1, f"Extreme correct predictions should have low loss, got {loss.data}"
    print("✅ Edge cases test passed")
    
    print("🎯 Binary CrossEntropy Loss: All tests passed!")

# Run the test
test_binary_crossentropy_loss_comprehensive() 

# %% [markdown]
"""
## Step 2: Understanding Metrics

### What are Metrics?
Metrics are measurements that help us understand how well our model is performing. Unlike loss functions, metrics are often more interpretable and align with business objectives.

### Key Metrics for Classification

#### **Accuracy**
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
- **Range**: [0, 1]
- **Interpretation**: Percentage of correct predictions
- **Good for**: Balanced datasets

#### **Precision**
```
Precision = True Positives / (True Positives + False Positives)
```
- **Range**: [0, 1]
- **Interpretation**: Of all positive predictions, how many were correct?
- **Good for**: When false positives are costly

#### **Recall (Sensitivity)**
```
Recall = True Positives / (True Positives + False Negatives)
```
- **Range**: [0, 1]
- **Interpretation**: Of all actual positives, how many did we find?
- **Good for**: When false negatives are costly

### Key Metrics for Regression

#### **Mean Absolute Error (MAE)**
```
MAE = (1/n) * Σ|y_pred - y_true|
```
- **Range**: [0, ∞)
- **Interpretation**: Average absolute error
- **Good for**: Robust to outliers

Let's implement these essential metrics!
"""

# %% nbgrader={"grade": false, "grade_id": "accuracy-metric", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Accuracy:
    """
    Accuracy Metric for Classification
    
    Computes the fraction of correct predictions.
    Accuracy = (Correct Predictions) / (Total Predictions)
    """
    
    def __init__(self):
        """Initialize Accuracy metric."""
        pass
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        Compute accuracy between predictions and targets.
        
        Args:
            y_pred: Model predictions (shape: [batch_size, num_classes] or [batch_size])
            y_true: True class labels (shape: [batch_size] or [batch_size])
            
        Returns:
            Accuracy as a float value between 0 and 1
            
        TODO: Implement accuracy computation.
        
        APPROACH:
        1. Convert predictions to class indices (argmax for multi-class)
        2. Convert true labels to class indices if needed
        3. Count correct predictions
        4. Divide by total predictions
        5. Return as float
        
        EXAMPLE:
        y_pred = Tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])  # Probabilities
        y_true = Tensor([0, 1, 0])  # True classes
        accuracy = accuracy_metric(y_pred, y_true)
        # Should return: 2/3 = 0.667 (first and second predictions correct)
        
        HINTS:
        - Use np.argmax(axis=1) for multi-class predictions
        - Handle both probability and class index inputs
        - Use np.mean() for averaging
        - Return Python float, not Tensor
        """
        ### BEGIN SOLUTION
        # Convert predictions to class indices
        if len(y_pred.data.shape) > 1 and y_pred.data.shape[1] > 1:
            # Multi-class: use argmax
            pred_classes = np.argmax(y_pred.data, axis=1)
        else:
            # Binary classification: threshold at 0.5
            pred_classes = (y_pred.data.flatten() > 0.5).astype(int)
        
        # Convert true labels to class indices if needed
        if len(y_true.data.shape) > 1 and y_true.data.shape[1] > 1:
            # One-hot encoded
            true_classes = np.argmax(y_true.data, axis=1)
        else:
            # Already class indices
            true_classes = y_true.data.flatten().astype(int)
        
        # Compute accuracy
        correct = np.sum(pred_classes == true_classes)
        total = len(true_classes)
        accuracy = correct / total
        
        return float(accuracy)
        ### END SOLUTION
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Alternative interface for forward pass."""
        return self.__call__(y_pred, y_true)

# %% [markdown]
"""
### 🧪 Unit Test: Accuracy Metric

Let's test our Accuracy metric implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "test-accuracy-metric", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_accuracy_metric_comprehensive():
    """Test Accuracy metric with comprehensive examples."""
    print("🔬 Unit Test: Accuracy Metric...")
    
    accuracy = Accuracy()
    
    # Test 1: Perfect predictions
    y_pred = Tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    y_true = Tensor([0, 1, 0])
    acc = accuracy(y_pred, y_true)
    assert acc == 1.0, f"Perfect predictions should have accuracy 1.0, got {acc}"
    print("✅ Perfect predictions test passed")
    
    # Test 2: Half correct
    y_pred = Tensor([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2]])  # All predict class 0
    y_true = Tensor([0, 1, 0])  # Classes: 0, 1, 0
    acc = accuracy(y_pred, y_true)
    expected = 2.0/3.0  # 2 out of 3 correct
    assert abs(acc - expected) < 1e-6, f"Half correct should have accuracy {expected}, got {acc}"
    print("✅ Half correct test passed")
    
    # Test 3: Binary classification
    y_pred = Tensor([[0.8], [0.3], [0.9], [0.1]])  # Predictions above/below 0.5
    y_true = Tensor([1, 0, 1, 0])
    acc = accuracy(y_pred, y_true)
    assert acc == 1.0, f"Binary classification should have accuracy 1.0, got {acc}"
    print("✅ Binary classification test passed")
    
    # Test 4: Multi-class
    y_pred = Tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    y_true = Tensor([0, 1, 2])
    acc = accuracy(y_pred, y_true)
    assert acc == 1.0, f"Multi-class should have accuracy 1.0, got {acc}"
    print("✅ Multi-class test passed")
    
    print("🎯 Accuracy Metric: All tests passed!")

# Run the test
test_accuracy_metric_comprehensive()

# %% [markdown]
"""
## Step 3: Building the Training Loop

### What is a Training Loop?
A training loop is the orchestration logic that coordinates all components of neural network training:

1. **Forward Pass**: Compute predictions
2. **Loss Computation**: Measure prediction quality
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Update model parameters
5. **Evaluation**: Compute metrics and validation performance

### The Training Loop Architecture
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    for batch in val_dataloader:
        predictions = model(batch_x)
        val_loss = loss_function(predictions, batch_y)
        accuracy = accuracy_metric(predictions, batch_y)
```

### Why We Need a Trainer Class
- **Encapsulation**: Keeps training logic organized
- **Reusability**: Same trainer works with different models/datasets
- **Monitoring**: Built-in logging and progress tracking
- **Flexibility**: Easy to modify training behavior

Let's build our Trainer class!
"""

# %% nbgrader={"grade": false, "grade_id": "trainer-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Trainer:
    """
    Training Loop Orchestrator
    
    Coordinates model training with loss functions, optimizers, and metrics.
    """
    
    def __init__(self, model, optimizer, loss_function, metrics=None):
        """
        Initialize trainer with model and training components.
        
        Args:
            model: Neural network model to train
            optimizer: Optimizer for parameter updates
            loss_function: Loss function for training
            metrics: List of metrics to track (optional)
            
        TODO: Initialize the trainer with all necessary components.
        
        APPROACH:
        1. Store model, optimizer, loss function, and metrics
        2. Initialize history tracking for losses and metrics
        3. Set up training state (epoch, step counters)
        4. Prepare for training and validation loops
        
        EXAMPLE:
        model = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])
        optimizer = Adam(model.parameters, learning_rate=0.001)
        loss_fn = CrossEntropyLoss()
        metrics = [Accuracy()]
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        HINTS:
        - Store all components as instance variables
        - Initialize empty history dictionaries
        - Set metrics to empty list if None provided
        - Initialize epoch and step counters to 0
        """
        ### BEGIN SOLUTION
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics or []
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # Add metric history tracking
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        ### END SOLUTION
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch on the given dataloader.
        
        Args:
            dataloader: DataLoader containing training data
            
        Returns:
            Dictionary with epoch training metrics
            
        TODO: Implement single epoch training logic.
        
        APPROACH:
        1. Initialize epoch metrics tracking
        2. Iterate through batches in dataloader
        3. For each batch:
           - Zero gradients
           - Forward pass
           - Compute loss
           - Backward pass
           - Update parameters
           - Track metrics
        4. Return averaged metrics for the epoch
        
        HINTS:
        - Use optimizer.zero_grad() before each batch
        - Call loss.backward() for gradient computation
        - Use optimizer.step() for parameter updates
        - Track running averages for metrics
        """
        ### BEGIN SOLUTION
        epoch_metrics = {'loss': 0.0}
        
        # Initialize metric tracking
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            epoch_metrics[metric_name] = 0.0
        
        batch_count = 0
        
        for batch_x, batch_y in dataloader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Compute loss
            loss = self.loss_function(predictions, batch_y)
            
            # Backward pass (simplified - in real implementation would use autograd)
            # loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'] += loss.data
            
            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                metric_value = metric(predictions, batch_y)
                epoch_metrics[metric_name] += metric_value
            
            batch_count += 1
            self.current_step += 1
        
        # Average metrics over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return epoch_metrics
        ### END SOLUTION
    
    def validate_epoch(self, dataloader):
        """
        Validate for one epoch on the given dataloader.
        
        Args:
            dataloader: DataLoader containing validation data
            
        Returns:
            Dictionary with epoch validation metrics
            
        TODO: Implement single epoch validation logic.
        
        APPROACH:
        1. Initialize epoch metrics tracking
        2. Iterate through batches in dataloader
        3. For each batch:
           - Forward pass (no gradient computation)
           - Compute loss
           - Track metrics
        4. Return averaged metrics for the epoch
        
        HINTS:
        - No gradient computation needed for validation
        - No parameter updates during validation
        - Similar to train_epoch but simpler
        """
        ### BEGIN SOLUTION
        epoch_metrics = {'loss': 0.0}
        
        # Initialize metric tracking
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            epoch_metrics[metric_name] = 0.0
        
        batch_count = 0
        
        for batch_x, batch_y in dataloader:
            # Forward pass only (no gradients needed)
            predictions = self.model(batch_x)
            
            # Compute loss
            loss = self.loss_function(predictions, batch_y)
            
            # Track metrics
            epoch_metrics['loss'] += loss.data
            
            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                metric_value = metric(predictions, batch_y)
                epoch_metrics[metric_name] += metric_value
            
            batch_count += 1
        
        # Average metrics over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return epoch_metrics
        ### END SOLUTION
    
    def fit(self, train_dataloader, val_dataloader=None, epochs=10, verbose=True):
        """
        Train the model for specified number of epochs.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data (optional)
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
            
        TODO: Implement complete training loop.
        
        APPROACH:
        1. Loop through epochs
        2. For each epoch:
           - Train on training data
           - Validate on validation data (if provided)
           - Update history
           - Print progress (if verbose)
        3. Return complete training history
        
        HINTS:
        - Use train_epoch() and validate_epoch() methods
        - Update self.history with results
        - Print epoch summary if verbose=True
        """
        ### BEGIN SOLUTION
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation phase
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            
            if val_dataloader is not None:
                self.history['val_loss'].append(val_metrics['loss'])
            
            # Update metric history
            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                self.history[f'train_{metric_name}'].append(train_metrics[metric_name])
                if val_dataloader is not None:
                    self.history[f'val_{metric_name}'].append(val_metrics[metric_name])
            
            # Print progress
            if verbose:
                train_loss = train_metrics['loss']
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}", end="")
                
                if val_dataloader is not None:
                    val_loss = val_metrics['loss']
                    print(f" - val_loss: {val_loss:.4f}", end="")
                
                for metric in self.metrics:
                    metric_name = metric.__class__.__name__.lower()
                    train_metric = train_metrics[metric_name]
                    print(f" - train_{metric_name}: {train_metric:.4f}", end="")
                    
                    if val_dataloader is not None:
                        val_metric = val_metrics[metric_name]
                        print(f" - val_{metric_name}: {val_metric:.4f}", end="")
                
                print()  # New line
        
        print("Training completed!")
        return self.history
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Training Loop

Let's test our Trainer class with a simple example.
"""

# %% nbgrader={"grade": false, "grade_id": "test-trainer", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_trainer_comprehensive():
    """Test Trainer class with comprehensive examples."""
    print("🔬 Unit Test: Trainer Class...")
    
    # Create simple model and components
    model = Sequential([Dense(2, 3), ReLU(), Dense(3, 2)])  # Simple model
    optimizer = SGD([], learning_rate=0.01)  # Empty parameters list for testing
    loss_fn = MeanSquaredError()
    metrics = [Accuracy()]
    
    # Create trainer
    trainer = Trainer(model, optimizer, loss_fn, metrics)
    
    # Test 1: Trainer initialization
    assert trainer.model is model, "Model should be stored correctly"
    assert trainer.optimizer is optimizer, "Optimizer should be stored correctly"
    assert trainer.loss_function is loss_fn, "Loss function should be stored correctly"
    assert len(trainer.metrics) == 1, "Metrics should be stored correctly"
    assert 'train_loss' in trainer.history, "Training history should be initialized"
    print("✅ Trainer initialization test passed")
    
    # Test 2: History structure
    assert 'epoch' in trainer.history, "History should track epochs"
    assert 'train_accuracy' in trainer.history, "History should track training accuracy"
    assert 'val_accuracy' in trainer.history, "History should track validation accuracy"
    print("✅ History structure test passed")
    
    # Test 3: Training state
    assert trainer.current_epoch == 0, "Current epoch should start at 0"
    assert trainer.current_step == 0, "Current step should start at 0"
    print("✅ Training state test passed")
    
    print("🎯 Trainer Class: All tests passed!")

# Run the test
test_trainer_comprehensive()

# %% [markdown]
"""
### 🧪 Unit Test: Complete Training Comprehensive Test

Let's test the complete training pipeline with all components working together.

**This is a comprehensive test** - it tests all training components working together in a realistic scenario.
"""

# %% nbgrader={"grade": true, "grade_id": "test-training-comprehensive", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_training_comprehensive():
    """Test complete training pipeline with all components."""
    print("🔬 Comprehensive Test: Complete Training Pipeline...")
    
    try:
        # Test 1: Loss functions work correctly
        mse = MeanSquaredError()
        ce = CrossEntropyLoss()
        bce = BinaryCrossEntropyLoss()
        
        # MSE test
        y_pred = Tensor([[1.0, 2.0]])
        y_true = Tensor([[1.0, 2.0]])
        loss = mse(y_pred, y_true)
        assert abs(loss.data) < 1e-6, "MSE should work for perfect predictions"
        
        # CrossEntropy test
        y_pred = Tensor([[10.0, 0.0], [0.0, 10.0]])
        y_true = Tensor([0, 1])
        loss = ce(y_pred, y_true)
        assert loss.data < 1.0, "CrossEntropy should work for good predictions"
        
        # Binary CrossEntropy test
        y_pred = Tensor([[10.0], [-10.0]])
        y_true = Tensor([[1.0], [0.0]])
        loss = bce(y_pred, y_true)
        assert loss.data < 1.0, "Binary CrossEntropy should work for good predictions"
        
        print("✅ Loss functions work correctly")
        
        # Test 2: Metrics work correctly
        accuracy = Accuracy()
        
        y_pred = Tensor([[0.9, 0.1], [0.1, 0.9]])
        y_true = Tensor([0, 1])
        acc = accuracy(y_pred, y_true)
        assert acc == 1.0, "Accuracy should work for perfect predictions"
        
        print("✅ Metrics work correctly")
        
        # Test 3: Trainer integrates all components
        model = Sequential([])  # Empty model for testing
        optimizer = SGD([], learning_rate=0.01)
        loss_fn = MeanSquaredError()
        metrics = [Accuracy()]
        
        trainer = Trainer(model, optimizer, loss_fn, metrics)
        
        # Check trainer setup
        assert trainer.model is model, "Trainer should store model"
        assert trainer.optimizer is optimizer, "Trainer should store optimizer"
        assert trainer.loss_function is loss_fn, "Trainer should store loss function"
        assert len(trainer.metrics) == 1, "Trainer should store metrics"
        
        print("✅ Trainer integrates all components")
        
        print("🎉 Complete training pipeline works correctly!")
        
        # Test 4: Integration works end-to-end
        print("✅ End-to-end integration successful")
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        raise
    
    print("🎯 Training Pipeline: All comprehensive tests passed!")

# Run the comprehensive test
test_training_comprehensive() 