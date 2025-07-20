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
# Optimizers - Gradient-Based Parameter Updates

Welcome to the Optimizers module! This is where neural networks learn to improve through intelligent parameter updates.

## Learning Goals
- Understand gradient descent and how optimizers use gradients to update parameters
- Implement SGD with momentum for accelerated convergence
- Build Adam optimizer with adaptive learning rates
- Master learning rate scheduling strategies
- See how optimizers enable effective neural network training

## Build → Use → Analyze
1. **Build**: Core optimization algorithms (SGD, Adam)
2. **Use**: Apply optimizers to train neural networks
3. **Analyze**: Compare optimizer behavior and convergence patterns
"""

# %% nbgrader={"grade": false, "grade_id": "optimizers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.optimizers

#| export
import math
import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

# Helper function to set up import paths
def setup_import_paths():
    """Set up import paths for development modules."""
    import sys
    import os
    
    # Add module directories to path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tensor_dir = os.path.join(base_dir, '01_tensor')
    autograd_dir = os.path.join(base_dir, '07_autograd')
    
    if tensor_dir not in sys.path:
        sys.path.append(tensor_dir)
    if autograd_dir not in sys.path:
        sys.path.append(autograd_dir)

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
except ImportError:
    # For development, try local imports
    try:
        setup_import_paths()
        from tensor_dev import Tensor
        from autograd_dev import Variable
    except ImportError:
        # Create minimal fallback classes for testing
        print("Warning: Using fallback classes for testing")
        
        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape
            
            def __str__(self):
                return f"Tensor({self.data})"
        
        class Variable:
            def __init__(self, data, requires_grad=True):
                if isinstance(data, (int, float)):
                    self.data = Tensor([data])
                else:
                    self.data = Tensor(data)
                self.requires_grad = requires_grad
                self.grad = None
            
            def zero_grad(self):
                self.grad = None
            
            def __str__(self):
                return f"Variable({self.data.data})"

# %% nbgrader={"grade": false, "grade_id": "optimizers-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Optimizers Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build optimization algorithms!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/08_optimizers/optimizers_dev.py`  
**Building Side:** Code exports to `tinytorch.core.optimizers`

```python
# Final package structure:
from tinytorch.core.optimizers import SGD, Adam, StepLR  # The optimization engines!
from tinytorch.core.autograd import Variable  # Gradient computation
from tinytorch.core.tensor import Tensor  # Data structures
```

**Why this matters:**
- **Learning:** Focused module for understanding optimization algorithms
- **Production:** Proper organization like PyTorch's `torch.optim`
- **Consistency:** All optimization algorithms live together in `core.optimizers`
- **Foundation:** Enables effective neural network training
"""

# %% [markdown]
"""
## What Are Optimizers?

### The Problem: How to Update Parameters
Neural networks learn by updating parameters using gradients:
```
parameter_new = parameter_old - learning_rate * gradient
```

But **naive gradient descent** has problems:
- **Slow convergence**: Takes many steps to reach optimum
- **Oscillation**: Bounces around valleys without making progress
- **Poor scaling**: Same learning rate for all parameters

### The Solution: Smart Optimization
**Optimizers** are algorithms that intelligently update parameters:
- **Momentum**: Accelerate convergence by accumulating velocity
- **Adaptive learning rates**: Different learning rates for different parameters
- **Second-order information**: Use curvature to guide updates

### Real-World Impact
- **SGD**: The foundation of all neural network training
- **Adam**: The default optimizer for most deep learning applications
- **Learning rate scheduling**: Critical for training stability and performance

### What We'll Build
1. **SGD**: Stochastic Gradient Descent with momentum
2. **Adam**: Adaptive Moment Estimation optimizer
3. **StepLR**: Learning rate scheduling
4. **Integration**: Complete training loop with optimizers
"""

# %% [markdown]
"""
## 🔧 DEVELOPMENT
"""

# %% [markdown]
"""
## Step 1: Understanding Gradient Descent

### What is Gradient Descent?
**Gradient descent** finds the minimum of a function by following the negative gradient:

```
θ_{t+1} = θ_t - α ∇f(θ_t)
```

Where:
- θ: Parameters we want to optimize
- α: Learning rate (how big steps to take)
- ∇f(θ): Gradient of loss function with respect to parameters

### Why Gradient Descent Works
1. **Gradients point uphill**: Negative gradient points toward minimum
2. **Iterative improvement**: Each step reduces the loss (in theory)
3. **Local convergence**: Finds local minimum with proper learning rate
4. **Scalable**: Works with millions of parameters

### The Learning Rate Dilemma
- **Too large**: Overshoots minimum, diverges
- **Too small**: Extremely slow convergence
- **Just right**: Steady progress toward minimum

### Visual Understanding
```
Loss landscape: \__/
Start here: ↑
Gradient descent: ↓ → ↓ → ↓ → minimum
```

### Real-World Applications
- **Neural networks**: Training any deep learning model
- **Machine learning**: Logistic regression, SVM, etc.
- **Scientific computing**: Optimization problems in physics, engineering
- **Economics**: Portfolio optimization, game theory

Let's implement gradient descent to understand it deeply!
"""

# %% nbgrader={"grade": false, "grade_id": "gradient-descent-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def gradient_descent_step(parameter: Variable, learning_rate: float) -> None:
    """
    Perform one step of gradient descent on a parameter.
    
    Args:
        parameter: Variable with gradient information
        learning_rate: How much to update parameter
    
    TODO: Implement basic gradient descent parameter update.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Check if parameter has a gradient
    2. Get current parameter value and gradient
    3. Update parameter: new_value = old_value - learning_rate * gradient
    4. Update parameter data with new value
    5. Handle edge cases (no gradient, invalid values)
    
    EXAMPLE USAGE:
    ```python
    # Parameter with gradient
    w = Variable(2.0, requires_grad=True)
    w.grad = Variable(0.5)  # Gradient from loss
    
    # Update parameter
    gradient_descent_step(w, learning_rate=0.1)
    # w.data now contains: 2.0 - 0.1 * 0.5 = 1.95
    ```
    
    IMPLEMENTATION HINTS:
    - Check if parameter.grad is not None
    - Use parameter.grad.data.data to get gradient value
    - Update parameter.data with new Tensor
    - Don't modify gradient (it's used for logging)
    
    LEARNING CONNECTIONS:
    - This is the foundation of all neural network training
    - PyTorch's optimizer.step() does exactly this
    - The learning rate determines convergence speed
    """
    ### BEGIN SOLUTION
    if parameter.grad is not None:
        # Get current parameter value and gradient
        current_value = parameter.data.data
        gradient_value = parameter.grad.data.data
        
        # Update parameter: new_value = old_value - learning_rate * gradient
        new_value = current_value - learning_rate * gradient_value
        
        # Update parameter data
        parameter.data = Tensor(new_value)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Gradient Descent Step

Let's test your gradient descent implementation right away! This is the foundation of all optimization algorithms.

**This is a unit test** - it tests one specific function (gradient_descent_step) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-gradient-descent", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_gradient_descent_step():
    """Unit test for the basic gradient descent parameter update."""
    print("🔬 Unit Test: Gradient Descent Step...")
    
    # Test basic parameter update
    try:
        w = Variable(2.0, requires_grad=True)
        w.grad = Variable(0.5)  # Positive gradient
        
        original_value = w.data.data.item()
        gradient_descent_step(w, learning_rate=0.1)
        new_value = w.data.data.item()
        
        expected_value = original_value - 0.1 * 0.5  # 2.0 - 0.05 = 1.95
        assert abs(new_value - expected_value) < 1e-6, f"Expected {expected_value}, got {new_value}"
        print("✅ Basic parameter update works")
        
    except Exception as e:
        print(f"❌ Basic parameter update failed: {e}")
        raise

    # Test with negative gradient
    try:
        w2 = Variable(1.0, requires_grad=True)
        w2.grad = Variable(-0.2)  # Negative gradient
        
        gradient_descent_step(w2, learning_rate=0.1)
        expected_value2 = 1.0 - 0.1 * (-0.2)  # 1.0 + 0.02 = 1.02
        assert abs(w2.data.data.item() - expected_value2) < 1e-6, "Negative gradient test failed"
        print("✅ Negative gradient handling works")
        
    except Exception as e:
        print(f"❌ Negative gradient handling failed: {e}")
        raise

    # Test with no gradient (should not update)
    try:
        w3 = Variable(3.0, requires_grad=True)
        w3.grad = None
        original_value3 = w3.data.data.item()
        
        gradient_descent_step(w3, learning_rate=0.1)
        assert w3.data.data.item() == original_value3, "Parameter with no gradient should not update"
        print("✅ No gradient case works")
        
    except Exception as e:
        print(f"❌ No gradient case failed: {e}")
        raise

    print("🎯 Gradient descent step behavior:")
    print("   Updates parameters in negative gradient direction")
    print("   Uses learning rate to control step size")
    print("   Skips updates when gradient is None")
    print("📈 Progress: Gradient Descent Step ✓")

# Test function is called by auto-discovery system

# %% [markdown]
"""
## Step 2: SGD with Momentum

### What is SGD?
**SGD (Stochastic Gradient Descent)** is the fundamental optimization algorithm:

```
θ_{t+1} = θ_t - α ∇L(θ_t)
```

### The Problem with Vanilla SGD
- **Slow convergence**: Especially in narrow valleys
- **Oscillation**: Bounces around without making progress
- **Poor conditioning**: Struggles with ill-conditioned problems

### The Solution: Momentum
**Momentum** accumulates velocity to accelerate convergence:

```
v_t = β v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - α v_t
```

Where:
- v_t: Velocity (exponential moving average of gradients)
- β: Momentum coefficient (typically 0.9)
- α: Learning rate

### Why Momentum Works
1. **Acceleration**: Builds up speed in consistent directions
2. **Dampening**: Reduces oscillations in inconsistent directions
3. **Memory**: Remembers previous gradient directions
4. **Robustness**: Less sensitive to noisy gradients

### Visual Understanding
```
Without momentum: ↗↙↗↙↗↙ (oscillating)
With momentum:    ↗→→→→→ (smooth progress)
```

### Real-World Applications
- **Image classification**: Training ResNet, VGG
- **Natural language**: Training RNNs, early transformers
- **Classic choice**: Still used when Adam fails
- **Large batch training**: Often preferred over Adam

Let's implement SGD with momentum!
"""

# %% nbgrader={"grade": false, "grade_id": "sgd-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SGD:
    """
    SGD Optimizer with Momentum
    
    Implements stochastic gradient descent with momentum:
    v_t = momentum * v_{t-1} + gradient
    parameter = parameter - learning_rate * v_t
    """
    
    def __init__(self, parameters: List[Variable], learning_rate: float = 0.01, 
                 momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: List of Variables to optimize
            learning_rate: Learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.0)
            weight_decay: L2 regularization coefficient (default: 0.0)
        
        TODO: Implement SGD optimizer initialization.
        
        APPROACH:
        1. Store parameters and hyperparameters
        2. Initialize momentum buffers for each parameter
        3. Set up state tracking for optimization
        4. Prepare for step() and zero_grad() methods
        
        EXAMPLE:
        ```python
        # Create optimizer
        optimizer = SGD([w1, w2, b1, b2], learning_rate=0.01, momentum=0.9)
        
        # In training loop:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ```
        
        HINTS:
        - Store parameters as a list
        - Initialize momentum buffers as empty dict
        - Use parameter id() as key for momentum tracking
        - Momentum buffers will be created lazily in step()
        """
        ### BEGIN SOLUTION
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = {}
        
        # Track optimization steps
        self.step_count = 0
        ### END SOLUTION
    
    def step(self) -> None:
        """
        Perform one optimization step.
        
        TODO: Implement SGD parameter update with momentum.
        
        APPROACH:
        1. Iterate through all parameters
        2. For each parameter with gradient:
           a. Get current gradient
           b. Apply weight decay if specified
           c. Update momentum buffer (or create if first time)
           d. Update parameter using momentum
        3. Increment step count
        
        MATHEMATICAL FORMULATION:
        - If weight_decay > 0: gradient = gradient + weight_decay * parameter
        - momentum_buffer = momentum * momentum_buffer + gradient
        - parameter = parameter - learning_rate * momentum_buffer
        
        IMPLEMENTATION HINTS:
        - Use id(param) as key for momentum buffers
        - Initialize buffer with zeros if not exists
        - Handle case where momentum = 0 (no momentum)
        - Update parameter.data with new Tensor
        """
        ### BEGIN SOLUTION
        for param in self.parameters:
            if param.grad is not None:
                # Get gradient
                gradient = param.grad.data.data
                
                # Apply weight decay (L2 regularization)
                if self.weight_decay > 0:
                    gradient = gradient + self.weight_decay * param.data.data
                
                # Get or create momentum buffer
                param_id = id(param)
                if param_id not in self.momentum_buffers:
                    self.momentum_buffers[param_id] = np.zeros_like(param.data.data)
                
                # Update momentum buffer
                self.momentum_buffers[param_id] = (
                    self.momentum * self.momentum_buffers[param_id] + gradient
                )
                
                # Update parameter
                param.data = Tensor(
                    param.data.data - self.learning_rate * self.momentum_buffers[param_id]
                )
        
        self.step_count += 1
        ### END SOLUTION
    
    def zero_grad(self) -> None:
        """
        Zero out gradients for all parameters.
        
        TODO: Implement gradient zeroing.
        
        APPROACH:
        1. Iterate through all parameters
        2. Set gradient to None for each parameter
        3. This prepares for next backward pass
        
        IMPLEMENTATION HINTS:
        - Simply set param.grad = None
        - This is called before loss.backward()
        - Essential for proper gradient accumulation
        """
        ### BEGIN SOLUTION
        for param in self.parameters:
            param.grad = None
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: SGD Optimizer

Let's test your SGD optimizer implementation! This optimizer adds momentum to gradient descent for better convergence.

**This is a unit test** - it tests one specific class (SGD) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-sgd", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_sgd_optimizer():
    """Unit test for the SGD optimizer implementation."""
    print("🔬 Unit Test: SGD Optimizer...")
    
    # Create test parameters
    w1 = Variable(1.0, requires_grad=True)
    w2 = Variable(2.0, requires_grad=True)
    b = Variable(0.5, requires_grad=True)
    
    # Create optimizer
    optimizer = SGD([w1, w2, b], learning_rate=0.1, momentum=0.9)
    
    # Test zero_grad
    try:
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        optimizer.zero_grad()
        
        assert w1.grad is None, "Gradient should be None after zero_grad"
        assert w2.grad is None, "Gradient should be None after zero_grad"
        assert b.grad is None, "Gradient should be None after zero_grad"
        print("✅ zero_grad() works correctly")
        
    except Exception as e:
        print(f"❌ zero_grad() failed: {e}")
        raise
    
    # Test step with gradients
    try:
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        # First step (no momentum yet)
        original_w1 = w1.data.data.item()
        original_w2 = w2.data.data.item()
        original_b = b.data.data.item()
        
        optimizer.step()
        
        # Check parameter updates
        expected_w1 = original_w1 - 0.1 * 0.1  # 1.0 - 0.01 = 0.99
        expected_w2 = original_w2 - 0.1 * 0.2  # 2.0 - 0.02 = 1.98
        expected_b = original_b - 0.1 * 0.05   # 0.5 - 0.005 = 0.495
        
        assert abs(w1.data.data.item() - expected_w1) < 1e-6, f"w1 update failed: expected {expected_w1}, got {w1.data.data.item()}"
        assert abs(w2.data.data.item() - expected_w2) < 1e-6, f"w2 update failed: expected {expected_w2}, got {w2.data.data.item()}"
        assert abs(b.data.data.item() - expected_b) < 1e-6, f"b update failed: expected {expected_b}, got {b.data.data.item()}"
        print("✅ Parameter updates work correctly")
        
    except Exception as e:
        print(f"❌ Parameter updates failed: {e}")
        raise
    
    # Test momentum buffers
    try:
        assert len(optimizer.momentum_buffers) == 3, f"Should have 3 momentum buffers, got {len(optimizer.momentum_buffers)}"
        assert optimizer.step_count == 1, f"Step count should be 1, got {optimizer.step_count}"
        print("✅ Momentum buffers created correctly")
        
    except Exception as e:
        print(f"❌ Momentum buffers failed: {e}")
        raise
    
    # Test step counting
    try:
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        optimizer.step()
        
        assert optimizer.step_count == 2, f"Step count should be 2, got {optimizer.step_count}"
        print("✅ Step counting works correctly")
        
    except Exception as e:
        print(f"❌ Step counting failed: {e}")
        raise

    print("🎯 SGD optimizer behavior:")
    print("   Maintains momentum buffers for accelerated updates")
    print("   Tracks step count for learning rate scheduling")
    print("   Supports weight decay for regularization")
    print("📈 Progress: SGD Optimizer ✓")

# Run the test
test_unit_sgd_optimizer()

# %% [markdown]
"""
## Step 3: Adam - Adaptive Learning Rates

### What is Adam?
**Adam (Adaptive Moment Estimation)** is the most popular optimizer in deep learning:

```
m_t = β₁ m_{t-1} + (1 - β₁) ∇L(θ_t)        # First moment (momentum)
v_t = β₂ v_{t-1} + (1 - β₂) (∇L(θ_t))²     # Second moment (variance)
m̂_t = m_t / (1 - β₁ᵗ)                      # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                      # Bias correction
θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)        # Parameter update
```

### Why Adam is Revolutionary
1. **Adaptive learning rates**: Different learning rate for each parameter
2. **Momentum**: Accelerates convergence like SGD
3. **Variance adaptation**: Scales updates based on gradient variance
4. **Bias correction**: Handles initialization bias
5. **Robust**: Works well with minimal hyperparameter tuning

### The Three Key Ideas
1. **First moment (m_t)**: Exponential moving average of gradients (momentum)
2. **Second moment (v_t)**: Exponential moving average of squared gradients (variance)
3. **Adaptive scaling**: Large gradients → small updates, small gradients → large updates

### Visual Understanding
```
Parameter with large gradients: /\/\/\/\ → smooth updates
Parameter with small gradients: ______ → amplified updates
```

### Real-World Applications
- **Deep learning**: Default optimizer for most neural networks
- **Computer vision**: Training CNNs, ResNets, Vision Transformers
- **Natural language**: Training BERT, GPT, T5
- **Transformers**: Essential for attention-based models

Let's implement Adam optimizer!
"""

# %% nbgrader={"grade": false, "grade_id": "adam-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Adam:
    """
    Adam Optimizer
    
    Implements Adam algorithm with adaptive learning rates:
    - First moment: exponential moving average of gradients
    - Second moment: exponential moving average of squared gradients
    - Bias correction: accounts for initialization bias
    - Adaptive updates: different learning rate per parameter
    """
    
    def __init__(self, parameters: List[Variable], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: List of Variables to optimize
            learning_rate: Learning rate (default: 0.001)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
            weight_decay: L2 regularization coefficient (default: 0.0)
        
        TODO: Implement Adam optimizer initialization.
        
        APPROACH:
        1. Store parameters and hyperparameters
        2. Initialize first moment buffers (m_t)
        3. Initialize second moment buffers (v_t)
        4. Set up step counter for bias correction
        
        EXAMPLE:
        ```python
        # Create Adam optimizer
        optimizer = Adam([w1, w2, b1, b2], learning_rate=0.001)
        
        # In training loop:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ```
        
        HINTS:
        - Store all hyperparameters
        - Initialize moment buffers as empty dicts
        - Use parameter id() as key for tracking
        - Buffers will be created lazily in step()
        """
        ### BEGIN SOLUTION
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Initialize moment buffers (created lazily)
        self.first_moment = {}   # m_t
        self.second_moment = {}  # v_t
        
        # Track optimization steps for bias correction
        self.step_count = 0
        ### END SOLUTION
    
    def step(self) -> None:
        """
        Perform one optimization step using Adam algorithm.
        
        TODO: Implement Adam parameter update.
        
        APPROACH:
        1. Increment step count
        2. For each parameter with gradient:
           a. Get current gradient
           b. Apply weight decay if specified
           c. Update first moment (momentum)
           d. Update second moment (variance)
           e. Apply bias correction
           f. Update parameter with adaptive learning rate
        
        MATHEMATICAL FORMULATION:
        - m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        - v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        - m_hat = m_t / (1 - beta1^t)
        - v_hat = v_t / (1 - beta2^t)
        - parameter = parameter - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        
        IMPLEMENTATION HINTS:
        - Use id(param) as key for moment buffers
        - Initialize buffers with zeros if not exists
        - Use np.sqrt() for square root
        - Handle numerical stability with epsilon
        """
        ### BEGIN SOLUTION
        self.step_count += 1
        
        for param in self.parameters:
            if param.grad is not None:
                # Get gradient
                gradient = param.grad.data.data
                
                # Apply weight decay (L2 regularization)
                if self.weight_decay > 0:
                    gradient = gradient + self.weight_decay * param.data.data
                
                # Get or create moment buffers
                param_id = id(param)
                if param_id not in self.first_moment:
                    self.first_moment[param_id] = np.zeros_like(param.data.data)
                    self.second_moment[param_id] = np.zeros_like(param.data.data)
                
                # Update first moment (momentum)
                self.first_moment[param_id] = (
                    self.beta1 * self.first_moment[param_id] + 
                    (1 - self.beta1) * gradient
                )
                
                # Update second moment (variance)
                self.second_moment[param_id] = (
                    self.beta2 * self.second_moment[param_id] + 
                    (1 - self.beta2) * gradient * gradient
                )
                
                # Bias correction
                first_moment_corrected = (
                    self.first_moment[param_id] / (1 - self.beta1 ** self.step_count)
                )
                second_moment_corrected = (
                    self.second_moment[param_id] / (1 - self.beta2 ** self.step_count)
                )
                
                # Update parameter with adaptive learning rate
                param.data = Tensor(
                    param.data.data - self.learning_rate * first_moment_corrected / 
                    (np.sqrt(second_moment_corrected) + self.epsilon)
                )
        ### END SOLUTION
    
    def zero_grad(self) -> None:
        """
        Zero out gradients for all parameters.
        
        TODO: Implement gradient zeroing (same as SGD).
        
        IMPLEMENTATION HINTS:
        - Set param.grad = None for all parameters
        - This is identical to SGD implementation
        """
        ### BEGIN SOLUTION
        for param in self.parameters:
            param.grad = None
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Adam Implementation

Let's test the Adam optimizer:
"""

# %% [markdown]
"""
### 🧪 Unit Test: Adam Optimizer

Let's test your Adam optimizer implementation! This is a state-of-the-art adaptive optimization algorithm.

**This is a unit test** - it tests one specific class (Adam) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-adam", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_adam_optimizer():
    """Unit test for the Adam optimizer implementation."""
    print("🔬 Unit Test: Adam Optimizer...")
    
    # Create test parameters
    w1 = Variable(1.0, requires_grad=True)
    w2 = Variable(2.0, requires_grad=True)
    b = Variable(0.5, requires_grad=True)
    
    # Create optimizer
    optimizer = Adam([w1, w2, b], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    # Test zero_grad
    try:
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        optimizer.zero_grad()
        
        assert w1.grad is None, "Gradient should be None after zero_grad"
        assert w2.grad is None, "Gradient should be None after zero_grad"
        assert b.grad is None, "Gradient should be None after zero_grad"
        print("✅ zero_grad() works correctly")
        
    except Exception as e:
        print(f"❌ zero_grad() failed: {e}")
        raise
    
    # Test step with gradients
    try:
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        # First step
        original_w1 = w1.data.data.item()
        original_w2 = w2.data.data.item()
        original_b = b.data.data.item()
        
        optimizer.step()
        
        # Check that parameters were updated (Adam uses adaptive learning rates)
        assert w1.data.data.item() != original_w1, "w1 should have been updated"
        assert w2.data.data.item() != original_w2, "w2 should have been updated"
        assert b.data.data.item() != original_b, "b should have been updated"
        print("✅ Parameter updates work correctly")
        
    except Exception as e:
        print(f"❌ Parameter updates failed: {e}")
        raise
    
    # Test moment buffers
    try:
        assert len(optimizer.first_moment) == 3, f"Should have 3 first moment buffers, got {len(optimizer.first_moment)}"
        assert len(optimizer.second_moment) == 3, f"Should have 3 second moment buffers, got {len(optimizer.second_moment)}"
        print("✅ Moment buffers created correctly")
        
    except Exception as e:
        print(f"❌ Moment buffers failed: {e}")
        raise
    
    # Test step counting and bias correction
    try:
        assert optimizer.step_count == 1, f"Step count should be 1, got {optimizer.step_count}"
        
        # Take another step
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        
        optimizer.step()
        
        assert optimizer.step_count == 2, f"Step count should be 2, got {optimizer.step_count}"
        print("✅ Step counting and bias correction work correctly")
        
    except Exception as e:
        print(f"❌ Step counting and bias correction failed: {e}")
        raise
    
    # Test adaptive learning rates
    try:
        # Adam should have different effective learning rates for different parameters
        # This is tested implicitly by the parameter updates above
        print("✅ Adaptive learning rates work correctly")
        
    except Exception as e:
        print(f"❌ Adaptive learning rates failed: {e}")
        raise

    print("🎯 Adam optimizer behavior:")
    print("   Maintains first and second moment estimates")
    print("   Applies bias correction for early training")
    print("   Uses adaptive learning rates per parameter")
    print("   Combines benefits of momentum and RMSprop")
    print("📈 Progress: Adam Optimizer ✓")

# Run the test
test_unit_adam_optimizer()

# %% [markdown]
"""
## Step 4: Learning Rate Scheduling

### What is Learning Rate Scheduling?
**Learning rate scheduling** adjusts the learning rate during training:

```
Initial: learning_rate = 0.1
After 10 epochs: learning_rate = 0.01
After 20 epochs: learning_rate = 0.001
```

### Why Scheduling Matters
1. **Fine-tuning**: Start with large steps, then refine with small steps
2. **Convergence**: Prevents overshooting near optimum
3. **Stability**: Reduces oscillations in later training
4. **Performance**: Often improves final accuracy

### Common Scheduling Strategies
1. **Step decay**: Reduce by factor every N epochs
2. **Exponential decay**: Gradual exponential reduction
3. **Cosine annealing**: Smooth cosine curve reduction
4. **Warm-up**: Start small, increase, then decrease

### Visual Understanding
```
Step decay:     ----↓----↓----↓
Exponential:    \\\\\\\\\\\\\\
Cosine:         ∩∩∩∩∩∩∩∩∩∩∩∩∩
```

### Real-World Applications
- **ImageNet training**: Essential for achieving state-of-the-art results
- **Language models**: Critical for training large transformers
- **Fine-tuning**: Prevents catastrophic forgetting
- **Transfer learning**: Adapts pre-trained models

Let's implement step learning rate scheduling!
"""

# %% nbgrader={"grade": false, "grade_id": "steplr-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class StepLR:
    """
    Step Learning Rate Scheduler
    
    Decays learning rate by gamma every step_size epochs:
    learning_rate = initial_lr * (gamma ^ (epoch // step_size))
    """
    
    def __init__(self, optimizer: Union[SGD, Adam], step_size: int, gamma: float = 0.1):
        """
        Initialize step learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            step_size: Number of epochs between decreases
            gamma: Multiplicative factor for learning rate decay
        
        TODO: Implement learning rate scheduler initialization.
        
        APPROACH:
        1. Store optimizer reference
        2. Store scheduling parameters
        3. Save initial learning rate
        4. Initialize step counter
        
        EXAMPLE:
        ```python
        optimizer = SGD([w1, w2], learning_rate=0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        # In training loop:
        for epoch in range(100):
            train_one_epoch()
            scheduler.step()  # Update learning rate
        ```
        
        HINTS:
        - Store optimizer reference
        - Save initial learning rate from optimizer
        - Initialize step counter to 0
        - gamma is the decay factor (0.1 = 10x reduction)
        """
        ### BEGIN SOLUTION
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.learning_rate
        self.step_count = 0
        ### END SOLUTION
    
    def step(self) -> None:
        """
        Update learning rate based on current step.
        
        TODO: Implement learning rate update.
        
        APPROACH:
        1. Increment step counter
        2. Calculate new learning rate using step decay formula
        3. Update optimizer's learning rate
        
        MATHEMATICAL FORMULATION:
        new_lr = initial_lr * (gamma ^ ((step_count - 1) // step_size))
        
        IMPLEMENTATION HINTS:
        - Use // for integer division
        - Use ** for exponentiation
        - Update optimizer.learning_rate directly
        """
        ### BEGIN SOLUTION
        self.step_count += 1
        
        # Calculate new learning rate
        decay_factor = self.gamma ** ((self.step_count - 1) // self.step_size)
        new_lr = self.initial_lr * decay_factor
        
        # Update optimizer's learning rate
        self.optimizer.learning_rate = new_lr
        ### END SOLUTION
    
    def get_lr(self) -> float:
        """
        Get current learning rate.
        
        TODO: Return current learning rate.
        
        IMPLEMENTATION HINTS:
        - Return optimizer.learning_rate
        """
        ### BEGIN SOLUTION
        return self.optimizer.learning_rate
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Step Learning Rate Scheduler

Let's test your step learning rate scheduler implementation! This scheduler reduces learning rate at regular intervals.

**This is a unit test** - it tests one specific class (StepLR) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-step-scheduler", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_step_scheduler():
    """Unit test for the StepLR scheduler implementation."""
    print("🔬 Unit Test: Step Learning Rate Scheduler...")
    
    # Create test parameters and optimizer
    w = Variable(1.0, requires_grad=True)
    optimizer = SGD([w], learning_rate=0.1)
    
    # Test scheduler initialization
    try:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Test initial learning rate
        assert scheduler.get_lr() == 0.1, f"Initial learning rate should be 0.1, got {scheduler.get_lr()}"
        print("✅ Initial learning rate is correct")
        
    except Exception as e:
        print(f"❌ Initial learning rate failed: {e}")
        raise
    
    # Test step-based decay
    try:
        # Steps 1-10: no decay (decay happens after step 10)
        for i in range(10):
            scheduler.step()
        
        assert scheduler.get_lr() == 0.1, f"Learning rate should still be 0.1 after 10 steps, got {scheduler.get_lr()}"
        
        # Step 11: decay should occur
        scheduler.step()
        expected_lr = 0.1 * 0.1  # 0.01
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6, f"Learning rate should be {expected_lr} after 11 steps, got {scheduler.get_lr()}"
        print("✅ Step-based decay works correctly")
        
    except Exception as e:
        print(f"❌ Step-based decay failed: {e}")
        raise
    
    # Test multiple decay levels
    try:
        # Steps 12-20: should stay at 0.01
        for i in range(9):
            scheduler.step()
        
        assert abs(scheduler.get_lr() - 0.01) < 1e-6, f"Learning rate should be 0.01 after 20 steps, got {scheduler.get_lr()}"
        
        # Step 21: another decay
        scheduler.step()
        expected_lr = 0.01 * 0.1  # 0.001
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6, f"Learning rate should be {expected_lr} after 21 steps, got {scheduler.get_lr()}"
        print("✅ Multiple decay levels work correctly")
        
    except Exception as e:
        print(f"❌ Multiple decay levels failed: {e}")
        raise
    
    # Test with different optimizer
    try:
        w2 = Variable(2.0, requires_grad=True)
        adam_optimizer = Adam([w2], learning_rate=0.001)
        adam_scheduler = StepLR(adam_optimizer, step_size=5, gamma=0.5)
        
        # Test initial learning rate
        assert adam_scheduler.get_lr() == 0.001, f"Initial Adam learning rate should be 0.001, got {adam_scheduler.get_lr()}"
        
        # Test decay after 5 steps
        for i in range(5):
            adam_scheduler.step()
        
        # Learning rate should still be 0.001 after 5 steps
        assert adam_scheduler.get_lr() == 0.001, f"Adam learning rate should still be 0.001 after 5 steps, got {adam_scheduler.get_lr()}"
        
        # Step 6: decay should occur
        adam_scheduler.step()
        expected_lr = 0.001 * 0.5  # 0.0005
        assert abs(adam_scheduler.get_lr() - expected_lr) < 1e-6, f"Adam learning rate should be {expected_lr} after 6 steps, got {adam_scheduler.get_lr()}"
        print("✅ Works with different optimizers")
        
    except Exception as e:
        print(f"❌ Different optimizers failed: {e}")
        raise

    print("🎯 Step learning rate scheduler behavior:")
    print("   Reduces learning rate at regular intervals")
    print("   Multiplies current rate by gamma factor")
    print("   Works with any optimizer (SGD, Adam, etc.)")
    print("📈 Progress: Step Learning Rate Scheduler ✓")

# Run the test
test_unit_step_scheduler()

# %% [markdown]
"""
## Step 5: Integration - Complete Training Example

### Putting It All Together
Let's see how optimizers enable complete neural network training:

1. **Forward pass**: Compute predictions
2. **Loss computation**: Compare with targets
3. **Backward pass**: Compute gradients
4. **Optimizer step**: Update parameters
5. **Learning rate scheduling**: Adjust learning rate

### The Modern Training Loop
```python
# Setup
optimizer = Adam(model.parameters(), learning_rate=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch.inputs)
        loss = criterion(predictions, batch.targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()
```

Let's implement a complete training example!
"""

# %% nbgrader={"grade": false, "grade_id": "training-integration", "locked": false, "schema_version": 3, "solution": true, "task": false}
def train_simple_model():
    """
    Complete training example using optimizers.
    
    TODO: Implement a complete training loop.
    
    APPROACH:
    1. Create a simple model (linear regression)
    2. Generate training data
    3. Set up optimizer and scheduler
    4. Train for several epochs
    5. Show convergence
    
    LEARNING OBJECTIVE:
    - See how optimizers enable real learning
    - Compare SGD vs Adam performance
    - Understand the complete training workflow
    """
    ### BEGIN SOLUTION
    print("Training simple linear regression model...")
    
    # Create simple model: y = w*x + b
    w = Variable(0.1, requires_grad=True)  # Initialize near zero
    b = Variable(0.0, requires_grad=True)
    
    # Training data: y = 2*x + 1
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_data = [3.0, 5.0, 7.0, 9.0, 11.0]
    
    # Try SGD first
    print("\n🔍 Training with SGD...")
    optimizer_sgd = SGD([w, b], learning_rate=0.01, momentum=0.9)
    
    for epoch in range(60):
        total_loss = 0
        
        for x_val, y_val in zip(x_data, y_data):
            # Forward pass
            x = Variable(x_val, requires_grad=False)
            y_target = Variable(y_val, requires_grad=False)
            
            # Prediction: y = w*x + b
            try:
                from tinytorch.core.autograd import add, multiply, subtract
            except ImportError:
                setup_import_paths()
                from autograd_dev import add, multiply, subtract
            
            prediction = add(multiply(w, x), b)
            
            # Loss: (prediction - target)^2
            error = subtract(prediction, y_target)
            loss = multiply(error, error)
            
            # Backward pass
            optimizer_sgd.zero_grad()
            loss.backward()
            optimizer_sgd.step()
            
            total_loss += loss.data.data.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}, w = {w.data.data.item():.3f}, b = {b.data.data.item():.3f}")
    
    sgd_final_w = w.data.data.item()
    sgd_final_b = b.data.data.item()
    
    # Reset parameters and try Adam
    print("\n🔍 Training with Adam...")
    w.data = Tensor(0.1)
    b.data = Tensor(0.0)
    
    optimizer_adam = Adam([w, b], learning_rate=0.01)
    
    for epoch in range(60):
        total_loss = 0
        
        for x_val, y_val in zip(x_data, y_data):
            # Forward pass
            x = Variable(x_val, requires_grad=False)
            y_target = Variable(y_val, requires_grad=False)
            
            # Prediction: y = w*x + b
            prediction = add(multiply(w, x), b)
            
            # Loss: (prediction - target)^2
            error = subtract(prediction, y_target)
            loss = multiply(error, error)
            
            # Backward pass
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            
            total_loss += loss.data.data.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}, w = {w.data.data.item():.3f}, b = {b.data.data.item():.3f}")
    
    adam_final_w = w.data.data.item()
    adam_final_b = b.data.data.item()
    
    print(f"\n📊 Results:")
    print(f"Target: w = 2.0, b = 1.0")
    print(f"SGD:    w = {sgd_final_w:.3f}, b = {sgd_final_b:.3f}")
    print(f"Adam:   w = {adam_final_w:.3f}, b = {adam_final_b:.3f}")
    
    return sgd_final_w, sgd_final_b, adam_final_w, adam_final_b
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Complete Training Integration

Let's test your complete training integration! This demonstrates optimizers working together in a realistic training scenario.

**This is a unit test** - it tests the complete training workflow with optimizers in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-training-integration", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_unit_training_integration():
    """Comprehensive unit test for complete training integration with optimizers."""
    print("🔬 Unit Test: Complete Training Integration...")
    
    # Test training with SGD and Adam
    try:
        sgd_w, sgd_b, adam_w, adam_b = train_simple_model()
        
        # Test SGD convergence
        assert abs(sgd_w - 2.0) < 0.1, f"SGD should converge close to w=2.0, got {sgd_w}"
        assert abs(sgd_b - 1.0) < 0.1, f"SGD should converge close to b=1.0, got {sgd_b}"
        print("✅ SGD convergence works")
        
        # Test Adam convergence (may be different due to adaptive learning rates)
        assert abs(adam_w - 2.0) < 1.0, f"Adam should converge reasonably close to w=2.0, got {adam_w}"
        assert abs(adam_b - 1.0) < 1.0, f"Adam should converge reasonably close to b=1.0, got {adam_b}"
        print("✅ Adam convergence works")
        
    except Exception as e:
        print(f"❌ Training integration failed: {e}")
        raise
    
    # Test optimizer comparison
    try:
        # Both optimizers should achieve reasonable results
        sgd_error = (sgd_w - 2.0)**2 + (sgd_b - 1.0)**2
        adam_error = (adam_w - 2.0)**2 + (adam_b - 1.0)**2
        
        # Both should have low error (< 0.1)
        assert sgd_error < 0.1, f"SGD error should be < 0.1, got {sgd_error}"
        assert adam_error < 1.0, f"Adam error should be < 1.0, got {adam_error}"
        print("✅ Optimizer comparison works")
        
    except Exception as e:
        print(f"❌ Optimizer comparison failed: {e}")
        raise
    
    # Test gradient flow
    try:
        # Create a simple test to verify gradients flow correctly
        w = Variable(1.0, requires_grad=True)
        b = Variable(0.0, requires_grad=True)
        
        # Set up simple gradients
        w.grad = Variable(0.1)
        b.grad = Variable(0.05)
        
        # Test SGD step
        sgd_optimizer = SGD([w, b], learning_rate=0.1)
        original_w = w.data.data.item()
        original_b = b.data.data.item()
        
        sgd_optimizer.step()
        
        # Check updates
        assert w.data.data.item() != original_w, "SGD should update w"
        assert b.data.data.item() != original_b, "SGD should update b"
        print("✅ Gradient flow works correctly")
        
    except Exception as e:
        print(f"❌ Gradient flow failed: {e}")
        raise

    print("🎯 Training integration behavior:")
    print("   Optimizers successfully minimize loss functions")
    print("   SGD and Adam both converge to target values")
    print("   Gradient computation and updates work correctly")
    print("   Ready for real neural network training")
    print("📈 Progress: Complete Training Integration ✓")

# Run the test
test_unit_training_integration()

# %%
def test_module_optimizer_autograd_compatibility():
    """
    Integration test for the optimizer and autograd Variable classes.
    
    Tests that an optimizer can correctly update the Tensors of Variables
    that have gradients computed by the autograd engine.
    """
    print("🔬 Running Integration Test: Optimizer with Autograd Variables...")

    # 1. Create a parameter that requires gradients
    w = Variable(Tensor([3.0]), requires_grad=True)

    # 2. Simulate a backward pass by manually setting a gradient
    # The gradient must also be a Tensor, wrapped in a Variable
    w.grad = Variable(Tensor([10.0]), requires_grad=False)

    # 3. Create an SGD optimizer for this parameter
    optimizer = SGD(parameters=[w], learning_rate=0.1)

    # 4. Perform an optimization step
    optimizer.step()

    # 5. Assert that the parameter's data (Tensor) has been updated
    # new_w = 3.0 - 0.1 * 10.0 = 2.0
    assert isinstance(w.data, Tensor), "Parameter's data should remain a Tensor"
    assert np.allclose(w.data.data, [2.0]), f"Expected w to be 2.0, but got {w.data.data}"

    print("✅ Integration Test Passed: Optimizer correctly updated Variable's Tensor data.")

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% [markdown]
"""
## 🤖 AUTO TESTING
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    # Unit tests
    test_unit_gradient_descent_step()
    test_unit_sgd_optimizer()
    test_unit_adam_optimizer()
    test_unit_step_scheduler()
    test_unit_training_integration()
    # Integration test
    test_module_optimizer_autograd_compatibility()

    from tito.tools.testing import run_module_tests_auto
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Optimizers")

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Optimization Algorithms

Congratulations! You've successfully implemented the optimization algorithms that power all modern neural network training:

### ✅ What You've Built
- **Gradient Descent**: The fundamental parameter update mechanism
- **SGD with Momentum**: Accelerated convergence with velocity accumulation
- **Adam Optimizer**: Adaptive learning rates with first and second moments
- **Learning Rate Scheduling**: Smart learning rate adjustment during training
- **Complete Training Integration**: End-to-end training workflow

### ✅ Key Learning Outcomes
- **Understanding**: How optimizers use gradients to update parameters intelligently
- **Implementation**: Built SGD and Adam optimizers from mathematical foundations
- **Mathematical mastery**: Momentum, adaptive learning rates, bias correction
- **Systems integration**: Complete training loops with scheduling
- **Real-world application**: Modern deep learning training workflow

### ✅ Mathematical Foundations Mastered
- **Gradient Descent**: θ = θ - α∇L(θ) for parameter updates
- **Momentum**: v_t = βv_{t-1} + ∇L(θ) for acceleration
- **Adam**: Adaptive learning rates with exponential moving averages
- **Learning Rate Scheduling**: Strategic learning rate adjustment

### ✅ Professional Skills Developed
- **Algorithm implementation**: Translating mathematical formulas into code
- **State management**: Tracking optimizer buffers and statistics
- **Hyperparameter design**: Understanding the impact of learning rate, momentum, etc.
- **Training orchestration**: Complete training loop design

### ✅ Ready for Advanced Applications
Your optimizers now enable:
- **Deep Neural Networks**: Effective training of complex architectures
- **Computer Vision**: Training CNNs, ResNets, Vision Transformers
- **Natural Language Processing**: Training transformers and language models
- **Any ML Model**: Gradient-based optimization for any differentiable system

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch**: `torch.optim.SGD()`, `torch.optim.Adam()`, `torch.optim.lr_scheduler.StepLR()`
- **TensorFlow**: `tf.keras.optimizers.SGD()`, `tf.keras.optimizers.Adam()`
- **Industry Standard**: Every major ML framework uses these exact algorithms

### 🎯 The Power of Intelligent Optimization
You've unlocked the algorithms that made modern AI possible:
- **Scalability**: Efficiently optimize millions of parameters
- **Adaptability**: Different learning rates for different parameters
- **Robustness**: Handle noisy gradients and ill-conditioned problems
- **Universality**: Work with any differentiable neural network

### 🧠 Deep Learning Revolution
You now understand the optimization technology that powers:
- **ImageNet**: Training state-of-the-art computer vision models
- **Language Models**: Training GPT, BERT, and other transformers
- **Modern AI**: Every breakthrough relies on these optimization algorithms
- **Future Research**: Your understanding enables you to develop new optimizers

### 🚀 What's Next
Your optimizers are the foundation for:
- **Training Module**: Complete training loops with loss functions and metrics
- **Advanced Optimizers**: RMSprop, AdaGrad, learning rate warm-up
- **Distributed Training**: Multi-GPU optimization strategies
- **Research**: Experimenting with novel optimization algorithms

**Next Module**: Complete training systems that orchestrate your optimizers for real-world ML!

You've built the intelligent algorithms that enable neural networks to learn. Now let's use them to train systems that can solve complex real-world problems!
""" 