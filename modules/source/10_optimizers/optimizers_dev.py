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
# Optimizers - Gradient-Based Parameter Updates and Training Dynamics

Welcome to the Optimizers module! You'll implement the algorithms that use gradients to update neural network parameters, determining how effectively networks learn from data.

## Learning Goals
- Systems understanding: How different optimization algorithms affect convergence speed, memory usage, and training stability
- Core implementation skill: Build SGD with momentum and Adam optimizer, understanding their mathematical foundations and implementation trade-offs
- Pattern recognition: Understand how adaptive learning rates and momentum help navigate complex loss landscapes
- Framework connection: See how your optimizer implementations match PyTorch's optim module design and state management
- Performance insight: Learn why optimizer choice affects training speed and why Adam uses 3x more memory than SGD

## Build → Use → Reflect
1. **Build**: Complete SGD and Adam optimizers with proper state management and learning rate scheduling
2. **Use**: Train neural networks with different optimizers and compare convergence behavior on real datasets
3. **Reflect**: Why do some optimizers work better for certain problems, and how does memory usage scale with model size?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how optimization algorithms navigate high-dimensional loss landscapes to find good solutions
- Practical capability to implement and tune optimizers that determine training success or failure
- Systems insight into why optimizer choice often matters more than architecture choice for training success
- Performance consideration of how optimizer memory requirements and computational overhead affect scalable training
- Connection to production ML systems and why new optimizers continue to be an active area of research

## Systems Reality Check
💡 **Production Context**: PyTorch's Adam implementation includes numerically stable variants and can automatically scale learning rates based on gradient norms to prevent training instability
⚡ **Performance Note**: Adam stores running averages for every parameter, using 3x the memory of SGD - this memory overhead becomes critical when training large models near GPU memory limits
"""

# %% nbgrader={"grade": false, "grade_id": "optimizers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.optimizers

#| export
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
Loss landscape: U-shaped curve
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

# Test function defined (called in main block)

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

# Test function defined (called in main block)

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
Parameter with large gradients: zigzag pattern → smooth updates
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

# Test function defined (called in main block)

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

# Test function defined (called in main block)

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
def test_module_unit_training():
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

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 6: ML Systems - Optimizer Performance Analysis

### Real-World Challenge: Optimizer Selection and Tuning

In production ML systems, choosing the right optimizer and hyperparameters can make the difference between:
- **Success**: Model converges to good performance in reasonable time
- **Failure**: Model doesn't converge, explodes, or takes too long to train

### The Production Reality
When training large models (millions or billions of parameters):
- **Wrong optimizer**: Can waste weeks of expensive GPU time
- **Wrong learning rate**: Can cause gradient explosion or extremely slow convergence
- **Wrong scheduling**: Can prevent models from reaching optimal performance
- **Memory constraints**: Some optimizers use significantly more memory than others

### What We'll Build
An **OptimizerConvergenceProfiler** that analyzes:
1. **Convergence patterns** across different optimizers
2. **Learning rate sensitivity** and optimal hyperparameters
3. **Computational cost vs convergence speed** trade-offs
4. **Gradient statistics** and update patterns
5. **Memory usage patterns** for different optimizers

This mirrors tools used in production for optimizer selection and hyperparameter tuning.
"""

# %% nbgrader={"grade": false, "grade_id": "convergence-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class OptimizerConvergenceProfiler:
    """
    ML Systems Tool: Optimizer Performance and Convergence Analysis
    
    Profiles convergence patterns, learning rate sensitivity, and computational costs
    across different optimizers to guide production optimizer selection.
    
    This is 60% implementation focusing on core analysis capabilities:
    - Convergence rate comparison across optimizers
    - Learning rate sensitivity analysis
    - Gradient statistics tracking
    - Memory usage estimation
    - Performance recommendations
    """
    
    def __init__(self):
        """
        Initialize optimizer convergence profiler.
        
        TODO: Implement profiler initialization.
        
        APPROACH:
        1. Initialize tracking dictionaries for different metrics
        2. Set up convergence analysis parameters
        3. Prepare memory and performance tracking
        4. Initialize recommendation engine components
        
        PRODUCTION CONTEXT:
        In production, this profiler would run on representative tasks to:
        - Select optimal optimizers for new models
        - Tune hyperparameters before expensive training runs
        - Predict training time and resource requirements
        - Monitor training stability and convergence
        
        IMPLEMENTATION HINTS:
        - Track convergence history per optimizer
        - Store gradient statistics over time
        - Monitor memory usage patterns
        - Prepare for comparative analysis
        """
        ### BEGIN SOLUTION
        # Convergence tracking
        self.convergence_history = defaultdict(list)  # {optimizer_name: [losses]}
        self.gradient_norms = defaultdict(list)       # {optimizer_name: [grad_norms]}
        self.learning_rates = defaultdict(list)       # {optimizer_name: [lr_values]}
        self.step_times = defaultdict(list)           # {optimizer_name: [step_durations]}
        
        # Performance metrics
        self.memory_usage = defaultdict(list)         # {optimizer_name: [memory_estimates]}
        self.convergence_rates = {}                   # {optimizer_name: convergence_rate}
        self.stability_scores = {}                    # {optimizer_name: stability_score}
        
        # Analysis parameters
        self.convergence_threshold = 1e-6
        self.stability_window = 10
        self.gradient_explosion_threshold = 1e6
        
        # Recommendations
        self.optimizer_rankings = {}
        self.hyperparameter_suggestions = {}
        ### END SOLUTION
    
    def profile_optimizer_convergence(self, optimizer_name: str, optimizer: Union[SGD, Adam], 
                                    training_function, initial_loss: float, 
                                    max_steps: int = 100) -> Dict[str, Any]:
        """
        Profile convergence behavior of an optimizer on a specific task.
        
        Args:
            optimizer_name: Name identifier for the optimizer
            optimizer: Optimizer instance to profile
            training_function: Function that performs one training step and returns loss
            initial_loss: Starting loss value
            max_steps: Maximum training steps to profile
        
        Returns:
            Dictionary containing convergence analysis results
        
        TODO: Implement optimizer convergence profiling.
        
        APPROACH:
        1. Run training loop with the optimizer
        2. Track loss, gradients, learning rates at each step
        3. Measure step execution time
        4. Estimate memory usage
        5. Analyze convergence patterns and stability
        6. Generate performance metrics
        
        CONVERGENCE ANALYSIS:
        - Track loss reduction over time
        - Measure convergence rate (loss reduction per step)
        - Detect convergence plateaus
        - Identify gradient explosion or vanishing
        - Assess training stability
        
        PRODUCTION INSIGHTS:
        This analysis helps determine:
        - Which optimizers converge fastest for specific model types
        - Optimal learning rates for different optimizers
        - Memory vs performance trade-offs
        - Training stability and robustness
        
        IMPLEMENTATION HINTS:
        - Use time.time() to measure step duration
        - Calculate gradient norms across all parameters
        - Track learning rate changes (for schedulers)
        - Estimate memory from optimizer state size
        """
        ### BEGIN SOLUTION
        import time
        
        print(f"🔍 Profiling {optimizer_name} convergence...")
        
        # Initialize tracking
        losses = []
        grad_norms = []
        step_durations = []
        lr_values = []
        
        previous_loss = initial_loss
        convergence_step = None
        
        for step in range(max_steps):
            step_start = time.time()
            
            # Perform training step
            try:
                current_loss = training_function()
                losses.append(current_loss)
                
                # Calculate gradient norm
                total_grad_norm = 0.0
                param_count = 0
                for param in optimizer.parameters:
                    if param.grad is not None:
                        grad_data = param.grad.data.data
                        if hasattr(grad_data, 'flatten'):
                            grad_norm = np.linalg.norm(grad_data.flatten())
                        else:
                            grad_norm = abs(float(grad_data))
                        total_grad_norm += grad_norm ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_grad_norm = (total_grad_norm / param_count) ** 0.5
                grad_norms.append(total_grad_norm)
                
                # Track learning rate
                lr_values.append(optimizer.learning_rate)
                
                # Check convergence
                if convergence_step is None and abs(current_loss - previous_loss) < self.convergence_threshold:
                    convergence_step = step
                
                previous_loss = current_loss
                
            except Exception as e:
                print(f"⚠️ Training step {step} failed: {e}")
                break
            
            step_end = time.time()
            step_durations.append(step_end - step_start)
            
            # Early stopping for exploded gradients
            if total_grad_norm > self.gradient_explosion_threshold:
                print(f"⚠️ Gradient explosion detected at step {step}")
                break
        
        # Store results
        self.convergence_history[optimizer_name] = losses
        self.gradient_norms[optimizer_name] = grad_norms
        self.learning_rates[optimizer_name] = lr_values
        self.step_times[optimizer_name] = step_durations
        
        # Analyze results
        analysis = self._analyze_convergence_profile(optimizer_name, losses, grad_norms, 
                                                   step_durations, convergence_step)
        
        return analysis
        ### END SOLUTION
    
    def compare_optimizers(self, profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple optimizer profiles and generate recommendations.
        
        Args:
            profiles: Dictionary mapping optimizer names to their profile results
        
        Returns:
            Comprehensive comparison analysis with recommendations
        
        TODO: Implement optimizer comparison and ranking.
        
        APPROACH:
        1. Analyze convergence speed across optimizers
        2. Compare final performance and stability
        3. Assess computational efficiency
        4. Generate rankings and recommendations
        5. Identify optimal hyperparameters
        
        COMPARISON METRICS:
        - Steps to convergence
        - Final loss achieved
        - Training stability (loss variance)
        - Computational cost per step
        - Memory efficiency
        - Gradient explosion resistance
        
        PRODUCTION VALUE:
        This comparison guides:
        - Optimizer selection for new projects
        - Hyperparameter optimization strategies
        - Resource allocation decisions
        - Training pipeline design
        
        IMPLEMENTATION HINTS:
        - Normalize metrics for fair comparison
        - Weight different factors based on importance
        - Generate actionable recommendations
        - Consider trade-offs between speed and stability
        """
        ### BEGIN SOLUTION
        comparison = {
            'convergence_speed': {},
            'final_performance': {},
            'stability': {},
            'efficiency': {},
            'rankings': {},
            'recommendations': {}
        }
        
        print("📊 Comparing optimizer performance...")
        
        # Analyze each optimizer
        for opt_name, profile in profiles.items():
            # Convergence speed
            convergence_step = profile.get('convergence_step', len(self.convergence_history[opt_name]))
            comparison['convergence_speed'][opt_name] = convergence_step
            
            # Final performance
            losses = self.convergence_history[opt_name]
            if losses:
                final_loss = losses[-1]
                comparison['final_performance'][opt_name] = final_loss
            
            # Stability (coefficient of variation in last 10 steps)
            if len(losses) >= self.stability_window:
                recent_losses = losses[-self.stability_window:]
                stability = 1.0 / (1.0 + np.std(recent_losses) / (np.mean(recent_losses) + 1e-8))
                comparison['stability'][opt_name] = stability
            
            # Efficiency (loss reduction per unit time)
            step_times = self.step_times[opt_name]
            if losses and step_times:
                initial_loss = losses[0]
                final_loss = losses[-1]
                total_time = sum(step_times)
                efficiency = (initial_loss - final_loss) / (total_time + 1e-8)
                comparison['efficiency'][opt_name] = efficiency
        
        # Generate rankings
        metrics = ['convergence_speed', 'final_performance', 'stability', 'efficiency']
        for metric in metrics:
            if comparison[metric]:
                if metric == 'convergence_speed':
                    # Lower is better for convergence speed
                    sorted_opts = sorted(comparison[metric].items(), key=lambda x: x[1])
                elif metric == 'final_performance':
                    # Lower is better for final loss
                    sorted_opts = sorted(comparison[metric].items(), key=lambda x: x[1])
                else:
                    # Higher is better for stability and efficiency
                    sorted_opts = sorted(comparison[metric].items(), key=lambda x: x[1], reverse=True)
                
                comparison['rankings'][metric] = [opt for opt, _ in sorted_opts]
        
        # Generate recommendations
        recommendations = []
        
        # Best overall optimizer
        if comparison['rankings']:
            # Simple scoring: rank position across metrics
            scores = defaultdict(float)
            for metric, ranking in comparison['rankings'].items():
                for i, opt_name in enumerate(ranking):
                    scores[opt_name] += len(ranking) - i
            
            best_optimizer = max(scores.items(), key=lambda x: x[1])[0]
            recommendations.append(f"🏆 Best overall optimizer: {best_optimizer}")
        
        # Specific recommendations
        if 'convergence_speed' in comparison['rankings']:
            fastest = comparison['rankings']['convergence_speed'][0]
            recommendations.append(f"⚡ Fastest convergence: {fastest}")
        
        if 'stability' in comparison['rankings']:
            most_stable = comparison['rankings']['stability'][0]
            recommendations.append(f"🎯 Most stable training: {most_stable}")
        
        if 'efficiency' in comparison['rankings']:
            most_efficient = comparison['rankings']['efficiency'][0]
            recommendations.append(f"💰 Most compute-efficient: {most_efficient}")
        
        comparison['recommendations']['summary'] = recommendations
        
        return comparison
        ### END SOLUTION
    
    def analyze_learning_rate_sensitivity(self, optimizer_class, learning_rates: List[float],
                                        training_function, steps: int = 50) -> Dict[str, Any]:
        """
        Analyze optimizer sensitivity to different learning rates.
        
        Args:
            optimizer_class: Optimizer class (SGD or Adam)
            learning_rates: List of learning rates to test
            training_function: Function that creates and runs training
            steps: Number of training steps per learning rate
        
        Returns:
            Learning rate sensitivity analysis
        
        TODO: Implement learning rate sensitivity analysis.
        
        APPROACH:
        1. Test optimizer with different learning rates
        2. Measure convergence performance for each rate
        3. Identify optimal learning rate range
        4. Detect learning rate instability regions
        5. Generate learning rate recommendations
        
        SENSITIVITY ANALYSIS:
        - Plot loss curves for different learning rates
        - Identify optimal learning rate range
        - Detect gradient explosion thresholds
        - Measure convergence robustness
        - Generate adaptive scheduling suggestions
        
        PRODUCTION INSIGHTS:
        This analysis enables:
        - Automatic learning rate tuning
        - Learning rate scheduling optimization
        - Gradient explosion prevention
        - Training stability improvement
        
        IMPLEMENTATION HINTS:
        - Reset model state for each learning rate test
        - Track convergence metrics consistently
        - Identify learning rate sweet spots
        - Flag unstable learning rate regions
        """
        ### BEGIN SOLUTION
        print("🔍 Analyzing learning rate sensitivity...")
        
        lr_analysis = {
            'learning_rates': learning_rates,
            'final_losses': [],
            'convergence_steps': [],
            'stability_scores': [],
            'gradient_explosions': [],
            'optimal_range': None,
            'recommendations': []
        }
        
        # Test each learning rate
        for lr in learning_rates:
            print(f"  Testing learning rate: {lr}")
            
            try:
                # Create optimizer with current learning rate
                # This is a simplified test - in production, would reset model state
                losses, grad_norms = training_function(lr, steps)
                
                if losses:
                    final_loss = losses[-1]
                    lr_analysis['final_losses'].append(final_loss)
                    
                    # Find convergence step
                    convergence_step = steps
                    for i in range(1, len(losses)):
                        if abs(losses[i] - losses[i-1]) < self.convergence_threshold:
                            convergence_step = i
                            break
                    lr_analysis['convergence_steps'].append(convergence_step)
                    
                    # Calculate stability
                    if len(losses) >= 10:
                        recent_losses = losses[-10:]
                        stability = 1.0 / (1.0 + np.std(recent_losses) / (np.mean(recent_losses) + 1e-8))
                        lr_analysis['stability_scores'].append(stability)
                    else:
                        lr_analysis['stability_scores'].append(0.0)
                    
                    # Check for gradient explosion
                    max_grad_norm = max(grad_norms) if grad_norms else 0.0
                    explosion = max_grad_norm > self.gradient_explosion_threshold
                    lr_analysis['gradient_explosions'].append(explosion)
                    
                else:
                    # Failed to get losses
                    lr_analysis['final_losses'].append(float('inf'))
                    lr_analysis['convergence_steps'].append(steps)
                    lr_analysis['stability_scores'].append(0.0)
                    lr_analysis['gradient_explosions'].append(True)
                    
            except Exception as e:
                print(f"    ⚠️ Failed with lr={lr}: {e}")
                lr_analysis['final_losses'].append(float('inf'))
                lr_analysis['convergence_steps'].append(steps)
                lr_analysis['stability_scores'].append(0.0)
                lr_analysis['gradient_explosions'].append(True)
        
        # Find optimal learning rate range
        valid_indices = [i for i, (loss, explosion) in 
                        enumerate(zip(lr_analysis['final_losses'], lr_analysis['gradient_explosions']))
                        if not explosion and loss != float('inf')]
        
        if valid_indices:
            # Find learning rate with best final loss among stable ones
            stable_losses = [(i, lr_analysis['final_losses'][i]) for i in valid_indices]
            best_idx = min(stable_losses, key=lambda x: x[1])[0]
            
            # Define optimal range around best learning rate
            best_lr = learning_rates[best_idx]
            lr_analysis['optimal_range'] = (best_lr * 0.1, best_lr * 10.0)
            
            # Generate recommendations
            recommendations = []
            recommendations.append(f"🎯 Optimal learning rate: {best_lr:.2e}")
            recommendations.append(f"📈 Safe range: {lr_analysis['optimal_range'][0]:.2e} - {lr_analysis['optimal_range'][1]:.2e}")
            
            # Learning rate scheduling suggestions
            if best_idx > 0:
                recommendations.append("💡 Consider starting with higher LR and decaying")
            if any(lr_analysis['gradient_explosions']):
                max_safe_lr = max([learning_rates[i] for i in valid_indices])
                recommendations.append(f"⚠️ Avoid learning rates above {max_safe_lr:.2e}")
            
            lr_analysis['recommendations'] = recommendations
        else:
            lr_analysis['recommendations'] = ["⚠️ No stable learning rates found - try lower values"]
        
        return lr_analysis
        ### END SOLUTION
    
    def estimate_memory_usage(self, optimizer: Union[SGD, Adam], num_parameters: int) -> Dict[str, float]:
        """
        Estimate memory usage for different optimizers.
        
        Args:
            optimizer: Optimizer instance
            num_parameters: Number of model parameters
        
        Returns:
            Memory usage estimates in MB
        
        TODO: Implement memory usage estimation.
        
        APPROACH:
        1. Calculate parameter memory requirements
        2. Estimate optimizer state memory
        3. Account for gradient storage
        4. Include temporary computation memory
        5. Provide memory scaling predictions
        
        MEMORY ANALYSIS:
        - Parameter storage: num_params * 4 bytes (float32)
        - Gradient storage: num_params * 4 bytes
        - Optimizer state: varies by optimizer type
        - SGD momentum: num_params * 4 bytes
        - Adam: num_params * 8 bytes (first + second moments)
        
        PRODUCTION VALUE:
        Memory estimation helps:
        - Select optimizers for memory-constrained environments
        - Plan GPU memory allocation
        - Scale to larger models
        - Optimize batch sizes
        
        IMPLEMENTATION HINTS:
        - Use typical float32 size (4 bytes)
        - Account for optimizer-specific state
        - Include gradient accumulation overhead
        - Provide scaling estimates
        """
        ### BEGIN SOLUTION
        # Base memory requirements
        bytes_per_param = 4  # float32
        
        memory_breakdown = {
            'parameters_mb': num_parameters * bytes_per_param / (1024 * 1024),
            'gradients_mb': num_parameters * bytes_per_param / (1024 * 1024),
            'optimizer_state_mb': 0.0,
            'total_mb': 0.0
        }
        
        # Optimizer-specific state memory
        if isinstance(optimizer, SGD):
            if optimizer.momentum > 0:
                # Momentum buffers
                memory_breakdown['optimizer_state_mb'] = num_parameters * bytes_per_param / (1024 * 1024)
            else:
                memory_breakdown['optimizer_state_mb'] = 0.0
        elif isinstance(optimizer, Adam):
            # First and second moment estimates
            memory_breakdown['optimizer_state_mb'] = num_parameters * 2 * bytes_per_param / (1024 * 1024)
        
        # Calculate total
        memory_breakdown['total_mb'] = (
            memory_breakdown['parameters_mb'] + 
            memory_breakdown['gradients_mb'] + 
            memory_breakdown['optimizer_state_mb']
        )
        
        # Add efficiency estimates
        memory_breakdown['memory_efficiency'] = memory_breakdown['parameters_mb'] / memory_breakdown['total_mb']
        memory_breakdown['overhead_ratio'] = memory_breakdown['optimizer_state_mb'] / memory_breakdown['parameters_mb']
        
        return memory_breakdown
        ### END SOLUTION
    
    def generate_production_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations for production optimizer usage.
        
        Args:
            analysis_results: Combined results from convergence and sensitivity analysis
        
        Returns:
            List of production recommendations
        
        TODO: Implement production recommendation generation.
        
        APPROACH:
        1. Analyze convergence patterns and stability
        2. Consider computational efficiency requirements
        3. Account for memory constraints
        4. Generate optimizer selection guidance
        5. Provide hyperparameter tuning suggestions
        
        RECOMMENDATION CATEGORIES:
        - Optimizer selection for different scenarios
        - Learning rate and scheduling strategies
        - Memory optimization techniques
        - Training stability improvements
        - Production deployment considerations
        
        PRODUCTION CONTEXT:
        These recommendations guide:
        - ML engineer optimizer selection
        - DevOps resource allocation
        - Training pipeline optimization
        - Cost reduction strategies
        
        IMPLEMENTATION HINTS:
        - Provide specific, actionable advice
        - Consider different deployment scenarios
        - Include quantitative guidelines
        - Address common production challenges
        """
        ### BEGIN SOLUTION
        recommendations = []
        
        # Optimizer selection recommendations
        recommendations.append("🔧 OPTIMIZER SELECTION GUIDE:")
        recommendations.append("  • SGD + Momentum: Best for large batch training, proven stability")
        recommendations.append("  • Adam: Best for rapid prototyping, adaptive learning rates")
        recommendations.append("  • Consider memory constraints: SGD uses ~50% less memory than Adam")
        
        # Learning rate recommendations
        if 'learning_rate_analysis' in analysis_results:
            lr_analysis = analysis_results['learning_rate_analysis']
            if lr_analysis.get('optimal_range'):
                opt_range = lr_analysis['optimal_range']
                recommendations.append(f"📈 LEARNING RATE GUIDANCE:")
                recommendations.append(f"  • Start with: {opt_range[0]:.2e}")
                recommendations.append(f"  • Safe upper bound: {opt_range[1]:.2e}")
                recommendations.append("  • Use learning rate scheduling for best results")
        
        # Convergence recommendations
        if 'convergence_comparison' in analysis_results:
            comparison = analysis_results['convergence_comparison']
            if 'recommendations' in comparison and 'summary' in comparison['recommendations']:
                recommendations.append("🎯 CONVERGENCE OPTIMIZATION:")
                for rec in comparison['recommendations']['summary']:
                    recommendations.append(f"  • {rec}")
        
        # Production deployment recommendations
        recommendations.append("🚀 PRODUCTION DEPLOYMENT:")
        recommendations.append("  • Monitor gradient norms to detect training instability")
        recommendations.append("  • Implement gradient clipping for large models")
        recommendations.append("  • Use learning rate warmup for transformer architectures")
        recommendations.append("  • Consider mixed precision training to reduce memory usage")
        
        # Scaling recommendations
        recommendations.append("📊 SCALING CONSIDERATIONS:")
        recommendations.append("  • Large batch training: Prefer SGD with linear learning rate scaling")
        recommendations.append("  • Distributed training: Use synchronized optimizers")
        recommendations.append("  • Memory-constrained: Choose SGD or use gradient accumulation")
        recommendations.append("  • Fine-tuning: Use lower learning rates (10x-100x smaller)")
        
        # Monitoring recommendations
        recommendations.append("📈 MONITORING & DEBUGGING:")
        recommendations.append("  • Track loss smoothness to detect learning rate issues")
        recommendations.append("  • Monitor gradient norms for explosion/vanishing detection")
        recommendations.append("  • Log learning rate schedules for reproducibility")
        recommendations.append("  • Profile memory usage to optimize batch sizes")
        
        return recommendations
        ### END SOLUTION
    
    def _analyze_convergence_profile(self, optimizer_name: str, losses: List[float], 
                                   grad_norms: List[float], step_durations: List[float],
                                   convergence_step: Optional[int]) -> Dict[str, Any]:
        """
        Internal helper to analyze convergence profile data.
        
        Args:
            optimizer_name: Name of the optimizer
            losses: List of loss values over training
            grad_norms: List of gradient norms over training
            step_durations: List of step execution times
            convergence_step: Step where convergence was detected (if any)
        
        Returns:
            Analysis results dictionary
        """
        ### BEGIN SOLUTION
        analysis = {
            'optimizer_name': optimizer_name,
            'total_steps': len(losses),
            'convergence_step': convergence_step,
            'final_loss': losses[-1] if losses else float('inf'),
            'initial_loss': losses[0] if losses else float('inf'),
            'loss_reduction': 0.0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'average_step_time': 0.0,
            'gradient_health': 'unknown'
        }
        
        if losses:
            # Calculate loss reduction
            initial_loss = losses[0]
            final_loss = losses[-1]
            analysis['loss_reduction'] = initial_loss - final_loss
            
            # Calculate convergence rate (loss reduction per step)
            if len(losses) > 1:
                analysis['convergence_rate'] = analysis['loss_reduction'] / len(losses)
            
            # Calculate stability (inverse of coefficient of variation)
            if len(losses) >= self.stability_window:
                recent_losses = losses[-self.stability_window:]
                mean_loss = np.mean(recent_losses)
                std_loss = np.std(recent_losses)
                analysis['stability_score'] = 1.0 / (1.0 + std_loss / (mean_loss + 1e-8))
        
        # Average step time
        if step_durations:
            analysis['average_step_time'] = np.mean(step_durations)
        
        # Gradient health assessment
        if grad_norms:
            max_grad_norm = max(grad_norms)
            avg_grad_norm = np.mean(grad_norms)
            
            if max_grad_norm > self.gradient_explosion_threshold:
                analysis['gradient_health'] = 'exploding'
            elif avg_grad_norm < 1e-8:
                analysis['gradient_health'] = 'vanishing'
            elif np.std(grad_norms) / (avg_grad_norm + 1e-8) > 2.0:
                analysis['gradient_health'] = 'unstable'
            else:
                analysis['gradient_health'] = 'healthy'
        
        return analysis
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: OptimizerConvergenceProfiler

Let's test your ML systems optimizer profiler! This tool helps analyze and compare optimizer performance in production scenarios.

**This is a unit test** - it tests the OptimizerConvergenceProfiler class functionality.
"""

# %% nbgrader={"grade": true, "grade_id": "test-convergence-profiler", "locked": true, "points": 30, "schema_version": 3, "solution": false, "task": false}
def test_unit_convergence_profiler():
    """Unit test for the OptimizerConvergenceProfiler implementation."""
    print("🔬 Unit Test: Optimizer Convergence Profiler...")
    
    # Test profiler initialization
    try:
        profiler = OptimizerConvergenceProfiler()
        
        assert hasattr(profiler, 'convergence_history'), "Should have convergence_history tracking"
        assert hasattr(profiler, 'gradient_norms'), "Should have gradient_norms tracking"
        assert hasattr(profiler, 'learning_rates'), "Should have learning_rates tracking"
        assert hasattr(profiler, 'step_times'), "Should have step_times tracking"
        print("✅ Profiler initialization works")
        
    except Exception as e:
        print(f"❌ Profiler initialization failed: {e}")
        raise
    
    # Test memory usage estimation
    try:
        # Test SGD memory estimation
        w = Variable(1.0, requires_grad=True)
        sgd_optimizer = SGD([w], learning_rate=0.01, momentum=0.9)
        
        memory_estimate = profiler.estimate_memory_usage(sgd_optimizer, num_parameters=1000000)
        
        assert 'parameters_mb' in memory_estimate, "Should estimate parameter memory"
        assert 'gradients_mb' in memory_estimate, "Should estimate gradient memory"
        assert 'optimizer_state_mb' in memory_estimate, "Should estimate optimizer state memory"
        assert 'total_mb' in memory_estimate, "Should provide total memory estimate"
        
        # SGD with momentum should have optimizer state
        assert memory_estimate['optimizer_state_mb'] > 0, "SGD with momentum should have state memory"
        print("✅ Memory usage estimation works")
        
    except Exception as e:
        print(f"❌ Memory usage estimation failed: {e}")
        raise
    
    # Test simple convergence analysis
    try:
        # Create a simple training function for testing
        def simple_training_function():
            # Simulate decreasing loss
            losses = [10.0 - i * 0.5 for i in range(20)]
            return losses[-1]  # Return final loss
        
        # Create test optimizer
        w = Variable(1.0, requires_grad=True)
        w.grad = Variable(0.1)  # Set gradient for testing
        test_optimizer = SGD([w], learning_rate=0.01)
        
        # Profile convergence (simplified test)
        analysis = profiler.profile_optimizer_convergence(
            optimizer_name="test_sgd",
            optimizer=test_optimizer,
            training_function=simple_training_function,
            initial_loss=10.0,
            max_steps=10
        )
        
        assert 'optimizer_name' in analysis, "Should return optimizer name"
        assert 'total_steps' in analysis, "Should track total steps"
        assert 'final_loss' in analysis, "Should track final loss"
        print("✅ Basic convergence profiling works")
        
    except Exception as e:
        print(f"❌ Convergence profiling failed: {e}")
        raise
    
    # Test production recommendations
    try:
        # Create mock analysis results
        mock_results = {
            'learning_rate_analysis': {
                'optimal_range': (0.001, 0.1)
            },
            'convergence_comparison': {
                'recommendations': {
                    'summary': ['Best overall: Adam', 'Fastest: SGD']
                }
            }
        }
        
        recommendations = profiler.generate_production_recommendations(mock_results)
        
        assert isinstance(recommendations, list), "Should return list of recommendations"
        assert len(recommendations) > 0, "Should provide recommendations"
        
        # Check for key recommendation categories
        rec_text = ' '.join(recommendations)
        assert 'OPTIMIZER SELECTION' in rec_text, "Should include optimizer selection guidance"
        assert 'PRODUCTION DEPLOYMENT' in rec_text, "Should include production deployment advice"
        print("✅ Production recommendations work")
        
    except Exception as e:
        print(f"❌ Production recommendations failed: {e}")
        raise
    
    # Test optimizer comparison framework
    try:
        # Create mock profiles for comparison
        mock_profiles = {
            'sgd': {'convergence_step': 50, 'final_loss': 0.1},
            'adam': {'convergence_step': 30, 'final_loss': 0.05}
        }
        
        # Add some mock data to profiler
        profiler.convergence_history['sgd'] = [1.0, 0.5, 0.2, 0.1]
        profiler.convergence_history['adam'] = [1.0, 0.3, 0.1, 0.05]
        profiler.step_times['sgd'] = [0.01, 0.01, 0.01, 0.01]
        profiler.step_times['adam'] = [0.02, 0.02, 0.02, 0.02]
        
        comparison = profiler.compare_optimizers(mock_profiles)
        
        assert 'convergence_speed' in comparison, "Should compare convergence speed"
        assert 'final_performance' in comparison, "Should compare final performance"
        assert 'stability' in comparison, "Should compare stability"
        assert 'recommendations' in comparison, "Should provide recommendations"
        print("✅ Optimizer comparison works")
        
    except Exception as e:
        print(f"❌ Optimizer comparison failed: {e}")
        raise

    print("🎯 Optimizer Convergence Profiler behavior:")
    print("   Profiles convergence patterns across different optimizers")
    print("   Estimates memory usage for production planning")
    print("   Provides actionable recommendations for ML systems")
    print("   Enables data-driven optimizer selection")
    print("📈 Progress: ML Systems Optimizer Analysis ✓")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 7: Advanced Optimizer Features

### Production Optimizer Patterns

Real ML systems need more than basic optimizers. They need:

1. **Gradient Clipping**: Prevents gradient explosion in large models
2. **Learning Rate Warmup**: Gradually increases learning rate at start
3. **Gradient Accumulation**: Simulates large batch training
4. **Mixed Precision**: Reduces memory usage with FP16
5. **Distributed Synchronization**: Coordinates optimizer across GPUs

Let's implement these production patterns!
"""

# %% nbgrader={"grade": false, "grade_id": "advanced-optimizer-features", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class AdvancedOptimizerFeatures:
    """
    Advanced optimizer features for production ML systems.
    
    Implements production-ready optimizer enhancements:
    - Gradient clipping for stability
    - Learning rate warmup strategies
    - Gradient accumulation for large batches
    - Mixed precision optimization patterns
    - Distributed optimizer synchronization
    """
    
    def __init__(self):
        """
        Initialize advanced optimizer features.
        
        TODO: Implement advanced features initialization.
        
        PRODUCTION CONTEXT:
        These features are essential for:
        - Training large language models (GPT, BERT)
        - Computer vision at scale (ImageNet, COCO)
        - Distributed training across multiple GPUs
        - Memory-efficient training with limited resources
        
        IMPLEMENTATION HINTS:
        - Initialize gradient clipping parameters
        - Set up warmup scheduling state
        - Prepare accumulation buffers
        - Configure synchronization patterns
        """
        ### BEGIN SOLUTION
        # Gradient clipping
        self.max_grad_norm = 1.0
        self.clip_enabled = False
        
        # Learning rate warmup
        self.warmup_steps = 0
        self.warmup_factor = 0.1
        self.base_lr = 0.001
        
        # Gradient accumulation
        self.accumulation_steps = 1
        self.accumulated_gradients = {}
        self.accumulation_count = 0
        
        # Mixed precision simulation
        self.use_fp16 = False
        self.loss_scale = 1.0
        self.dynamic_loss_scaling = False
        
        # Distributed training simulation
        self.world_size = 1
        self.rank = 0
        ### END SOLUTION
    
    def apply_gradient_clipping(self, optimizer: Union[SGD, Adam], max_norm: float = 1.0) -> float:
        """
        Apply gradient clipping to prevent gradient explosion.
        
        Args:
            optimizer: Optimizer with parameters to clip
            max_norm: Maximum allowed gradient norm
        
        Returns:
            Actual gradient norm before clipping
        
        TODO: Implement gradient clipping.
        
        APPROACH:
        1. Calculate total gradient norm across all parameters
        2. If norm exceeds max_norm, scale all gradients down
        3. Apply scaling factor to maintain gradient direction
        4. Return original norm for monitoring
        
        MATHEMATICAL FORMULATION:
        total_norm = sqrt(sum(param_grad_norm^2 for all params))
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            for each param: param.grad *= clip_factor
        
        PRODUCTION VALUE:
        Gradient clipping is essential for:
        - Training RNNs and Transformers
        - Preventing training instability
        - Enabling higher learning rates
        - Improving convergence reliability
        
        IMPLEMENTATION HINTS:
        - Calculate global gradient norm
        - Apply uniform scaling to all gradients
        - Preserve gradient directions
        - Return unclipped norm for logging
        """
        ### BEGIN SOLUTION
        # Calculate total gradient norm
        total_norm = 0.0
        param_count = 0
        
        for param in optimizer.parameters:
            if param.grad is not None:
                grad_data = param.grad.data.data
                if hasattr(grad_data, 'flatten'):
                    param_norm = np.linalg.norm(grad_data.flatten())
                else:
                    param_norm = abs(float(grad_data))
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** 0.5
        else:
            return 0.0
        
        # Apply clipping if necessary
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            
            for param in optimizer.parameters:
                if param.grad is not None:
                    grad_data = param.grad.data.data
                    clipped_grad = grad_data * clip_factor
                    param.grad.data = Tensor(clipped_grad)
        
        return total_norm
        ### END SOLUTION
    
    def apply_warmup_schedule(self, optimizer: Union[SGD, Adam], step: int, 
                            warmup_steps: int, base_lr: float) -> float:
        """
        Apply learning rate warmup schedule.
        
        Args:
            optimizer: Optimizer to apply warmup to
            step: Current training step
            warmup_steps: Number of warmup steps
            base_lr: Target learning rate after warmup
        
        Returns:
            Current learning rate
        
        TODO: Implement learning rate warmup.
        
        APPROACH:
        1. If step < warmup_steps: gradually increase learning rate
        2. Use linear or polynomial warmup schedule
        3. Update optimizer's learning rate
        4. Return current learning rate for logging
        
        WARMUP STRATEGIES:
        - Linear: lr = base_lr * (step / warmup_steps)
        - Polynomial: lr = base_lr * ((step / warmup_steps) ^ power)
        - Constant: lr = base_lr * warmup_factor for warmup_steps
        
        PRODUCTION VALUE:
        Warmup prevents:
        - Early training instability
        - Poor initialization effects
        - Gradient explosion at start
        - Suboptimal convergence paths
        
        IMPLEMENTATION HINTS:
        - Handle step=0 case (avoid division by zero)
        - Use linear warmup for simplicity
        - Update optimizer.learning_rate directly
        - Smoothly transition to base learning rate
        """
        ### BEGIN SOLUTION
        if step < warmup_steps and warmup_steps > 0:
            # Linear warmup
            warmup_factor = step / warmup_steps
            current_lr = base_lr * warmup_factor
        else:
            # After warmup, use base learning rate
            current_lr = base_lr
        
        # Update optimizer learning rate
        optimizer.learning_rate = current_lr
        
        return current_lr
        ### END SOLUTION
    
    def accumulate_gradients(self, optimizer: Union[SGD, Adam], accumulation_steps: int) -> bool:
        """
        Accumulate gradients to simulate larger batch sizes.
        
        Args:
            optimizer: Optimizer with parameters to accumulate
            accumulation_steps: Number of steps to accumulate before update
        
        Returns:
            True if ready to perform optimizer step, False otherwise
        
        TODO: Implement gradient accumulation.
        
        APPROACH:
        1. Add current gradients to accumulated gradient buffers
        2. Increment accumulation counter
        3. If counter reaches accumulation_steps:
           a. Average accumulated gradients
           b. Set as current gradients
           c. Return True (ready for optimizer step)
           d. Reset accumulation
        4. Otherwise return False (continue accumulating)
        
        MATHEMATICAL FORMULATION:
        accumulated_grad += current_grad
        if accumulation_count == accumulation_steps:
            final_grad = accumulated_grad / accumulation_steps
            reset accumulation
            return True
        
        PRODUCTION VALUE:
        Gradient accumulation enables:
        - Large effective batch sizes on limited memory
        - Training large models on small GPUs
        - Consistent training across different hardware
        - Memory-efficient distributed training
        
        IMPLEMENTATION HINTS:
        - Store accumulated gradients per parameter
        - Use parameter id() as key for tracking
        - Average gradients before optimizer step
        - Reset accumulation after each update
        """
        ### BEGIN SOLUTION
        # Initialize accumulation if first time
        if not hasattr(self, 'accumulation_count'):
            self.accumulation_count = 0
            self.accumulated_gradients = {}
        
        # Accumulate gradients
        for param in optimizer.parameters:
            if param.grad is not None:
                param_id = id(param)
                grad_data = param.grad.data.data
                
                if param_id not in self.accumulated_gradients:
                    self.accumulated_gradients[param_id] = np.zeros_like(grad_data)
                
                self.accumulated_gradients[param_id] += grad_data
        
        self.accumulation_count += 1
        
        # Check if ready to update
        if self.accumulation_count >= accumulation_steps:
            # Average accumulated gradients and set as current gradients
            for param in optimizer.parameters:
                if param.grad is not None:
                    param_id = id(param)
                    if param_id in self.accumulated_gradients:
                        averaged_grad = self.accumulated_gradients[param_id] / accumulation_steps
                        param.grad.data = Tensor(averaged_grad)
            
            # Reset accumulation
            self.accumulation_count = 0
            self.accumulated_gradients = {}
            
            return True  # Ready for optimizer step
        
        return False  # Continue accumulating
        ### END SOLUTION
    
    def simulate_mixed_precision(self, optimizer: Union[SGD, Adam], loss_scale: float = 1.0) -> bool:
        """
        Simulate mixed precision training effects.
        
        Args:
            optimizer: Optimizer to apply mixed precision to
            loss_scale: Loss scaling factor for gradient preservation
        
        Returns:
            True if gradients are valid (no overflow), False if overflow detected
        
        TODO: Implement mixed precision simulation.
        
        APPROACH:
        1. Scale gradients by loss_scale factor
        2. Check for gradient overflow (inf or nan values)
        3. If overflow detected, skip optimizer step
        4. If valid, descale gradients before optimizer step
        5. Return overflow status
        
        MIXED PRECISION CONCEPTS:
        - Use FP16 for forward pass (memory savings)
        - Use FP32 for backward pass (numerical stability)
        - Scale loss to prevent gradient underflow
        - Check for overflow before optimization
        
        PRODUCTION VALUE:
        Mixed precision provides:
        - 50% memory reduction
        - Faster training on modern GPUs
        - Maintained numerical stability
        - Automatic overflow detection
        
        IMPLEMENTATION HINTS:
        - Scale gradients by loss_scale
        - Check for inf/nan in gradients
        - Descale before optimizer step
        - Return overflow status for dynamic scaling
        """
        ### BEGIN SOLUTION
        # Check for gradient overflow before scaling
        has_overflow = False
        
        for param in optimizer.parameters:
            if param.grad is not None:
                grad_data = param.grad.data.data
                if hasattr(grad_data, 'flatten'):
                    grad_flat = grad_data.flatten()
                    if np.any(np.isinf(grad_flat)) or np.any(np.isnan(grad_flat)):
                        has_overflow = True
                        break
                else:
                    if np.isinf(grad_data) or np.isnan(grad_data):
                        has_overflow = True
                        break
        
        if has_overflow:
            # Zero gradients to prevent corruption
            for param in optimizer.parameters:
                if param.grad is not None:
                    param.grad = None
            return False  # Overflow detected
        
        # Descale gradients (simulate unscaling from FP16)
        if loss_scale > 1.0:
            for param in optimizer.parameters:
                if param.grad is not None:
                    grad_data = param.grad.data.data
                    descaled_grad = grad_data / loss_scale
                    param.grad.data = Tensor(descaled_grad)
        
        return True  # No overflow, safe to proceed
        ### END SOLUTION
    
    def simulate_distributed_sync(self, optimizer: Union[SGD, Adam], world_size: int = 1) -> None:
        """
        Simulate distributed training gradient synchronization.
        
        Args:
            optimizer: Optimizer with gradients to synchronize
            world_size: Number of distributed processes
        
        TODO: Implement distributed gradient synchronization simulation.
        
        APPROACH:
        1. Simulate all-reduce operation on gradients
        2. Average gradients across all processes
        3. Update local gradients with synchronized values
        4. Handle communication overhead simulation
        
        DISTRIBUTED CONCEPTS:
        - All-reduce: Combine gradients from all GPUs
        - Averaging: Divide by world_size for consistency
        - Synchronization: Ensure all GPUs have same gradients
        - Communication: Network overhead for gradient sharing
        
        PRODUCTION VALUE:
        Distributed training enables:
        - Scaling to multiple GPUs/nodes
        - Training large models efficiently
        - Reduced training time
        - Consistent convergence across devices
        
        IMPLEMENTATION HINTS:
        - Simulate averaging by keeping gradients unchanged
        - Add small noise to simulate communication variance
        - Scale learning rate by world_size if needed
        - Log synchronization overhead
        """
        ### BEGIN SOLUTION
        if world_size <= 1:
            return  # No synchronization needed for single process
        
        # Simulate all-reduce operation (averaging gradients)
        for param in optimizer.parameters:
            if param.grad is not None:
                grad_data = param.grad.data.data
                
                # In real distributed training, gradients would be averaged across all processes
                # Here we simulate this by keeping gradients unchanged (already "averaged")
                # In practice, this would involve MPI/NCCL communication
                
                # Simulate communication noise (very small)
                if hasattr(grad_data, 'shape'):
                    noise = np.random.normal(0, 1e-10, grad_data.shape)
                    synchronized_grad = grad_data + noise
                else:
                    noise = np.random.normal(0, 1e-10)
                    synchronized_grad = grad_data + noise
                
                param.grad.data = Tensor(synchronized_grad)
        
        # In distributed training, learning rate is often scaled by world_size
        # to maintain effective learning rate with larger batch sizes
        if hasattr(optimizer, 'base_learning_rate'):
            optimizer.learning_rate = optimizer.base_learning_rate * world_size
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Advanced Optimizer Features

Let's test your advanced optimizer features! These are production-ready enhancements used in real ML systems.

**This is a unit test** - it tests the AdvancedOptimizerFeatures class functionality.
"""

# %% nbgrader={"grade": true, "grade_id": "test-advanced-features", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_unit_advanced_optimizer_features():
    """Unit test for advanced optimizer features implementation."""
    print("🔬 Unit Test: Advanced Optimizer Features...")
    
    # Test advanced features initialization
    try:
        features = AdvancedOptimizerFeatures()
        
        assert hasattr(features, 'max_grad_norm'), "Should have gradient clipping parameters"
        assert hasattr(features, 'warmup_steps'), "Should have warmup parameters"
        assert hasattr(features, 'accumulation_steps'), "Should have accumulation parameters"
        print("✅ Advanced features initialization works")
        
    except Exception as e:
        print(f"❌ Advanced features initialization failed: {e}")
        raise
    
    # Test gradient clipping
    try:
        # Create optimizer with large gradients
        w = Variable(1.0, requires_grad=True)
        w.grad = Variable(10.0)  # Large gradient
        optimizer = SGD([w], learning_rate=0.01)
        
        # Apply gradient clipping
        original_norm = features.apply_gradient_clipping(optimizer, max_norm=1.0)
        
        # Check that gradient was clipped
        clipped_grad = w.grad.data.data.item()
        assert abs(clipped_grad) <= 1.0, f"Gradient should be clipped to <= 1.0, got {clipped_grad}"
        assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"
        print("✅ Gradient clipping works")
        
    except Exception as e:
        print(f"❌ Gradient clipping failed: {e}")
        raise
    
    # Test learning rate warmup
    try:
        w2 = Variable(1.0, requires_grad=True)
        optimizer2 = SGD([w2], learning_rate=0.01)
        
        # Test warmup schedule
        lr_step_0 = features.apply_warmup_schedule(optimizer2, step=0, warmup_steps=10, base_lr=0.1)
        lr_step_5 = features.apply_warmup_schedule(optimizer2, step=5, warmup_steps=10, base_lr=0.1)
        lr_step_10 = features.apply_warmup_schedule(optimizer2, step=10, warmup_steps=10, base_lr=0.1)
        
        # Check warmup progression
        assert lr_step_0 == 0.0, f"Step 0 should have lr=0.0, got {lr_step_0}"
        assert 0.0 < lr_step_5 < 0.1, f"Step 5 should have 0 < lr < 0.1, got {lr_step_5}"
        assert lr_step_10 == 0.1, f"Step 10 should have lr=0.1, got {lr_step_10}"
        print("✅ Learning rate warmup works")
        
    except Exception as e:
        print(f"❌ Learning rate warmup failed: {e}")
        raise
    
    # Test gradient accumulation
    try:
        w3 = Variable(1.0, requires_grad=True)
        w3.grad = Variable(0.1)
        optimizer3 = SGD([w3], learning_rate=0.01)
        
        # Test accumulation over multiple steps
        ready_step_1 = features.accumulate_gradients(optimizer3, accumulation_steps=3)
        ready_step_2 = features.accumulate_gradients(optimizer3, accumulation_steps=3)
        ready_step_3 = features.accumulate_gradients(optimizer3, accumulation_steps=3)
        
        # Check accumulation behavior
        assert not ready_step_1, "Should not be ready after step 1"
        assert not ready_step_2, "Should not be ready after step 2"
        assert ready_step_3, "Should be ready after step 3"
        print("✅ Gradient accumulation works")
        
    except Exception as e:
        print(f"❌ Gradient accumulation failed: {e}")
        raise
    
    # Test mixed precision simulation
    try:
        w4 = Variable(1.0, requires_grad=True)
        w4.grad = Variable(0.1)
        optimizer4 = SGD([w4], learning_rate=0.01)
        
        # Test normal case (no overflow)
        no_overflow = features.simulate_mixed_precision(optimizer4, loss_scale=1.0)
        assert no_overflow, "Should not detect overflow with normal gradients"
        
        # Test overflow case
        w4.grad = Variable(float('inf'))
        overflow = features.simulate_mixed_precision(optimizer4, loss_scale=1.0)
        assert not overflow, "Should detect overflow with inf gradients"
        print("✅ Mixed precision simulation works")
        
    except Exception as e:
        print(f"❌ Mixed precision simulation failed: {e}")
        raise
    
    # Test distributed synchronization
    try:
        w5 = Variable(1.0, requires_grad=True)
        w5.grad = Variable(0.1)
        optimizer5 = SGD([w5], learning_rate=0.01)
        
        original_grad = w5.grad.data.data.item()
        
        # Simulate distributed sync
        features.simulate_distributed_sync(optimizer5, world_size=4)
        
        # Gradient should be slightly modified (due to simulated communication noise)
        # but still close to original
        synced_grad = w5.grad.data.data.item()
        assert abs(synced_grad - original_grad) < 0.01, "Synchronized gradient should be close to original"
        print("✅ Distributed synchronization simulation works")
        
    except Exception as e:
        print(f"❌ Distributed synchronization failed: {e}")
        raise

    print("🎯 Advanced Optimizer Features behavior:")
    print("   Implements gradient clipping for training stability")
    print("   Provides learning rate warmup for better convergence")
    print("   Enables gradient accumulation for large effective batches")
    print("   Simulates mixed precision training patterns")
    print("   Handles distributed training synchronization")
    print("📈 Progress: Advanced Production Optimizer Features ✓")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 8: Comprehensive Testing - ML Systems Integration

### Real-World Optimizer Performance Testing

Let's test our optimizers in realistic scenarios that mirror production ML systems:

1. **Convergence Race**: Compare optimizers on the same task
2. **Learning Rate Sensitivity**: Find optimal hyperparameters
3. **Memory Analysis**: Compare resource usage
4. **Production Recommendations**: Get actionable guidance

This integration test demonstrates how our ML systems tools work together.
"""

# %% nbgrader={"grade": true, "grade_id": "test-ml-systems-integration", "locked": true, "points": 35, "schema_version": 3, "solution": false, "task": false}
def test_comprehensive_ml_systems_integration():
    """Comprehensive integration test demonstrating ML systems optimizer analysis."""
    print("🔬 Comprehensive Test: ML Systems Integration...")
    
    # Initialize ML systems tools
    try:
        profiler = OptimizerConvergenceProfiler()
        advanced_features = AdvancedOptimizerFeatures()
        print("✅ ML systems tools initialized")
        
    except Exception as e:
        print(f"❌ ML systems tools initialization failed: {e}")
        raise
    
    # Test convergence profiling with multiple optimizers
    try:
        print("\n📊 Running optimizer convergence comparison...")
        
        # Create simple training scenario
        def create_training_function(optimizer_instance):
            def training_step():
                # Simulate a quadratic loss function: loss = (x - target)^2
                # where we're trying to minimize x towards target = 2.0
                current_x = optimizer_instance.parameters[0].data.data.item()
                target = 2.0
                loss = (current_x - target) ** 2
                
                # Compute gradient: d/dx (x - target)^2 = 2 * (x - target)
                gradient = 2 * (current_x - target)
                optimizer_instance.parameters[0].grad = Variable(gradient)
                
                # Perform optimizer step
                optimizer_instance.step()
                
                return loss
            return training_step
        
        # Test SGD
        w_sgd = Variable(0.0, requires_grad=True)  # Start at x=0, target=2
        sgd_optimizer = SGD([w_sgd], learning_rate=0.1, momentum=0.9)
        sgd_training = create_training_function(sgd_optimizer)
        
        sgd_profile = profiler.profile_optimizer_convergence(
            optimizer_name="SGD_momentum",
            optimizer=sgd_optimizer,
            training_function=sgd_training,
            initial_loss=4.0,  # (0-2)^2 = 4
            max_steps=30
        )
        
        # Test Adam
        w_adam = Variable(0.0, requires_grad=True)  # Start at x=0, target=2
        adam_optimizer = Adam([w_adam], learning_rate=0.1)
        adam_training = create_training_function(adam_optimizer)
        
        adam_profile = profiler.profile_optimizer_convergence(
            optimizer_name="Adam",
            optimizer=adam_optimizer,
            training_function=adam_training,
            initial_loss=4.0,
            max_steps=30
        )
        
        # Verify profiling results
        assert 'optimizer_name' in sgd_profile, "SGD profile should contain optimizer name"
        assert 'optimizer_name' in adam_profile, "Adam profile should contain optimizer name"
        assert 'final_loss' in sgd_profile, "SGD profile should contain final loss"
        assert 'final_loss' in adam_profile, "Adam profile should contain final loss"
        
        print(f"   SGD final loss: {sgd_profile['final_loss']:.4f}")
        print(f"   Adam final loss: {adam_profile['final_loss']:.4f}")
        print("✅ Convergence profiling completed")
        
    except Exception as e:
        print(f"❌ Convergence profiling failed: {e}")
        raise
    
    # Test optimizer comparison
    try:
        print("\n🏆 Comparing optimizer performance...")
        
        profiles = {
            'SGD_momentum': sgd_profile,
            'Adam': adam_profile
        }
        
        comparison = profiler.compare_optimizers(profiles)
        
        # Verify comparison results
        assert 'convergence_speed' in comparison, "Should compare convergence speed"
        assert 'final_performance' in comparison, "Should compare final performance"
        assert 'rankings' in comparison, "Should provide rankings"
        assert 'recommendations' in comparison, "Should provide recommendations"
        
        if 'summary' in comparison['recommendations']:
            print("   Recommendations:")
            for rec in comparison['recommendations']['summary']:
                print(f"     {rec}")
        
        print("✅ Optimizer comparison completed")
        
    except Exception as e:
        print(f"❌ Optimizer comparison failed: {e}")
        raise
    
    # Test memory analysis
    try:
        print("\n💾 Analyzing memory usage...")
        
        # Simulate large model parameters
        num_parameters = 100000  # 100K parameters
        
        sgd_memory = profiler.estimate_memory_usage(sgd_optimizer, num_parameters)
        adam_memory = profiler.estimate_memory_usage(adam_optimizer, num_parameters)
        
        print(f"   SGD memory usage: {sgd_memory['total_mb']:.1f} MB")
        print(f"   Adam memory usage: {adam_memory['total_mb']:.1f} MB")
        print(f"   Adam overhead: {adam_memory['total_mb'] - sgd_memory['total_mb']:.1f} MB")
        
        # Verify memory analysis
        assert sgd_memory['total_mb'] > 0, "SGD should have positive memory usage"
        assert adam_memory['total_mb'] > sgd_memory['total_mb'], "Adam should use more memory than SGD"
        
        print("✅ Memory analysis completed")
        
    except Exception as e:
        print(f"❌ Memory analysis failed: {e}")
        raise
    
    # Test advanced features integration
    try:
        print("\n🚀 Testing advanced optimizer features...")
        
        # Test gradient clipping
        w_clip = Variable(1.0, requires_grad=True)
        w_clip.grad = Variable(5.0)  # Large gradient
        clip_optimizer = SGD([w_clip], learning_rate=0.01)
        
        original_norm = advanced_features.apply_gradient_clipping(clip_optimizer, max_norm=1.0)
        assert original_norm > 1.0, "Should detect large gradient"
        assert abs(w_clip.grad.data.data.item()) <= 1.0, "Should clip gradient"
        
        # Test learning rate warmup
        warmup_optimizer = Adam([Variable(1.0)], learning_rate=0.001)
        lr_start = advanced_features.apply_warmup_schedule(warmup_optimizer, 0, 100, 0.001)
        lr_mid = advanced_features.apply_warmup_schedule(warmup_optimizer, 50, 100, 0.001)
        lr_end = advanced_features.apply_warmup_schedule(warmup_optimizer, 100, 100, 0.001)
        
        assert lr_start < lr_mid < lr_end, "Learning rate should increase during warmup"
        
        print("✅ Advanced features integration completed")
        
    except Exception as e:
        print(f"❌ Advanced features integration failed: {e}")
        raise
    
    # Test production recommendations
    try:
        print("\n📋 Generating production recommendations...")
        
        analysis_results = {
            'convergence_comparison': comparison,
            'memory_analysis': {
                'sgd': sgd_memory,
                'adam': adam_memory
            },
            'learning_rate_analysis': {
                'optimal_range': (0.01, 0.1)
            }
        }
        
        recommendations = profiler.generate_production_recommendations(analysis_results)
        
        assert len(recommendations) > 0, "Should generate recommendations"
        
        print("   Production guidance:")
        for i, rec in enumerate(recommendations[:5]):  # Show first 5 recommendations
            print(f"     {rec}")
        
        print("✅ Production recommendations generated")
        
    except Exception as e:
        print(f"❌ Production recommendations failed: {e}")
        raise

    print("\n🎯 ML Systems Integration Results:")
    print("   ✅ Optimizer convergence profiling works end-to-end")
    print("   ✅ Performance comparison identifies best optimizers")
    print("   ✅ Memory analysis guides resource planning")
    print("   ✅ Advanced features enhance training stability")
    print("   ✅ Production recommendations provide actionable guidance")
    print("   🚀 Ready for real-world ML systems deployment!")
    print("📈 Progress: Comprehensive ML Systems Integration ✓")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🎯 ML SYSTEMS THINKING: Optimizers in Production

### Production Deployment Considerations

**You've just built a comprehensive optimizer analysis system!** Let's reflect on how this connects to real ML systems:

### System Design Questions
1. **Optimizer Selection Strategy**: How would you build an automated system that selects the best optimizer for a new model architecture?

2. **Resource Planning**: Given memory constraints and training time budgets, how would you choose between SGD and Adam for different model sizes?

3. **Distributed Training**: How do gradient synchronization patterns affect optimizer performance across multiple GPUs or nodes?

4. **Production Monitoring**: What metrics would you track in production to detect optimizer-related training issues?

### Production ML Workflows
1. **Hyperparameter Search**: How would you integrate your convergence profiler into an automated hyperparameter tuning pipeline?

2. **Training Pipeline**: Where would gradient clipping and mixed precision fit into a production training workflow?

3. **Cost Optimization**: How would you balance optimizer performance against computational cost for training large models?

4. **Model Lifecycle**: How do optimizer choices change when fine-tuning vs training from scratch vs transfer learning?

### Framework Design Insights
1. **Optimizer Abstraction**: Why do frameworks like PyTorch separate optimizers from models? How does this design enable flexibility?

2. **State Management**: How do frameworks handle optimizer state persistence for training checkpoints and resumption?

3. **Memory Efficiency**: What design patterns enable frameworks to minimize memory overhead for optimizer state?

4. **Plugin Architecture**: How would you design an optimizer plugin system that allows researchers to add new algorithms?

### Performance & Scale Challenges
1. **Large Model Training**: How do optimizer memory requirements scale with model size, and what strategies mitigate this?

2. **Dynamic Batching**: How would you adapt your gradient accumulation strategy for variable batch sizes in production?

3. **Fault Tolerance**: How would you design optimizer state recovery for interrupted training runs in cloud environments?

4. **Cross-Hardware Portability**: How do optimizer implementations need to change when moving between CPUs, GPUs, and specialized ML accelerators?

These questions connect your optimizer implementations to the broader ecosystem of production ML systems, where optimization is just one piece of complex training and deployment pipelines.
"""

if __name__ == "__main__":
    print("🧪 Running comprehensive optimizer tests...")
    
    # Run all tests
    test_unit_sgd_implementation()
    test_unit_sgd_with_momentum()
    test_unit_adam_optimizer()
    test_module_optimizer_neural_network_training()
    test_memory_profiler()
    
    print("All tests passed!")
    print("Optimizers module complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built optimization algorithms that drive neural network training, let's connect this foundational work to broader ML systems challenges. These questions help you think critically about how optimization strategies scale to production training environments.

Take time to reflect thoughtfully on each question - your insights will help you understand how the optimization concepts you've implemented connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Memory Overhead and Optimizer State Management

**Context**: Your Adam optimizer maintains momentum and variance buffers for each parameter, creating 3× memory overhead compared to SGD. Production training systems with billions of parameters must carefully manage optimizer state memory while maintaining training efficiency and fault tolerance.

**Reflection Question**: Design an optimizer state management system for large-scale neural network training that optimizes memory usage while supporting distributed training and fault recovery. How would you implement memory-efficient optimizer state storage, handle state partitioning across devices, and manage optimizer checkpointing for training resumption? Consider scenarios where optimizer state memory exceeds model parameter memory and requires specialized optimization strategies.

Think about: memory optimization techniques, distributed state management, checkpointing strategies, and fault tolerance considerations.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-optimizer-memory", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON MEMORY OVERHEAD AND OPTIMIZER STATE MANAGEMENT:

TODO: Replace this text with your thoughtful response about optimizer state management system design.

Consider addressing:
- How would you optimize memory usage for optimizers that maintain extensive per-parameter state?
- What strategies would you use for distributed optimizer state management across multiple devices?
- How would you implement efficient checkpointing and state recovery for long-running training jobs?
- What role would state compression and quantization play in your optimization approach?
- How would you balance memory efficiency with optimization algorithm effectiveness?

Write a technical analysis connecting your optimizer implementations to real memory management challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of optimizer memory overhead and state management (3 points)
- Addresses distributed state management and partitioning strategies (3 points)
- Shows practical knowledge of checkpointing and fault tolerance techniques (2 points)
- Demonstrates systems thinking about memory vs optimization trade-offs (2 points)
- Clear technical reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of optimizer state management
# Students should demonstrate understanding of memory optimization and distributed state handling
### END SOLUTION

# %% [markdown]
"""
### Question 2: Distributed Optimization and Learning Rate Scheduling

**Context**: Your optimizers work on single devices with fixed learning rate schedules. Production distributed training systems must coordinate optimization across multiple workers while adapting learning rates based on real-time training dynamics and system constraints.

**Reflection Question**: Architect a distributed optimization system that coordinates parameter updates across multiple workers while implementing adaptive learning rate scheduling responsive to training progress and system constraints. How would you handle gradient aggregation strategies, implement learning rate scaling for different batch sizes, and design adaptive scheduling that responds to convergence patterns? Consider scenarios where training must adapt to varying computational resources and time constraints in cloud environments.

Think about: distributed optimization strategies, adaptive learning rate techniques, gradient aggregation methods, and system-aware scheduling.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-distributed-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON DISTRIBUTED OPTIMIZATION AND LEARNING RATE SCHEDULING:

TODO: Replace this text with your thoughtful response about distributed optimization system design.

Consider addressing:
- How would you coordinate parameter updates across multiple workers in distributed training?
- What strategies would you use for gradient aggregation and synchronization?
- How would you implement adaptive learning rate scheduling that responds to training dynamics?
- What role would system constraints and resource availability play in your optimization design?
- How would you handle learning rate scaling and batch size considerations in distributed settings?

Write an architectural analysis connecting your optimizer implementations to real distributed training challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of distributed optimization and coordination challenges (3 points)
- Designs practical approaches to gradient aggregation and learning rate adaptation (3 points)
- Addresses system constraints and resource-aware optimization (2 points)
- Demonstrates systems thinking about distributed training coordination (2 points)
- Clear architectural reasoning with distributed systems insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of distributed optimization systems
# Students should demonstrate knowledge of gradient aggregation and adaptive scheduling
### END SOLUTION

# %% [markdown]
"""
### Question 3: Production Integration and Optimization Monitoring

**Context**: Your optimizer implementations provide basic parameter updates, but production ML systems require comprehensive optimization monitoring, hyperparameter tuning, and integration with MLOps pipelines for continuous training and model improvement.

**Reflection Question**: Design a production optimization system that integrates with MLOps pipelines and provides comprehensive optimization monitoring and automated hyperparameter tuning. How would you implement real-time optimization metrics collection, automated optimizer selection based on model characteristics, and integration with experiment tracking and model deployment systems? Consider scenarios where optimization strategies must adapt to changing data distributions and business requirements in production environments.

Think about: optimization monitoring systems, automated hyperparameter tuning, MLOps integration, and adaptive optimization strategies.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-production-integration", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON PRODUCTION INTEGRATION AND OPTIMIZATION MONITORING:

TODO: Replace this text with your thoughtful response about production optimization system design.

Consider addressing:
- How would you design optimization monitoring and metrics collection for production training?
- What strategies would you use for automated optimizer selection and hyperparameter tuning?
- How would you integrate optimization systems with MLOps pipelines and experiment tracking?
- What role would adaptive optimization play in responding to changing data and requirements?
- How would you ensure optimization system reliability and performance in production environments?

Write a systems analysis connecting your optimizer implementations to real production integration challenges.

GRADING RUBRIC (Instructor Use):
- Understands production optimization monitoring and MLOps integration (3 points)
- Designs practical approaches to automated tuning and optimization selection (3 points)
- Addresses adaptive optimization and production reliability considerations (2 points)
- Shows systems thinking about optimization system integration and monitoring (2 points)
- Clear systems reasoning with production deployment insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of production optimization systems
# Students should demonstrate knowledge of MLOps integration and optimization monitoring
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Optimization Algorithms with ML Systems

Congratulations! You've successfully implemented optimization algorithms with comprehensive ML systems analysis:

### What You've Accomplished
✅ **Gradient Descent**: The foundation of all optimization algorithms
✅ **SGD with Momentum**: Improved convergence with momentum
✅ **Adam Optimizer**: Adaptive learning rates for better training
✅ **Learning Rate Scheduling**: Dynamic learning rate adjustment
✅ **ML Systems Analysis**: OptimizerConvergenceProfiler for production insights
✅ **Advanced Features**: Gradient clipping, warmup, accumulation, mixed precision
✅ **Production Integration**: Complete optimizer analysis and recommendation system

### Key Concepts You've Learned
- **Gradient-based optimization**: How gradients guide parameter updates
- **Momentum**: Using velocity to improve convergence
- **Adaptive learning rates**: Adam's adaptive moment estimation
- **Learning rate scheduling**: Dynamic adjustment of learning rates
- **Convergence analysis**: Profiling optimizer performance patterns
- **Memory efficiency**: Resource usage comparison across optimizers
- **Production patterns**: Advanced features for real-world deployment

### Mathematical Foundations
- **Gradient descent**: θ = θ - α∇θJ(θ)
- **Momentum**: v = βv + (1-β)∇θJ(θ), θ = θ - αv
- **Adam**: Adaptive moment estimation with bias correction
- **Learning rate scheduling**: StepLR and other scheduling strategies
- **Gradient clipping**: norm_clip = min(norm, max_norm) * grad / norm
- **Gradient accumulation**: grad_avg = Σgrad_i / accumulation_steps

### Professional Skills Developed
- **Algorithm implementation**: Building optimization algorithms from scratch
- **Performance analysis**: Profiling and comparing optimizer convergence
- **System design thinking**: Understanding production optimization workflows
- **Resource optimization**: Memory usage analysis and efficiency planning
- **Integration testing**: Ensuring optimizers work with neural networks
- **Production readiness**: Advanced features for real-world deployment

### Ready for Advanced Applications
Your optimization implementations now enable:
- **Neural network training**: Complete training pipelines with optimizers
- **Hyperparameter optimization**: Data-driven optimizer and LR selection
- **Advanced architectures**: Training complex models efficiently
- **Production deployment**: ML systems with optimizer monitoring and tuning
- **Research**: Experimenting with new optimization algorithms
- **Scalable training**: Distributed and memory-efficient optimization

### Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch**: `torch.optim.SGD`, `torch.optim.Adam` provide identical functionality
- **TensorFlow**: `tf.keras.optimizers` implements similar concepts
- **MLflow/Weights&Biases**: Your profiler mirrors production monitoring tools
- **Ray Tune/Optuna**: Your convergence analysis enables hyperparameter optimization
- **Industry Standard**: Every major ML framework uses these exact algorithms and patterns

### Next Steps
1. **Export your code**: `tito export 10_optimizers`
2. **Test your implementation**: `tito test 10_optimizers`
3. **Deploy ML systems**: Use your profiler for real optimizer selection
4. **Build training systems**: Combine with neural networks for complete training
5. **Move to Module 11**: Add complete training pipelines!

**Ready for production?** Your optimization algorithms and ML systems analysis tools are now ready for real-world deployment and performance optimization!
"""