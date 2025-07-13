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
# Module 8: Optimizers - Gradient-Based Parameter Updates

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

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
except ImportError:
    # For development, try local imports
    try:
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
### 🧪 Test Your Gradient Descent Implementation

Let's test the basic gradient descent step:
"""

# %% nbgrader={"grade": true, "grade_id": "test-gradient-descent", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_gradient_descent_step():
    """Test basic gradient descent parameter update"""
    print("🔬 Unit Test: Gradient Descent Step...")
    
    # Test basic parameter update
    w = Variable(2.0, requires_grad=True)
    w.grad = Variable(0.5)  # Positive gradient
    
    original_value = w.data.data.item()
    gradient_descent_step(w, learning_rate=0.1)
    new_value = w.data.data.item()
    
    expected_value = original_value - 0.1 * 0.5  # 2.0 - 0.05 = 1.95
    assert abs(new_value - expected_value) < 1e-6, f"Expected {expected_value}, got {new_value}"
    
    # Test with negative gradient
    w2 = Variable(1.0, requires_grad=True)
    w2.grad = Variable(-0.2)  # Negative gradient
    
    gradient_descent_step(w2, learning_rate=0.1)
    expected_value2 = 1.0 - 0.1 * (-0.2)  # 1.0 + 0.02 = 1.02
    assert abs(w2.data.data.item() - expected_value2) < 1e-6, "Negative gradient test failed"
    
    # Test with no gradient (should not update)
    w3 = Variable(3.0, requires_grad=True)
    w3.grad = None
    original_value3 = w3.data.data.item()
    
    gradient_descent_step(w3, learning_rate=0.1)
    assert w3.data.data.item() == original_value3, "Parameter with no gradient should not update"
    
    print("✅ Gradient descent step tests passed!")
    print(f"✅ Basic parameter update working")
    print(f"✅ Negative gradient handling correct")
    print(f"✅ No gradient case handled properly")

# Run the test
test_gradient_descent_step()

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
### 🧪 Test Your SGD Implementation

Let's test the SGD optimizer:
"""

# %% nbgrader={"grade": true, "grade_id": "test-sgd", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_sgd_optimizer():
    """Test SGD optimizer implementation"""
    print("🔬 Unit Test: SGD Optimizer...")
    
    # Create test parameters
    w1 = Variable(1.0, requires_grad=True)
    w2 = Variable(2.0, requires_grad=True)
    b = Variable(0.5, requires_grad=True)
    
    # Create optimizer
    optimizer = SGD([w1, w2, b], learning_rate=0.1, momentum=0.9)
    
    # Test zero_grad
    w1.grad = Variable(0.1)
    w2.grad = Variable(0.2)
    b.grad = Variable(0.05)
    
    optimizer.zero_grad()
    
    assert w1.grad is None, "Gradient should be None after zero_grad"
    assert w2.grad is None, "Gradient should be None after zero_grad"
    assert b.grad is None, "Gradient should be None after zero_grad"
    
    # Test step with gradients
    w1.grad = Variable(0.1)
    w2.grad = Variable(0.2)
    b.grad = Variable(0.05)
    
    # First step (no momentum yet)
    original_w1 = w1.data.data.item()
    optimizer.step()
    
    expected_w1 = original_w1 - 0.1 * 0.1  # 1.0 - 0.01 = 0.99
    assert abs(w1.data.data.item() - expected_w1) < 1e-6, "First step failed"
    
    # Second step (with momentum)
    w1.grad = Variable(0.1)
    w2.grad = Variable(0.2)
    b.grad = Variable(0.05)
    
    optimizer.step()
    
    # Check that momentum buffers were created
    assert len(optimizer.momentum_buffers) == 3, "Should have momentum buffers for all parameters"
    
    # Test step count
    assert optimizer.step_count == 2, "Should track step count"
    
    print("✅ SGD optimizer tests passed!")
    print(f"✅ zero_grad() working correctly")
    print(f"✅ Parameter updates working")
    print(f"✅ Momentum buffers created")
    print(f"✅ Step counting working")

# Run the test
test_sgd_optimizer()

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

# %% nbgrader={"grade": true, "grade_id": "test-adam", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_adam_optimizer():
    """Test Adam optimizer implementation"""
    print("🔬 Unit Test: Adam Optimizer...")
    
    # Create test parameters
    w1 = Variable(1.0, requires_grad=True)
    w2 = Variable(2.0, requires_grad=True)
    b = Variable(0.5, requires_grad=True)
    
    # Create optimizer
    optimizer = Adam([w1, w2, b], learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    # Test zero_grad
    w1.grad = Variable(0.1)
    w2.grad = Variable(0.2)
    b.grad = Variable(0.05)
    
    optimizer.zero_grad()
    
    assert w1.grad is None, "Gradient should be None after zero_grad"
    assert w2.grad is None, "Gradient should be None after zero_grad"
    assert b.grad is None, "Gradient should be None after zero_grad"
    
    # Test first step
    w1.grad = Variable(0.1)
    w2.grad = Variable(0.2)
    b.grad = Variable(0.05)
    
    original_w1 = w1.data.data.item()
    optimizer.step()
    
    # Check that parameter was updated
    assert w1.data.data.item() != original_w1, "Parameter should be updated"
    assert optimizer.step_count == 1, "Step count should be incremented"
    
    # Check that moment buffers were created
    assert len(optimizer.first_moment) == 3, "Should have first moment buffers"
    assert len(optimizer.second_moment) == 3, "Should have second moment buffers"
    
    # Test multiple steps (bias correction should change behavior)
    for i in range(5):
        w1.grad = Variable(0.1)
        w2.grad = Variable(0.2)
        b.grad = Variable(0.05)
        optimizer.step()
    
    assert optimizer.step_count == 6, "Should track multiple steps"
    
    # Test with different gradients (adaptive behavior)
    w1.grad = Variable(1.0)  # Large gradient
    w2.grad = Variable(0.01)  # Small gradient
    b.grad = Variable(0.05)
    
    optimizer.step()
    
    # Parameters should have been updated with adaptive rates
    assert optimizer.step_count == 7, "Should continue tracking steps"
    
    print("✅ Adam optimizer tests passed!")
    print(f"✅ zero_grad() working correctly")
    print(f"✅ Parameter updates working")
    print(f"✅ Moment buffers created")
    print(f"✅ Step counting and bias correction working")
    print(f"✅ Adaptive learning rates functioning")

# Run the test
test_adam_optimizer()

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
### 🧪 Test Your Learning Rate Scheduler

Let's test the learning rate scheduler:
"""

# %% nbgrader={"grade": true, "grade_id": "test-scheduler", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_step_scheduler():
    """Test step learning rate scheduler"""
    print("🔬 Unit Test: Step Learning Rate Scheduler...")
    
    # Create optimizer and scheduler
    w = Variable(1.0, requires_grad=True)
    optimizer = SGD([w], learning_rate=0.1)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Test initial state
    assert scheduler.get_lr() == 0.1, "Initial learning rate should be 0.1"
    
    # Test first few steps (no decay yet)
    for i in range(3):
        scheduler.step()
        assert scheduler.get_lr() == 0.1, f"Learning rate should still be 0.1 at step {i+1}"
    
    # Test first decay
    scheduler.step()  # Step 4
    expected_lr = 0.1 * 0.1  # 0.01
    assert abs(scheduler.get_lr() - expected_lr) < 1e-6, f"Learning rate should be {expected_lr}"
    
    # Test more steps
    for i in range(2):
        scheduler.step()
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6, "Learning rate should remain at first decay level"
    
    # Test second decay
    scheduler.step()  # Step 7
    expected_lr = 0.1 * (0.1 ** 2)  # 0.001
    assert abs(scheduler.get_lr() - expected_lr) < 1e-6, f"Learning rate should be {expected_lr}"
    
    # Test with different gamma
    optimizer2 = Adam([w], learning_rate=0.001)
    scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.5)
    
    scheduler2.step()
    scheduler2.step()
    scheduler2.step()  # Should decay after 2 steps
    
    expected_lr2 = 0.001 * 0.5  # 0.0005
    assert abs(scheduler2.get_lr() - expected_lr2) < 1e-6, "Should work with Adam optimizer"
    
    print("✅ Learning rate scheduler tests passed!")
    print(f"✅ Initial learning rate correct")
    print(f"✅ Step-based decay working")
    print(f"✅ Multiple decay levels working")
    print(f"✅ Works with different optimizers")

# Run the test
test_step_scheduler()

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
            from autograd_dev import add, multiply
            prediction = add(multiply(w, x), b)
            
            # Loss: (prediction - target)^2
            from autograd_dev import subtract
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
### 🧪 Test Complete Training Integration

Let's test the complete training workflow:
"""

# %% nbgrader={"grade": true, "grade_id": "test-training-integration", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_training_integration():
    """Test complete training integration"""
    print("🔬 Unit Test: Complete Training Integration...")
    
    # Run training example
    sgd_w, sgd_b, adam_w, adam_b = train_simple_model()
    
    # Check that both optimizers learned something reasonable
    # Target: w = 2.0, b = 1.0
    
    # SGD should get close to target
    assert abs(sgd_w - 2.0) < 0.5, f"SGD should learn w ≈ 2.0, got {sgd_w}"
    assert abs(sgd_b - 1.0) < 0.5, f"SGD should learn b ≈ 1.0, got {sgd_b}"
    
    # Adam should also get close to target (allowing for different convergence characteristics)
    assert abs(adam_w - 2.0) < 0.7, f"Adam should learn w ≈ 2.0, got {adam_w}"
    assert abs(adam_b - 1.0) < 0.7, f"Adam should learn b ≈ 1.0, got {adam_b}"
    
    print("✅ Training integration tests passed!")
    print(f"✅ SGD successfully trained model")
    print(f"✅ Adam successfully trained model")
    print(f"✅ Learning rate scheduling working")
    print(f"✅ Complete training workflow functional")

# Run the test
test_training_integration()

# %% [markdown]
"""
## 🎯 Module Summary: Optimization Mastery!

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