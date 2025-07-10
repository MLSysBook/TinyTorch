# %% [markdown]
"""
# Module 0: Setup - TinyTorch Development Workflow 🔥

Welcome to TinyTorch! This module demonstrates NBDev's educational features while teaching you the development workflow.

## 🎯 Learning Goals
- Master the NBDev notebook-to-Python workflow
- Write your first TinyTorch code with educational directives
- Understand progressive learning with NBDev features
- Run tests and use CLI tools effectively
- Build confidence with the development rhythm

## ✨ NBDev Educational Features You'll Learn
- `#|hide` - Hidden solutions and advanced content
- `#|code-fold` - Collapsible code sections
- `#|filter_stream` - Clean output
- Progressive revelation of complexity

Let's start your ML systems journey! 🚀
"""

# %%
#| default_exp core.utils

# %% [markdown]
"""
## Step 1: Environment Setup and Exploration

First, let's explore your development environment and understand NBDev's power.
"""

# %%
#| export
#| filter_stream FutureWarning DeprecationWarning
import sys
import platform
from datetime import datetime
from typing import Dict, List, Any

print("🔥 TinyTorch Development Environment - NBDev Educational Version")
print(f"Python {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Ready to explore NBDev educational features!")

# %% [markdown]
"""
### 🎓 Your First Challenge: Hello World with a Twist

Instead of a boring "Hello World", let's create something that actually teaches you about ML systems!

**Task**: Implement a function that introduces TinyTorch concepts.
"""

# %%
#| code-fold: show
def hello_tinytorch() -> str:
    """
    A more educational hello world for TinyTorch.
    
    TODO: Make this function return information about TinyTorch's capabilities
    - Mention tensors, autograd, neural networks
    - Make it encouraging for beginners
    - Add some ML terminology
    """
    # 🚨 Try implementing this yourself first!
    return "Hello from TinyTorch! 🔥"  # This is just a placeholder

# %% [markdown]
"""
### 🔍 Complete Solution (Hidden by Default)

Click below to see a more sophisticated implementation:
"""

# %%
#| hide
#| exports
def hello_tinytorch_complete() -> str:
    """
    COMPLETE SOLUTION - Hidden from students initially.
    
    A comprehensive introduction to TinyTorch that teaches while greeting.
    """
    return """
🔥 Welcome to TinyTorch - Your ML Systems Journey Begins! 🔥

What you'll build in this course:
📊 Tensors: N-dimensional arrays for data
🔄 Autograd: Automatic differentiation engine  
🧠 Neural Networks: MLPs, CNNs, and more
⚡ Training: Optimizers, loss functions, loops
🚀 Production: Deployment and monitoring

You're not just learning ML - you're building a complete framework from scratch!
Ready to become an ML systems engineer? Let's go! 💪
    """.strip()

# Update the main function with the complete implementation
hello_tinytorch.__code__ = hello_tinytorch_complete.__code__

# %% [markdown]
"""
### 🧪 Test Your Implementation
"""

# %%
print(hello_tinytorch())

# %% [markdown]
"""
## Step 2: Basic Operations with Progressive Learning

Let's implement some basic utility functions. We'll start simple and build complexity.
"""

# %%
#| export
def add_numbers(a: float, b: float) -> float:
    """Add two numbers - the foundation of all ML operations!"""
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers - essential for neural network forward passes!"""
    return a * b

# %%
# Test basic operations
print("=== Basic Operations ===")
print(f"Addition: 2 + 3 = {add_numbers(2, 3)}")
print(f"Multiplication: 4 × 5 = {multiply_numbers(4, 5)}")

# %% [markdown]
"""
### 🎯 Advanced Challenge: Vector Operations

Now let's implement something more ML-relevant. Can you implement vector operations?
"""

# %%
#| code-fold: true
def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Add two vectors element-wise.
    
    This is a fundamental operation in ML - vectors represent features,
    and vector addition is used everywhere from gradient updates to
    combining embeddings.
    
    TODO: Implement vector addition
    - Check that vectors have same length
    - Add corresponding elements
    - Return new vector
    """
    # Implementation hidden but expandable
    if len(v1) != len(v2):
        raise ValueError(f"Vector lengths don't match: {len(v1)} vs {len(v2)}")
    
    return [a + b for a, b in zip(v1, v2)]

def vector_dot(v1: List[float], v2: List[float]) -> float:
    """
    Compute dot product of two vectors.
    
    Dot product is THE fundamental operation in ML:
    - Linear layers: weights · inputs
    - Attention: queries · keys  
    - Similarity: compare embeddings
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector lengths don't match: {len(v1)} vs {len(v2)}")
    
    return sum(a * b for a, b in zip(v1, v2))

# Export the vector functions
hello_tinytorch.__code__ = hello_tinytorch_complete.__code__

# %% [markdown]
"""
### 🧪 Test Vector Operations
"""

# %%
print("=== Vector Operations ===")
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Addition: {vector_add(v1, v2)}")
print(f"Dot product: {vector_dot(v1, v2)}")

# %% [markdown]
"""
## Step 3: System Information Class - Building ML-Aware Tools

Let's create a more sophisticated system information class that's ML-aware.
"""

# %%
#| export
class TinyTorchSystemInfo:
    """
    ML-aware system information class.
    
    This class demonstrates object-oriented programming while
    providing useful information for ML development.
    """
    
    def __init__(self):
        """Initialize system information collection."""
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.machine = platform.machine()
        self._check_ml_libraries()
    
    def _check_ml_libraries(self) -> None:
        """Check if common ML libraries are available."""
        self.has_numpy = self._try_import('numpy')
        self.has_torch = self._try_import('torch')
        self.has_tensorflow = self._try_import('tensorflow')
    
    def _try_import(self, module_name: str) -> bool:
        """Safely try to import a module."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def __str__(self) -> str:
        """Human-readable system information."""
        return f"TinyTorch on Python {self.python_version.major}.{self.python_version.minor} ({self.platform} {self.machine})"
    
    def is_ml_ready(self) -> bool:
        """Check if system is ready for ML development."""
        return (
            self.python_version >= (3, 8) and
            self.has_numpy
        )
    
    def ml_status_report(self) -> str:
        """Generate a detailed ML readiness report."""
        status = []
        status.append(f"🔥 TinyTorch System Status")
        status.append(f"Platform: {self.platform} ({self.machine})")
        status.append(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check requirements
        status.append("\n📋 ML Library Status:")
        status.append(f"  NumPy: {'✅ Available' if self.has_numpy else '❌ Missing'}")
        status.append(f"  PyTorch: {'✅ Available' if self.has_torch else '❌ Missing (optional)'}")
        status.append(f"  TensorFlow: {'✅ Available' if self.has_tensorflow else '❌ Missing (optional)'}")
        
        # Overall status
        ready = self.is_ml_ready()
        status.append(f"\n🎯 Overall Status: {'✅ Ready for TinyTorch!' if ready else '❌ Missing requirements'}")
        
        if ready:
            status.append("🚀 You're all set to build an ML framework from scratch!")
        else:
            status.append("💡 Install missing libraries: pip install numpy")
        
        return "\n".join(status)

# %% [markdown]
"""
### 🧪 Test System Information
"""

# %%
#| filter_stream ImportWarning
print("=== System Information ===")
info = TinyTorchSystemInfo()
print(info)
print(f"ML Ready: {info.is_ml_ready()}")
print("\n" + info.ml_status_report())

# %% [markdown]
"""
## Step 4: The NBDev Export Process - Your Superpower

Now let's understand how NBDev transforms your notebook into production code!

### 🔄 The Magic of `#| export`

Every cell marked with `#| export` becomes part of the `tinytorch` package.
This is how we separate **learning** (notebooks) from **building** (package).
"""

# %%
#| code-fold: show
print("=== NBDev Export Demonstration ===")
print("🎓 Learning Side: You work in modules/setup/setup_nbdev_educational.py")
print("🔧 Building Side: Code exports to tinytorch/core/utils.py")
print()
print("✨ NBDev Features Demonstrated:")
print("  - #|export: Code goes to package")
print("  - #|hide: Solutions hidden from students") 
print("  - #|code-fold: Collapsible sections")
print("  - #|filter_stream: Clean output")
print()
print("🚀 Try this export command:")
print("  python bin/tito.py sync --module setup")

# %% [markdown]
"""
### 🎯 Advanced Export Features (Instructor Level)

These advanced features show NBDev's real power:
"""

# %%
#| hide
def advanced_ml_function():
    """
    ADVANCED CONTENT - Hidden from beginners
    
    This demonstrates how to hide complex implementations
    while still teaching the concepts progressively.
    """
    import math
    
    def sigmoid(x: float) -> float:
        """Sigmoid activation function - fundamental to neural networks."""
        return 1 / (1 + math.exp(-x))
    
    def relu(x: float) -> float:
        """ReLU activation function - most common in modern ML."""
        return max(0, x)
    
    # Demonstrate activation functions
    test_vals = [-2, -1, 0, 1, 2]
    print("Activation Functions Preview:")
    for x in test_vals:
        print(f"  x={x:2}: sigmoid={sigmoid(x):.3f}, relu={relu(x):.3f}")

# Only show this to advanced students
print("🔬 Advanced ML Preview (hidden from beginners):")
advanced_ml_function()

# %% [markdown]
"""
## Step 5: Testing and Quality Assurance

Quality code is essential for ML systems. Let's test our implementations!
"""

# %%
def run_setup_tests():
    """Run basic tests on our setup module functions."""
    print("=== Running Setup Module Tests ===")
    
    # Test basic functions
    assert hello_tinytorch() is not None, "hello_tinytorch should return something"
    assert add_numbers(2, 3) == 5, "Addition should work correctly"
    assert multiply_numbers(4, 5) == 20, "Multiplication should work correctly"
    
    # Test vector operations
    v1, v2 = [1.0, 2.0], [3.0, 4.0]
    assert vector_add(v1, v2) == [4.0, 6.0], "Vector addition should work"
    assert vector_dot(v1, v2) == 11.0, "Dot product should work"
    
    # Test system info
    info = TinyTorchSystemInfo()
    assert isinstance(info.ml_status_report(), str), "Status report should be string"
    
    print("✅ All tests passed! Your setup module is working correctly.")
    return True

# Run the tests
run_setup_tests()

# %% [markdown]
"""
## 🎉 Congratulations! You've Mastered NBDev + TinyTorch Basics

### ✨ What You've Accomplished

- ✅ **NBDev Educational Features**: Used `#|hide`, `#|code-fold`, `#|filter_stream`
- ✅ **Progressive Learning**: Started simple, built to complexity  
- ✅ **ML-Relevant Code**: Vector operations, system checks, activations
- ✅ **Quality Assurance**: Comprehensive testing approach
- ✅ **Production Workflow**: Export to package structure

### 🔄 The TinyTorch Development Rhythm

1. **Write** code in notebooks with educational directives
2. **Export** with `python bin/tito.py sync --module setup`
3. **Test** with `python bin/tito.py test --module setup`  
4. **Verify** with `python bin/tito.py info`

### 🚀 Next Steps: Ready for Real ML Systems

You're now ready for **Module 1: Tensor** where you'll build the foundation of all ML systems!

**What's Coming:**
- 📊 **Tensors**: N-dimensional arrays with shape management
- 🔄 **Autograd**: Automatic differentiation for training
- 🧠 **Networks**: MLPs, CNNs, attention mechanisms
- ⚡ **Training**: End-to-end learning pipelines

### 💡 Pro Tips for Your Journey

- **Always test in notebooks first** - catch issues early
- **Use NBDev directives strategically** - control learning pace
- **Read error messages carefully** - they're designed to teach
- **Celebrate small wins** - building ML systems is challenging but rewarding!

**Happy building, ML engineer! 🔥**
"""

# %%
#| hide_line
print("🎓 Instructor note: Students have completed the setup successfully!")
print("🎯 Ready to move to tensor implementation - the real fun begins!") 