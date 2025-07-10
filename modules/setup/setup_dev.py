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
# Module 0: Setup - Tiny🔥Torch Development Workflow

Welcome to TinyTorch! This module teaches you the development workflow you'll use throughout the course.

> **📚 Educational Mode**: This notebook uses NBDev's educational features. Instructors see complete solutions, students see exercises with hidden answers.

## Learning Goals
- Understand the NBDev notebook-to-Python workflow with educational directives
- Write your first TinyTorch code with instructor/student mode support
- Master the hide/show pattern for progressive learning
- Run tests and use the CLI tools
- Get comfortable with the development rhythm

## ✨ NBDev Educational Features
- **`#|hide`** - Hide complete solutions from students (click to reveal)
- **`#|code-fold`** - Collapsible code sections for optional details
- **Single source** - One notebook serves both instructors and students

## The TinyTorch Development Cycle

1. **Write code** in this notebook using `#| export` and educational directives
2. **Export code** with `python bin/tito.py sync --module setup`
3. **Run tests** with `python bin/tito.py test --module setup`
4. **Check progress** with `python bin/tito.py info`

Let's get started!
"""

# %%
#| default_exp core.utils

# Setup imports and environment
import sys
import platform
from datetime import datetime

print("🔥 TinyTorch Development Environment - Educational Mode")
print(f"Python {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
"""
## Step 1: Understanding the Module → Package Structure

**🎓 Teaching vs. 🔧 Building**: This course has two sides:
- **Teaching side**: You work in `modules/setup/setup_dev.ipynb` (learning-focused)
- **Building side**: Your code exports to `tinytorch/core/utils.py` (production package)

**Key Concept**: The `#| default_exp core.utils` directive at the top tells nbdev to export all `#| export` cells to `tinytorch/core/utils.py`.

This separation allows us to:
- Organize learning by **concepts** (modules)  
- Organize code by **function** (package structure)
- Build a real ML framework while learning systematically

### 🎯 Your First Challenge: Educational Hello World

Let's implement a hello world function that teaches ML concepts. **Students**: Try implementing this yourself first!
"""

# %%
#| export
def hello_tinytorch():
    """
    A hello world function that introduces TinyTorch concepts.
    
    Students: Implement a function that returns a welcoming message 
    mentioning tensors, autograd, and neural networks.
    
    Hint: Make it inspiring and educational!
    """
    # TODO: Replace this placeholder with an educational greeting
    # TODO: Mention what students will build (tensors, autograd, networks)
    return "Hello from TinyTorch! 🔥"

def add_numbers(a, b):
    """Add two numbers - the foundation of all ML operations!"""
    return a + b

# %% [markdown]
"""
### 🔍 Instructor Solution (Hidden from Students)

Click the cell below to see the complete educational implementation:
"""

# %%
#| hide
#| export
def hello_tinytorch():
    """INSTRUCTOR SOLUTION: A comprehensive TinyTorch introduction."""
    return """🔥 Welcome to TinyTorch - Your ML Systems Journey! 🔥

What you'll build in this course:
📊 Tensors: N-dimensional arrays for data
🔄 Autograd: Automatic differentiation engine  
🧠 Neural Networks: MLPs, CNNs, and more
⚡ Training: Optimizers, loss functions, loops
🚀 Production: Deployment and monitoring

You're not just learning ML - you're building a complete framework from scratch!
Ready to become an ML systems engineer? Let's go! 💪"""

# %% [markdown]
"""
### 🧪 Test Your Implementation

Run the cell below to test your hello world function:
"""

# %%
# Test the functions in the notebook
print("=== Testing Hello World Function ===")
print(hello_tinytorch())
print()
print("=== Testing Basic Operations ===")
print(f"2 + 3 = {add_numbers(2, 3)}")
print(f"This is the foundation of neural network math!")

# %% [markdown]
"""
### 🎯 Advanced Challenge: Vector Operations

Let's implement something more ML-relevant. Can you implement vector operations that are fundamental to ML?
"""

# %%
#| export
#| code-fold: true
def vector_add(v1, v2):
    """
    Add two vectors element-wise.
    
    Students: Implement vector addition
    - Check that vectors have same length
    - Add corresponding elements  
    - Return new vector
    
    This is fundamental to ML: gradient updates, combining embeddings, etc.
    """
    # TODO: Implement vector addition
    # Hint: Use zip() to pair up elements
    if len(v1) != len(v2):
        raise ValueError(f"Vector lengths don't match: {len(v1)} vs {len(v2)}")
    
    return [a + b for a, b in zip(v1, v2)]

def vector_dot(v1, v2):
    """
    Compute dot product of two vectors.
    
    Students: Implement dot product
    - Multiply corresponding elements
    - Sum the results
    
    Dot product is THE core ML operation (linear layers, attention, etc.)
    """
    # TODO: Implement dot product
    if len(v1) != len(v2):
        raise ValueError(f"Vector lengths don't match: {len(v1)} vs {len(v2)}")
    
    return sum(a * b for a, b in zip(v1, v2))

# %%
# Test vector operations
print("=== Vector Operations Test ===")
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Addition: {vector_add(v1, v2)}")
print(f"Dot product: {vector_dot(v1, v2)}")
print("These operations power all of machine learning!")

# %% [markdown]
"""
## Step 2: ML-Aware System Information

Let's create a more sophisticated system class that's ML-aware. This demonstrates object-oriented programming while providing useful ML development information.
"""

# %%
#| export
class SystemInfo:
    """ML-aware system information class."""
    
    def __init__(self):
        """Initialize system information collection."""
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.machine = platform.machine()
        self._check_ml_libraries()
    
    def _check_ml_libraries(self):
        """Check if common ML libraries are available."""
        self.has_numpy = self._try_import('numpy')
        self.has_torch = self._try_import('torch')
        self.has_tensorflow = self._try_import('tensorflow')
    
    def _try_import(self, module_name):
        """Safely try to import a module."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def __str__(self):
        """Human-readable system information."""
        return f"TinyTorch on Python {self.python_version.major}.{self.python_version.minor} ({self.platform} {self.machine})"
    
    def is_compatible(self):
        """Check if system meets minimum requirements."""
        return self.python_version >= (3, 8)
    
    def is_ml_ready(self):
        """Check if system is ready for ML development."""
        return self.is_compatible() and self.has_numpy
    
    def ml_status_report(self):
        """Generate a detailed ML readiness report."""
        status = []
        status.append("🔥 TinyTorch System Status")
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

# %%
# Test the enhanced SystemInfo class
print("=== Enhanced System Information ===")
info = SystemInfo()
print(f"System: {info}")
print(f"Compatible: {info.is_compatible()}")
print(f"ML Ready: {info.is_ml_ready()}")
print()
print(info.ml_status_report())

# %% [markdown]
"""
## Step 3: The NBDev Export Process - Your Educational Superpower

Now let's understand how NBDev transforms your notebook into production code while maintaining the educational experience!

### 🔄 The Magic of Educational `#| export`

Every cell marked with `#| export` becomes part of the `tinytorch` package, but NBDev's educational directives control what students see vs what instructors see.
"""

# %%
#| code-fold: show
print("=== NBDev Educational Export Demonstration ===")
print("🎓 Learning Side: You work in modules/setup/setup_dev.ipynb")
print("🔧 Building Side: Code exports to tinytorch/core/utils.py")
print()
print("✨ Educational Directives Used:")
print("  #|export - Code goes to package")
print("  #|hide - Solutions hidden from students") 
print("  #|code-fold - Collapsible sections")
print("  Single source - One notebook, two audiences")
print()
print("🚀 Try this export command:")
print("  python bin/tito.py sync --module setup")

# %% [markdown]
"""
### 🎯 Advanced ML Preview (Instructor Level)

The following cell demonstrates advanced concepts that will be hidden from beginners but visible to instructors:
"""

# %%
#| hide
#| filter_stream ImportWarning DeprecationWarning
def advanced_ml_preview():
    """
    ADVANCED CONTENT - Hidden from beginners
    
    This demonstrates how to hide complex implementations
    while still teaching the concepts progressively.
    """
    import math
    
    def sigmoid(x):
        """Sigmoid activation function - fundamental to neural networks."""
        return 1 / (1 + math.exp(-x))
    
    def relu(x):
        """ReLU activation function - most common in modern ML."""
        return max(0, x)
    
    # Demonstrate activation functions
    test_vals = [-2, -1, 0, 1, 2]
    print("🔬 Activation Functions Preview:")
    for x in test_vals:
        print(f"  x={x:2}: sigmoid={sigmoid(x):.3f}, relu={relu(x):.3f}")

# Show preview to instructors
print("🔬 Advanced ML Preview (hidden from beginners):")
advanced_ml_preview()

# %% [markdown]
"""
## Step 4: Testing and Quality Assurance

Quality code is essential for ML systems. Let's test our implementations!
"""

# %%
def run_setup_tests():
    """Run comprehensive tests on our setup module functions."""
    print("=== Running Setup Module Tests ===")
    
    # Test basic functions
    assert hello_tinytorch() is not None, "hello_tinytorch should return something"
    assert len(hello_tinytorch()) > 20, "hello_tinytorch should be educational"
    assert add_numbers(2, 3) == 5, "Addition should work correctly"
    
    # Test vector operations
    v1, v2 = [1.0, 2.0], [3.0, 4.0]
    assert vector_add(v1, v2) == [4.0, 6.0], "Vector addition should work"
    assert vector_dot(v1, v2) == 11.0, "Dot product should work"
    
    # Test system info
    info = SystemInfo()
    assert isinstance(info.ml_status_report(), str), "Status report should be string"
    assert info.is_compatible(), "Should be compatible with Python 3.8+"
    
    print("✅ All tests passed! Your setup module is working correctly.")
    print("📚 Ready for production ML systems development!")
    return True

# Run the comprehensive tests
run_setup_tests()

# %% [markdown]
"""
## Step 5: Export and Build Process

Now let's export our code! In your terminal, run:

```bash
python bin/tito.py sync --module setup
```

This will export the code marked with `#| export` to `tinytorch/core/utils.py`.

**What happens during educational export:**
1. NBDev scans this notebook for `#| export` cells
2. **Students see**: Exercise versions with TODOs and hints
3. **Instructors see**: Complete solutions with `#|hide` directive
4. **Package gets**: The complete implementation (instructor version)
5. **Documentation shows**: Educational progression with hide/show buttons

**🔍 Verification**: After export, check `tinytorch/core/utils.py` - you'll see the complete functions with auto-generated headers!
"""

# %% [markdown]
"""
## Step 6: Run Tests

After exporting, run the tests:

```bash
python bin/tito.py test --module setup
```

This will run all tests for the setup module and verify your implementation works correctly.

## Step 7: Check Your Progress

See your overall progress:

```bash
python bin/tito.py info
```

This shows which modules are complete and which are pending.
"""

# %% [markdown]
"""
## 🎉 Congratulations! You've Mastered NBDev Educational Features

### ✨ What You've Accomplished

- ✅ **NBDev Educational Directives**: Used `#|hide`, `#|code-fold`, `#|filter_stream`
- ✅ **Progressive Learning**: Started simple, built to ML complexity  
- ✅ **Single Source Truth**: One notebook serves students AND instructors
- ✅ **ML-Relevant Code**: Vector operations, activation previews, system checks
- ✅ **Quality Assurance**: Comprehensive testing approach
- ✅ **Production Workflow**: Export to package with educational metadata

### 🔄 The Educational TinyTorch Development Rhythm

1. **Write** code with educational directives (`#|hide` for solutions)
2. **Test** implementations in notebook (both student and instructor versions)
3. **Export** with `python bin/tito.py sync --module setup`
4. **Verify** with `python bin/tito.py test --module setup`
5. **Progress** with `python bin/tito.py info`

### 🚀 Next Steps: Ready for Real ML Systems

You're now ready for **Module 1: Tensor** where you'll build the foundation of all ML systems using this same educational pattern!

**What's Coming:**
- 📊 **Tensors**: N-dimensional arrays with educational progression
- 🔄 **Autograd**: Automatic differentiation with hidden complexity
- 🧠 **Networks**: MLPs, CNNs with step-by-step revelation
- ⚡ **Training**: End-to-end learning with instructor solutions

### 💡 Educational Development Tips

- **Use `#|hide` strategically** - provide complete solutions but let students try first
- **Progressive revelation** - start simple, build complexity with fold/hide
- **Test both versions** - ensure student stubs and instructor solutions work
- **Single source truth** - maintain one notebook, serve two audiences
- **Quality first** - educational code should be production-ready

**Happy building, ML educator! 🔥**
"""
