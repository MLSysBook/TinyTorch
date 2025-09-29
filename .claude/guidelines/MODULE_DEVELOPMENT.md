# TinyTorch Module Development Standards

## 🎯 Core Principle

**Modules teach ML systems engineering through building, not just ML algorithms through reading.**

## 📁 File Structure

### One Module = One .py File

```
modules/XX_modulename/
├── modulename_dev.py     # The ONLY file you edit
├── modulename_dev.ipynb  # Auto-generated from .py (DO NOT EDIT)
└── README.md            # Module overview
```

**Critical Rules:**
- ✅ ALWAYS edit `.py` files only
- ❌ NEVER edit `.ipynb` notebooks directly
- ✅ Use jupytext to sync .py → .ipynb

## 📚 Module Structure Pattern

Every module MUST follow this exact structure:

```python
# %% [markdown]
"""
# Module XX: [Name]

**Learning Objectives:**
- Build [component] from scratch
- Understand [systems concept]
- Analyze performance implications
"""

# %% [markdown]
"""
## Part 1: Mathematical Foundations
[Theory and complexity analysis]
"""

# %% [code]
# Implementation

# %% [markdown]
"""
### Testing [Component]
Let's verify our implementation works correctly.
"""

# %% [code]
# Immediate test

# %% [markdown]
"""
## Part 2: Systems Analysis
### Memory Profiling
Let's understand the memory implications.
"""

# %% [code]
# Memory profiling code

# %% [markdown]
"""
## Part 3: Production Context
In real ML systems like PyTorch...
"""

# ... continue pattern ...

# %% [code]
if __name__ == "__main__":
    run_all_tests()

# %% [markdown]
"""
## 🤔 ML Systems Thinking
[Interactive questions analyzing implementation]
"""

# %% [markdown]
"""
## 🎯 Module Summary
[What was learned - ALWAYS LAST]
"""
```

## 📊 Visual Learning with ASCII Diagrams - MANDATORY

**Every module MUST include ASCII diagrams to visualize key concepts.**

ASCII diagrams provide immediate visual understanding without dependencies. They work in terminals, IDEs, notebooks, and are part of CS education tradition.

### When to Use ASCII Diagrams:
- **Data structures**: Show memory layout and object relationships
- **Algorithms**: Visualize step-by-step execution
- **Gradient flow**: Forward and backward passes
- **Computation graphs**: Operation dependencies
- **Performance**: Memory usage, complexity scaling
- **Systems concepts**: Cache patterns, parallelization

### ASCII Art Toolkit:
```
Box drawings: ┌─┐│└┘├┤┬┴┼╔═╗║╚╝╠╣╦╩╬
Arrows: → ← ↓ ↑ ⇒ ⇐ ⇓ ⇑
Math symbols: ∂ ∇ Σ ∏ ∫ ≈ ≠ ≤ ≥ ± ×
Progress bars: ░▒▓█
```

### Example Excellence from Module 05:
```
    Forward Pass:                 Backward Pass:
    x(2.0) ────┐                 x.grad ←── 1.0
               ├─► [+] ──► z(5.0)         ↑
    y(3.0) ────┘              │           │
                               ▼           │
                        z.backward(1.0) ───┘
```

### ASCII vs Equations:
Let the content decide. Sometimes ASCII explains flow, sometimes equations explain math. Use both:
```
Product Rule: ∂z/∂x = y, ∂z/∂y = x

    x(2.0) ──┐
             ├─[×]→ z(6.0): grad_x = grad × y
    y(3.0) ──┘              grad_y = grad × x
```

## 📝 Markdown Flow - Natural Narrative

**CRITICAL: Markdown cells should flow naturally, explaining what's about to happen.**

### Opening Explanation Pattern:
Start each module with clear context about WHAT we're implementing and WHY:

```python
# %% [markdown]
"""
# Module 05: Autograd - Automatic Differentiation

Here's what we're actually implementing: a system that automatically computes gradients by tracking operations and applying the chain rule backward. This requires:

1. Extending our Tensor data structure to remember gradients
2. Making operations "smart" so they record how to reverse themselves
3. Building a computation graph as we compute forward
4. Traversing that graph backward to compute all gradients

Let's build this step by step, with immediate validation at each stage.
"""
```

### Step Introduction Pattern:
Each implementation step should explain the immediate goal:

```python
# %% [markdown]
"""
Now we'll make addition smart. When we compute z = x + y, we need z to remember
how to send gradients back to both x and y. Since ∂z/∂x = 1 and ∂z/∂y = 1,
both inputs will receive the same gradient unchanged.

Here's what the gradient flow looks like:
[ASCII diagram here]

Let's implement this enhancement:
"""
```

### Natural Flow Rules:
- ❌ Don't add redundant section headers within explanations
- ✅ Let the narrative flow from concept → visual → implementation → test
- ✅ Use diagrams inline where they clarify understanding
- ✅ Explain WHAT you're about to do and WHY before showing code

## 🧪 Implementation → Test Pattern

**MANDATORY**: Every implementation must be immediately followed by a test.

```python
# ✅ CORRECT Pattern:

# %% [markdown]
"""
## Building the Dense Layer
"""

# %% [code]
class Dense:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)
    
    def forward(self, x):
        return x @ self.weights + self.bias

# %% [markdown]
"""
### Testing Dense Layer
Let's verify our dense layer handles shapes correctly.
"""

# %% [code]
def test_dense_layer():
    layer = Dense(10, 5)
    x = np.random.randn(32, 10)  # Batch of 32, 10 features
    output = layer.forward(x)
    assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
    print("✅ Dense layer forward pass works!")

test_dense_layer()
```

## 🚨 CRITICAL: Module Dependency Rules

### Tensor Evolution Pattern - THE CLEAN APPROACH

**CRITICAL: Use ONE evolving Tensor class, NOT separate Tensor/Variable classes**

Following PyTorch's design philosophy, TinyTorch uses a single `Tensor` class that gains capabilities over time:

#### Module Evolution Plan

```
Module 01 (Tensor):
├── Create basic Tensor class with data storage
├── Add requires_grad=False by default
├── Add placeholder grad=None
├── Add NotImplementedError for backward()
└── Basic operations (__add__, __mul__) without gradient tracking

Module 02-04 (Activations, Layers, Losses):
├── Use existing Tensor class as-is
├── Work with requires_grad=False tensors
├── Build layers, activations, losses on basic Tensor
└── No gradient functionality needed yet

Module 05 (Autograd):
├── STUDENTS UPDATE the existing Tensor class
├── Implement the backward() method (replace NotImplementedError)
├── Update operations (__add__, __mul__) to build computation graph
├── Add grad_fn tracking for chain rule
└── Now requires_grad=True works everywhere automatically

Module 06+ (Optimizers, Training, etc.):
├── Use enhanced Tensor class with full gradient capabilities
├── All previous code works unchanged (backward compatibility)
├── New code can use requires_grad=True for automatic differentiation
└── Single clean interface throughout
```

#### Implementation Examples

**Module 01: Basic Tensor (Focus ONLY on data storage)**
```python
class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    # No gradient-related code AT ALL
```

**Module 02-04: Use basic Tensor as-is**
```python
class Linear:
    def __init__(self, in_features, out_features):
        # Use basic Tensor - no gradient tracking yet
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.1)
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weights + self.bias  # Basic operations
```

**Module 05: Students EXTEND Tensor using Python class decoration**
```python
def add_autograd(cls):
    """Decorator that adds gradient tracking to existing Tensor class"""

    # Save original methods
    original_init = cls.__init__
    original_add = cls.__add__
    original_mul = cls.__mul__

    def new_init(self, data, requires_grad=False):
        # Call original constructor
        original_init(self, data)
        # Add gradient tracking
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def new_add(self, other):
        result = original_add(self, other)
        # Add gradient computation graph building
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            def grad_fn(gradient):
                if self.requires_grad:
                    self.backward(gradient)
                if hasattr(other, 'requires_grad') and other.requires_grad:
                    other.backward(gradient)
            result.grad_fn = grad_fn
        return result

    def backward(self, gradient=None):
        """New method added to Tensor class"""
        if not self.requires_grad:
            raise RuntimeError("Tensor doesn't require gradients")
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += gradient
        if self.grad_fn:
            self.grad_fn(gradient)

    # Enhance the existing class
    cls.__init__ = new_init
    cls.__add__ = new_add
    cls.__mul__ = new_mul
    cls.backward = backward

    return cls

# Students apply the decorator to enhance their existing Tensor
Tensor = add_autograd(Tensor)

# Now all existing code works with gradient tracking!
x = Tensor([1, 2, 3], requires_grad=True)
y = x + x  # Builds computation graph automatically
y.backward([1, 1, 1])  # Computes gradients
```

### 🪄 Python Class Extension Magic

**CRITICAL: Use Python metaprogramming to extend classes cleanly**

The decorator pattern uses Python's dynamic nature to add capabilities to existing classes:

```python
# This is "monkey patching" - modifying classes after definition
original_method = MyClass.some_method

def enhanced_method(self, *args, **kwargs):
    # Add new behavior
    result = original_method(self, *args, **kwargs)
    # Add more behavior
    return result

MyClass.some_method = enhanced_method  # Replace the method
```

**Educational Benefits:**
- ✅ **Each module focuses ONLY on its concepts** (no forward-looking code)
- ✅ **Clean separation** - Module 01 is pure data structure, Module 05 is pure autograd
- ✅ **Python learning** - Students learn metaprogramming and decorators
- ✅ **Real-world pattern** - This is how many frameworks actually work

### Key Benefits
- ✅ **Zero hasattr() checks anywhere** - decorator handles all compatibility
- ✅ **Single class throughout** - always just `Tensor`
- ✅ **Module focus** - each module teaches exactly one concept
- ✅ **Python mastery** - students learn advanced Python patterns
- ✅ **Clean backward compatibility** - all old code works automatically

### Forbidden Patterns (UPDATED)
- ❌ **BAD**: `if hasattr(x, 'data'): x.data else: x`
- ❌ **BAD**: Separate Tensor and Variable classes
- ❌ **BAD**: Parameter wrappers with hasattr() checks
- ❌ **BAD**: `requires_grad` in Module 01 constructor
- ✅ **GOOD**: Pure data class in Module 01, enhanced via decorator in Module 05
- ✅ **GOOD**: Python metaprogramming to extend functionality
- ✅ **GOOD**: Each module contains only what students need to learn at that stage

## 🔬 ML Systems Focus

### MANDATORY Systems Analysis Sections

Every module MUST include:

1. **Complexity Analysis**
```python
# %% [markdown]
"""
### Computational Complexity
- Matrix multiply: O(batch × in_features × out_features)
- Memory usage: O(in_features × out_features) for weights
- This becomes the bottleneck when...
"""
```

2. **Memory Profiling**
```python
# %% [code]
def profile_memory():
    import tracemalloc
    tracemalloc.start()
    
    layer = Dense(1000, 1000)
    x = np.random.randn(128, 1000)
    output = layer.forward(x)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    print("This shows why large models need GPUs!")
```

3. **Production Context**
```python
# %% [markdown]
"""
### In Production Systems
PyTorch's nn.Linear does the same thing but with:
- GPU acceleration via CUDA kernels
- Automatic differentiation support
- Optimized BLAS operations
- Memory pooling for efficiency
"""
```

## 📝 NBGrader Integration

### Cell Metadata Structure

```python
# %% [code] {"nbgrader": {"grade": false, "locked": false, "solution": true, "grade_id": "dense_implementation"}}
### BEGIN SOLUTION
class Dense:
    # Full implementation for instructors
    ...
### END SOLUTION

### BEGIN HIDDEN TESTS
# Instructor-only tests
...
### END HIDDEN TESTS
```

### Critical NBGrader Rules

1. **Every cell needs unique grade_id**
2. **Scaffolding stays OUTSIDE solution blocks**
3. **Hidden tests validate student work**
4. **Points should reflect complexity**

## 🎓 Educational Patterns

### The "Build → Measure → Understand" Pattern

```python
# 1. BUILD
class LayerNorm:
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)

# 2. MEASURE
def measure_performance():
    layer = LayerNorm()
    x = np.random.randn(1000, 512)
    
    start = time.time()
    for _ in range(100):
        output = layer.forward(x)
    elapsed = time.time() - start
    
    print(f"Time per forward pass: {elapsed/100*1000:.2f}ms")
    print(f"Throughput: {100*1000*512/elapsed:.0f} tokens/sec")

# 3. UNDERSTAND
"""
With 512 dimensions, normalization adds ~2ms overhead.
This is why large models use fused kernels!
"""
```

### Progressive Complexity

Start simple, build up:

```python
# Step 1: Simplest possible version
def relu_v1(x):
    return np.maximum(0, x)

# Step 2: Add complexity
def relu_v2(x):
    # Handle gradients
    output = np.maximum(0, x)
    output.grad_fn = lambda grad: grad * (x > 0)
    return output

# Step 3: Production version
class ReLU:
    def forward(self, x):
        self.input = x  # Save for backward
        return np.maximum(0, x)
    
    def backward(self, grad):
        return grad * (self.input > 0)
```

## 🎨 Visual Excellence Examples

### Memory Layout Visualization:
```
    Tensor Without Gradients:        Tensor With Gradients:
    ┌─────────────────┐             ┌─────────────────────────┐
    │ data: [1,2,3]   │             │ data: [1,2,3]           │
    │ shape: (3,)     │             │ requires_grad: True     │
    └─────────────────┘             │ grad: None → [∇₁,∇₂,∇₃] │
                                    │ grad_fn: <AddBackward>   │
                                    └─────────────────────────┘
```

### Algorithm Step Visualization:
```
    Convolution Sliding Window:
    Input:          Kernel:       Step 1:        Step 2:
    ┌─┬─┬─┬─┐      ┌─┬─┐        ╔═╦═╗┬─┬─┐     ┌─╔═╦═╗─┐
    ├─┼─┼─┼─┤      ├─┼─┤   →    ╠═╬═╣┼─┼─┤  →  ├─╠═╬═╣─┤
    ├─┼─┼─┼─┤      └─┴─┘        ╚═╩═╝┴─┴─┘     └─╚═╩═╝─┘
    └─┴─┴─┴─┘                   Output: 6       Output: 8
```

### Performance Scaling:
```
    Memory Usage vs Network Depth:

    10 layers:  ████████░░░░░░░░░░░░ 40%
    20 layers:  ████████████████░░░░ 80%
    30 layers:  ████████████████████ 100% (OOM)

    Solution: Gradient checkpointing
```

## ⚠️ Common Pitfalls

1. **Too Much Theory**
   - Students want to BUILD, not read
   - Show through code, not exposition

2. **Missing Systems Analysis**
   - Not just algorithms, but engineering
   - Always discuss memory and performance

3. **Tests at the End**
   - Loses educational flow
   - Test immediately after implementation

4. **No Production Context**
   - Students need to see real-world relevance
   - Compare with PyTorch/TensorFlow

## 📌 Module Checklist

Before considering a module complete:

- [ ] All code in .py file (not notebook)
- [ ] Clear upfront explanation of WHAT and WHY
- [ ] ASCII diagrams for key concepts
- [ ] Natural markdown flow without redundant headers
- [ ] Every implementation preceded by explanation
- [ ] Every implementation has immediate test
- [ ] Memory profiling with visual representation
- [ ] Complexity analysis with scaling diagrams
- [ ] Production context with system diagrams
- [ ] NBGrader metadata correct
- [ ] ML systems thinking questions with visuals
- [ ] Summary is LAST section
- [ ] Tests run when module executed

## 🎯 Visual Learning Philosophy

**Great modules teach through seeing, not just reading.**

Module 05's success comes from:
1. **Clear goal setting**: Explaining WHAT we're building upfront
2. **Visual-first teaching**: Concepts shown before implementation
3. **Natural flow**: Explanations that lead naturally into code
4. **Immediate feedback**: Tests that validate understanding
5. **Systems thinking**: Visuals that show real-world implications

When developing modules:
- Start with the big picture explanation
- Use ASCII diagrams liberally (they're free and universal!)
- Let content determine visual needs (not every concept needs a diagram)
- Ensure markdown flows as natural narrative
- Show, don't just tell

## 🎯 Remember

> We're teaching ML systems engineering, not just ML algorithms.

Every module should help students understand:
- How to BUILD ML systems
- Why performance matters
- Where bottlenecks occur
- How production systems work