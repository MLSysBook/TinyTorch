# TinyTorch Module Development Standards

## 🎯 Core Principle

**Modules teach ML systems engineering through building, not just ML algorithms through reading.**

## 📁 File Structure

### One Module = One .py File

```
modules/source/XX_modulename/
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
- [ ] Follows exact structure pattern
- [ ] Every implementation has immediate test
- [ ] Includes memory profiling
- [ ] Includes complexity analysis
- [ ] Shows production context
- [ ] NBGrader metadata correct
- [ ] ML systems thinking questions
- [ ] Summary is LAST section
- [ ] Tests run when module executed

## 🎯 Remember

> We're teaching ML systems engineering, not just ML algorithms.

Every module should help students understand:
- How to BUILD ML systems
- Why performance matters
- Where bottlenecks occur
- How production systems work