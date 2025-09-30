# The Real Solution: Clean, Simple, Educational

## The Approach: Single Tensor Class with Progressive Activation

### Module 01: Simple Tensor with Dormant Features
```python
class Tensor:
    """Educational tensor that grows with student knowledge."""

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.shape = self.data.shape

        # Gradient features (dormant until explained)
        self.requires_grad = requires_grad
        self.grad = None

    def __add__(self, other):
        """Add two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        """Multiply two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data * other.data)

    def backward(self):
        """Compute gradients (implemented in Module 05)."""
        pass  # Explained in Module 05: Autograd
```

### Why This Works Pedagogically:

**Module 01 Introduction:**
```python
"""
We're building a Tensor class that will grow throughout the course.
For now, focus on:
- data: holds the actual numbers
- shape: the dimensions
- Basic operations: +, *, etc.

Ignore these for now (we'll use them later):
- requires_grad: for automatic differentiation (Module 05)
- grad: stores gradients (Module 05)
- backward(): computes gradients (Module 05)
"""
```

**Module 05 Introduction:**
```python
"""
Remember those mysterious attributes from Module 01?
Now we'll bring them to life!

- requires_grad=True: tells TinyTorch to track operations
- grad: stores computed gradients
- backward(): triggers gradient computation

Let's implement autograd by filling in the backward() method!
"""
```

## The Key Insight: Educational Scaffolding

This is like a textbook with:
- **Forward references**: "We'll explain this in Chapter 5"
- **Consistent structure**: Same class throughout
- **Progressive disclosure**: Features explained when needed
- **No magic**: Students can read the full code from day 1

## Implementation Details:

### Module 01-04: Focus on Forward Pass
```python
# Students work with:
x = Tensor([1, 2, 3])  # requires_grad defaults to False
y = x * 2 + 1
print(y.data)  # [3, 5, 7]

# They see but ignore:
x.backward()  # Does nothing (yet)
print(x.grad)  # None (always)
```

### Module 05: Activate Gradients
```python
# Now we implement backward() properly:
class Tensor:
    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            grad_output = np.ones_like(self.data)

        # Accumulate gradients
        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad += grad_output

        # Backpropagate through the computation graph
        if hasattr(self, '_backward_fn'):
            self._backward_fn(grad_output)
```

### Module 05: Track Operations
```python
# Override operations to build computation graph
def __mul__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)

    result = Tensor(self.data * other.data)

    # NEW in Module 05: Track gradients
    if self.requires_grad or other.requires_grad:
        result.requires_grad = True

        def _backward_fn(grad_output):
            if self.requires_grad:
                self.grad = grad_output * other.data
            if other.requires_grad:
                other.grad = grad_output * self.data

        result._backward_fn = _backward_fn

    return result
```

## Why This is Superior:

### 1. **Honest Education**
- "This method exists but isn't implemented yet" is honest
- Real frameworks have deprecated/future methods too
- Students learn to read documentation critically

### 2. **IDE Friendly**
- Autocomplete works from day 1
- Type hints work correctly
- Debugger shows real class structure

### 3. **Testable**
```python
# Module 01 test
def test_tensor_basic():
    x = Tensor([1, 2, 3])
    assert x.grad is None
    x.backward()  # Shouldn't crash
    assert x.grad is None  # Still None

# Module 05 test
def test_tensor_autograd():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x * 2
    y.sum().backward()
    assert np.allclose(x.grad, [2, 2, 2])
```

### 4. **Clear Mental Model**
- One Tensor class throughout
- Features are dormant → active, not missing → added
- Like a Swiss Army knife where blades unfold as needed

### 5. **Production-Ready Pattern**
```python
# This is how PyTorch actually works:
torch.Tensor  # Has grad, requires_grad, backward from start
# They're just inactive until you set requires_grad=True
```

## The Educational Journey:

```
Module 01: "Here's Tensor. Focus on the data operations."
    ↓
Module 02-04: "Keep using Tensor for layers and losses."
    ↓
Module 05: "Remember backward()? Let's implement it!"
    ↓
Module 06+: "Now our Tensor is fully functional!"
```

## Final Recommendation:

**Use the single Tensor class with progressive activation because:**

1. ✅ **Honest** - We tell students upfront what's coming
2. ✅ **Clean** - No monkey-patching or runtime modifications
3. ✅ **Testable** - Consistent behavior across all modules
4. ✅ **IDE-friendly** - Full autocomplete and type checking
5. ✅ **Pedagogical** - Shows how real frameworks organize code
6. ✅ **Pythonic** - Uses standard OOP patterns

**This is what PyTorch actually does**, and it's the right educational choice.

## Alternative for Purists: Two Classes

If you absolutely hate having dormant features:

```python
# Module 01-04
class BasicTensor:
    def __init__(self, data):
        self.data = np.array(data)

# Module 05+
class Tensor(BasicTensor):
    def __init__(self, data, requires_grad=False):
        super().__init__(data)
        self.requires_grad = requires_grad
        self.grad = None

    def backward(self):
        # Full implementation

# Then in tinytorch/__init__.py:
# Module 01-04: from .tensor_basic import BasicTensor as Tensor
# Module 05+:   from .tensor_grad import Tensor
```

But this creates more confusion than the dormant features approach.