# CRITICAL FIX: Forward-Compatible Tensor Design

## The Problem
Module 05 (Autograd) cannot cleanly retrofit gradient support to Tensor class created in Module 01.

## The Solution: Design Tensor with Future Autograd in Mind

### Module 01: Tensor (Forward-Compatible Version)

```python
class Tensor:
    """Tensor with hooks for future autograd support."""

    def __init__(self, data, requires_grad=False, _grad_fn=None):
        """
        Initialize tensor with forward-compatibility for autograd.

        Args:
            data: The tensor data
            requires_grad: Whether to track gradients (inactive until Module 05)
            _grad_fn: Gradient function (used by autograd in Module 05)
        """
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad_fn = _grad_fn  # Placeholder for autograd
        self.grad = None  # Placeholder for gradients

    def backward(self, grad=None):
        """Placeholder for backward pass - implemented in Module 05."""
        if not self.requires_grad:
            return
        # Module 05 will implement this
        pass

    def zero_grad(self):
        """Clear gradients - functional even before autograd."""
        self.grad = None
```

### Module 02-04: Work Normally
- Activations, Layers, Losses all work with basic Tensor
- They ignore `requires_grad` flag (it's always False)
- `backward()` exists but does nothing

### Module 05: Autograd Activates the System

```python
# autograd_dev.py
from tinytorch import Tensor as BaseTensor

# Monkey-patch the backward method with actual implementation
def backward_with_autograd(self, grad=None):
    """Actual backward implementation."""
    if not self.requires_grad or self._grad_fn is None:
        return

    if grad is None:
        grad = np.ones_like(self.data)

    # Accumulate gradients
    if self.grad is None:
        self.grad = grad
    else:
        self.grad += grad

    # Propagate to dependencies
    if self._grad_fn:
        self._grad_fn.backward(grad)

# Replace the placeholder
BaseTensor.backward = backward_with_autograd

# Now add Function classes that set _grad_fn
class AddBackward:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def backward(self, grad):
        if self.x.requires_grad:
            self.x.backward(grad)
        if self.y.requires_grad:
            self.y.backward(grad)

# Override arithmetic operations to track gradients
original_add = BaseTensor.__add__

def tracked_add(self, other):
    result = original_add(self, other)
    if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
        result.requires_grad = True
        result._grad_fn = AddBackward(self, other)
    return result

BaseTensor.__add__ = tracked_add
```

### Module 06+: Everything Just Works!
Optimizers work because Tensor always had `grad` attribute:

```python
class SGD:
    def step(self):
        for param in self.params:
            if param.grad is not None:  # Works even pre-autograd
                param.data -= self.lr * param.grad
```

## Why This Works

1. **No Breaking Changes**: Tensor API is consistent from Module 01
2. **Progressive Enhancement**: Features activate when implemented
3. **No Variable Class**: Single Tensor type throughout
4. **Clean Dependency Chain**: Each module only uses what came before
5. **Python Decorators**: Can cleanly wrap methods when needed

## Implementation Strategy

### Stage 1: Foundation (Modules 01-04)
- Tensor has gradient infrastructure but inactive
- All operations work without gradients
- Tests verify basic functionality

### Stage 2: Activation (Module 05)
- Autograd "switches on" the gradient system
- Monkey-patches methods with real implementations
- Previous modules still work unchanged

### Stage 3: Utilization (Modules 06+)
- Optimizers use the now-active gradient system
- Training loops work with full backprop
- No code changes to earlier modules

## Alternative: Pure Decorator Approach

```python
# Module 05 could use decorators instead of monkey-patching
def track_gradients(op):
    """Decorator to add gradient tracking to operations."""
    def wrapper(self, other=None):
        result = op(self, other)
        if should_track_gradients(self, other):
            result.requires_grad = True
            result._grad_fn = create_backward_fn(op, self, other)
        return result
    return wrapper

# Apply decorators
Tensor.__add__ = track_gradients(Tensor.__add__)
Tensor.__mul__ = track_gradients(Tensor.__mul__)
```

## Testing Strategy

```python
# Module 01-04 tests
def test_tensor_basic():
    t = Tensor([1, 2, 3])
    assert t.grad is None  # Exists but None
    t.backward()  # Should not crash
    assert t.grad is None  # Still None (autograd not active)

# Module 05 tests
def test_autograd_active():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x * 2
    y.sum().backward()
    assert x.grad is not None  # Now gradients work!

# Module 06+ tests work without modification
def test_optimizer():
    param = Tensor([1, 2, 3], requires_grad=True)
    optimizer = SGD([param], lr=0.01)
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()  # Works seamlessly
```

## Benefits Over Current Approach

| Current Approach | This Approach |
|-----------------|---------------|
| Variable vs Tensor confusion | Single Tensor class |
| hasattr() checks everywhere | Clean attributes from start |
| Module 06 needs ugly fallbacks | Module 06 just works |
| Students learn wrong patterns | Students see clean design |
| Breaks if run out of order | Graceful degradation |

## Summary

**This forward-compatible design is MANDATORY for the new module plan to work properly.**

The module-developer MUST implement Tensor in Module 01 with:
1. `requires_grad` parameter (default False)
2. `grad` attribute (starts as None)
3. `_grad_fn` attribute (for autograd hook)
4. `backward()` method (placeholder)
5. `zero_grad()` method (functional immediately)

This ensures Modules 05+ can cleanly extend functionality without breaking Modules 01-04.