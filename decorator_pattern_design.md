# 🎯 Decorator-Based Tensor Evolution Pattern for Module 05

## Overview
This document specifies how Module 05 (Autograd) will use Python decorators to extend the existing pure Tensor class with gradient tracking capabilities, following the clean evolution pattern.

## Core Principle: Extend, Don't Replace
- **Modules 01-04**: Use pure Tensor class with no gradient code
- **Module 05**: Apply decorator to extend Tensor with gradient capabilities
- **Result**: Same Tensor class, but with gradient tracking when needed

## Implementation Pattern

### 1. Pure Tensor Class (Modules 01-04)
```python
class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    # NO gradient code: no requires_grad, no grad, no backward()
```

### 2. Autograd Decorator (Module 05)
```python
def add_autograd(cls):
    """
    Decorator that extends existing Tensor class with gradient tracking.

    This decorator:
    1. Preserves all existing functionality
    2. Adds gradient tracking capabilities
    3. Enhances operations to build computation graphs
    4. Adds backward() method for gradient computation
    """

    # Save original methods
    original_init = cls.__init__
    original_add = cls.__add__
    original_mul = cls.__mul__
    # ... save all operations

    def new_init(self, data, requires_grad=False):
        # Call original constructor first
        original_init(self, data)

        # Add gradient tracking attributes
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def new_add(self, other):
        # Call original add operation
        result = original_add(self, other)

        # Add computation graph building if needed
        if self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad):
            result.requires_grad = True
            result.grad_fn = AddBackward(self, other)

        return result

    def backward(self, gradient=None):
        """New method: backward propagation"""
        if gradient is None:
            gradient = Tensor(np.ones_like(self.data))

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        if self.grad_fn:
            self.grad_fn.backward(gradient)

    # Replace methods on the class
    cls.__init__ = new_init
    cls.__add__ = new_add
    cls.__mul__ = new_mul
    cls.backward = backward

    return cls

# Apply decorator to extend Tensor
Tensor = add_autograd(Tensor)
```

### 3. Computation Graph Classes
```python
class Function:
    """Base class for computation graph nodes."""

    def __init__(self, *tensors):
        self.inputs = tensors

    def backward(self, grad_output):
        """Implement in subclasses."""
        pass

class AddBackward(Function):
    """Backward function for addition."""

    def __init__(self, tensor1, tensor2):
        super().__init__(tensor1, tensor2)

    def backward(self, grad_output):
        if self.inputs[0].requires_grad:
            self.inputs[0].backward(grad_output)
        if self.inputs[1].requires_grad:
            self.inputs[1].backward(grad_output)

class MulBackward(Function):
    """Backward function for multiplication."""

    def __init__(self, tensor1, tensor2):
        super().__init__(tensor1, tensor2)

    def backward(self, grad_output):
        if self.inputs[0].requires_grad:
            # d/dx (x * y) = y
            grad = grad_output * self.inputs[1]
            self.inputs[0].backward(grad)
        if self.inputs[1].requires_grad:
            # d/dy (x * y) = x
            grad = grad_output * self.inputs[0]
            self.inputs[1].backward(grad)
```

## Educational Benefits

### 1. Clear Learning Progression
- **Modules 01-04**: Focus on core concepts without gradient complexity
- **Module 05**: Learn how autograd extends existing functionality
- **Python Skills**: Understand decorators and metaprogramming

### 2. Realistic Software Engineering
- **Extension Pattern**: How real codebases evolve features
- **Decorator Pattern**: Professional Python development technique
- **Backward Compatibility**: Original code continues working

### 3. Deep Understanding
- **Students see the "magic" happen**: How PyTorch-like functionality is built
- **Explicit transformation**: From pure data structure to autograd-capable
- **Clean separation**: Algorithm logic vs gradient tracking logic

## Implementation Guidelines for Module 05

### Structure
```
Module 05: Autograd - Extending Tensor with Gradient Tracking

Part 1: Understanding the Problem
- Why pure tensors can't learn
- What gradient tracking enables

Part 2: Python Decorators Introduction
- What decorators do
- How to extend classes
- Preserving original functionality

Part 3: Building the Autograd Decorator
- Step 1: Save original methods
- Step 2: Create enhanced methods
- Step 3: Add backward propagation
- Step 4: Apply decorator

Part 4: Computation Graph Classes
- Function base class
- Specific backward functions
- Chain rule implementation

Part 5: Testing the Extension
- Original functionality preserved
- New gradient capabilities work
- Integration with modules 01-04
```

### Key Teaching Points
1. **Extension, not replacement**: Original tensor code still works
2. **Metaprogramming power**: How Python enables elegant solutions
3. **Computation graphs**: How autograd systems track operations
4. **Chain rule**: Mathematical foundation of backpropagation

## Success Criteria
- ✅ Modules 01-04 continue working unchanged
- ✅ Module 05 adds gradient tracking seamlessly
- ✅ Students understand decorator pattern
- ✅ Clear path to understanding PyTorch internals
- ✅ Professional Python development patterns demonstrated

## Alternative Approaches Considered

### 1. Inheritance (Rejected)
```python
class AutogradTensor(Tensor):
    # Problems: requires changing all existing code
```

### 2. Composition (Rejected)
```python
class Variable:
    def __init__(self, tensor):
        self.tensor = tensor
    # Problems: wrapper complexity, duplication
```

### 3. Monkey Patching (Rejected)
```python
def add_grad(tensor):
    tensor.requires_grad = True
# Problems: not systematic, hard to understand
```

## Why Decorator Pattern Wins
- **Systematic**: Enhances entire class at once
- **Educational**: Teaches professional Python patterns
- **Clean**: No code duplication or wrapper complexity
- **Realistic**: How real frameworks evolve features
- **Preserving**: All existing functionality remains