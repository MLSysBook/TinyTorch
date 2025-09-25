# TinyTorch Architecture Fix: Unified Data Interface

## Problem: Inconsistent Data Access Patterns

Current broken architecture:
- `Tensor.data` returns `np.ndarray`
- `Variable.data` returns `Tensor` 
- Operations need complex conditional logic: `if hasattr(x, 'data') and hasattr(x.data, 'data'):`

## PyTorch-Inspired Solution: Single Data Extraction Interface

### 1. Universal `.numpy()` Method

**Every tensor-like object should have a `.numpy()` method that returns `np.ndarray`:**

```python
class Tensor:
    def numpy(self) -> np.ndarray:
        """Convert tensor to numpy array - ALWAYS returns np.ndarray"""
        return self._data
        
class Variable:
    def numpy(self) -> np.ndarray:
        """Convert variable to numpy array - ALWAYS returns np.ndarray"""
        return self.data.numpy()  # Delegate to underlying tensor
        
def Parameter(data):
    """Parameter is just a Tensor with requires_grad=True"""
    return Tensor(data, requires_grad=True)
```

### 2. Consistent `.data` Property

**Make `.data` consistent - either always returns np.ndarray OR always returns same type:**

**Option A: Always return np.ndarray**
```python
class Tensor:
    @property
    def data(self) -> np.ndarray:
        return self._data
        
class Variable:
    @property  
    def data(self) -> np.ndarray:
        return self._tensor.data  # Always np.ndarray
```

**Option B: Always return same type (PyTorch way)**
```python  
class Tensor:
    @property
    def data(self) -> 'Tensor':
        return Tensor(self._data, requires_grad=False)  # Detached tensor
        
class Variable:
    @property
    def data(self) -> 'Tensor':
        return self._tensor  # Always Tensor
```

### 3. Operations Use Single Interface

**With universal `.numpy()`, operations become clean:**

```python
def conv2d_operation(x, weight, bias=None):
    # BEFORE: Complex conditional logic
    # if hasattr(x, 'data') and hasattr(x.data, 'data'):
    #     input_data = x.data.data
    # elif hasattr(x, 'data'):
    #     input_data = x.data
    
    # AFTER: Clean single interface
    input_data = x.numpy()
    weight_data = weight.numpy()
    bias_data = bias.numpy() if bias is not None else None
    
    # Perform operation
    result = actual_convolution(input_data, weight_data, bias_data)
    return Tensor(result)
```

## Implementation Steps

### Step 1: Add `.numpy()` to All Tensor Types

```python
# In Tensor class (modules/02_tensor/tensor_dev.py)
def numpy(self) -> np.ndarray:
    """Convert to numpy array - the universal interface."""
    return self._data

# In Variable class (autograd module)  
def numpy(self) -> np.ndarray:
    """Convert to numpy array - the universal interface."""
    return self.data.numpy()
```

### Step 2: Update All Operations

Replace conditional data extraction:
```python
# OLD BROKEN WAY:
if hasattr(x, 'data') and hasattr(x.data, 'data'):
    x_array = x.data.data
elif hasattr(x, 'data'):
    x_array = x.data  
else:
    x_array = x

# NEW CLEAN WAY:
x_array = x.numpy()
```

### Step 3: Fix Variable.data Property

Make Variable.data consistent with Tensor.data:
```python
class Variable:
    @property
    def data(self) -> np.ndarray:  # Return same type as Tensor.data
        return self._tensor.data  # Delegate to underlying tensor
```

## Benefits of This Fix

1. **Eliminates all conditional logic** in operations
2. **Consistent interface** - `.numpy()` always returns `np.ndarray`
3. **PyTorch-compatible** - mirrors `tensor.numpy()` pattern
4. **Type safety** - operations know what they're getting
5. **Performance** - no more complex type checking

## Files to Fix

1. `modules/02_tensor/tensor_dev.py` - Add `.numpy()` method
2. Autograd module - Fix `Variable.data` property and add `.numpy()`
3. `tinytorch/core/spatial.py` - Replace conditional logic with `.numpy()`
4. Any other operations with complex data extraction

This is the fundamental architectural fix that will eliminate your hacky workarounds.