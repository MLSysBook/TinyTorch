# Gradient Flow Testing Guide

## Overview
This guide documents the testing strategy for preventing gradient flow regressions in TinyTorch. After fixing critical gradient flow issues in CNN and Transformer milestones, we've established a comprehensive testing framework to catch these issues early.

---

## Test Hierarchy

### 1. **Unit Tests** (Fastest, Most Specific)
Test individual backward functions in isolation.

**Location**: `tests/{module_number}_{module_name}/test_{component}_gradient_flow.py`

**Examples**:
- `tests/09_spatial/test_spatial_gradient_flow.py` - Conv2d, MaxPool2d
- `tests/11_embeddings/test_embedding_gradient_flow.py` - Embedding
- `tests/05_autograd/test_gradient_flow.py` - Basic operations
- `tests/13_transformers/test_transformer_gradient_flow.py` - Attention, LayerNorm

**What they test**:
- ✅ `_grad_fn` is attached to output tensors
- ✅ Gradients flow to all parameters (weight, bias)
- ✅ Gradients flow to inputs (for backprop through layers)
- ✅ End-to-end gradient chains (e.g., Conv2d → MaxPool2d)

**Run them**:
```bash
# Individual module
python3 tests/09_spatial/test_spatial_gradient_flow.py
python3 tests/11_embeddings/test_embedding_gradient_flow.py

# All unit tests in a module
cd tests/09_spatial && python3 run_all_tests.py
```

---

### 2. **Integration Tests** (Medium Speed, Cross-Module)
Test interactions between multiple modules.

**Location**: `tests/integration/`

**What they test**:
- Module combinations work together
- Gradients flow across module boundaries
- Data loaders integrate with autograd

**Run them**:
```bash
python3 tests/integration/test_cnn_gradient_flow.py
```

---

### 3. **Milestone Learning Tests** (Slowest, Most Comprehensive)
Test that complete historical models actually learn.

**Location**: `tests/milestones/test_learning_verification.py`

**What they test**:
- ✅ Loss decreases over training
- ✅ All parameters receive gradients
- ✅ All parameters update during training
- ✅ Model achieves target accuracy
- ✅ Specific component checks (Conv, Embedding, Attention)

**Run them**:
```bash
# All milestones
python3 tests/milestones/test_learning_verification.py

# Individual milestone
python3 -c "from tests.milestones.test_learning_verification import test_cnn_learning; test_cnn_learning()"
```

**Coverage**:
1. Perceptron (1957) - 2 params
2. XOR (1969) - 4 params
3. MLP Digits (1986) - 4 params
4. CNN (1998) - 6 params ✅ **Fixed: Conv2d, MaxPool2d**
5. Transformer (2017) - 19 params ✅ **Fixed: Embedding, Attention**

---

## Common Gradient Flow Issues

### Issue 1: Using `.data` Bypasses Autograd ❌

**Problem**:
```python
# WRONG - breaks gradient flow
output = self.weight.data[indices.data]
result = Tensor(output)  # No _grad_fn attached!
return result
```

**Solution**:
```python
# CORRECT - preserves gradient flow
output = self.weight.data[indices.data]
result = Tensor(output, requires_grad=True)
result._grad_fn = EmbeddingBackward(self.weight, indices)
return result
```

**Tests that catch this**:
- `test_{component}_has_backward_function()` - Checks `_grad_fn` is attached
- `test_{component}_weight_gradient_flow()` - Verifies gradients reach parameters

---

### Issue 2: Creating New Tensors in Residual Connections ❌

**Problem**:
```python
# WRONG - breaks computation graph
x = Tensor(x.data + attn_out.data)  # New Tensor from .data
```

**Solution**:
```python
# CORRECT - uses autograd-tracked addition
x = x + attn_out  # Preserves _grad_fn chain
```

**Tests that catch this**:
- Milestone learning tests (integration level)
- `test_conv2d_maxpool2d_chain()` - Tests gradient chains

---

### Issue 3: Using Reshape Incorrectly ❌

**Problem**:
```python
# WRONG - creates new Tensor without _grad_fn
x = Tensor(x.data.reshape(batch_size, -1))
```

**Solution**:
```python
# CORRECT - uses Tensor.reshape() which preserves autograd
x = x.reshape(batch_size, -1)  # Attaches ReshapeBackward
```

**Tests that catch this**:
- Milestone learning tests (checks end-to-end gradient flow)

---

### Issue 4: Not Registering Parameters with Autograd ❌

**Problem**:
```python
class Conv2dBackward(Function):
    def __init__(self, x, weight, bias, ...):
        super().__init__(x, weight)  # Forgot bias!
```

**Solution**:
```python
class Conv2dBackward(Function):
    def __init__(self, x, weight, bias, ...):
        if bias is not None:
            super().__init__(x, weight, bias)  # Include all params
        else:
            super().__init__(x, weight)
```

**Tests that catch this**:
- `test_conv2d_bias_gradient_flow()` - Specifically checks bias gradients
- Milestone learning tests (checks all parameters)

---

## How to Add Tests for New Modules

When implementing a new operation with a backward function:

### 1. Create Unit Tests

**Template**: `tests/{module_num}_{module_name}/test_{operation}_gradient_flow.py`

```python
def test_{operation}_has_backward_function():
    """Test that {Operation} attaches _grad_fn to output."""
    op = YourOperation(...)
    x = Tensor(np.random.randn(...), requires_grad=True)
    
    output = op(x)
    
    assert hasattr(output, '_grad_fn'), "Output should have _grad_fn"
    assert output._grad_fn is not None
    assert type(output._grad_fn).__name__ == "YourBackward"

def test_{operation}_parameter_gradient_flow():
    """Test that {Operation} parameters receive gradients."""
    op = YourOperation(...)
    op.weight.requires_grad = True
    
    x = Tensor(np.random.randn(...), requires_grad=True)
    output = op(x)
    loss = output.sum()
    loss.backward()
    
    assert op.weight.grad is not None
    assert not np.allclose(op.weight.grad.data, 0)

def test_{operation}_input_gradient_flow():
    """Test that {Operation} propagates gradients to input."""
    op = YourOperation(...)
    x = Tensor(np.random.randn(...), requires_grad=True)
    
    output = op(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not np.allclose(x.grad.data, 0)
```

### 2. Add to Integration Tests

If your operation is part of a larger architecture (e.g., new attention mechanism), add integration tests.

### 3. Optionally Add Milestone Test

If your operation is central to a historical model, consider adding a milestone learning test.

---

## Regression Prevention Checklist

When implementing new operations, verify:

- [ ] **`_grad_fn` Attachment**: Output tensors have backward function attached
- [ ] **Parameter Registration**: All parameters passed to `Function.__init__()`
- [ ] **Return Type**: `apply()` returns tuple of gradients (not single gradient)
- [ ] **`.data` Usage**: Never use `.data` except for final numpy operations
- [ ] **Tensor Creation**: Use Tensor operations (reshape, transpose) not `Tensor(...data...)`
- [ ] **Unit Tests**: Created `test_{operation}_gradient_flow.py`
- [ ] **All Tests Pass**: Run unit, integration, and milestone tests

---

## Quick Reference

### Run All Gradient Flow Tests
```bash
# Unit tests (fast)
python3 tests/09_spatial/test_spatial_gradient_flow.py
python3 tests/11_embeddings/test_embedding_gradient_flow.py
python3 tests/05_autograd/test_gradient_flow.py
python3 tests/13_transformers/test_transformer_gradient_flow.py

# Milestone tests (comprehensive)
python3 tests/milestones/test_learning_verification.py
```

### Debug Gradient Flow Issues
```python
# Check if _grad_fn is attached
output = layer(x)
print(f"Has _grad_fn: {hasattr(output, '_grad_fn')}")
print(f"Type: {type(output._grad_fn).__name__ if hasattr(output, '_grad_fn') else 'None'}")

# Check if gradients flow
loss.backward()
for param in layer.parameters():
    print(f"Param grad: {param.grad is not None}, non-zero: {not np.allclose(param.grad.data, 0) if param.grad else False}")
```

---

## Summary

**The Test Pyramid**:
```
        Milestone Tests (5)
       ↗                   ↖
    Integration Tests (~10)
   ↗                         ↖
Unit Tests (50+)
```

**Coverage**:
- ✅ **Unit Tests**: 14 tests across 4 modules
- ✅ **Integration Tests**: Existing cross-module tests
- ✅ **Milestone Tests**: 5 comprehensive end-to-end tests

**Protection**:
- Catches `.data` bypass issues
- Catches missing `_grad_fn` attachment
- Catches parameter registration issues
- Validates end-to-end learning

This testing framework ensures that gradient flow regressions are caught at the earliest possible level, preventing silent failures where models "run" but don't learn.

