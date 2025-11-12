# TinyTorch Testing Quick Reference

## 🚀 Quick Start

### **For Students**
```bash
# 1. Run inline tests (fast feedback)
python modules/XX_modulename/modulename.py

# 2. Export to package
tito export XX_modulename

# 3. Run module tests
pytest tests/XX_modulename/ -v

# 4. Run critical integration tests
pytest tests/integration/test_gradient_flow.py -v
```

### **For Maintainers**
```bash
# Run all tests
pytest tests/ -v

# Run critical tests only
pytest tests/integration/test_gradient_flow.py -v

# Run tests for specific module
pytest tests/XX_modulename/ -v
```

---

## 📋 Test Categories Checklist

For each module, verify:

- [ ] **Core Functionality** - Does it work?
- [ ] **Gradient Flow** - Do gradients flow? (if trainable)
- [ ] **Integration** - Works with other modules?
- [ ] **Shape Correctness** - Shapes handled correctly?
- [ ] **Edge Cases** - Handles edge cases?
- [ ] **Export/Import** - Exports correctly?

---

## 🔥 Critical Tests (Must Pass)

These tests **must pass** before merging:

1. **Gradient Flow**: `tests/integration/test_gradient_flow.py`
   - If this fails, training is broken

2. **Module Integration**: `tests/XX_modulename/test_progressive_integration.py`
   - Ensures module works with previous modules

3. **Export/Import**: Verify module exports to `tinytorch.*`
   - Students need to import from package

---

## 📊 Module Status Quick Check

| Module | Core | Gradients | Integration | Status |
|--------|------|-----------|-------------|--------|
| 01_tensor | ✅ | N/A | ✅ | ✅ Good |
| 02_activations | ✅ | ⚠️ | ✅ | ⚠️ Missing gradients |
| 03_layers | ✅ | ✅ | ✅ | ✅ Good |
| 04_losses | ✅ | ✅ | ✅ | ✅ Good |
| 05_autograd | ✅ | ✅ | ✅ | ✅ Excellent |
| 06_optimizers | ⚠️ | ⚠️ | ✅ | ⚠️ Missing core |
| 07_training | ✅ | ✅ | ⚠️ | ⚠️ Missing convergence |
| 08_dataloader | ✅ | N/A | ⚠️ | ⚠️ Missing edge cases |
| 09_spatial | ✅ | ✅ | ✅ | ✅ Good |
| 10_tokenization | ⚠️ | N/A | ✅ | ⚠️ Missing core |
| 11_embeddings | ✅ | ✅ | ✅ | ✅ Good |
| 12_attention | ⚠️ | ⚠️ | ✅ | ⚠️ Missing core |
| 13_transformers | ✅ | ✅ | ✅ | ✅ Excellent |
| 14_profiling | ✅ | N/A | ✅ | ✅ Good |
| 15-20 | ⚠️ | ⚠️ | ⚠️ | ⚠️ Needs assessment |

**Legend**: ✅ Complete | ⚠️ Gaps | ❌ Missing | N/A Not Applicable

---

## 🎯 Priority Actions

### **High Priority** (Do First)
1. Module 02_activations: Add gradient flow tests
2. Module 06_optimizers: Add core functionality tests
3. Module 07_training: Add convergence tests

### **Medium Priority** (Do Next)
4. Module 08_dataloader: Add edge case tests
5. Module 10_tokenization: Add core tests
6. Module 12_attention: Add core tests

### **Low Priority** (Nice to Have)
7. Modules 15-20: Assess and add tests
8. All modules: Add export/import tests

---

## 📝 Test File Structure

For module `XX_modulename`:

```
tests/XX_modulename/
├── test_[modulename]_core.py              # Core functionality
├── test_gradient_flow.py                  # Gradient flow (if applicable)
├── test_[modulename]_integration.py       # Integration
├── test_progressive_integration.py        # Progressive integration
├── test_edge_cases.py                     # Edge cases
└── test_real_world_usage.py               # Real-world usage
```

---

## 🔍 Common Test Patterns

### **Gradient Flow Test**
```python
def test_component_gradient_flow():
    component = Component(...)
    x = Tensor(..., requires_grad=True)
    output = component(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    for param in component.parameters():
        assert param.grad is not None
```

### **Integration Test**
```python
def test_module_integration():
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Linear
    
    # Test components work together
    x = Tensor([[1.0, 2.0]])
    layer = Linear(2, 3)
    output = layer(x)
    assert output.shape == (1, 3)
```

### **Edge Case Test**
```python
def test_edge_cases():
    # Empty input
    result = component(Tensor([]))
    
    # Zero values
    result = component(Tensor([0.0]))
    
    # Large values
    result = component(Tensor([1e10]))
```

---

## 📚 Full Documentation

- **Test Separation Plan**: `docs/development/TEST_SEPARATION_PLAN.md` - **START HERE** - What goes where
- **Master Plan**: `docs/development/MASTER_TESTING_PLAN.md`
- **Testing Architecture**: `docs/development/testing-architecture.md`
- **Gradient Flow Strategy**: `docs/development/gradient-flow-testing-strategy.md`
- **Comprehensive Plan**: `docs/development/comprehensive-module-testing-plan.md`

---

**Last Updated**: 2025-01-XX  
**Quick Reference**: For detailed plans, see MASTER_TESTING_PLAN.md

