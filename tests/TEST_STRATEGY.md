# TinyTorch Test Strategy

> **📚 For the complete testing plan**: See `docs/development/MASTER_TESTING_PLAN.md`  
> **🚀 Quick Reference**: See `docs/development/TESTING_QUICK_REFERENCE.md`

## 🎯 Testing Philosophy

TinyTorch uses a **two-tier testing approach** that separates component validation from system integration:

1. **Inline Tests** (in module source files) - Component validation
2. **Integration Tests** (in `tests/` directory) - Inter-module integration

This separation follows ML engineering best practices: validate components in isolation, then test how they work together.

---

## 📋 Tier 1: Inline Tests (Component Validation)

### **Location**: `modules/XX_modulename/*_dev.py`

### **Purpose**:
- Validate individual components work correctly
- Test in isolation from other modules
- Provide immediate feedback during development
- Educate students about expected behavior

### **What to Test**:
✅ Individual class/function correctness
✅ Mathematical operations (forward passes)
✅ Shape transformations
✅ Edge cases and error handling
✅ Basic functionality

### **Format**:
```python
def test_unit_componentname():
    """🧪 Unit Test: Component Name
    
    **This is a unit test** - it tests [component] in isolation.
    """
    print("🔬 Unit Test: Component...")
    
    # Test implementation
    assert condition, "✅ Component works"
    
    print("✅ Component test passed")
    print("📈 Progress: Component ✓")
```

### **Execution**:
```bash
# Run inline tests only
tito test 01_tensor --inline-only

# Tests run when you execute the module file
python modules/01_tensor/tensor_dev.py
```

### **Current Status** (Modules 01-15):
- ✅ **Passing**: 11/15 modules (73%)
  - 01_tensor, 03_layers, 04_losses, 05_autograd
  - 07_training, 08_dataloader, 09_spatial, 10_tokenization
  - 11_embeddings, 13_transformers, 14_profiling

- ❌ **Failing**: 4/15 modules (27%)
  - 02_activations - Script execution error
  - 06_optimizers - Script execution error
  - 12_attention - Assertion error
  - 15_memoization - Missing matplotlib dependency

---

## 📊 Tier 2: Integration Tests (System Validation)

### **Location**: `tests/XX_modulename/test_*_integration.py`

### **Purpose**:
- Test how modules work together
- Validate cross-module dependencies
- Test realistic workflows
- Ensure system-level correctness

### **What to Test**:
✅ Module interactions (e.g., Tensor → Autograd → Optimizer)
✅ End-to-end workflows (e.g., training loop)
✅ Data flow through pipeline
✅ Real-world use cases
✅ Progressive integration (modules 1-N)

### **Test Types**:

#### 1. **Progressive Integration**
Tests that module N works with all previous modules (1 through N-1):
```python
# tests/05_autograd/test_progressive_integration.py
def test_autograd_with_all_previous_modules():
    # Use Tensor (01), Activations (02), Layers (03), Losses (04)
    # Then test Autograd (05) with all of them
```

#### 2. **Feature Integration**
Tests specific feature combinations:
```python
# tests/07_training/test_training_integration.py
def test_complete_training_loop():
    # Combine: Tensor + Layers + Losses + Autograd + Optimizers + Training
```

#### 3. **Benchmark Integration**  
Tests realistic end-to-end scenarios:
```python
# tests/14_profiling/test_benchmarking_integration.py
def test_profile_real_model():
    # Profile actual transformer with real data
```

### **Execution**:
```bash
# Run integration tests only
tito test 01_tensor --external-only

# Run both inline and integration
tito test 01_tensor

# Run all tests
tito test --all
```

### **Current Structure**:
```
tests/
├── 01_tensor/           ✅ (4 test files)
├── 02_activations/      ✅ (5 test files)
├── ...
├── 15_memoization/      ✅ (4 test files)
├── 16_quantization/     ✅ (2 files - pending implementation)
├── 17_compression/      ✅ (2 files - pending implementation)
├── 18_acceleration/     ✅ (2 files - pending implementation)
├── 19_benchmarking/     ✅ (2 files - pending implementation)
├── 20_capstone/         ✅ (2 files - pending implementation)
├── integration/         ✅ (27 cross-module tests)
├── checkpoints/         ✅ (23 milestone tests)
├── milestones/          ✅ (4 historical milestone tests)
└── TEST_STRATEGY.md     ✅ (this document)
```

---

## 🔄 Testing Workflow

### For Students:

```bash
# 1. Work on module
cd modules/01_tensor
vim tensor_dev.py

# 2. Run inline tests (fast feedback)
python tensor_dev.py
# or
tito test 01_tensor --inline-only

# 3. Export to package
tito export 01_tensor

# 4. Run integration tests (full validation)
tito test 01_tensor

# 5. Run progressive tests (ensure nothing broke)
pytest tests/integration/
```

### For Instructors:

```bash
# Comprehensive test suite
tito test --comprehensive

# Specific module deep dive
tito test 05_autograd --detailed

# All inline tests only (quick check)
tito test --all --inline-only
```

---

## 📈 Test Coverage Matrix

| Module | Inline Tests | Integration Tests | Status |
|--------|-------------|-------------------|--------|
| 01_tensor | ✅ Pass | ✅ Implemented | Complete |
| 02_activations | ❌ Fail | ✅ Implemented | Needs Fix |
| 03_layers | ✅ Pass | ✅ Implemented | Complete |
| 04_losses | ✅ Pass | ✅ Implemented | Complete |
| 05_autograd | ✅ Pass | ✅ Implemented | Complete |
| 06_optimizers | ❌ Fail | ✅ Implemented | Needs Fix |
| 07_training | ✅ Pass | ✅ Implemented | Complete |
| 08_dataloader | ✅ Pass | ✅ Implemented | Complete |
| 09_spatial | ✅ Pass | ✅ Implemented | Complete |
| 10_tokenization | ✅ Pass | ✅ Implemented | Complete |
| 11_embeddings | ✅ Pass | ✅ Implemented | Complete |
| 12_attention | ❌ Fail | ✅ Implemented | Needs Fix |
| 13_transformers | ✅ Pass | ✅ Implemented | Complete |
| 14_profiling | ✅ Pass | ✅ Implemented | Complete |
| 15_memoization | ❌ Fail | ✅ Implemented | Needs Fix |
| 16_quantization | ⏳ N/A | 📝 Pending | Needs Implementation |
| 17_compression | ⏳ N/A | 📝 Pending | Needs Implementation |
| 18_acceleration | ⏳ N/A | 📝 Pending | Needs Implementation |
| 19_benchmarking | ⏳ N/A | 📝 Pending | Needs Implementation |
| 20_capstone | ⏳ N/A | 📝 Pending | Needs Implementation |

**Overall**: 11/15 modules passing inline tests (73%), all modules have test infrastructure

---

## 🚀 Best Practices

### **DO**:
✅ Write inline tests immediately after implementing a component
✅ Test one thing per inline test function
✅ Use descriptive test function names (`test_unit_sigmoid`, not `test1`)
✅ Add integration tests when combining multiple modules
✅ Run inline tests frequently during development
✅ Run full test suite before committing

### **DON'T**:
❌ Mix inline and integration test concerns
❌ Test implementation details in integration tests
❌ Skip inline tests and jump to integration
❌ Test mocked/fake components (use real ones)
❌ Create dependencies between test files

---

## 🔧 Common Patterns

### **Pattern 1: Test Component in Isolation**
```python
# Inline test in 02_activations/activations_dev.py
def test_unit_sigmoid():
    sigmoid = Sigmoid()
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.269, 0.5, 0.731], atol=0.01)
```

### **Pattern 2: Test Module Integration**
```python
# Integration test in tests/05_autograd/test_progressive_integration.py
def test_autograd_with_layers():
    # Uses real Tensor, real Layers, real Autograd
    x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    layer = Linear(2, 3)
    output = layer.forward(x)
    output.backward()
    assert x.grad is not None
```

### **Pattern 3: Test Full Pipeline**
```python
# Integration test in tests/13_transformers/test_transformer_integration.py
def test_complete_transformer_pipeline():
    # Tokenization → Embedding → Attention → Transformer → Generation
    tokenizer = CharTokenizer("Hello")
    model = GPT(vocab_size=tokenizer.vocab_size)
    output = model.forward(tokenizer.encode("Hi"))
    assert output.shape == (1, len("Hi"), vocab_size)
```

---

## 📚 Additional Resources

- **Test Module Template**: `tests/module_template/`
- **Integration Test Examples**: `tests/integration/`
- **Checkpoint Tests**: `tests/checkpoints/`
- **Historical Milestones**: `tests/milestones/`
- **TinyTorch Testing Guide**: `docs/development/testing-guide.md`

---

## 🎓 For Educators

This testing structure provides:
1. **Immediate Feedback**: Inline tests give instant validation
2. **Progressive Learning**: Students see components work before integration
3. **Real Systems**: Integration tests use actual components, not mocks
4. **Industry Practices**: Mirrors professional ML engineering workflows
5. **Debugging Aid**: Clear separation helps identify where issues occur

Students learn that **component correctness ≠ system correctness**, a crucial lesson for building reliable ML systems.

---

**Last Updated**: 2025-11-10
**Test Infrastructure**: Complete (20/20 modules have test directories)
**Inline Test Coverage**: 73% passing (11/15 implemented modules)
**Integration Test Coverage**: 100% infrastructure ready, 75% implemented (15/20 modules)

