# TinyTorch Testing Best Practices
## Creating a Robust Learning Sandbox

### 🎯 Core Principle: The Framework Must Be Invisible

**Students should focus on ML concepts, not framework debugging.**

**When we discover a bug, we immediately:**
1. **Document it** - What broke and why
2. **Fix it** - Implement the solution
3. **Test it** - Write a regression test to prevent recurrence
4. **Categorize it** - Place the test in the appropriate location

---

## 📂 Test Organization Strategy

### **1. Student-Facing Tests (In Modules)**
**Location**: `modules/XX_module/module_dev.py`
**Purpose**: Educational, concept-focused
**What goes here**:
- Tests that teach concepts
- Simple validation of their implementations
- "Did I understand this correctly?" checks
- Clear, pedagogical test cases

**Example**:
```python
def test_unit_conv2d():
    """Test that Conv2d produces correct output shape."""
    conv = Conv2d(3, 32, kernel_size=3)
    x = Tensor(np.random.randn(1, 3, 32, 32))
    output = conv(x)
    assert output.shape == (1, 32, 30, 30), "Conv2d output shape incorrect"
```

### **2. Integration Tests (System Validation)**
**Location**: `tests/integration/`
**Purpose**: Verify modules work together
**What goes here**:
- Cross-module compatibility tests
- Data flow validation
- Shape/dimension compatibility
- API contract tests

**Example**:
```python
# tests/integration/test_conv_to_linear_integration.py
def test_conv_output_matches_linear_input():
    """Regression test for CNN shape mismatch bug found 2024-11-25."""
    # This is the bug we found in alexnet example
    conv1 = Conv2d(3, 32, kernel_size=3)
    conv2 = Conv2d(32, 64, kernel_size=3)
    
    x = Tensor(np.random.randn(1, 3, 32, 32))  # CIFAR image
    x = conv1(x)  # -> (1, 32, 30, 30)
    x = F.max_pool2d(x, 2)  # -> (1, 32, 15, 15)
    x = conv2(x)  # -> (1, 64, 13, 13)
    x = F.max_pool2d(x, 2)  # -> (1, 64, 6, 6)
    
    flat_size = 64 * 6 * 6  # 2304
    fc = Linear(flat_size, 128)
    x_flat = x.reshape(1, -1)
    
    # This should not raise ValueError
    output = fc(x_flat)
    assert output.shape == (1, 128)
```

### **3. Sandbox Integrity Tests**
**Location**: `tests/regression/`
**Purpose**: Keep the student sandbox robust
**What goes here**:
- Infrastructure that must work perfectly
- Common integration patterns students will use
- Shape compatibility guarantees
- "This must always work" tests

**Example**:
```python
# tests/regression/test_transformer_output_dimensions.py
def test_transformer_3d_to_linear_2d():
    """
    Regression test for TinyGPT bug: transformer outputs 3D but Linear expects 2D.
    Bug discovered: 2024-11-25 in gpt_2018 example
    """
    transformer = TransformerBlock(embed_dim=128, num_heads=4)
    linear = Linear(128, 1000)  # vocab projection
    
    x = Tensor(np.random.randn(2, 10, 128))  # (batch, seq, embed)
    transformer_out = transformer(x)  # Still (2, 10, 128)
    
    # Should handle reshaping gracefully
    batch, seq, embed = transformer_out.shape
    reshaped = transformer_out.reshape(batch * seq, embed)
    output = linear(reshaped)
    
    assert output.shape == (20, 1000), "Linear should handle reshaped transformer output"
```

### **4. System Tests (End-to-End Validation)**
**Location**: `tests/system/`
**Purpose**: Validate complete pipelines work
**What goes here**:
- Full training loop tests
- Complete model architectures
- Data loading to training pipelines
- Milestone validation tests

---

## 🔧 Bug Discovery Workflow

### **When You Find a Bug:**

```python
# 1. DOCUMENT: Create a regression test immediately
# tests/regression/test_issue_YYYYMMDD_description.py
"""
BUG REPORT:
Date: 2024-11-25
Found in: examples/alexnet_2012/train_cnn.py
Issue: Conv output size (2304) doesn't match FC input (1600)
Root cause: Incorrect calculation of conv output dimensions
Fix: Calculate actual dimensions after pooling
"""

def test_conv_dimension_calculation():
    """Ensure conv output dimensions are calculated correctly."""
    # Test that reproduces the exact bug
    ...

# 2. FIX: Implement the solution
# (fix in the actual module)

# 3. VERIFY: Run the regression test
pytest tests/regression/test_issue_20241125_conv_dims.py

# 4. INTEGRATE: Add to CI/CD pipeline
# The test now runs on every commit
```

---

## 📊 Test Categories by Purpose

| Test Type | Location | Purpose | Who Sees It | Example |
|-----------|----------|---------|-------------|---------|
| **Unit Tests** | `modules/*/` | Teach & validate basic functionality | Students | "Conv2d produces correct shape" |
| **Integration Tests** | `tests/integration/` | Verify modules work together | Developers | "Conv output fits Linear input" |
| **Regression Tests** | `tests/regression/` | Prevent bug recurrence | Developers | "Fix for issue #123" |
| **System Tests** | `tests/system/` | End-to-end validation | Developers | "Train CNN on CIFAR-10" |
| **Performance Tests** | `tests/performance/` | Benchmark & optimization | Developers | "Conv2d under 100ms" |

---

## 🎯 Best Practices

### **1. Name Tests Descriptively**
```python
# ❌ Bad
def test_conv():
    
# ✅ Good  
def test_conv2d_output_shape_with_padding():
```

### **2. Include Bug Context**
```python
def test_regression_conv_fc_shape_mismatch():
    """
    Regression test for bug found 2024-11-25.
    Issue: Conv output (2304) != FC input (1600) in CNN example.
    PR: #456
    """
```

### **3. Test the Actual Bug**
```python
# Don't just test general functionality
# Test the EXACT scenario that failed
def test_cifar10_cnn_architecture_shapes():
    """Test exact architecture from alexnet_2012 example."""
    # Use exact same layer sizes that failed
    model = SimpleCNN(num_classes=10)
    x = Tensor(np.random.randn(32, 3, 32, 32))  # CIFAR batch
    
    # This exact forward pass failed before
    output = model(x)
    assert output.shape == (32, 10)
```

### **4. Separate Concerns**
- **Unit tests**: Test one thing in isolation
- **Integration tests**: Test how things connect
- **System tests**: Test complete workflows
- **Regression tests**: Test specific fixed bugs

### **5. Fast Feedback Loop**
```bash
# After fixing a bug, immediately:
1. Write the test
2. Verify it catches the bug (test should fail without fix)
3. Verify the fix works (test should pass with fix)
4. Commit both together
```

---

## 🚀 Implementation Strategy

### **Immediate Action Items:**
1. Create `tests/regression/` directory
2. Move complex integration tests out of student modules
3. Document every bug we find with a regression test
4. Add regression suite to CI/CD pipeline

### **File Structure:**
```
tests/
├── unit/                  # Basic functionality (mirrors modules/)
├── integration/           # Module interactions
├── regression/           # Bug prevention (NEW)
│   ├── test_issue_20241125_conv_dims.py
│   ├── test_issue_20241125_transformer_reshape.py
│   └── README.md        # Bug index and descriptions
├── system/              # End-to-end workflows
└── performance/         # Benchmarks and optimization

modules/XX_module/
└── module_dev.py        # Simple, educational tests only
```

---

## 📝 Bug Tracking Template

```python
"""
BUG TRACKING:
============
Bug ID: BUG-YYYY-MM-DD-001
Date Found: YYYY-MM-DD
Found By: [Name/System]
Severity: [Critical/High/Medium/Low]

DESCRIPTION:
What broke and under what conditions

REPRODUCTION:
Exact steps to reproduce

ROOT CAUSE:
Why it happened

FIX:
What was changed to fix it

PREVENTION:
This regression test ensures it never happens again
"""

def test_regression_bug_YYYYMMDD_001():
    """Test that [specific bug] is fixed."""
    # Exact reproduction of the bug scenario
    # Should pass with fix, fail without it
```

---

## 🏆 Success Metrics

**We know we're doing this right when:**
1. ✅ Every bug discovered has a corresponding regression test
2. ✅ No bug resurfaces after being fixed
3. ✅ Students see clean, simple tests in modules
4. ✅ Developers have comprehensive regression coverage
5. ✅ Integration issues are caught before merging

---

## 🎓 Educational Impact

**For Students:**
- They see clean, focused unit tests that teach concepts
- Not overwhelmed by complex regression/integration tests
- Learn good testing practices by example

**For Maintainers:**
- Complete regression coverage prevents bugs from returning
- Integration tests catch composition issues early
- Clear separation of educational vs. system tests

---

## 🔄 Continuous Improvement

**Monthly Review:**
1. Count bugs found vs. bugs with tests
2. Review regression test effectiveness
3. Move stable regression tests to integration tests
4. Update this document with new patterns

**Remember**: The goal is not just to fix bugs, but to build a system where bugs CAN'T return. Every test we write is an investment in TinyTorch's reliability and educational value.