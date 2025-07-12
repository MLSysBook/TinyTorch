# TinyTorch Testing Guidelines

## Overview

TinyTorch uses a **two-tier testing system** designed to provide immediate feedback during development while ensuring comprehensive validation for production use.

## The Two-Tier Testing System

### **Tier 1: Inline Testing (For Learning)**
- **Purpose**: Immediate feedback during development
- **Location**: Within `*_dev.py` files as `🧪 Test Your Implementation` sections
- **Style**: Simple, visual, encouraging
- **When**: After each major implementation step
- **Audience**: Students learning the concepts

### **Tier 2: Comprehensive Testing (For Validation)**
- **Purpose**: Thorough validation and grading
- **Location**: `tests/test_*.py` files using pytest
- **Style**: Professional test suites with edge cases
- **When**: After completing the module
- **Audience**: Instructors and automated systems

## Why This Approach Works

### **Prevents Late-Stage Failures**
Students get immediate feedback as they build, preventing the frustration of implementing an entire module only to discover fundamental errors during final testing.

### **Builds Confidence**
Each successful inline test provides positive reinforcement and confirms the student is on the right track.

### **Teaches Testing Culture**
Students learn to test incrementally, a crucial skill for professional development.

### **Maintains Professional Standards**
The comprehensive test suites ensure that the final package meets production-quality standards.

## Inline Testing Guidelines

### **When to Add Inline Tests**

✅ **Add inline tests after:**
- Each major class implementation
- Each significant function or method
- Complex algorithms or mathematical operations
- Data loading or preprocessing steps
- Any component that students might struggle with

❌ **Don't add inline tests for:**
- Trivial getters/setters
- Simple utility functions
- Already well-tested components

### **Inline Test Structure**

```python
# %% [markdown]
"""
### 🧪 Test Your [Component] Implementation

Let's test your [Component] implementation to ensure it's working correctly:
"""

# %%
# Test [Component] implementation
print("Testing [Component] Implementation:")
print("=" * 40)

try:
    # Test 1: Basic functionality
    component = Component()
    test_input = create_test_input()
    output = component(test_input)
    
    print(f"✅ Input: {test_input}")
    print(f"✅ Output: {output}")
    
    # Check if implementation is correct
    if validate_output(output):
        print("🎉 [Component] implementation is CORRECT!")
    else:
        print("❌ [Component] implementation needs fixing")
        print("   [Specific guidance on what to check]")
    
    # Test 2: Edge cases or properties
    edge_case_test()
    
    print("✅ [Component] tests complete!")
    
except NotImplementedError:
    print("⚠️  [Component] not implemented yet - complete the method above!")
except Exception as e:
    print(f"❌ Error in [Component]: {e}")
    print("   Check your implementation in the [method] method")

print()  # Add spacing
```

### **Best Practices for Inline Tests**

#### **1. Immediate Feedback**
```python
# ✅ Good: Immediate, specific feedback
if np.allclose(output.data.flatten(), expected):
    print("🎉 ReLU implementation is CORRECT!")
else:
    print("❌ ReLU implementation needs fixing")
    print("   Make sure negative values become 0, positive values stay unchanged")

# ❌ Bad: Vague or delayed feedback
assert np.allclose(output.data.flatten(), expected)  # Just crashes
```

#### **2. Visual and Intuitive**
```python
# ✅ Good: Visual confirmation
print(f"✅ Input: {test_input.data.flatten()}")
print(f"✅ Output: {output.data.flatten()}")
print(f"✅ Expected: {expected}")

# ❌ Bad: No visual feedback
result = component(test_input)
```

#### **3. Property-Based Testing**
```python
# ✅ Good: Test mathematical properties
all_positive = np.all(output.data > 0)
sums_to_one = abs(np.sum(output.data) - 1.0) < 1e-6
print(f"✅ All outputs positive: {all_positive}")
print(f"✅ Sum equals 1.0: {sums_to_one}")

# ❌ Bad: Only test specific values
assert output.data[0] == 0.665  # Too specific, fragile
```

#### **4. Progressive Complexity**
```python
# ✅ Good: Start simple, add complexity
# Test 1: Basic functionality
basic_test()

# Test 2: Edge cases
edge_case_test()

# Test 3: Numerical stability
stability_test()

# ❌ Bad: Jump to complex cases immediately
complex_edge_case_test()  # Students get overwhelmed
```

#### **5. Helpful Error Messages**
```python
# ✅ Good: Actionable guidance
except NotImplementedError:
    print("⚠️  Sigmoid not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in Sigmoid: {e}")
    print("   Check your implementation in the forward method")
    print("   Make sure: 0 < output < 1 and sigmoid(0) = 0.5")

# ❌ Bad: Generic or unhelpful messages
except Exception as e:
    print(f"Error: {e}")  # Not helpful
```

## Comprehensive Testing Guidelines

### **Test File Structure**

```python
"""
Tests for TinyTorch [Module] module.

These tests validate [description of what module does].
Focus on [key aspects being tested].
"""

import pytest
import numpy as np
from tinytorch.core.[module] import [Classes]

class Test[Component]:
    """Test the [Component] class."""
    
    def test_[component]_basic_functionality(self):
        """Test basic [component] behavior."""
        # Test implementation
    
    def test_[component]_edge_cases(self):
        """Test [component] with edge cases."""
        # Edge case testing
    
    def test_[component]_properties(self):
        """Test mathematical properties of [component]."""
        # Property-based testing
```

### **Test Categories**

#### **1. Correctness Tests**
- Verify mathematical correctness
- Test against known expected outputs
- Validate algorithm implementations

#### **2. Property Tests**
- Test mathematical properties (e.g., symmetry, monotonicity)
- Verify invariants (e.g., probability sums to 1)
- Check boundary conditions

#### **3. Edge Case Tests**
- Extreme values (very large, very small)
- Boundary conditions (zero, negative)
- Empty inputs, single elements

#### **4. Integration Tests**
- Test components working together
- Verify data flow through pipelines
- Check compatibility between modules

#### **5. Performance Tests**
- Memory usage validation
- Reasonable execution times
- Scalability with data size

## Module-Specific Guidelines

### **Tensor Module**
- **Inline tests**: After each arithmetic operation
- **Focus**: Shape handling, broadcasting, data types
- **Visual feedback**: Print shapes and values

### **Activations Module**
- **Inline tests**: After each activation function
- **Focus**: Mathematical properties, numerical stability
- **Visual feedback**: Input/output ranges, function properties

### **Layers Module**
- **Inline tests**: After matrix multiplication, after Dense layer
- **Focus**: Weight initialization, forward pass correctness
- **Visual feedback**: Weight shapes, output dimensions

### **Networks Module**
- **Inline tests**: After Sequential class, after each network type
- **Focus**: Layer composition, architecture correctness
- **Visual feedback**: Network structure, data flow

### **DataLoader Module**
- **Inline tests**: After Dataset, DataLoader, Normalizer
- **Focus**: Data integrity, batching correctness, preprocessing
- **Visual feedback**: Sample images, batch shapes, statistics

## Implementation Checklist

### **For Each Module**

- [ ] **Inline tests after major components**
  - [ ] Basic functionality test
  - [ ] Property validation test
  - [ ] Edge case test
  - [ ] Visual feedback
  - [ ] Helpful error messages

- [ ] **Comprehensive test suite**
  - [ ] Correctness tests
  - [ ] Property tests
  - [ ] Edge case tests
  - [ ] Integration tests
  - [ ] Performance tests

- [ ] **Documentation**
  - [ ] Clear test descriptions
  - [ ] Expected behavior documented
  - [ ] Error message guidance
  - [ ] Examples of usage

### **Quality Checks**

- [ ] **Inline tests provide immediate feedback**
- [ ] **Error messages are actionable**
- [ ] **Tests cover the most likely failure modes**
- [ ] **Visual feedback helps build intuition**
- [ ] **Progressive complexity from simple to advanced**

## Examples from Existing Modules

### **Excellent Inline Testing: Activations Module**

```python
# Test ReLU implementation
print("Testing ReLU Implementation:")
print("=" * 40)

try:
    relu = ReLU()
    
    # Test 1: Basic functionality
    test_input = Tensor([[-3, -1, 0, 1, 3]])
    output = relu(test_input)
    expected = [0, 0, 0, 1, 3]
    
    print(f"✅ Input: {test_input.data.flatten()}")
    print(f"✅ Output: {output.data.flatten()}")
    print(f"✅ Expected: {expected}")
    
    # Check if implementation is correct
    if np.allclose(output.data.flatten(), expected):
        print("🎉 ReLU implementation is CORRECT!")
    else:
        print("❌ ReLU implementation needs fixing")
        print("   Make sure negative values become 0, positive values stay unchanged")
    
    print("✅ ReLU tests complete!")
    
except NotImplementedError:
    print("⚠️  ReLU not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in ReLU: {e}")
    print("   Check your implementation in the forward method")
```

### **Good Progressive Testing: Tensor Module**

```python
# Test basic tensor creation
print("Testing Tensor creation...")

try:
    # Test scalar
    t1 = Tensor(5)
    print(f"✅ Scalar: {t1} (shape: {t1.shape}, size: {t1.size})")
    
    # Test vector
    t2 = Tensor([1, 2, 3, 4])
    print(f"✅ Vector: {t2} (shape: {t2.shape}, size: {t2.size})")
    
    # Test matrix
    t3 = Tensor([[1, 2], [3, 4]])
    print(f"✅ Matrix: {t3} (shape: {t3.shape}, size: {t3.size})")
    
    print("\n🎉 All basic tests passed! Your Tensor class is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement all the required methods!")
```

## Conclusion

The two-tier testing system ensures that students receive immediate, helpful feedback during development while maintaining professional-quality validation standards. This approach:

1. **Reduces frustration** by catching errors early
2. **Builds confidence** through positive reinforcement
3. **Teaches good practices** for incremental testing
4. **Maintains quality** through comprehensive validation

By following these guidelines, every TinyTorch module provides an excellent learning experience while producing production-ready code. 