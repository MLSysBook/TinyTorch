# NBGrader Testing Cell Template

## 🎯 Standardized Module Structure

Every TinyTorch module should follow this structure for consistent testing:

### 1. Educational Content
```python
# %% [markdown]
"""
# Module X: Topic Name
[Educational content, implementation guidance, etc.]
"""

# %%
#| default_exp core.module_name

# Student implementation sections...
# [Student code here]
```

### 2. Individual Test Functions
```python
# Test functions that students can run during development
def test_feature_1_comprehensive():
    """Test feature 1 functionality comprehensively."""
    # Detailed test implementation
    assert feature_works()
    print("✅ Feature 1 tests passed!")

def test_feature_2_integration():
    """Test feature 2 integration with other components."""
    # Integration test implementation
    assert integration_works()
    print("✅ Feature 2 integration tests passed!")

def test_module_integration():
    """Test overall module integration."""
    # Overall integration tests
    assert module_works()
    print("✅ Module integration tests passed!")
```

### 3. Dedicated Testing Section (Auto-Discovery)
```python
# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework with **automatic test discovery**.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    from tinytorch.utils.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("ModuleName")
```

### 4. Module Summary (After Testing)
```python
# %% [markdown]
"""
## 🎯 Module Summary: [Topic] Mastery!

Congratulations! You've successfully implemented [module topic]:

### What You've Accomplished
✅ **Feature 1**: Description of what was implemented
✅ **Feature 2**: Description of what was implemented
✅ **Integration**: How features work together

### Key Concepts You've Learned
- **Concept 1**: Explanation
- **Concept 2**: Explanation
- **Concept 3**: Explanation

### Next Steps
1. **Export your code**: `tito package nbdev --export module_name`
2. **Test your implementation**: `tito test module_name`
3. **Move to next module**: Brief description of what's next
"""
```

## 🎯 **Critical: Correct Section Ordering**

The order of sections **must** follow this logical flow:

1. **Educational Content** - Students learn the concepts
2. **Implementation Sections** - Students build the functionality
3. **🧪 Module Testing** - Students verify their implementation works
4. **🎯 Module Summary** - Students celebrate success and move forward

### ❌ **Wrong Order (Confusing)**:
```
Implementation → Summary ("Congratulations!") → Testing → "Wait, did it work?"
```

### ✅ **Correct Order (Natural)**:
```
Implementation → Testing → Summary ("Congratulations! It works!") → Next Steps
```

**Why This Matters**: 
- Testing **validates** the implementation before celebrating
- Summary **confirms** success after verification
- Natural flow: Build → Test → Celebrate → Advance
- Mirrors real software development practices

## 🔍 Automatic Test Discovery

The new testing framework **automatically discovers** test functions, eliminating manual registration:

### ✅ **Discovered Test Patterns**
The system automatically finds and runs functions matching these patterns:
- `test_*_comprehensive`: Comprehensive testing of individual features
- `test_*_integration`: Integration testing with other components
- `test_*_activation`: Specific activation function tests (ReLU, Sigmoid, etc.)

### ✅ **Benefits**
- **Zero Manual Work**: No need to register functions manually
- **Error Prevention**: Won't miss test functions
- **Consistent Naming**: Enforces good test naming conventions
- **Automatic Ordering**: Tests run in alphabetical order
- **Clean Output**: Standardized reporting format

### ✅ **Example Output**
```
🔍 Auto-discovered 4 test functions

🧪 Running Tensor Module Tests...
==================================================
✅ Tensor Arithmetic: PASSED
✅ Tensor Creation: PASSED
✅ Tensor Integration: PASSED
✅ Tensor Properties: PASSED
==================================================
🎉 All tests passed! (4/4)
✅ Tensor module is working correctly!
```

### ✅ **Safety Features**
- **Pattern Matching**: Only discovers functions matching expected patterns
- **Protected Framework**: NBGrader locked cells prevent student modifications
- **Fallback Support**: Manual registration still available if needed
- **Error Handling**: Graceful handling of malformed test functions

## 📝 Implementation Notes

### Test Function Requirements
1. **Naming Convention**: Must start with `test_` and contain expected patterns
2. **Self-Contained**: Each test should be independent
3. **Clear Output**: Print success messages for educational feedback
4. **Proper Assertions**: Use assert statements for validation

### Module Integration
1. **Single Entry Point**: Each module has one standardized testing entry
2. **Consistent Interface**: Same API across all modules
3. **CLI Integration**: `tito test module_name` uses the auto-discovery
4. **Development Workflow**: Students can run individual tests during development

### Educational Benefits
1. **Immediate Feedback**: Students see results as they develop
2. **Professional Practices**: Mirrors real software development workflows
3. **Consistent Experience**: Same testing approach across all modules
4. **Assessment Ready**: NBGrader can evaluate student implementations 