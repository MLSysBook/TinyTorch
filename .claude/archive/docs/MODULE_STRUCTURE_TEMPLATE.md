# TinyTorch Module Structure Template

This template documents the exact structure every TinyTorch module should follow to provide students with a consistent, predictable learning experience.

## Module Structure Overview

Every TinyTorch module follows the **Build → Use → Understand** pedagogical framework with consistent naming patterns, test placement, and educational flow.

## 1. File Header Structure

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---
```

## 2. Module Introduction Pattern

```python
# %% [markdown]
"""
# [Module Title] - [Descriptive Subtitle]

Welcome to the [Module Name] module! [Brief description of what students will accomplish]

## Learning Goals
- [Goal 1: Understanding concept]
- [Goal 2: Implementation skill] 
- [Goal 3: Integration capability]
- [Goal 4: Real-world connection]
- [Goal 5: Testing methodology]

## Build → Use → Understand
1. **Build**: [What students will implement]
2. **Use**: [How they'll use it immediately]
3. **Understand**: [Deeper conceptual insight gained]
"""
```

## 3. Setup and Imports Cell

```python
# %% nbgrader={"grade": false, "grade_id": "[module]-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.[module_name]

#| export
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import [relevant types]

# Import dependencies with fallback pattern
try:
    from tinytorch.core.[dependency] import [Class]
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '[dependency_module]'))
    from [dependency]_dev import [Class]

# %% nbgrader={"grade": false, "grade_id": "[module]-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch [Module Title] Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build [module functionality]!")
```

## 4. Package Context Section

```python
# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/XX_[module]/[module]_dev.py`  
**Building Side:** Code exports to `tinytorch.core.[module]`

```python
# Final package structure:
from tinytorch.core.[module] import [MainClass]  # What you're building!
from tinytorch.core.[dependency] import [DepClass]  # Prerequisites
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's structure
- **Consistency:** All [module] operations live together
- **Integration:** Works seamlessly with other TinyTorch components
"""
```

## 5. Development Section Header

```python
# %% [markdown]
"""
## 🔧 DEVELOPMENT
"""
```

## 6. Implementation Pattern (Repeated for Each Concept)

### Step Introduction
```python
# %% [markdown]
"""
## Step [N]: [Concept Name] - [Brief Description]

### What is [Concept]?
[Clear definition with mathematical formulation]

### Why [Concept] Matters
1. **[Reason 1]**: [Explanation]
2. **[Reason 2]**: [Explanation]
3. **[Reason 3]**: [Explanation]
4. **[Reason 4]**: [Explanation]

### Visual Understanding
```
[Simple example with inputs/outputs]
```

### Real-World Applications
- **[Domain 1]**: [Examples]
- **[Domain 2]**: [Examples]
- **[Domain 3]**: [Examples]

### Mathematical Properties
- **[Property 1]**: [Description]
- **[Property 2]**: [Description]
- **[Property 3]**: [Description]
"""
```

### Implementation Cell
```python
# %% nbgrader={"grade": false, "grade_id": "[concept]-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class [ConceptClass]:
    """
    [Brief description of the class]
    
    [Longer description explaining purpose and usage]
    """
    
    def [method_name](self, [parameters]):
        """
        [Method description]
        
        TODO: Implement [method functionality].
        
        STEP-BY-STEP IMPLEMENTATION:
        1. [Step 1 with clear action]
        2. [Step 2 with clear action]
        3. [Step 3 with clear action]
        4. [Step 4 with clear action]
        
        EXAMPLE USAGE:
        ```python
        [clear example showing usage]
        ```
        
        IMPLEMENTATION HINTS:
        - [Hint 1 with specific function/approach]
        - [Hint 2 with specific function/approach]
        - [Hint 3 with specific function/approach]
        - [Hint 4 with specific function/approach]
        
        LEARNING CONNECTIONS:
        - This is like [PyTorch equivalent]
        - Used in [real applications]
        - [Connection to broader concepts]
        """
        ### BEGIN SOLUTION
        [Implementation code]
        ### END SOLUTION
```

### Immediate Unit Test
```python
# %% [markdown]
"""
### 🧪 Test Your [Concept] Implementation

Once you implement the [concept] method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-[concept]-immediate", "locked": true, "points": [N], "schema_version": 3, "solution": false, "task": false}
def test_unit_[concept_name]():
    """Unit test for the [Concept] [functionality]."""
    print("🔬 Unit Test: [Concept Description]...")

    # Test case 1: [Basic functionality]
    [test code]
    assert [condition], f"[Error message with expected vs actual]"
    
    # Test case 2: [Edge case or different scenario]
    [test code]
    assert [condition], f"[Error message]"
    
    # Test case 3: [Integration or advanced scenario]
    [test code]
    assert [condition], f"[Error message]"
    
    print("✅ [Concept] tests passed!")
    print(f"✅ [Specific behavior verified]")
    print(f"✅ [Another behavior verified]")
    print(f"✅ [Third behavior verified]")

# Run the test
test_unit_[concept_name]()
```

## 7. Comprehensive Testing Section

### Unit Tests for Each Function
```python
# %% [markdown]
"""
### 🧪 Unit Test: [Function Name]

This test validates your `[function_name]` implementation, ensuring it correctly [specific functionality description].
"""

# %%
def test_unit_[function_name]():
    """Comprehensive test of [function] with [specific scenarios]."""
    print("🔬 Testing comprehensive [function] functionality...")
    
    [comprehensive test implementation]
    
    print("✅ [Function] tests passed!")

# Run the test
test_unit_[function_name]()
```

### Integration Tests
```python
# %%
def test_module_[module]_[integration_type]_integration():
    """
    Integration test for [module] with [other components].
    
    Tests that [module] properly integrates with [other systems]
    and maintains compatibility for [use cases].
    """
    print("🔬 Running Integration Test: [Module]-[Component] Integration...")
    
    # Test 1: [Integration scenario 1]
    [test code]
    assert [condition], "[Error description]"
    
    # Test 2: [Integration scenario 2] 
    [test code]
    assert [condition], "[Error description]"
    
    # Test 3: [Integration scenario 3]
    [test code]
    assert [condition], "[Error description]"
    
    print("✅ Integration Test Passed: [Module] integration works correctly.")

# Run the integration test
test_module_[module]_[integration_type]_integration()
```

## 8. Module Summary

```python
# %% [markdown]
"""
## 🎯 MODULE SUMMARY: [Module Title]

Congratulations! You've successfully implemented [main accomplishment]:

### ✅ What You've Built
- **[Component 1]**: [Description and importance]
- **[Component 2]**: [Description and importance]
- **[Component 3]**: [Description and importance]
- **[Component 4]**: [Description and importance]

### ✅ Key Learning Outcomes
- **Understanding**: [Conceptual knowledge gained]
- **Implementation**: [Technical skills developed]
- **Testing**: [Testing methodology learned]
- **Integration**: [How components work together]
- **Real-world context**: [Connection to production systems]

### ✅ Mathematical Foundations Mastered
- **[Concept 1]**: [Mathematical formula/principle]
- **[Concept 2]**: [Mathematical formula/principle]
- **[Concept 3]**: [Mathematical formula/principle]

### ✅ Professional Skills Developed
- **[Skill 1]**: [Description of professional capability]
- **[Skill 2]**: [Description of professional capability]
- **[Skill 3]**: [Description of professional capability]

### ✅ Ready for Advanced Applications
Your [module] implementation now enables:
- **[Application 1]**: [What's now possible]
- **[Application 2]**: [What's now possible]
- **[Application 3]**: [What's now possible]

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch**: [PyTorch equivalent and similarity]
- **TensorFlow**: [TensorFlow equivalent and similarity]
- **Industry Standard**: [How this applies in real systems]

### 🎯 The Power of [Module Concept]
You've [major accomplishment description]:
- **[Capability 1]**: [Description]
- **[Capability 2]**: [Description]
- **[Capability 3]**: [Description]

### 🚀 What's Next
Your [module] implementation is the foundation for:
- **[Next Module]**: [Connection to next learning]
- **[Future Capability]**: [Advanced applications]

**Next Module**: [Next module name] - [What students will learn next]!

[Motivational closing statement about their progress!]
"""
```

## Testing Naming Conventions

### Unit Tests
- **Pattern**: `test_unit_[function_name]()`
- **Purpose**: Test individual functions/methods in isolation
- **Placement**: Immediately after implementation
- **Points**: 5-15 points based on complexity

### Integration Tests  
- **Pattern**: `test_module_[module]_[component]_integration()`
- **Purpose**: Test how module components work together
- **Placement**: After all unit tests
- **Points**: 15-25 points based on complexity

### Comprehensive Tests
- **Pattern**: `test_unit_[module]_comprehensive()` 
- **Purpose**: Test multiple functions working together
- **Placement**: Near end of module
- **Points**: 15-20 points

## NBGrader Cell Configuration

### Implementation Cells
```python
# %% nbgrader={"grade": false, "grade_id": "[unique-id]", "locked": false, "schema_version": 3, "solution": true, "task": false}
```

### Test Cells
```python  
# %% nbgrader={"grade": true, "grade_id": "test-[concept]-immediate", "locked": true, "points": [N], "schema_version": 3, "solution": false, "task": false}
```

### Documentation Cells
```python
# %% nbgrader={"grade": false, "grade_id": "[concept]-[type]", "locked": false, "schema_version": 3, "solution": false, "task": false}
```

## Educational Principles

1. **Immediate Feedback**: Test each concept immediately after implementation
2. **Progressive Complexity**: Start simple, build systematically  
3. **Clear Scaffolding**: TODO/APPROACH/EXAMPLE/HINTS pattern
4. **Real Connections**: Link to PyTorch/TensorFlow/industry
5. **Comprehensive Validation**: Multiple test types ensure understanding
6. **Professional Patterns**: Use real ML development practices

## Module Integration Requirements

1. **Import Pattern**: Try package first, fallback to local
2. **Export Pattern**: Use `#| export` for package code
3. **Dependency Chain**: Verify prerequisites in module.yaml
4. **Type Consistency**: Return same types as inputs when appropriate
5. **Error Handling**: Clear, educational error messages

This template ensures every TinyTorch module provides students with a consistent, high-quality learning experience that builds professional ML development skills.