# 📖 TinyTorch Module Development Guide

A complete guide for creating educational modules that automatically generate student exercise versions.

## 🎯 Philosophy

**Write once, teach everywhere.** Create complete, working implementations with embedded pedagogical markers that automatically generate student exercise notebooks.

## 🏗️ Module Development Workflow

### Step 1: Plan the Learning Journey
Before coding, decide:
- **What should students implement?** (core learning objectives)
- **What should be provided?** (utilities, complex setup, advanced features)
- **What's the difficulty progression?** (easy → medium → hard)

### Step 2: Write Complete Implementation
Create `modules/{module}/{module}_dev.py` with:
```python
# %% [markdown]
# # Module Title: Learning Objectives
# 
# Brief description of what students will build and learn.

# %%
#| keep_imports
import numpy as np
from typing import Union, List

# %%
# Complete working implementation with markers...
```

### Step 3: Add Pedagogical Markers
Mark what students should implement vs. what to provide.

### Step 4: Convert and Generate
```bash
# Convert Python to notebook
python bin/tito.py notebooks --module {module}

# Generate student version
python3 bin/generate_student_notebooks.py --module {module}
```

## 🏷️ Marker System

### Exercise Markers
```python
#| exercise_start
#| difficulty: easy|medium|hard
#| hint: Helpful guidance without giving away the solution
#| hint: Multiple hints allowed for complex methods
#| solution_test: What students should verify after implementation
def method_to_implement(self, params):
    """Complete function signature and docstring."""
    # Full working implementation
    return result
#| exercise_end
```

### Preservation Markers
```python
#| keep_imports
import statements  # Preserved in student version

#| keep_complete  
def utility_method(self):
    """Keep entire implementation - too complex/not educational."""
    return complex_implementation()

#| remove_cell
# This entire cell removed from student version
instructor_only_code()
```

## 🎨 Difficulty System

Use visual indicators that students immediately understand:

- **🟢 Easy (5-10 min)**: Basic implementation, clear patterns
  - Constructor with simple type conversion
  - Property getters  
  - Basic arithmetic operations

- **🟡 Medium (10-20 min)**: Moderate complexity, some edge cases
  - Methods with conditional logic
  - Shape manipulation
  - Error handling

- **🔴 Hard (20+ min)**: Complex logic, multiple concepts
  - Broadcasting operations
  - Advanced algorithms
  - Integration between components

## 📋 What Students Should Implement vs. Receive

### ✅ **Students Implement** (Active Learning)
- **Core functionality**: Main learning objectives
- **Basic operations**: Addition, multiplication, etc.
- **Essential properties**: Shape, size, dtype
- **Error handling**: Type checking, validation
- **Simple algorithms**: Reshape, transpose, reductions

### 🎁 **Students Receive** (Focus on Learning Goals)
- **Complex setup code**: Initialization boilerplate
- **Utility functions**: String formatting, type checking helpers
- **Advanced features**: GPU support, optimization, broadcasting
- **Infrastructure**: Test frameworks, import statements
- **Edge case handling**: Complex error messages, corner cases

## 📝 Writing Guidelines

### Function Structure
Always provide complete signatures and docstrings:
```python
#| exercise_start
#| difficulty: medium
#| hint: Use numpy broadcasting for element-wise operations
#| solution_test: Result should be same shape as inputs
def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
    """
    Add this tensor with another tensor or scalar.
    
    Args:
        other: Another Tensor or scalar value
        
    Returns:
        Tensor: New tensor with element-wise addition result
        
    Raises:
        TypeError: If other is not a compatible type
    """
    if isinstance(other, Tensor):
        return Tensor(self._data + other._data)
    else:
        return Tensor(self._data + other)
#| exercise_end
```

### Hint Guidelines
- **Be helpful, not prescriptive**: Guide thinking, don't give code
- **Progressive disclosure**: Start general, get more specific
- **Reference relevant concepts**: "Use numpy broadcasting", "Check instance type"

Good hints:
```python
#| hint: Convert input to numpy array for consistent handling
#| hint: Use isinstance() to check if input is a Tensor or scalar
#| hint: Remember to return a new Tensor object, not numpy array
```

Bad hints:
```python
#| hint: Write: return Tensor(self._data + other._data)  # Too specific!
#| hint: Add the data  # Too vague!
```

### Solution Tests
Help students verify their implementation:
```python
#| solution_test: Tensor([1,2]) + Tensor([3,4]) should equal Tensor([4,6])
#| solution_test: Tensor([1,2]) + 5 should equal Tensor([6,7])
#| solution_test: Result should be a Tensor object, not numpy array
```

## 🗂️ Module Structure

```
modules/{module}/
├── {module}_dev.py          # 🔧 Complete implementation (instructor)
├── {module}_dev.ipynb       # 📓 Generated from .py file  
├── {module}_dev_student.ipynb # 🎓 Auto-generated exercise version
├── test_{module}.py         # 🧪 Comprehensive test suite
├── check_{module}.py        # ✅ Manual verification script
└── README.md               # 📖 Module overview and setup
```

## 🎯 Module Examples

### Easy Module (Tensor Basics)
Students implement:
- 🟢 Constructor (`__init__`)
- 🟢 Properties (`shape`, `size`, `dtype`)  
- 🟢 Basic arithmetic (`__add__`, `__sub__`)
- 🟡 Utilities (`reshape`, `transpose`)

### Medium Module (Autograd)
Students implement:
- 🟡 Forward pass computation
- 🟡 Gradient accumulation
- 🔴 Backward pass algorithm
- 🔴 Chain rule application

### Hard Module (Neural Networks)
Students implement:
- 🟡 Layer forward pass
- 🔴 Loss computation  
- 🔴 Backpropagation
- 🔴 Parameter updates

## 🛠️ Development Tools

### Convert Python to Notebook
```bash
python bin/tito.py notebooks --module {module}
```

### Generate Student Version
```bash
python bin/generate_student_notebooks.py --module {module}
```

### Test Complete Workflow
```bash
# Test instructor version
cd modules/{module}
jupyter lab {module}_dev.ipynb

# Test student version  
jupyter lab {module}_dev_student.ipynb

# Verify exports work
python bin/tito.py sync --module {module}
python bin/tito.py test --module {module}
```

## ✅ Quality Checklist

Before releasing a module:

**Instructor Version:**
- [ ] Complete implementation works and passes all tests
- [ ] All methods have proper type hints and docstrings
- [ ] Markers are correctly placed and consistent
- [ ] Code follows TinyTorch style guidelines

**Student Version:**
- [ ] Generated notebook preserves function signatures
- [ ] Hints are helpful but not prescriptive
- [ ] Difficulty progression makes sense
- [ ] Tests provide clear verification guidance
- [ ] Students can run and get feedback immediately

**Integration:**
- [ ] Module exports correctly to tinytorch package
- [ ] Tests pass with student implementation stubs
- [ ] README explains module purpose and workflow
- [ ] Follows established module naming conventions

## 💡 Best Practices

### For Instructors
1. **Start with learning goals**: What should students understand after this module?
2. **Write complete first**: Get the full solution working before adding markers
3. **Think like a student**: What would be confusing? What needs guidance?
4. **Test the student path**: Try implementing following your own hints

### For Markers
1. **Be consistent**: Use same difficulty criteria across modules
2. **Progressive complexity**: Easy concepts early, hard concepts later  
3. **Meaningful hints**: Guide understanding, not just implementation
4. **Clear boundaries**: Be explicit about what students implement vs. receive

### For Generator
1. **Preserve signatures**: Students need to know the interface
2. **Keep docstrings**: Essential for understanding purpose
3. **Clean output**: Remove implementation but keep structure
4. **Test guidance**: Help students verify their work

## 🔄 Iteration and Improvement

After teaching with a module:
1. **Collect feedback**: What confused students? What was too easy/hard?
2. **Update markers**: Adjust difficulty, improve hints
3. **Refine scope**: Move methods between implement/provide categories
4. **Share learnings**: Update this guide with discoveries

Remember: The goal is **active learning through hands-on implementation**, not just demonstration! 