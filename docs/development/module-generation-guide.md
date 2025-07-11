# TinyTorch Module Generation Guide

## 🎯 Overview

This guide explains how to create and generate student/instructor versions of TinyTorch modules using NBDev's educational features.

## 📝 Module Structure Pattern

Each module should follow this pattern:

### Student Version (Visible)
```python
#| export
def my_function():
    """
    Function description with learning goals.
    
    TODO: Clear implementation instructions
    - Bullet point guidance
    - Hints about approach
    """
    raise NotImplementedError("Student implementation required")
```

### Instructor Solution (Hidden)
```python
#| hide
#| export  
def my_function():
    """Function description."""
    # Complete working implementation
    return actual_solution
```

## 🔧 Generation Commands

### Generate Student Notebook
```bash
# Convert Python to notebook (students work in notebooks)
cd modules/{module_name}
jupytext --to notebook {module_name}_dev.py

# The notebook will show:
# - Student stubs with NotImplementedError
# - Hidden instructor solutions (expandable)
# - Clear TODO instructions
```

### Generate Package Code (Instructor Solutions)
```bash
# Export to package (gets complete implementations)
python bin/tito.py sync --module {module_name}

# This exports the #|hide instructor solutions to tinytorch package
# Package always has working code even if students haven't implemented
```

### Generate Documentation
```bash
# Generate docs with student view by default
python bin/tito.py docs

# Or use nbdev directly
nbdev_docs

# Students see exercise versions
# Instructors can click to reveal solutions
```

## ✅ Module Checklist

When creating a new module, ensure:

### Student Version Requirements
- [ ] All functions have `raise NotImplementedError("Student implementation required")`
- [ ] Clear TODO instructions with specific guidance
- [ ] Descriptive docstrings explaining the purpose
- [ ] Test cells that handle NotImplementedError gracefully

### Instructor Version Requirements  
- [ ] Complete working implementations in `#|hide` cells
- [ ] Same function signatures as student version
- [ ] All functions have `#|export` directive
- [ ] Code is production-ready and well-tested

### Module Structure
- [ ] `{module_name}_dev.py` - Source Python file
- [ ] `{module_name}_dev.ipynb` - Generated notebook for students
- [ ] Clear learning progression from simple to complex
- [ ] Proper `#| default_exp` directive at top

## 📋 Example Module Template

```python
# %% [markdown]
"""
# Module X: {Module Name} - {Purpose}

Learning goals and overview...
"""

# %%
#| default_exp {target_package_location}

# Imports and setup

# %% [markdown]
"""
## Step 1: {First Concept}
Explanation of what students will implement...
"""

# %%
#| export
def student_function():
    """
    Description of function purpose.
    
    TODO: Implementation instructions
    - Step 1: Do this
    - Step 2: Do that
    - Hint: Use this approach
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def student_function():
    """Complete instructor solution."""
    # Working implementation
    return result

# %% [markdown]
"""
### 🧪 Test Your Implementation
"""

# %%
try:
    result = student_function()
    print(f"Success: {result}")
except NotImplementedError as e:
    print(f"⚠️  {e}")
    print("Implement the function above first!")
```

## 🎯 Key Principles

1. **Single Source Truth**: One `.py` file contains both versions
2. **Student-First**: Student version should be clear and instructive
3. **Hidden Solutions**: Instructor code hidden but accessible
4. **Package Quality**: Exported code is production-ready
5. **Progressive Learning**: Build complexity gradually

## 🚀 Workflow Summary

1. **Create**: Write `{module}_dev.py` with student stubs and hidden solutions
2. **Convert**: `jupytext --to notebook {module}_dev.py`
3. **Export**: `python bin/tito.py sync --module {module}`
4. **Test**: `python bin/tito.py test --module {module}`
5. **Document**: `python bin/tito.py docs`

This approach ensures students get a challenging learning experience while the TinyTorch package remains fully functional! 