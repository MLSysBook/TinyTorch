# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
"""
# Assignment 0: Setup - TinyTorch System Configuration

Welcome to TinyTorch! This assignment configures your personal TinyTorch installation and teaches you the NBGrader workflow.

## 🎯 Learning Objectives
By the end of this assignment, you will:
- ✅ Configure your personal TinyTorch installation
- ✅ Learn the NBGrader workflow: implement → test → export
- ✅ Create a function that will be part of your tinytorch package
- ✅ Understand solution blocks and hidden tests

## 📋 Assignment Overview
You'll implement **1 simple function** that configures your TinyTorch installation:

| Problem | Points | Description |
|---------|--------|-------------|
| System Info Function | 100 | Return your personal TinyTorch configuration |
| **Total** | **100** | **Complete Setup** |

## 🚀 The Goal
After completing this assignment and running `tito module export 00_setup`, you'll be able to:

```python
from tinytorch.core.setup import system_info
print(system_info())
```

And see your personalized TinyTorch configuration!

---

Let's configure your TinyTorch installation! 🔥
"""

# %%
#| default_exp core.setup

# %% [markdown]
"""
## Problem 1: System Info Function (100 points) ⚙️

**Goal**: Create a function that returns your personal TinyTorch system configuration.

### 📝 Requirements
- Function name: `system_info()`
- Return a dictionary with your information
- Use your actual details (not the example)
- Include: developer, email, institution, system_name, version

### 💡 Example Output
```python
{
    'developer': 'Vijay Janapa Reddi',
    'email': 'vj@eecs.harvard.edu',
    'institution': 'Harvard University',
    'system_name': 'VJ-TinyTorch-Dev',
    'version': '1.0.0'
}
```

### 🚨 Important
- Replace the example information with **your actual details**
- Use your real name, email, and institution
- Create a unique system name for your installation
- Keep version as '1.0.0' for now

### 🎯 This Will Become Part of Your TinyTorch Package
After export, you'll be able to call `tinytorch.system_info()` and see your personalized configuration!
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 100 points if:
# - Function returns a dictionary
# - All required keys are present: developer, email, institution, system_name, version
# - All values are non-empty strings
# - Information appears to be real (not placeholder text)
# - Dictionary structure is correct
# 
# Deduct 20 points per missing/invalid field.
# === END MARK SCHEME ===

#| export
def system_info():
    """
    Return system information for this TinyTorch installation.
    
    Returns:
        dict: System configuration with developer info, email, institution, 
              system name, and version
    """
    ### BEGIN SOLUTION
    return {
        'developer': 'Vijay Janapa Reddi',
        'email': 'vj@eecs.harvard.edu',
        'institution': 'Harvard University',
        'system_name': 'VJ-TinyTorch-Dev',
        'version': '1.0.0'
    }
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
# Test system_info function
info = system_info()

# Test return type
assert isinstance(info, dict), "system_info should return a dictionary"

# Test required keys
required_keys = ['developer', 'email', 'institution', 'system_name', 'version']
for key in required_keys:
    assert key in info, f"Dictionary should have '{key}' key"

# Test non-empty values
for key, value in info.items():
    assert isinstance(value, str), f"Value for '{key}' should be a string"
    assert len(value) > 0, f"Value for '{key}' cannot be empty"

# Test email format
assert '@' in info['email'], "Email should contain @ symbol"
assert '.' in info['email'], "Email should contain domain"

# Test version format
assert info['version'] == '1.0.0', "Version should be '1.0.0'"

# Test system name (should be unique/personalized)
assert len(info['system_name']) > 5, "System name should be descriptive"

print("✅ System info function tests passed!")
print(f"✅ TinyTorch configured for: {info['developer']}")
print(f"✅ System: {info['system_name']}")
### END HIDDEN TESTS

# %% [markdown]
"""
## 🎉 Setup Complete!

Congratulations! You've successfully configured your personal TinyTorch installation.

### What You've Accomplished:
✅ **Created your system_info function** - This will be part of your tinytorch package  
✅ **Learned NBGrader workflow** - Solution blocks and hidden tests  
✅ **Configured your personal installation** - Your TinyTorch is now personalized  

### Next Steps:
1. **Export your code**: `tito module export 00_setup`
2. **Test your installation**: 
   ```python
   from tinytorch.core.setup import system_info
   print(system_info())
   ```
3. **Move to Assignment 1**: Start building your first tensors!

### Key Takeaways:
- **NBGrader workflow** - Write solutions between BEGIN/END SOLUTION markers
- **Hidden tests** - Tests that verify your implementation automatically
- **Code export** - Your functions become part of the tinytorch package
- **Personalization** - Your TinyTorch installation is now uniquely yours

**Welcome to TinyTorch - let's build ML systems from scratch! 🚀**
""" 