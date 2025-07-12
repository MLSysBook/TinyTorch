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
- ✅ Learn to query system information using Python
- ✅ Learn the NBGrader workflow: implement → test → export
- ✅ Create functions that will be part of your tinytorch package
- ✅ Understand solution blocks and hidden tests

## 📋 Assignment Overview
You'll implement **2 functions** that configure your TinyTorch installation:

| Problem | Points | Description |
|---------|--------|-------------|
| Personal Info Function | 50 | Return your personal configuration |
| System Info Function | 50 | Query and return system information |
| **Total** | **100** | **Complete Setup** |

## 🚀 The Goal
After completing this assignment and running `tito module export 00_setup`, you'll be able to:

```python
from tinytorch.core.setup import personal_info, system_info
print(personal_info())  # Your personal details
print(system_info())    # System information
```

And see your personalized TinyTorch configuration!

---

Let's configure your TinyTorch installation! 🔥
"""

# %%
#| default_exp core.setup

# %%
#| export
# Required imports for system information
import sys
import platform
import psutil
import os

# %% [markdown]
"""
## Problem 1: Personal Info Function (50 points) 👤

**Goal**: Create a function that returns your personal TinyTorch configuration.

### 📝 Requirements
- Function name: `personal_info()`
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
After export, you'll be able to call `tinytorch.personal_info()` and see your personalized configuration!
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 50 points if:
# - Function returns a dictionary
# - All required keys are present: developer, email, institution, system_name, version
# - All values are non-empty strings
# - Information appears to be real (not placeholder text)
# - Dictionary structure is correct
# 
# Deduct 10 points per missing/invalid field.
# === END MARK SCHEME ===

#| export
def personal_info():
    """
    Return personal information for this TinyTorch installation.
    
    Returns:
        dict: Personal configuration with developer info, email, institution, 
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
# Test personal_info function
info = personal_info()

# Test return type
assert isinstance(info, dict), "personal_info should return a dictionary"

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

print("✅ Personal info function tests passed!")
print(f"✅ TinyTorch configured for: {info['developer']}")
print(f"✅ System: {info['system_name']}")
### END HIDDEN TESTS

# %% [markdown]
"""
## Problem 2: System Info Function (50 points) 🖥️

**Goal**: Create a function that queries and returns useful system information.

### 📝 Requirements
- Function name: `system_info()`
- Query system information using Python modules
- Return a dictionary with system details
- Include: python_version, platform, memory_gb, cpu_count, architecture

### 💡 System Information to Query
You'll need to use these Python modules to get system information:
- `sys.version_info` - Get Python version
- `platform.system()` - Get operating system (Windows, Darwin, Linux)
- `platform.machine()` - Get CPU architecture (x86_64, arm64, etc.)
- `psutil.cpu_count()` - Get number of CPU cores
- `psutil.virtual_memory().total` - Get total RAM in bytes (convert to GB)

### 💡 Example Output
```python
{
    'python_version': '3.9.7',
    'platform': 'Darwin',
    'architecture': 'arm64',
    'cpu_count': 8,
    'memory_gb': 16.0
}
```

### 🚨 Important
- Use the imported modules to query real system information
- Convert memory from bytes to GB (divide by 1024^3)
- Handle any potential errors gracefully
- Make sure all values are the correct data types

### 🎯 Learning System Queries
This teaches you how to gather system information that's useful for ML development!
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 50 points if:
# - Function returns a dictionary with system information
# - All required keys are present: python_version, platform, architecture, cpu_count, memory_gb
# - Values are queried from actual system (not hardcoded)
# - Memory is converted to GB correctly
# - Data types are correct (strings for text, int for cpu_count, float for memory_gb)
# 
# Deduct 10 points per missing/incorrect field.
# === END MARK SCHEME ===

#| export
def system_info():
    """
    Query and return system information for this TinyTorch installation.
    
    Returns:
        dict: System information including Python version, platform, 
              architecture, CPU count, and memory
    """
    ### BEGIN SOLUTION
    # Get Python version
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    # Get platform information
    platform_name = platform.system()
    architecture = platform.machine()
    
    # Get CPU information
    cpu_count = psutil.cpu_count()
    
    # Get memory information (convert bytes to GB)
    memory_bytes = psutil.virtual_memory().total
    memory_gb = round(memory_bytes / (1024**3), 1)
    
    return {
        'python_version': python_version,
        'platform': platform_name,
        'architecture': architecture,
        'cpu_count': cpu_count,
        'memory_gb': memory_gb
    }
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
# Test system_info function
sys_info = system_info()

# Test return type
assert isinstance(sys_info, dict), "system_info should return a dictionary"

# Test required keys
required_keys = ['python_version', 'platform', 'architecture', 'cpu_count', 'memory_gb']
for key in required_keys:
    assert key in sys_info, f"Dictionary should have '{key}' key"

# Test data types
assert isinstance(sys_info['python_version'], str), "python_version should be string"
assert isinstance(sys_info['platform'], str), "platform should be string"
assert isinstance(sys_info['architecture'], str), "architecture should be string"
assert isinstance(sys_info['cpu_count'], int), "cpu_count should be integer"
assert isinstance(sys_info['memory_gb'], (int, float)), "memory_gb should be number"

# Test reasonable values
assert sys_info['cpu_count'] > 0, "CPU count should be positive"
assert sys_info['memory_gb'] > 0, "Memory should be positive"
assert len(sys_info['python_version']) > 0, "Python version should not be empty"

# Test that values are actually queried (not hardcoded)
# These should match the actual system
actual_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
assert sys_info['python_version'] == actual_version, "Python version should match actual system"

print("✅ System info function tests passed!")
print(f"✅ Python: {sys_info['python_version']} on {sys_info['platform']}")
print(f"✅ Hardware: {sys_info['cpu_count']} cores, {sys_info['memory_gb']} GB RAM")
### END HIDDEN TESTS

# %% [markdown]
"""
## 🎉 Setup Complete!

Congratulations! You've successfully configured your personal TinyTorch installation.

### What You've Accomplished:
✅ **Created personal_info function** - Your personal TinyTorch configuration  
✅ **Created system_info function** - System information queries  
✅ **Learned NBGrader workflow** - Solution blocks and hidden tests  
✅ **Learned system queries** - How to gather useful system information  
✅ **Configured your personal installation** - Your TinyTorch is now personalized  

### Next Steps:
1. **Export your code**: `tito module export 00_setup`
2. **Test your installation**: 
   ```python
   from tinytorch.core.setup import personal_info, system_info
   print(personal_info())  # Your personal details
   print(system_info())    # System information
   ```
3. **Move to Assignment 1**: Start building your first tensors!

### Key Takeaways:
- **NBGrader workflow** - Write solutions between BEGIN/END SOLUTION markers
- **Hidden tests** - Tests that verify your implementation automatically
- **System queries** - How to gather useful system information for ML development
- **Code export** - Your functions become part of the tinytorch package
- **Personalization** - Your TinyTorch installation is now uniquely yours

**Welcome to TinyTorch - let's build ML systems from scratch! 🚀**
""" 