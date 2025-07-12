# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Module 0: Setup - Tiny🔥Torch Development Workflow

Welcome to TinyTorch! This module teaches you the development workflow you'll use throughout the course.

## Learning Goals
- Understand the nbdev notebook-to-Python workflow
- Write your first TinyTorch code
- Run tests and use the CLI tools
- Get comfortable with the development rhythm

## The TinyTorch Development Cycle

1. **Write code** in this notebook using `#| export` 
2. **Export code** with `python bin/tito.py sync --module setup`
3. **Run tests** with `python bin/tito.py test --module setup`
4. **Check progress** with `python bin/tito.py info`

Let's get started!
"""

# %%
#| default_exp core.utils

# Setup imports and environment
import sys
import platform
from datetime import datetime
import os
from pathlib import Path

print("🔥 TinyTorch Development Environment")
print(f"Python {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
"""
## Step 1: Understanding the Module → Package Structure

**🎓 Teaching vs. 🔧 Building**: This course has two sides:
- **Teaching side**: You work in `modules/setup/setup_dev.ipynb` (learning-focused)
- **Building side**: Your code exports to `tinytorch/core/utils.py` (production package)

**Key Concept**: The `#| default_exp core.utils` directive at the top tells nbdev to export all `#| export` cells to `tinytorch/core/utils.py`.

This separation allows us to:
- Organize learning by **concepts** (modules)  
- Organize code by **function** (package structure)
- Build a real ML framework while learning systematically

Let's write a simple "Hello World" function with the `#| export` directive:
"""

# %% nbgrader={"grade": false, "grade_id": "hello-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def hello_tinytorch():
    """
    A simple hello world function for TinyTorch.
    
    TODO: Implement this function to display TinyTorch ASCII art and welcome message.
    Load the flame art from tinytorch_flame.txt file with graceful fallback.
    
    HINTS:
    1. Try to load ASCII art from 'tinytorch_flame.txt' in current directory
    2. If file exists, read and print the content
    3. Add "Tiny🔥Torch" and "Build ML Systems from Scratch!" messages
    4. If file doesn't exist, just print the emoji version
    5. Handle any exceptions gracefully
    
    EXAMPLE OUTPUT:
    [ASCII art from file]
    Tiny🔥Torch
    Build ML Systems from Scratch!
    """
    # YOUR CODE HERE
    raise NotImplementedError()

# %% nbgrader={"grade": false, "grade_id": "add-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def add_numbers(a, b):
    """
    Add two numbers together.
    
    TODO: Implement addition of two numbers.
    This is the foundation of all mathematical operations in ML.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
        
    Returns:
        Sum of a and b
        
    EXAMPLE:
    add_numbers(2, 3) should return 5
    add_numbers(1.5, 2.5) should return 4.0
    """
    # YOUR CODE HERE
    raise NotImplementedError()

# %% [markdown]
"""
### 🧪 Test Your Implementation

Once you implement the functions above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-hello-function", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
# Test hello_tinytorch function
print("Testing hello_tinytorch():")
try:
    hello_tinytorch()
    print("✅ hello_tinytorch() executed successfully!")
except NotImplementedError:
    print("❌ hello_tinytorch() not implemented yet")
    raise

# %% nbgrader={"grade": true, "grade_id": "test-add-function", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Test add_numbers function
print("Testing add_numbers():")
assert add_numbers(2, 3) == 5, "add_numbers(2, 3) should return 5"
assert add_numbers(0, 0) == 0, "add_numbers(0, 0) should return 0"
assert add_numbers(-1, 1) == 0, "add_numbers(-1, 1) should return 0"
assert abs(add_numbers(1.5, 2.5) - 4.0) < 1e-10, "add_numbers(1.5, 2.5) should return 4.0"
print("✅ All addition tests passed!")

# %% [markdown]
"""
## Step 2: A Simple Class

Let's create a simple class that will help us understand system information. This is still basic, but shows how to structure classes in TinyTorch.
"""

# %% nbgrader={"grade": false, "grade_id": "systeminfo-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SystemInfo:
    """
    Simple system information class.
    
    TODO: Implement this class to collect and display system information.
    
    REQUIREMENTS:
    1. __init__: Collect Python version, platform, and machine information
    2. __str__: Return formatted system info string
    3. is_compatible: Check if Python version >= 3.8
    
    HINTS:
    - Use sys.version_info for Python version
    - Use platform.system() for platform name  
    - Use platform.machine() for machine architecture
    - Store these as instance attributes in __init__
    """
    
    def __init__(self):
        """
        Initialize system information collection.
        
        TODO: Collect Python version, platform, and machine information.
        Store as instance attributes: self.python_version, self.platform, self.machine
        """
        # YOUR CODE HERE
        raise NotImplementedError()
    
    def __str__(self):
        """
        Return human-readable system information.
        
        TODO: Format system info as a readable string.
        FORMAT: "Python X.Y on Platform (Architecture)"
        EXAMPLE: "Python 3.9 on Darwin (arm64)"
        """
        # YOUR CODE HERE
        raise NotImplementedError()
    
    def is_compatible(self):
        """
        Check if system meets minimum requirements.
        
        TODO: Check if Python version >= 3.8
        Return True if compatible, False otherwise
        """
        # YOUR CODE HERE
        raise NotImplementedError()

# %% [markdown]
"""
### 🧪 Test Your SystemInfo Class

Once you implement the SystemInfo class above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-systeminfo-creation", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Test SystemInfo creation
print("Testing SystemInfo creation...")
info = SystemInfo()
assert hasattr(info, 'python_version'), "SystemInfo should have python_version attribute"
assert hasattr(info, 'platform'), "SystemInfo should have platform attribute"
assert hasattr(info, 'machine'), "SystemInfo should have machine attribute"
print("✅ SystemInfo creation test passed!")

# %% nbgrader={"grade": true, "grade_id": "test-systeminfo-str", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Test SystemInfo string representation
print("Testing SystemInfo string representation...")
info = SystemInfo()
info_str = str(info)
assert isinstance(info_str, str), "SystemInfo.__str__() should return a string"
assert len(info_str) > 0, "SystemInfo string should not be empty"
assert 'Python' in info_str, "SystemInfo string should contain 'Python'"
print(f"✅ SystemInfo string: {info_str}")

# %% nbgrader={"grade": true, "grade_id": "test-systeminfo-compatibility", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Test SystemInfo compatibility check
print("Testing SystemInfo compatibility...")
info = SystemInfo()
compatible = info.is_compatible()
assert isinstance(compatible, bool), "is_compatible() should return a boolean"
# Since we're running this test, Python should be >= 3.8
assert compatible == True, "Current Python version should be compatible (>= 3.8)"
print("✅ SystemInfo compatibility test passed!")

# %% [markdown]
"""
## Step 3: Developer Profile (Optional Challenge)

For students who want an extra challenge, implement a DeveloperProfile class:
"""

# %% nbgrader={"grade": false, "grade_id": "developer-profile", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class DeveloperProfile:
    """
    Developer profile for personalizing TinyTorch experience.
    
    TODO: OPTIONAL CHALLENGE - Implement this class for extra credit!
    
    REQUIREMENTS:
    1. Store developer information (name, email, etc.)
    2. Load ASCII art from file with fallback
    3. Generate formatted profile display
    4. Create professional signature
    
    This is an advanced exercise - only attempt after completing the required parts!
    """
    
    def __init__(self, name="Student", email="student@example.com"):
        """
        Initialize developer profile.
        
        TODO: Store developer information with defaults.
        Feel free to customize with your own info!
        """
        # YOUR CODE HERE (OPTIONAL)
        self.name = name
        self.email = email
    
    def get_signature(self):
        """
        Get a short signature for code headers.
        
        TODO: Return a signature like "Built by Name (email)"
        """
        # YOUR CODE HERE (OPTIONAL)
        return f"Built by {self.name} ({self.email})"

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've completed the TinyTorch setup module:

### What You've Accomplished
✅ **Environment Setup**: Learned the development workflow  
✅ **First Function**: Implemented hello_tinytorch() with file handling  
✅ **Math Operations**: Built add_numbers() for ML foundations  
✅ **Object-Oriented Programming**: Created SystemInfo class with properties  
✅ **Testing**: Verified your implementations with automated tests  
✅ **Package Export**: Used nbdev to build the tinytorch package  

### Key Concepts You've Learned
- **nbdev workflow**: From notebook to production package
- **File handling**: Reading ASCII art with graceful fallbacks
- **System information**: Collecting platform and version data
- **Object-oriented design**: Classes, properties, and methods
- **Error handling**: Using try/except and fallback strategies

### Next Steps
1. **Export your code**: Run `python bin/tito.py sync --module setup`
2. **Run tests**: Use `python bin/tito.py test --module setup`
3. **Check your work**: Import your functions with `from tinytorch.core.utils import hello_tinytorch`

**Ready for the next challenge?** Let's move on to building tensors!
"""
