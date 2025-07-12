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
# Assignment 0: Setup - TinyTorch Development Environment

Welcome to TinyTorch! In this assignment, you'll set up your development environment and create your first utilities for the TinyTorch ML framework.

## Learning Objectives
- Set up and verify your TinyTorch development environment
- Create utility functions for the framework
- Learn the development workflow: implement → export → test → use
- Get familiar with the TinyTorch CLI tools

## Assignment Overview
You'll implement 4 core utilities that will be used throughout the TinyTorch framework:
1. A welcome function with ASCII art loading
2. A simple math utility function
3. A system information collector
4. A developer profile manager

Let's get started!
"""

# %%
#| default_exp core.utils

# %%
#| export
# Required imports for TinyTorch utilities
import sys
import platform
from datetime import datetime
import os
from pathlib import Path

# %% [markdown]
"""
## Problem 1: Hello Function (5 points)

Create a function that displays a welcome message for TinyTorch. This function should try to load ASCII art from a file, but gracefully fall back to a simple banner if the file doesn't exist.

**Requirements:**
- Try to read 'tinytorch_flame.txt' from the current directory
- If the file exists, print its contents
- If the file doesn't exist, print a simple "TinyTorch" banner
- Always print "Build ML Systems from Scratch!" after the banner
- Handle any file reading errors gracefully

**Example Output:**
```
TinyTorch
Build ML Systems from Scratch!
```
"""

# %% nbgrader={"grade": false, "grade_id": "hello_function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def hello_tinytorch():
    """
    Display a welcome message for TinyTorch.
    
    This function should:
    1. Try to load ASCII art from 'tinytorch_flame.txt' if it exists
    2. If the file doesn't exist, display a simple text banner
    3. Print "TinyTorch" and "Build ML Systems from Scratch!"
    4. Handle any exceptions gracefully
    """
    ### BEGIN SOLUTION
    try:
        # Try to read the ASCII art file
        flame_file = Path('tinytorch_flame.txt')
        if flame_file.exists():
            print(flame_file.read_text().strip())
        else:
            print("TinyTorch")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        # If file doesn't exist or can't be read, show simple banner
        print("TinyTorch")
    
    # Always print the tagline
    print("Build ML Systems from Scratch!")
    ### END SOLUTION

# %% [markdown]
"""
## Problem 2: Multi-Step Math Function (10 points)

Create a function that demonstrates multiple solution blocks within one function. This shows how NBGrader can guide you through step-by-step implementation.

**Requirements:**
- Step 1: Add 2 to each input variable
- Step 2: Sum the modified variables  
- Step 3: Multiply the result by 10
- Return the final result

**Example:**
```python
complex_calculation(3, 4)  # Step 1: 5, 6  Step 2: 11  Step 3: 110
```

**Note:** This function demonstrates how you can have multiple solution blocks within a single function for guided learning!
"""

# %% nbgrader={"grade": false, "grade_id": "multi_step_function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def complex_calculation(a, b):
    """
    Perform a multi-step calculation with guided implementation.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result of multi-step calculation
    """
    # Step 1: Add 2 to each input variable
    # a_plus_2 = ...
    ### BEGIN SOLUTION
    a_plus_2 = a + 2
    b_plus_2 = b + 2
    ### END SOLUTION
    
    # Step 2: Sum everything
    # everything_summed = ...
    ### BEGIN SOLUTION
    everything_summed = a_plus_2 + b_plus_2
    ### END SOLUTION
    
    # Step 3: Multiply your previous result by 10
    # Hint: you can use np.multiply if you want people to hate you
    # everything_summed_times_10 = ...
    ### BEGIN SOLUTION
    everything_summed_times_10 = everything_summed * 10
    ### END SOLUTION
    
    return everything_summed_times_10

# %% [markdown]
"""
## Problem 3: Basic Math Function (5 points)

Create a simple addition function. This might seem trivial, but it's important to verify our basic development workflow is working correctly.

**Requirements:**
- Accept two parameters (a and b)
- Return their sum
- Handle both integers and floats

**Example:**
```python
add_numbers(3, 4)      # Returns 7
add_numbers(2.5, 1.5)  # Returns 4.0
```
"""

# %% nbgrader={"grade": false, "grade_id": "add_function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a: First number (int or float)
        b: Second number (int or float)
        
    Returns:
        Sum of a and b
    """
    ### BEGIN SOLUTION
    return a + b
    ### END SOLUTION

# %% [markdown]
"""
## Problem 4: System Information Class (20 points)

Create a class that collects and displays system information. This will help us understand the environment where TinyTorch is running.

**Requirements:**
- Collect Python version, platform, and machine architecture in `__init__`
- Implement `__str__` to return formatted system info
- Implement `is_compatible()` to check if Python version >= 3.8
- Store information as instance variables

**Example Output:**
```
Python 3.9.7 on Darwin (arm64)
```
"""

# %% nbgrader={"grade": false, "grade_id": "systeminfo_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SystemInfo:
    """
    A class for collecting and displaying system information.
    """
    
    def __init__(self):
        """
        Initialize the SystemInfo object.
        Collect Python version, platform, and machine information.
        """
        ### BEGIN SOLUTION
        # Get Python version info
        version_info = sys.version_info
        self.python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        # Get platform information
        self.platform = platform.system()
        
        # Get machine architecture
        self.machine = platform.machine()
        ### END SOLUTION
    
    def __str__(self):
        """
        Return a formatted string representation of system information.
        Format: "Python X.Y.Z on Platform (Architecture)"
        """
        ### BEGIN SOLUTION
        return f"Python {self.python_version} on {self.platform} ({self.machine})"
        ### END SOLUTION
    
    def is_compatible(self):
        """
        Check if the Python version is compatible (>= 3.8).
        Returns True if compatible, False otherwise.
        """
        ### BEGIN SOLUTION
        return sys.version_info[:2] >= (3, 8)
        ### END SOLUTION

# %% [markdown]
"""
## Problem 5: Developer Profile Class (30 points)

Create a class to manage developer profiles. This will be used to track who's working on different parts of the TinyTorch framework.

**Requirements:**
- Store developer information (name, email, affiliation, specialization)
- Implement `__str__` for basic representation
- Implement `get_signature()` for formatted signature
- Implement `get_profile_info()` to return all info as a dictionary

**Example:**
```python
dev = DeveloperProfile("Alice", "alice@example.com", "University", "Neural Networks")
print(dev)  # "Alice (alice@example.com)"
print(dev.get_signature())  # Formatted signature with all info
```
"""

# %% nbgrader={"grade": false, "grade_id": "developer_profile_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class DeveloperProfile:
    """
    A class representing a developer profile.
    """
    
    def __init__(self, name="Student", email="student@example.com", affiliation="TinyTorch Community", specialization="ML Systems"):
        """
        Initialize a developer profile.
        
        Args:
            name: Developer's name
            email: Developer's email
            affiliation: Developer's affiliation or organization
            specialization: Developer's area of specialization
        """
        ### BEGIN SOLUTION
        self.name = name
        self.email = email
        self.affiliation = affiliation
        self.specialization = specialization
        ### END SOLUTION
    
    def __str__(self):
        """
        Return a basic string representation of the developer.
        Format: "Name (email)"
        """
        ### BEGIN SOLUTION
        return f"{self.name} ({self.email})"
        ### END SOLUTION
    
    def get_signature(self):
        """
        Return a formatted signature for the developer.
        Should include name, affiliation, and specialization.
        """
        ### BEGIN SOLUTION
        return f"{self.name}\n{self.affiliation}\nSpecialization: {self.specialization}"
        ### END SOLUTION
    
    def get_profile_info(self):
        """
        Return comprehensive profile information as a dictionary.
        """
        ### BEGIN SOLUTION
        return {
            'name': self.name,
            'email': self.email,
            'affiliation': self.affiliation,
            'specialization': self.specialization
        }
        ### END SOLUTION

# %% [markdown]
"""
## Testing Your Implementation

Once you've implemented all the functions above, run the cells below to test your work!

Remember the TinyTorch workflow:
1. **Implement** the functions above
2. **Export** to package: `tito module export 00_setup`
3. **Test** your work: `pytest tests/ -v`
4. **Use** your code: `from tinytorch.core.utils import hello_tinytorch`
"""

# %% [markdown]
"""
## Problem 6: Integration Test (30 points)

Test that all your components work together correctly. This ensures your implementation is complete and ready for export to the TinyTorch package.

**Requirements:**
- Test all functions and classes work correctly
- Test the multi-step function with multiple solution blocks
- Verify system compatibility 
- Display a complete developer profile
- Show successful framework initialization

**Total Points: 95/95**
"""

# %% nbgrader={"grade": true, "grade_id": "integration_test", "locked": false, "points": 30, "schema_version": 3, "solution": true, "task": false}
def test_integration():
    """
    Integration test to verify all components work together.
    This function tests the complete TinyTorch setup workflow.
    """
    ### BEGIN SOLUTION
    # Test 1: Welcome function
    print("🧪 Testing hello_tinytorch()...")
    hello_tinytorch()
    print("✅ Welcome function works!\n")
    
    # Test 2: Multi-step calculation (demonstrates multiple solution blocks)
    print("🧪 Testing complex_calculation() with multiple solution blocks...")
    result = complex_calculation(3, 4)
    expected = 110  # (3+2) + (4+2) = 11, 11*10 = 110
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✅ Multi-step calculation: complex_calculation(3, 4) = {result}")
    print("✅ Multiple solution blocks working correctly!\n")
    
    # Test 3: Simple math function
    print("🧪 Testing add_numbers()...")
    result = add_numbers(2.5, 1.5)
    assert result == 4.0, f"Expected 4.0, got {result}"
    print(f"✅ Math function: add_numbers(2.5, 1.5) = {result}\n")
    
    # Test 4: System information
    print("🧪 Testing SystemInfo class...")
    sys_info = SystemInfo()
    print(f"✅ System info: {sys_info}")
    print(f"✅ Python compatible: {sys_info.is_compatible()}\n")
    
    # Test 5: Developer profile
    print("🧪 Testing DeveloperProfile class...")
    dev = DeveloperProfile("TinyTorch Student", "student@tinytorch.edu", "TinyTorch University", "ML Systems")
    print(f"✅ Developer: {dev}")
    print(f"✅ Profile info: {dev.get_profile_info()}\n")
    
    # Test 6: Complete workflow
    print("🎉 All components working together!")
    print("✅ Ready for module export and package building!")
    return True
    ### END SOLUTION

# %% [markdown]
"""
## Next Steps

After completing this assignment:
1. Export your code to the TinyTorch package
2. Run the tests to verify everything works
3. Try importing and using your functions
4. Move on to the next assignment!

You've just created the foundation utilities for TinyTorch. Great job! 🎉
""" 