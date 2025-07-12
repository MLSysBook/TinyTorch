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

## 🎯 Learning Objectives
By the end of this assignment, you will:
- ✅ Set up and verify your TinyTorch development environment
- ✅ Create utility functions using proper Python practices
- ✅ Learn the development workflow: implement → export → test → use
- ✅ Get familiar with the TinyTorch CLI tools
- ✅ Understand NBGrader solution blocks and guided implementation
- ✅ Practice error handling and graceful fallbacks

## 📋 Assignment Overview
You'll implement **5 core utilities** that demonstrate different programming concepts:

| Problem | Points | Concept | Difficulty |
|---------|--------|---------|------------|
| 1. Hello Function | 5 | File I/O, Error Handling | ⭐ Easy |
| 2. Multi-Step Function | 10 | Guided Implementation | ⭐⭐ Medium |
| 3. Basic Math | 5 | Simple Functions | ⭐ Easy |
| 4. System Info Class | 20 | OOP, System APIs | ⭐⭐ Medium |
| 5. Developer Profile | 30 | Advanced OOP | ⭐⭐⭐ Hard |
| 6. Integration Test | 25 | Testing, Workflow | ⭐⭐ Medium |
| **Total** | **95** | **Complete Workflow** | |

## 🛠️ Development Workflow
1. **Implement** functions in this notebook
2. **Test** your work locally: `python -c "from setup_dev import function_name; function_name()"`
3. **Export** to package: `tito module export 00_setup`
4. **Verify** with tests: `pytest tests/ -v`
5. **Use** your code: `from tinytorch.core.utils import function_name`

## 💡 General Tips
- **Read error messages carefully** - they often tell you exactly what's wrong
- **Test incrementally** - don't wait until everything is done to test
- **Use print statements** for debugging - they're your friend!
- **Check the examples** - they show exactly what output is expected
- **Ask for help** - if you're stuck for >30 minutes, reach out!

---

Let's get started! 🚀
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
## Problem 1: Hello Function (5 points) 👋

**Goal**: Create a function that displays a welcome message for TinyTorch with graceful error handling.

### 📝 Requirements
- Try to read 'tinytorch_flame.txt' from the current directory
- If the file exists, print its contents (without extra newlines)
- If the file doesn't exist, print a simple "TinyTorch" banner
- Always print "Build ML Systems from Scratch!" after the banner
- Handle any file reading errors gracefully (no crashes!)

### 💡 Approach & Hints
1. **Use Path from pathlib** - it's more robust than raw file operations
2. **Try-except pattern** - wrap file operations to catch errors
3. **Check file existence** - use `Path.exists()` before reading
4. **Strip whitespace** - use `.strip()` to remove extra newlines from file content
5. **Multiple exception types** - catch `FileNotFoundError`, `OSError`, `UnicodeDecodeError`

### 🎯 Expected Behavior
```python
# Case 1: File exists and readable
hello_tinytorch()
# Prints: [ASCII art from file]
#         Build ML Systems from Scratch!

# Case 2: File missing or unreadable  
hello_tinytorch()
# Prints: TinyTorch
#         Build ML Systems from Scratch!
```

### 🚨 Common Pitfalls
- ❌ Not handling exceptions (your function crashes)
- ❌ Forgetting to print the tagline
- ❌ Not stripping whitespace (extra blank lines)
- ❌ Using hardcoded file paths instead of Path

### 🧪 Quick Test
```python
# Test your function
hello_tinytorch()  # Should print welcome message without crashing
```
"""

# %% 
# nbgrader: grade, solution
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

# %% 
# nbgrader: tests
# Test hello_tinytorch function
def test_hello_tinytorch():
    """Test that hello_tinytorch runs without crashing."""
    import io
    import sys
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        hello_tinytorch()
        output = captured_output.getvalue()
        
        # Should have some output
        assert len(output.strip()) > 0, "Function should produce output"
        
        # Should contain the tagline
        assert "Build ML Systems from Scratch!" in output, "Should contain tagline"
        
        # Should contain either ASCII art or simple banner
        assert "TinyTorch" in output or len(output.split('\n')) > 2, "Should contain banner"
        
    finally:
        sys.stdout = old_stdout

test_hello_tinytorch()  # Run the test

# %% [markdown]
"""
## Problem 2: Multi-Step Math Function (10 points) 🔢

**Goal**: Create a function with **multiple solution blocks** to demonstrate guided implementation.

### 📝 Requirements
- **Step 1**: Add 2 to each input variable
- **Step 2**: Sum the modified variables  
- **Step 3**: Multiply the result by 10
- Return the final result

### 💡 Approach & Hints
This problem demonstrates **multiple solution blocks** within one function:
- Each step has its own solution block
- Complete each step before moving to the next
- Use descriptive variable names as suggested in comments
- Follow the math carefully: `(a+2) + (b+2) = a+b+4`, then `×10`

### 🎯 Step-by-Step Walkthrough
```python
# Example: complex_calculation(3, 4)
# Step 1: a_plus_2 = 3+2 = 5, b_plus_2 = 4+2 = 6
# Step 2: everything_summed = 5+6 = 11  
# Step 3: everything_summed_times_10 = 11*10 = 110
# Return: 110
```

### 🧮 More Examples
```python
complex_calculation(1, 2)  # (1+2)+(2+2) = 7, 7*10 = 70
complex_calculation(0, 0)  # (0+2)+(0+2) = 4, 4*10 = 40
complex_calculation(-1, 1) # (-1+2)+(1+2) = 4, 4*10 = 40
```

### 🚨 Common Pitfalls
- ❌ Using wrong variable names (doesn't match the comments)
- ❌ Skipping intermediate variables (storing directly in final result)
- ❌ Math errors (forgetting to add 2 to both variables)
- ❌ Not following the exact steps in order

### 🧪 Quick Test
```python
result = complex_calculation(3, 4)
print(f"Result: {result}")  # Should print: Result: 110
```

**Note**: This demonstrates how NBGrader can guide you through complex functions step-by-step!
"""

# %% 
# nbgrader: grade, solution
#| export
def complex_calculation(a, b):
    """
    Perform a multi-step calculation with guided implementation.
    
    This function demonstrates multiple solution blocks:
    1. Add 2 to both input variables
    2. Sum the modified variables
    3. Multiply by 10
    
    Args:
        a (int): First number
        b (int): Second number
        
    Returns:
        int: Result of (a+2) + (b+2), then multiplied by 10
    """
    # Step 1: Add 2 to each variable
    a_plus_2 = a + 2
    b_plus_2 = b + 2
    
    # Step 2: Sum the modified variables
    everything_summed = a_plus_2 + b_plus_2
    
    # Step 3: Multiply by 10
    everything_summed_times_10 = everything_summed * 10
    
    return everything_summed_times_10

# %% 
# nbgrader: tests
# Test complex_calculation function
assert complex_calculation(3, 4) == 110, "complex_calculation(3, 4) should equal 110"
assert complex_calculation(1, 2) == 70, "complex_calculation(1, 2) should equal 70"
assert complex_calculation(0, 0) == 40, "complex_calculation(0, 0) should equal 40"
assert complex_calculation(-1, 1) == 40, "complex_calculation(-1, 1) should equal 40"
print("✅ complex_calculation tests passed!")

# %% [markdown]
"""
## Problem 3: Basic Math Function (5 points) ➕

**Goal**: Create a simple function that adds two numbers.

### 📝 Requirements
- Take two parameters: `a` and `b`
- Return their sum
- Handle any numeric types (int, float)

### 💡 Approach & Hints
1. **Simple addition** - just use the `+` operator
2. **No type conversion needed** - Python handles int + float automatically
3. **One line implementation** - this is straightforward!

### 🎯 Expected Behavior
```python
add_numbers(2, 3)    # Returns: 5
add_numbers(1.5, 2.5)  # Returns: 4.0
add_numbers(-1, 1)   # Returns: 0
```

### 🚨 Common Pitfalls
- ❌ Overthinking it - this is really simple!
- ❌ Forgetting to return the result
- ❌ Trying to do type conversion (not needed)

### 🧪 Quick Test
```python
result = add_numbers(5, 7)
print(f"5 + 7 = {result}")  # Should print: 5 + 7 = 12
```
"""

# %% 
# nbgrader: grade, solution
#| export
def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a (int or float): First number
        b (int or float): Second number
        
    Returns:
        int or float: Sum of a and b
    """
    return a + b

# %% 
# nbgrader: tests
# Test add_numbers function
assert add_numbers(2, 3) == 5, "add_numbers(2, 3) should equal 5"
assert add_numbers(1.5, 2.5) == 4.0, "add_numbers(1.5, 2.5) should equal 4.0"
assert add_numbers(-1, 1) == 0, "add_numbers(-1, 1) should equal 0"
assert add_numbers(0, 0) == 0, "add_numbers(0, 0) should equal 0"
print("✅ add_numbers tests passed!")

# %% [markdown]
"""
## Problem 4: System Info Class (20 points) 🖥️

**Goal**: Create a class that gathers and displays system information.

### 📝 Requirements
- Create a class called `SystemInfo`
- `__init__()`: Store system information (Python version, platform, timestamp)
- `__str__()`: Return a formatted string with all system info
- `is_compatible()`: Return True if Python version >= 3.8

### 💡 Approach & Hints
1. **Use sys.version** - gets Python version string
2. **Use platform.system()** - gets OS name (Windows, Darwin, Linux)
3. **Use datetime.now()** - gets current timestamp
4. **Parse version string** - extract major.minor version for compatibility check
5. **String formatting** - use f-strings for clean output

### 🎯 Expected Behavior
```python
info = SystemInfo()
print(info)
# Output:
# TinyTorch System Info
# Python Version: 3.9.7
# Platform: Darwin
# Timestamp: 2024-01-15 10:30:45.123456

print(info.is_compatible())  # True (if Python >= 3.8)
```

### 🚨 Common Pitfalls
- ❌ Not storing data in __init__ (computing it in __str__ instead)
- ❌ Version parsing errors (handling edge cases in version string)
- ❌ Incorrect string formatting (missing newlines or proper spacing)
- ❌ Not using instance variables (self.variable_name)

### 🧪 Quick Test
```python
info = SystemInfo()
print(f"Compatible: {info.is_compatible()}")  # Should print: Compatible: True
print(info)  # Should print formatted system info
```
"""

# %% 
# nbgrader: grade, solution
#| export  
class SystemInfo:
    """
    A class for gathering and displaying system information.
    
    This class collects Python version, platform, and timestamp information
    when instantiated and provides methods to display and check compatibility.
    """
    
    def __init__(self):
        """Initialize system info by collecting current system data."""
        self.python_version = sys.version.split()[0]  # Get clean version string
        self.platform = platform.system()
        self.timestamp = datetime.now()
    
    def __str__(self):
        """Return formatted system information string."""
        return f"""TinyTorch System Info
Python Version: {self.python_version}
Platform: {self.platform}
Timestamp: {self.timestamp}"""
    
    def is_compatible(self):
        """Check if Python version is compatible (>= 3.8)."""
        try:
            version_parts = self.python_version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            return major > 3 or (major == 3 and minor >= 8)
        except (ValueError, IndexError):
            return False

# %% 
# nbgrader: tests
# Test SystemInfo class
info = SystemInfo()

# Test that attributes exist
assert hasattr(info, 'python_version'), "SystemInfo should have python_version attribute"
assert hasattr(info, 'platform'), "SystemInfo should have platform attribute"
assert hasattr(info, 'timestamp'), "SystemInfo should have timestamp attribute"

# Test string representation
info_str = str(info)
assert "TinyTorch System Info" in info_str, "String should contain title"
assert "Python Version:" in info_str, "String should contain Python version"
assert "Platform:" in info_str, "String should contain platform"
assert "Timestamp:" in info_str, "String should contain timestamp"

# Test compatibility check
compatibility = info.is_compatible()
assert isinstance(compatibility, bool), "is_compatible should return boolean"

print("✅ SystemInfo tests passed!")

# %% [markdown]
"""
## Problem 5: Developer Profile Class (30 points) 👨‍💻

**Goal**: Create an advanced class representing a developer profile with multiple methods.

### 📝 Requirements
- Create a class called `DeveloperProfile`
- `__init__()`: Accept name, email, affiliation, specialization (with defaults)
- `__str__()`: Return a formatted profile card
- `get_signature()`: Return a signature line with name and specialization
- `get_profile_info()`: Return a dictionary with all profile information

### 💡 Approach & Hints
1. **Default parameters** - use defaults in __init__ method signature
2. **Instance variables** - store all parameters as self.variable_name
3. **String formatting** - create a nice "business card" format
4. **Dictionary creation** - return all instance variables as key-value pairs
5. **Method chaining** - each method should work independently

### 🎯 Expected Behavior
```python
dev = DeveloperProfile("Alice", "alice@example.com", "MIT", "Deep Learning")
print(dev)
# Output:
# ═══════════════════════════════════════
# 🚀 DEVELOPER PROFILE
# ═══════════════════════════════════════
# Name: Alice
# Email: alice@example.com
# Affiliation: MIT
# Specialization: Deep Learning
# ═══════════════════════════════════════

print(dev.get_signature())
# Output: Alice - Deep Learning Specialist

profile_dict = dev.get_profile_info()
print(profile_dict['name'])  # Output: Alice
```

### 🚨 Common Pitfalls
- ❌ Not using default parameters correctly
- ❌ Missing the decorative formatting (borders, emoji)
- ❌ Incorrect dictionary key names
- ❌ Not storing all parameters as instance variables
- ❌ String formatting issues (missing newlines, spacing)

### 🧪 Quick Test
```python
dev = DeveloperProfile()  # Should work with defaults
print(dev.get_signature())  # Should print default signature
```
"""

# %% 
# nbgrader: grade, solution
#| export
class DeveloperProfile:
    """
    A class representing a developer profile with personal and professional information.
    
    This class stores developer information and provides methods to display
    and access the profile data in various formats.
    """
    
    def __init__(self, name="Student", email="student@example.com", 
                 affiliation="TinyTorch Community", specialization="ML Systems"):
        """
        Initialize a developer profile.
        
        Args:
            name (str): Developer's name
            email (str): Developer's email address
            affiliation (str): Developer's organization or school
            specialization (str): Developer's area of expertise
        """
        self.name = name
        self.email = email
        self.affiliation = affiliation
        self.specialization = specialization
    
    def __str__(self):
        """Return a formatted developer profile card."""
        return f"""═══════════════════════════════════════
🚀 DEVELOPER PROFILE
═══════════════════════════════════════
Name: {self.name}
Email: {self.email}
Affiliation: {self.affiliation}
Specialization: {self.specialization}
═══════════════════════════════════════"""
    
    def get_signature(self):
        """Return a signature line with name and specialization."""
        return f"{self.name} - {self.specialization} Specialist"
    
    def get_profile_info(self):
        """Return profile information as a dictionary."""
        return {
            'name': self.name,
            'email': self.email,
            'affiliation': self.affiliation,
            'specialization': self.specialization
        }

# %% 
# nbgrader: tests
# Test DeveloperProfile class
dev = DeveloperProfile("Alice", "alice@example.com", "MIT", "Deep Learning")

# Test attributes
assert dev.name == "Alice", "Name should be stored correctly"
assert dev.email == "alice@example.com", "Email should be stored correctly"
assert dev.affiliation == "MIT", "Affiliation should be stored correctly"
assert dev.specialization == "Deep Learning", "Specialization should be stored correctly"

# Test string representation
dev_str = str(dev)
assert "DEVELOPER PROFILE" in dev_str, "String should contain title"
assert "Alice" in dev_str, "String should contain name"
assert "alice@example.com" in dev_str, "String should contain email"
assert "MIT" in dev_str, "String should contain affiliation"
assert "Deep Learning" in dev_str, "String should contain specialization"

# Test signature
signature = dev.get_signature()
assert "Alice - Deep Learning Specialist" == signature, "Signature should be formatted correctly"

# Test profile info dictionary
profile_info = dev.get_profile_info()
assert isinstance(profile_info, dict), "get_profile_info should return dict"
assert profile_info['name'] == "Alice", "Profile info should contain correct name"
assert profile_info['email'] == "alice@example.com", "Profile info should contain correct email"
assert profile_info['affiliation'] == "MIT", "Profile info should contain correct affiliation"
assert profile_info['specialization'] == "Deep Learning", "Profile info should contain correct specialization"

# Test default initialization
default_dev = DeveloperProfile()
assert default_dev.name == "Student", "Default name should be 'Student'"
assert default_dev.email == "student@example.com", "Default email should be correct"

print("✅ DeveloperProfile tests passed!")

# %% [markdown]
"""
## Problem 6: Integration Test (25 points) 🔧

**Goal**: Create a comprehensive test function that verifies all previous functions work together.

### 📝 Requirements
- Create a function called `test_integration()`
- Test all previously implemented functions
- Use proper assertions with descriptive error messages
- Handle any exceptions gracefully
- Return a success message if all tests pass

### 💡 Approach & Hints
1. **Test each function systematically** - call each function with known inputs
2. **Use assert statements** - verify expected outputs
3. **Descriptive error messages** - help debug what went wrong
4. **Exception handling** - catch and report any unexpected errors
5. **Comprehensive coverage** - test both normal and edge cases

### 🎯 Expected Behavior
```python
test_integration()
# Output:
# ✅ Testing hello_tinytorch... passed
# ✅ Testing complex_calculation... passed
# ✅ Testing add_numbers... passed
# ✅ Testing SystemInfo... passed
# ✅ Testing DeveloperProfile... passed
# 🎉 All integration tests passed! TinyTorch setup is complete.
```

### 🚨 Common Pitfalls
- ❌ Not testing all functions thoroughly
- ❌ Missing error handling for unexpected exceptions
- ❌ Unclear error messages (hard to debug failures)
- ❌ Not returning a success indicator
- ❌ Testing only happy path (not edge cases)

### 🧪 Quick Test
```python
result = test_integration()
print(result)  # Should print success message
```
"""

# %% 
# nbgrader: grade, solution
#| export
def test_integration():
    """
    Comprehensive integration test for all TinyTorch setup functions.
    
    This function tests all implemented functions to ensure they work
    correctly together and individually.
    
    Returns:
        str: Success message if all tests pass
        
    Raises:
        AssertionError: If any test fails
        Exception: If any unexpected error occurs
    """
    try:
        # Test hello_tinytorch
        print("✅ Testing hello_tinytorch... ", end="")
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        hello_tinytorch()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        assert len(output.strip()) > 0, "hello_tinytorch should produce output"
        assert "Build ML Systems from Scratch!" in output, "hello_tinytorch should contain tagline"
        print("passed")
        
        # Test complex_calculation
        print("✅ Testing complex_calculation... ", end="")
        result = complex_calculation(3, 4)
        assert result == 110, f"complex_calculation(3, 4) should return 110, got {result}"
        result = complex_calculation(0, 0)
        assert result == 40, f"complex_calculation(0, 0) should return 40, got {result}"
        print("passed")
        
        # Test add_numbers
        print("✅ Testing add_numbers... ", end="")
        result = add_numbers(5, 7)
        assert result == 12, f"add_numbers(5, 7) should return 12, got {result}"
        result = add_numbers(1.5, 2.5)
        assert result == 4.0, f"add_numbers(1.5, 2.5) should return 4.0, got {result}"
        print("passed")
        
        # Test SystemInfo
        print("✅ Testing SystemInfo... ", end="")
        info = SystemInfo()
        assert hasattr(info, 'python_version'), "SystemInfo should have python_version attribute"
        assert hasattr(info, 'platform'), "SystemInfo should have platform attribute"
        assert hasattr(info, 'timestamp'), "SystemInfo should have timestamp attribute"
        info_str = str(info)
        assert "TinyTorch System Info" in info_str, "SystemInfo string should contain title"
        assert isinstance(info.is_compatible(), bool), "is_compatible should return boolean"
        print("passed")
        
        # Test DeveloperProfile
        print("✅ Testing DeveloperProfile... ", end="")
        dev = DeveloperProfile("Test User", "test@example.com", "Test Org", "Testing")
        assert dev.name == "Test User", "DeveloperProfile should store name correctly"
        assert dev.email == "test@example.com", "DeveloperProfile should store email correctly"
        dev_str = str(dev)
        assert "DEVELOPER PROFILE" in dev_str, "DeveloperProfile string should contain title"
        signature = dev.get_signature()
        assert "Test User - Testing Specialist" == signature, "Signature should be formatted correctly"
        profile_info = dev.get_profile_info()
        assert isinstance(profile_info, dict), "get_profile_info should return dict"
        assert profile_info['name'] == "Test User", "Profile info should contain correct name"
        print("passed")
        
        # All tests passed
        success_msg = "🎉 All integration tests passed! TinyTorch setup is complete."
        print(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"❌ Integration test failed: {str(e)}"
        print(error_msg)
        raise

# %% 
# nbgrader: tests
# Test integration function
try:
    result = test_integration()
    assert "All integration tests passed" in result, "Integration test should return success message"
    print("✅ Integration test verification passed!")
except Exception as e:
    print(f"❌ Integration test verification failed: {e}")
    raise

# %% [markdown]
"""
## 🎉 Assignment Complete!

Congratulations! You've successfully completed the TinyTorch setup assignment. 

### What You've Accomplished:
✅ **File I/O & Error Handling** - Created robust file reading with graceful fallbacks  
✅ **Multi-Step Implementation** - Learned NBGrader's guided solution approach  
✅ **Basic Functions** - Implemented fundamental mathematical operations  
✅ **Object-Oriented Programming** - Built classes with multiple methods  
✅ **System Integration** - Created comprehensive testing workflows  
✅ **Real-World Skills** - Practiced debugging, testing, and validation  

### Next Steps:
1. **Export your code**: `tito module export 00_setup`
2. **Run the tests**: `pytest tests/ -v`
3. **Use your functions**: `from tinytorch.core.utils import hello_tinytorch`

### Key Takeaways:
- **Error handling is crucial** - Always plan for things to go wrong
- **Testing saves time** - Comprehensive tests catch bugs early
- **Documentation matters** - Clear docstrings help future you
- **Incremental development** - Build and test one piece at a time

**Welcome to the TinyTorch journey! 🚀**
""" 