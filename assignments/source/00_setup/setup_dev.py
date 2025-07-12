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
    
    Hints:
    - Use Path('filename').exists() to check if file exists
    - Use try/except to handle file reading errors
    - Use .strip() to remove extra whitespace
    - Always print the tagline regardless of file status
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
## Problem 2: Multi-Step Math Function (10 points) 🔢

**Goal**: Create a function with **multiple solution blocks** to demonstrate guided implementation.

### 📝 Requirements
- **Step 1**: Add 2 to each input variable
- **Step 2**: Sum the modified variables  
- **Step 3**: Multiply the result by 10
- Return the final result

### 💡 Approach & Hints
This problem demonstrates **multiple solution blocks** within one function:
- Each step has its own `### BEGIN SOLUTION` / `### END SOLUTION` block
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
        
    This function demonstrates multiple solution blocks within one function.
    Complete each step in order using the suggested variable names.
    """
    # Step 1: Add 2 to each input variable
    # Create variables: a_plus_2 and b_plus_2
    # Hint: Simple addition - a_plus_2 = a + ?
    ### BEGIN SOLUTION
    a_plus_2 = a + 2
    b_plus_2 = b + 2
    ### END SOLUTION
    
    # Step 2: Sum everything
    # Create variable: everything_summed
    # Hint: Add the two variables from Step 1
    ### BEGIN SOLUTION
    everything_summed = a_plus_2 + b_plus_2
    ### END SOLUTION
    
    # Step 3: Multiply your previous result by 10
    # Create variable: everything_summed_times_10
    # Hint: You can use * operator (np.multiply is overkill and will make people hate you 😄)
    ### BEGIN SOLUTION
    everything_summed_times_10 = everything_summed * 10
    ### END SOLUTION
    
    return everything_summed_times_10

# %% [markdown]
"""
## Problem 3: Basic Math Function (5 points) ➕

**Goal**: Create a simple addition function to verify our basic workflow.

### 📝 Requirements
- Accept two parameters (a and b)
- Return their sum
- Handle both integers and floats
- Keep it simple - this is a workflow verification!

### 💡 Approach & Hints
- This is intentionally simple - focus on getting the workflow right
- Use the `+` operator (works for int, float, and even some other types)
- No error checking needed - assume valid inputs
- One line of code inside the function is enough!

### 🎯 Expected Behavior
```python
add_numbers(3, 4)      # Returns: 7
add_numbers(2.5, 1.5)  # Returns: 4.0
add_numbers(-1, 1)     # Returns: 0
add_numbers(0, 0)      # Returns: 0
```

### 🧪 Quick Test
```python
result = add_numbers(2.5, 1.5)
print(f"2.5 + 1.5 = {result}")  # Should print: 2.5 + 1.5 = 4.0
```

### 🚨 Common Pitfalls
- ❌ Overthinking it - this is meant to be simple!
- ❌ Adding unnecessary error checking
- ❌ Using complex math when simple + works fine
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
        
    Hint: This is intentionally simple - just use the + operator!
    """
    ### BEGIN SOLUTION
    return a + b
    ### END SOLUTION

# %% [markdown]
"""
## Problem 4: System Information Class (20 points) 🖥️

**Goal**: Create a class that collects and displays system information for debugging and compatibility.

### 📝 Requirements
- **`__init__`**: Collect Python version, platform, and machine architecture
- **`__str__`**: Return formatted system info as "Python X.Y.Z on Platform (Architecture)"
- **`is_compatible()`**: Check if Python version >= 3.8
- Store information as instance variables (self.*)

### 💡 Approach & Hints

#### For `__init__`:
- Use `sys.version_info` to get Python version as tuple
- Convert to string: `f"{major}.{minor}.{micro}"`
- Use `platform.system()` for OS name (Darwin, Windows, Linux)
- Use `platform.machine()` for architecture (arm64, x86_64, etc.)

#### For `__str__`:
- Return exact format: `"Python {version} on {platform} ({machine})"`
- Use f-strings for clean formatting
- Example: `"Python 3.9.7 on Darwin (arm64)"`

#### For `is_compatible()`:
- Compare `sys.version_info[:2]` with `(3, 8)`
- Use `>=` operator on tuples
- Return boolean (True/False)

### 🎯 Expected Behavior
```python
info = SystemInfo()
print(info)  # "Python 3.9.7 on Darwin (arm64)"
print(info.is_compatible())  # True (if Python >= 3.8)

# Access individual properties
print(info.python_version)  # "3.9.7"
print(info.platform)        # "Darwin"  
print(info.machine)         # "arm64"
```

### 🧪 Quick Test
```python
info = SystemInfo()
print(f"System: {info}")
print(f"Compatible: {info.is_compatible()}")
```

### 🚨 Common Pitfalls
- ❌ Not storing as instance variables (using local variables instead)
- ❌ Wrong string formatting in `__str__`
- ❌ Using `sys.version` (string) instead of `sys.version_info` (tuple)
- ❌ Hardcoding version check instead of using `sys.version_info`
"""

# %% nbgrader={"grade": false, "grade_id": "systeminfo_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SystemInfo:
    """
    A class for collecting and displaying system information.
    
    This class helps with debugging and compatibility checking.
    """
    
    def __init__(self):
        """
        Initialize the SystemInfo object.
        Collect Python version, platform, and machine information.
        
        Hints:
        - Use sys.version_info to get version tuple
        - Convert version to string: f"{major}.{minor}.{micro}"
        - Use platform.system() and platform.machine()
        - Store as self.attribute_name
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
        
        Hints:
        - Use f-string formatting
        - Exact format: "Python {version} on {platform} ({machine})"
        - Example: "Python 3.9.7 on Darwin (arm64)"
        """
        ### BEGIN SOLUTION
        return f"Python {self.python_version} on {self.platform} ({self.machine})"
        ### END SOLUTION
    
    def is_compatible(self):
        """
        Check if the Python version is compatible (>= 3.8).
        Returns True if compatible, False otherwise.
        
        Hints:
        - Use sys.version_info[:2] to get (major, minor) tuple
        - Compare with (3, 8) using >= operator
        - Tuple comparison works element by element
        """
        ### BEGIN SOLUTION
        return sys.version_info[:2] >= (3, 8)
        ### END SOLUTION

# %% [markdown]
"""
## Problem 5: Developer Profile Class (30 points) 👨‍💻

**Goal**: Create a class to manage developer profiles with multiple methods and data handling.

### 📝 Requirements
- **`__init__`**: Store developer information (name, email, affiliation, specialization)
- **`__str__`**: Return basic representation as "Name (email)"
- **`get_signature()`**: Return formatted signature with name, affiliation, and specialization
- **`get_profile_info()`**: Return all info as a dictionary

### 💡 Approach & Hints

#### For `__init__`:
- Use default parameters as shown in the function signature
- Store all parameters as instance variables: `self.name = name`, etc.
- Default values make the class easy to use: `DeveloperProfile()` works!

#### For `__str__`:
- Simple format: `"Name (email)"`
- Use f-string: `f"{self.name} ({self.email})"`
- Example: `"Alice (alice@example.com)"`

#### For `get_signature()`:
- Multi-line string with `\\n` separators
- Format: `"Name\\nAffiliation\\nSpecialization: specialization"`
- Example:
  ```
  Alice
  University
  Specialization: Neural Networks
  ```

#### For `get_profile_info()`:
- Return dictionary with all four attributes
- Keys: 'name', 'email', 'affiliation', 'specialization'
- Values: the corresponding instance variable values

### 🎯 Expected Behavior
```python
# Using defaults
profile = DeveloperProfile()
print(profile)  # "Student (student@example.com)"

# Custom values
dev = DeveloperProfile("Alice", "alice@example.com", "University", "Neural Networks")
print(dev)  # "Alice (alice@example.com)"

print(dev.get_signature())
# Alice
# University  
# Specialization: Neural Networks

print(dev.get_profile_info())
# {'name': 'Alice', 'email': 'alice@example.com', 'affiliation': 'University', 'specialization': 'Neural Networks'}
```

### 🧪 Quick Test
```python
profile = DeveloperProfile("Test", "test@example.com")
print(f"Profile: {profile}")
print(f"Signature:\\n{profile.get_signature()}")
print(f"Info: {profile.get_profile_info()}")
```

### 🚨 Common Pitfalls
- ❌ Not using `self.` when storing or accessing instance variables
- ❌ Wrong dictionary keys in `get_profile_info()` 
- ❌ Incorrect string formatting in `get_signature()`
- ❌ Forgetting to return values from methods
- ❌ Not using the default parameters properly
"""

# %% nbgrader={"grade": false, "grade_id": "developer_profile_class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class DeveloperProfile:
    """
    A class representing a developer profile.
    
    This class manages developer information and provides
    multiple ways to display and access the data.
    """
    
    def __init__(self, name="Student", email="student@example.com", affiliation="TinyTorch Community", specialization="ML Systems"):
        """
        Initialize a developer profile.
        
        Args:
            name: Developer's name
            email: Developer's email
            affiliation: Developer's affiliation or organization
            specialization: Developer's area of specialization
            
        Hints:
        - Store each parameter as an instance variable
        - Use self.attribute_name = parameter_name
        - Default values are already provided
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
        
        Hints:
        - Use f-string formatting
        - Format: f"{self.name} ({self.email})"
        - Example: "Alice (alice@example.com)"
        """
        ### BEGIN SOLUTION
        return f"{self.name} ({self.email})"
        ### END SOLUTION
    
    def get_signature(self):
        """
        Return a formatted signature for the developer.
        Should include name, affiliation, and specialization.
        
        Hints:
        - Multi-line string with `\\n` separators
        - Format: "Name\\nAffiliation\\nSpecialization: specialization"
        - Use f-strings for clean formatting
        """
        ### BEGIN SOLUTION
        return f"{self.name}\n{self.affiliation}\nSpecialization: {self.specialization}"
        ### END SOLUTION
    
    def get_profile_info(self):
        """
        Return comprehensive profile information as a dictionary.
        
        Hints:
        - Return dict with keys: 'name', 'email', 'affiliation', 'specialization'
        - Values should be the corresponding self.attribute values
        - Use exact key names as shown above
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
## Testing Your Implementation 🧪

Once you've implemented all the functions above, run the cells below to test your work!

### 🔄 TinyTorch Workflow Reminder
1. **Implement** the functions above ✅
2. **Export** to package: `tito module export 00_setup`
3. **Test** your work: `pytest tests/ -v`
4. **Use** your code: `from tinytorch.core.utils import hello_tinytorch`

### 🚨 Before You Continue
Make sure you can run this without errors:
```python
# Quick test all functions
hello_tinytorch()
print(complex_calculation(3, 4))  # Should be 110
print(add_numbers(2.5, 1.5))      # Should be 4.0
print(SystemInfo())               # Should show your system info
print(DeveloperProfile())         # Should show default profile
```
"""

# %% [markdown]
"""
## Problem 6: Integration Test (25 points) 🔗

**Goal**: Test that all your components work together correctly and demonstrate the complete workflow.

### 📝 Requirements
- Test all functions and classes work correctly
- Test the multi-step function with multiple solution blocks
- Verify system compatibility 
- Display a complete developer profile
- Show successful framework initialization
- Return `True` if all tests pass

### 💡 Approach & Hints
- Use the functions you just implemented
- Include print statements for clear output
- Test with specific values to verify correctness
- Use assertions to check expected results
- Catch and handle any exceptions gracefully

### 🎯 Expected Output
Your integration test should produce output like:
```
🧪 Testing hello_tinytorch()...
[ASCII art or TinyTorch banner]
Build ML Systems from Scratch!
✅ Welcome function works!

🧪 Testing complex_calculation() with multiple solution blocks...
✅ Multi-step calculation: complex_calculation(3, 4) = 110
✅ Multiple solution blocks working correctly!

... [more tests] ...

🎉 All components working together!
✅ Ready for module export and package building!
```

### 🧪 Quick Test Structure
```python
def test_integration():
    print("🧪 Testing...")
    
    # Test each function
    # Use assertions to verify correctness
    # Print success messages
    
    print("🎉 All tests passed!")
    return True
```

### 🚨 Common Pitfalls
- ❌ Not testing all functions
- ❌ Not checking return values with assertions
- ❌ Not handling potential exceptions
- ❌ Forgetting to return True at the end
"""

# %% nbgrader={"grade": true, "grade_id": "integration_test", "locked": false, "points": 25, "schema_version": 3, "solution": true, "task": false}
def test_integration():
    """
    Integration test to verify all components work together.
    This function tests the complete TinyTorch setup workflow.
    
    Returns:
        bool: True if all tests pass
        
    Hints:
    - Test each function you implemented
    - Use assertions to verify expected results
    - Include informative print statements
    - Handle any exceptions gracefully
    - Return True if everything works
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
## Next Steps 🚀

Congratulations! You've completed your first TinyTorch assignment! 

### 🎯 What You've Accomplished
- ✅ Set up your TinyTorch development environment
- ✅ Implemented 5 core utility functions
- ✅ Learned the complete development workflow
- ✅ Practiced error handling and defensive programming
- ✅ Mastered NBGrader solution blocks and guided implementation
- ✅ Created your first real TinyTorch components!

### 🔄 Complete the Workflow
1. **Export** your code to the TinyTorch package:
   ```bash
   tito module export 00_setup
   ```

2. **Run tests** to verify everything works:
   ```bash
   pytest tests/ -v
   ```

3. **Try using** your functions:
   ```python
   from tinytorch.core.utils import hello_tinytorch, add_numbers
   hello_tinytorch()
   print(add_numbers(5, 3))
   ```

4. **Move on** to the next assignment: `01_tensor`

### 🎉 You're Ready!
You've just created the foundation utilities for TinyTorch. These functions will be used throughout the framework. Great job! 

Welcome to the world of building ML systems from scratch! 🔥
""" 