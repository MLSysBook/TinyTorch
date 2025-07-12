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
# Module 0: Setup - TinyTorch System Configuration

Welcome to TinyTorch! This setup module configures your personal TinyTorch installation and teaches you the NBGrader workflow.

## Learning Goals
- Configure your personal TinyTorch installation with custom information
- Learn to query system information using Python modules
- Master the NBGrader workflow: implement → test → export
- Create functions that become part of your tinytorch package
- Understand solution blocks, hidden tests, and automated grading
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

# Setup and imports
import sys
import platform
import psutil
import os
from typing import Dict, Any

print("🔥 TinyTorch Setup Module")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Platform: {platform.system()}")
print("Ready to configure your TinyTorch installation!")

# %% [markdown]
"""
## Step 1: What is System Configuration?

### Definition
**System configuration** is the process of setting up your development environment with personalized information and system diagnostics. In TinyTorch, this means:

- **Personal Information**: Your name, email, institution for identification
- **System Information**: Hardware specs, Python version, platform details
- **Customization**: Making your TinyTorch installation uniquely yours

### Why Configuration Matters in ML Systems
Proper system configuration is crucial because:
- **Reproducibility**: Your setup can be documented and shared
- **Debugging**: System info helps troubleshoot ML performance issues
- **Personalization**: Your work is clearly identified and attributed
- **Workflow**: Learn the NBGrader development process

Let's start configuring!
"""

# %% [markdown]
"""
## Step 2: Personal Information Configuration

### Concept
Your **personal information** identifies you as the developer and configures your TinyTorch installation. This includes your name, email, institution, and a custom system name.

### Why Personal Info Matters
- **Attribution**: Your work is properly credited
- **Collaboration**: Others can contact you about your code
- **Professionalism**: Shows proper development practices
- **Customization**: Makes your installation uniquely yours
"""

# %% nbgrader={"grade": false, "grade_id": "personal-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def personal_info() -> Dict[str, str]:
    """
    Return personal information for this TinyTorch installation.
    
    TODO: Implement personal information configuration.
    
    STEP-BY-STEP:
    1. Create a dictionary with your personal details
    2. Include: developer (your name), email, institution, system_name, version
    3. Use your actual information (not placeholder text)
    4. Make system_name unique and descriptive
    5. Keep version as '1.0.0' for now
    
    EXAMPLE:
    {
        'developer': 'Vijay Janapa Reddi',
        'email': 'vj@eecs.harvard.edu', 
        'institution': 'Harvard University',
        'system_name': 'VJ-TinyTorch-Dev',
        'version': '1.0.0'
    }
    
    HINTS:
    - Replace the example with your real information
    - Use a descriptive system_name (e.g., 'YourName-TinyTorch-Dev')
    - Keep email format valid (contains @ and domain)
    - Make sure all values are strings
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

# %% [markdown]
"""
## Step 3: System Information Queries

### Concept
**System information** provides details about your hardware and software environment. This is crucial for ML development because:

- **Performance**: CPU cores and memory affect training speed
- **Compatibility**: Python version and platform determine what works
- **Debugging**: Architecture and platform help troubleshoot issues
- **Optimization**: Hardware specs guide performance tuning

### System Information to Query
You'll learn to use these Python modules:
- `sys.version_info` - Python version information
- `platform.system()` - Operating system (Windows, Darwin, Linux)
- `platform.machine()` - CPU architecture (x86_64, arm64, etc.)
- `psutil.cpu_count()` - Number of CPU cores
- `psutil.virtual_memory().total` - Total RAM in bytes
"""

# %% nbgrader={"grade": false, "grade_id": "system-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def system_info() -> Dict[str, Any]:
    """
    Query and return system information for this TinyTorch installation.
    
    TODO: Implement system information queries.
    
    STEP-BY-STEP:
    1. Get Python version using sys.version_info
    2. Get platform using platform.system()
    3. Get architecture using platform.machine()
    4. Get CPU count using psutil.cpu_count()
    5. Get memory using psutil.virtual_memory().total
    6. Convert memory from bytes to GB (divide by 1024^3)
    7. Return all information in a dictionary
    
    EXAMPLE OUTPUT:
    {
        'python_version': '3.9.7',
        'platform': 'Darwin', 
        'architecture': 'arm64',
        'cpu_count': 8,
        'memory_gb': 16.0
    }
    
    HINTS:
    - Use f-string formatting for Python version: f"{major}.{minor}.{micro}"
    - Memory conversion: bytes / (1024^3) = GB
    - Round memory to 1 decimal place for readability
    - Make sure data types are correct (strings for text, int for cpu_count, float for memory_gb)
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

# %% [markdown]
"""
### 🧪 Test Your Configuration Functions

Once you implement both functions above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-personal-info", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test personal information configuration
print("Testing personal information...")

# Test personal_info function
personal = personal_info()

# Test return type
assert isinstance(personal, dict), "personal_info should return a dictionary"

# Test required keys
required_keys = ['developer', 'email', 'institution', 'system_name', 'version']
for key in required_keys:
    assert key in personal, f"Dictionary should have '{key}' key"

# Test non-empty values
for key, value in personal.items():
    assert isinstance(value, str), f"Value for '{key}' should be a string"
    assert len(value) > 0, f"Value for '{key}' cannot be empty"

# Test email format
assert '@' in personal['email'], "Email should contain @ symbol"
assert '.' in personal['email'], "Email should contain domain"

# Test version format
assert personal['version'] == '1.0.0', "Version should be '1.0.0'"

# Test system name (should be unique/personalized)
assert len(personal['system_name']) > 5, "System name should be descriptive"

print("✅ Personal info function tests passed!")
print(f"✅ TinyTorch configured for: {personal['developer']}")
print(f"✅ System: {personal['system_name']}")

# %% nbgrader={"grade": true, "grade_id": "test-system-info", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test system information queries
print("Testing system information...")

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
actual_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
assert sys_info['python_version'] == actual_version, "Python version should match actual system"

print("✅ System info function tests passed!")
print(f"✅ Python: {sys_info['python_version']} on {sys_info['platform']}")
print(f"✅ Hardware: {sys_info['cpu_count']} cores, {sys_info['memory_gb']} GB RAM")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully configured your TinyTorch installation:

### What You've Accomplished
✅ **Personal Configuration**: Set up your identity and custom system name  
✅ **System Queries**: Learned to gather hardware and software information  
✅ **NBGrader Workflow**: Mastered solution blocks and automated testing  
✅ **Code Export**: Created functions that become part of your tinytorch package  
✅ **Professional Setup**: Established proper development practices  

### Key Concepts You've Learned
- **System configuration** personalizes your development environment
- **System queries** provide crucial information for ML development
- **NBGrader workflow** enables automated grading and feedback
- **Code export** makes your functions available in the tinytorch package

### Next Steps
1. **Export your code**: `tito module export 00_setup`
2. **Test your installation**: 
   ```python
   from tinytorch.core.setup import personal_info, system_info
   print(personal_info())  # Your personal details
   print(system_info())    # System information
   ```
3. **Move to Module 1**: Start building your first tensors!

**Ready for the next challenge?** Let's build the foundation of ML systems!
""" 