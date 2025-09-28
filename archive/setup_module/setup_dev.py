# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Setup - Building Your ML Development Foundation

Welcome to TinyTorch Setup! You'll configure your development environment for machine learning.

## 🔗 Building on Previous Learning
**What You Need**:
- Python 3.8+ installed on your system
- Basic command line familiarity

**What's Working**: You have Python installed and ready to go.

**The Gap**: Raw Python isn't sufficient for ML computation - we need optimized libraries.

**This Module's Solution**: Set up NumPy foundation and validate your environment.

**Connection Map**:
```
Python → Setup → Tensor
(base)   (tools)  (computation)
```

## Learning Objectives
1. **Environment Setup**: Install and validate ML dependencies
2. **Basic Validation**: Check versions and system compatibility  
3. **Development Profile**: Create user configuration for projects
4. **Testing Skills**: Validate setup with immediate feedback

## Build → Test → Use
1. **Build**: Install packages and create validation functions
2. **Test**: Verify each function works correctly
3. **Use**: Apply setup in real ML development workflow
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

#| export
import sys
import platform

# %% [markdown]
"""
## Step 1: Package Installation 📦

Install the essential packages for ML development.
"""

# %% nbgrader={"grade": false, "grade_id": "setup-function", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def setup():
    """Install required packages for TinyTorch development.
    
    TODO: Install NumPy and matplotlib using pip
    
    APPROACH:
    1. Use subprocess to run pip install commands
    2. Install the essential packages we need
    3. Print success message
    
    EXAMPLE:
    >>> setup()
    ✅ Packages installed successfully!
    
    HINT: Use subprocess.run() with ["pip", "install", "package_name"]
    """
    ### BEGIN SOLUTION
    import subprocess
    
    # Install essential packages
    packages = ["numpy", "matplotlib"]
    
    print("📦 Installing TinyTorch dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(["pip", "install", package], check=True)
    
    print("✅ Packages installed successfully!")
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Package Installation

This test validates the setup function works correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-setup", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_setup():
    """Test setup function."""
    print("🔬 Unit Test: Package Installation...")
    
    # Test that function exists and is callable
    assert callable(setup), "setup should be callable"
    
    # Run setup (should not crash)
    setup()
    
    print("✅ Setup function works!")

# Run the test immediately
test_unit_setup()

# %% [markdown]
"""
## Step 2: Version Checking ✅

Verify that essential packages are installed and working.
"""

# %% nbgrader={"grade": false, "grade_id": "check-versions", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def check_versions():
    """Check versions of essential packages.
    
    TODO: Import and display version information for key packages
    
    APPROACH:
    1. Try importing NumPy and display version
    2. Show Python and platform information
    3. Handle import errors gracefully
    
    EXAMPLE:
    >>> check_versions()
    🐍 Python: 3.11
    🔢 NumPy: 1.24.3
    💻 Platform: Darwin
    
    HINT: Use try/except to handle missing packages
    """
    ### BEGIN SOLUTION
    try:
        import numpy as np
        print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"🔢 NumPy: {np.__version__}")
        print(f"💻 Platform: {platform.system()}")
        print("✅ All packages available!")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Run setup() first to install packages")
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Version Check

This test validates the version checking function.
"""

# %% nbgrader={"grade": true, "grade_id": "test-versions", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_check_versions():
    """Test check_versions function."""
    print("🔬 Unit Test: Version Check...")
    
    # Test that function exists and is callable
    assert callable(check_versions), "check_versions should be callable"
    
    # Run version check (should not crash)
    check_versions()
    
    print("✅ Version check function works!")

# Run the test immediately
test_unit_check_versions()

# %% [markdown]
"""
## Step 3: User Information 👋

Create a development profile for project tracking.
"""

# %% nbgrader={"grade": false, "grade_id": "user-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def get_info():
    """Create a development profile with user and system information.
    
    A development profile helps track:
    - Who is working on the project (name, email)
    - What system they're using (platform, Python version)
    - When the environment was set up (timestamp)
    
    TODO: Build a profile dictionary with user input and system detection
    
    APPROACH:
    1. Collect user identity (name and email for project attribution)
    2. Detect system information (platform and Python version for compatibility)
    3. Add timestamp (when this environment was configured)
    4. Return complete profile dictionary
    
    EXAMPLE:
    >>> profile = get_info()
    Your name: Alice Smith
    Your email: alice@university.edu
    >>> print(profile)
    {'name': 'Alice Smith', 'email': 'alice@university.edu', 
     'platform': 'Darwin', 'python_version': '3.11', 'timestamp': '2024-01-15T10:30:00'}
    
    HINT: Use input() for user data, platform/sys modules for system info, datetime for timestamp
    """
    ### BEGIN SOLUTION
    import datetime
    
    print("👋 Creating your TinyTorch development profile...")
    print("This helps track who's working on projects and their system setup.")
    
    # Get user information
    name = input("Your name: ").strip()
    if not name:
        name = "TinyTorch Developer"
    
    email = input("Your email: ").strip() 
    if not email:
        email = "dev@tinytorch.local"
    
    # Detect system information automatically
    current_time = datetime.datetime.now().isoformat()
    
    # Create comprehensive profile
    profile = {
        "name": name,
        "email": email,
        "platform": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "timestamp": current_time,
        "setup_complete": True
    }
    
    print(f"\n✅ Profile created for {profile['name']}")
    print(f"📧 Email: {profile['email']}")
    print(f"💻 Platform: {profile['platform']}")
    print(f"🐍 Python: {profile['python_version']}")
    print(f"⏰ Created: {profile['timestamp']}")
    
    return profile
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: User Information

This test validates the user information function.
"""

# %% nbgrader={"grade": true, "grade_id": "test-info", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_get_info():
    """Test get_info function."""
    print("🔬 Unit Test: User Information...")
    
    # Test that function exists and is callable
    assert callable(get_info), "get_info should be callable"
    
    # Mock input to avoid interactive prompt in tests
    import unittest.mock
    with unittest.mock.patch('builtins.input', return_value=''):
        profile = get_info()
    
    # Verify profile structure
    assert isinstance(profile, dict), "get_info should return a dictionary"
    assert 'name' in profile, "Profile should have 'name' field"
    assert 'platform' in profile, "Profile should have 'platform' field"
    
    print("✅ User information function works!")

# Run the test immediately
test_unit_get_info()

# %% [markdown]
"""
## 🧪 Complete Module Testing

Let's run all tests to ensure everything works together.
"""

# %%
def test_unit_all():
    """Run all unit tests for this module."""
    print("🧪 Running all setup tests...")
    
    test_unit_setup()
    test_unit_check_versions() 
    test_unit_get_info()
    
    print("✅ All tests passed! Setup module complete.")

# Run all tests
test_unit_all()


# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Setup Complete!

Congratulations! You've successfully configured your ML development environment!

### What You've Accomplished
✅ **Package Installation**: Automated setup with error handling
✅ **Environment Validation**: Version checking and compatibility testing  
✅ **Development Profile**: User configuration for project tracking
✅ **Testing Framework**: Immediate validation with clear feedback

### Key Learning Outcomes
- **Environment Management**: Install and validate ML dependencies
- **Error Handling**: Graceful failure management with helpful messages
- **System Information**: Platform and version detection techniques
- **Testing Patterns**: Immediate validation after each implementation

### Ready for Next Steps
Your setup implementation now enables:
- **Immediate Application**: Ready for ML development with NumPy foundation
- **Next Module Preparation**: Solid environment for tensor operations
- **Real-World Connection**: Professional development workflow patterns

### Next Steps
1. **Export your module**: `tito module complete 01_setup`
2. **Validate integration**: `tito test --module setup`  
3. **Ready for Module 02**: Tensor operations build on this foundation

**Your environment is ready - let's start building ML systems!** 🚀
"""