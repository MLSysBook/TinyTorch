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
# Assignment 0: Setup - Student Information Configuration

Welcome to TinyTorch! This setup assignment configures your personal information for the course and validates your development environment.

## 🎯 Learning Objectives
By the end of this assignment, you will:
- ✅ Configure your student profile information
- ✅ Validate information formats are correct
- ✅ Verify your development environment is ready
- ✅ Understand the NBGrader workflow

## 📋 Assignment Overview
Simple configuration tasks to get you started:

| Problem | Points | Description |
|---------|--------|-------------|
| 1. Student Profile | 40 | Configure your name, email, institution, student ID |
| 2. Email Validation | 20 | Implement email format validation |
| 3. Environment Check | 20 | Verify Python and system compatibility |
| 4. Configuration Test | 20 | Test that all information is valid |
| **Total** | **100** | **Complete Setup** |

## 💡 Instructions
- Replace the example information with **your actual details**
- Make sure all validation functions work correctly
- Test your setup before submitting

---

Let's configure your TinyTorch profile! 🚀
"""

# %%
#| default_exp core.setup

# %%
#| export
# Required imports for setup utilities
import sys
import platform
import re
from datetime import datetime

# %% [markdown]
"""
## Problem 1: Student Profile Configuration (40 points) 👤

**Goal**: Configure your personal information for the TinyTorch course.

### 📝 Requirements
- Return a dictionary with your student information
- Use your actual name, email, institution, and student ID
- Follow the exact format shown in the example

### 💡 Example
```python
{
    'name': 'Vijay Janapa Reddi',
    'email': 'vj@eecs.harvard.edu', 
    'institution': 'Harvard University',
    'student_id': '406737410'
}
```

### 🚨 Important
- Use **your actual information** (not the example)
- Double-check spelling and formatting
- Student ID should be a string, not integer
"""

# %%
# === BEGIN MARK SCHEME ===
# Award full 40 points if:
# - All fields are present and non-empty
# - Email contains @ symbol and domain
# - Information appears to be real (not placeholder text)
# - Dictionary format is correct
# 
# Deduct 10 points per missing/invalid field.
# === END MARK SCHEME ===

#| export
def get_student_profile():
    """
    Return student profile information.
    
    Returns:
        dict: Student information with keys 'name', 'email', 'institution', 'student_id'
    """
    ### BEGIN SOLUTION
    return {
        'name': 'Vijay Janapa Reddi',
        'email': 'vj@eecs.harvard.edu',
        'institution': 'Harvard University', 
        'student_id': '406737410'
    }
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
profile = get_student_profile()

# Test dictionary structure
assert isinstance(profile, dict), "Should return a dictionary"
assert 'name' in profile, "Dictionary should have 'name' key"
assert 'email' in profile, "Dictionary should have 'email' key"
assert 'institution' in profile, "Dictionary should have 'institution' key"
assert 'student_id' in profile, "Dictionary should have 'student_id' key"

# Test non-empty values
assert len(profile['name']) > 0, "Name cannot be empty"
assert len(profile['email']) > 0, "Email cannot be empty"
assert len(profile['institution']) > 0, "Institution cannot be empty"
assert len(profile['student_id']) > 0, "Student ID cannot be empty"

# Test basic email format
assert '@' in profile['email'], "Email should contain @ symbol"
assert '.' in profile['email'], "Email should contain domain"

print("✅ Student profile configuration tests passed!")
### END HIDDEN TESTS

# %% [markdown]
"""
## Problem 2: Email Validation (20 points) 📧

**Goal**: Implement a function to validate email format.

### 📝 Requirements
- Check if email contains @ symbol
- Check if email has a domain (contains . after @)
- Check if email is not empty
- Return True if valid, False otherwise

### 💡 Examples
```python
is_valid_email('vj@eecs.harvard.edu')  # True
is_valid_email('student@university.edu')  # True
is_valid_email('invalid-email')  # False
is_valid_email('no@domain')  # False
is_valid_email('')  # False
```
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 20 points if:
# - Correctly validates emails with @ and domain
# - Rejects invalid formats
# - Handles empty strings
# - Returns boolean values
# 
# Deduct 5 points for each test case that fails.
# === END MARK SCHEME ===

#| export
def is_valid_email(email):
    """
    Validate email format.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    ### BEGIN SOLUTION
    if not email or not isinstance(email, str):
        return False
    
    # Check for @ symbol
    if '@' not in email:
        return False
    
    # Split on @ and check parts
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    local_part, domain_part = parts
    
    # Check non-empty parts
    if not local_part or not domain_part:
        return False
    
    # Check domain has at least one dot
    if '.' not in domain_part:
        return False
    
    return True
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
# Test valid emails
assert is_valid_email('vj@eecs.harvard.edu') == True, "Should accept valid email"
assert is_valid_email('student@university.edu') == True, "Should accept valid email"
assert is_valid_email('test@example.com') == True, "Should accept valid email"

# Test invalid emails
assert is_valid_email('invalid-email') == False, "Should reject email without @"
assert is_valid_email('no@domain') == False, "Should reject email without domain"
assert is_valid_email('') == False, "Should reject empty email"
assert is_valid_email('@domain.com') == False, "Should reject email without local part"
assert is_valid_email('user@') == False, "Should reject email without domain"
assert is_valid_email('user@@domain.com') == False, "Should reject email with multiple @"

print("✅ Email validation tests passed!")
### END HIDDEN TESTS

# %% [markdown]
"""
## Problem 3: Environment Check (20 points) 🖥️

**Goal**: Check that your development environment is ready for TinyTorch.

### 📝 Requirements
- Return a dictionary with system information
- Include Python version, platform, and compatibility status
- Mark compatible if Python >= 3.8

### 💡 Expected Output
```python
{
    'python_version': '3.9.7',
    'platform': 'Darwin',
    'compatible': True
}
```
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 20 points if:
# - Returns dictionary with correct keys
# - Python version is extracted correctly
# - Platform is identified
# - Compatibility check works (True for Python >= 3.8)
# 
# Deduct 5 points per missing/incorrect field.
# === END MARK SCHEME ===

#| export
def check_environment():
    """
    Check development environment compatibility.
    
    Returns:
        dict: System information with python_version, platform, compatible
    """
    ### BEGIN SOLUTION
    # Get Python version
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    # Get platform
    system_platform = platform.system()
    
    # Check compatibility (Python >= 3.8)
    compatible = version_info >= (3, 8)
    
    return {
        'python_version': python_version,
        'platform': system_platform,
        'compatible': compatible
    }
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
env_info = check_environment()

# Test dictionary structure
assert isinstance(env_info, dict), "Should return a dictionary"
assert 'python_version' in env_info, "Should have python_version key"
assert 'platform' in env_info, "Should have platform key"
assert 'compatible' in env_info, "Should have compatible key"

# Test value types
assert isinstance(env_info['python_version'], str), "Python version should be string"
assert isinstance(env_info['platform'], str), "Platform should be string"
assert isinstance(env_info['compatible'], bool), "Compatible should be boolean"

# Test version format
version_parts = env_info['python_version'].split('.')
assert len(version_parts) >= 2, "Version should have at least major.minor"

print("✅ Environment check tests passed!")
### END HIDDEN TESTS

# %% [markdown]
"""
## Problem 4: Configuration Test (20 points) 🔧

**Goal**: Test that all your configuration is valid and ready for TinyTorch.

### 📝 Requirements
- Validate your student profile using the email validation function
- Check that your environment is compatible
- Return a summary of the validation results
- Print a success message if everything is valid

### 💡 Expected Behavior
```python
test_configuration()
# Should print: "✅ TinyTorch setup complete! Ready for ML systems development."
# Returns: True if all tests pass, False otherwise
```
"""

# %%
# === BEGIN MARK SCHEME ===
# Award 20 points if:
# - Uses both previously defined functions
# - Validates student profile email
# - Checks environment compatibility
# - Returns correct boolean
# - Prints appropriate message
# 
# Deduct 5 points for each missing validation check.
# === END MARK SCHEME ===

#| export
def test_configuration():
    """
    Test that all configuration is valid.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    ### BEGIN SOLUTION
    try:
        # Get student profile
        profile = get_student_profile()
        
        # Validate email
        email_valid = is_valid_email(profile['email'])
        if not email_valid:
            print("❌ Invalid email format")
            return False
        
        # Check environment
        env_info = check_environment()
        if not env_info['compatible']:
            print("❌ Python version incompatible (need >= 3.8)")
            return False
        
        # All tests passed
        print("✅ TinyTorch setup complete! Ready for ML systems development.")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    ### END SOLUTION

# %%
### BEGIN HIDDEN TESTS
# Test configuration
result = test_configuration()

# Should return boolean
assert isinstance(result, bool), "Should return boolean"

# Test components work together
profile = get_student_profile()
email_valid = is_valid_email(profile['email'])
env_info = check_environment()

# Profile should be valid
assert email_valid, "Student profile email should be valid"
assert env_info['compatible'], "Environment should be compatible"

print("✅ Configuration test passed!")
### END HIDDEN TESTS

# %% [markdown]
"""
## 🎉 Setup Complete!

Congratulations! You've successfully configured your TinyTorch development environment.

### What You've Accomplished:
✅ **Student Profile** - Configured your personal information  
✅ **Email Validation** - Implemented format checking  
✅ **Environment Check** - Verified system compatibility  
✅ **Integration Test** - Confirmed everything works together  

### Next Steps:
1. **Export your code**: `tito module export 00_setup`
2. **Move to Assignment 1**: Start building your first tensors!

### Key Takeaways:
- **Validation is important** - Always check data formats
- **Environment compatibility** - Verify system requirements
- **Testing integration** - Make sure components work together

**Welcome to TinyTorch - let's build ML systems from scratch! 🚀**
""" 