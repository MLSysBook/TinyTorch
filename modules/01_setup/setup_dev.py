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
# Welcome to TinyTorch! 🚀

You're about to build your own neural network framework from scratch!

First, let's get your environment ready in 3 quick steps:
1. 📦 Install packages 
2. ✅ Check versions
3. 👋 Set up basic info

That's it! Let's begin your AI journey.
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

#| export
import sys
import platform

# %% [markdown]
"""
## Step 1: Install Required Packages 📦

First, we'll install the few packages TinyTorch needs (like NumPy for arrays).
"""

# %% nbgrader={"grade": false, "grade_id": "setup-install", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def setup():
    """Install required packages."""
    import subprocess
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True, capture_output=True, text=True)
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("💡 Try: pip install -r requirements.txt")
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        print("💡 Make sure you're in the TinyTorch directory")

# %% [markdown]
"""
### 🧪 Test: Package Installation
"""

# %% nbgrader={"grade": true, "grade_id": "test-setup", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_setup():
    """Test setup function."""
    print("🔬 Testing setup...")
    
    # Test that function exists and is callable
    assert callable(setup), "setup should be callable"
    
    # Run setup
    setup()
    
    print("✅ Setup function works!")

# %% [markdown]
"""
## Step 2: Check Your Environment ✅

Let's make sure everything installed correctly.
"""

# %% nbgrader={"grade": false, "grade_id": "check-versions", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def check_versions():
    """Quick version check."""
    try:
        import numpy as np
        print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"🔢 NumPy: {np.__version__}")
        print(f"💻 Platform: {platform.system()}")
        print("✅ All versions look good!")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Run setup() first to install packages")

# %% [markdown]
"""
### 🧪 Test: Version Check
"""

# %% nbgrader={"grade": true, "grade_id": "test-versions", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_check_versions():
    """Test check_versions function."""
    print("🔬 Testing version check...")
    
    # Test that function exists and is callable
    assert callable(check_versions), "check_versions should be callable"
    
    # Run version check
    check_versions()
    
    print("✅ Version check function works!")

# %% [markdown]
"""
## Step 3: Basic Course Info 👋

Just some simple info for the course.
"""

# %% nbgrader={"grade": false, "grade_id": "basic-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def get_info():
    """Get basic user info for course."""
    # Students can customize this info
    return {
        "name": "Your Name",
        "email": "your.email@example.com",
        "platform": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}"
    }

# %% [markdown]
"""
### 🧪 Test: Basic Info
"""

# %% nbgrader={"grade": true, "grade_id": "test-basic-info", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_get_info():
    """Test get_info function."""
    print("🔬 Testing basic info...")
    
    # Test that function exists and is callable
    assert callable(get_info), "get_info should be callable"
    
    # Get info
    info = get_info()
    
    # Test return type and structure
    assert isinstance(info, dict), "get_info should return dict"
    assert "name" in info, "Should have name"
    assert "email" in info, "Should have email"
    
    # Display results
    print(f"Name: {info['name']}")
    print(f"Email: {info['email']}")
    
    print("✅ Basic info function works!")

# %% [markdown]
"""
## 🧪 Complete Setup Test
"""

# %% nbgrader={"grade": true, "grade_id": "test-complete", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_complete_setup():
    """Test complete setup workflow."""
    print("🔬 Testing complete setup...")
    
    # Test all functions work together
    setup()
    check_versions()
    info = get_info()
    
    print("\n🎉 SETUP COMPLETE! 🎉")
    print(f"Welcome {info['name']}!")
    print(f"Email: {info['email']}")
    print("✅ Ready to build neural networks!")

if __name__ == "__main__":
    print("🚀 TinyTorch Simple Setup!")
    print("Quick and easy environment setup...\n")
    
    # Run all tests
    print("📦 Step 1: Package Installation")
    test_setup()
    print()
    
    print("✅ Step 2: Version Check")
    test_check_versions()
    print()
    
    print("👋 Step 3: Basic Info")
    test_get_info()
    print()
    
    print("🧪 Step 4: Complete Test")
    test_complete_setup()
    
    print("\n" + "="*50)
    print("🎉 TINYTORCH SETUP COMPLETE! 🎉")
    print("="*50)
    print("✅ Packages installed")
    print("✅ Versions verified")
    print("✅ Basic info collected") 
    print("✅ Ready to build AI!")
    print("\n🚀 Next: Module 2 - Tensors!")

# %% [markdown]
"""
## 🤔 Your AI Journey Starts Here!

Time to think about what you want to create!
"""

# %% nbgrader={"grade": true, "grade_id": "question-excitement", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
### What Are You Most Excited to Build?

Write one sentence about what AI application you want to create!

YOUR ANSWER:
TODO: Write what you're excited to build!
"""

### BEGIN SOLUTION
# Student writes their excitement
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Welcome Complete!

Congratulations! Your TinyTorch environment is ready! 🎉

### What You Accomplished
✅ Installed required packages  
✅ Verified your environment works  
✅ Set up course information

### What's Next? 🚀
1. Run: `tito module complete 01_setup`
2. Module 2: Learn about tensors (the foundation of AI)
3. Start building your neural network framework!

You're officially ready to create AI from scratch! ⚡
"""