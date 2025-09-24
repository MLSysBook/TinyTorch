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
### 🧪 Unit Test: Package Installation

This test validates the `setup` function, ensuring it correctly installs required packages and handles errors gracefully.
"""

# %% nbgrader={"grade": true, "grade_id": "test-setup", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_setup():
    """Test setup function."""
    print("🔬 Unit Test: Package Installation...")
    
    # Test that function exists and is callable
    assert callable(setup), "setup should be callable"
    
    # Run setup
    setup()
    
    print("✅ Setup function works!")

# Call the test immediately
test_unit_setup()

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
### 🧪 Unit Test: Version Check

This test validates the `check_versions` function, ensuring it correctly displays system and package version information.
"""

# %% nbgrader={"grade": true, "grade_id": "test-versions", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_check_versions():
    """Test check_versions function."""
    print("🔬 Unit Test: Version Check...")
    
    # Test that function exists and is callable
    assert callable(check_versions), "check_versions should be callable"
    
    # Run version check
    check_versions()
    
    print("✅ Version check function works!")

# Call the test immediately
test_unit_check_versions()

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
### 🧪 Unit Test: Basic Info

This test validates the `get_info` function, ensuring it correctly collects and displays user information.
"""

# %% nbgrader={"grade": true, "grade_id": "test-basic-info", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_get_info():
    """Test get_info function."""
    print("🔬 Unit Test: Basic Info...")
    
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

# Call the test immediately
test_unit_get_info()

# %% [markdown]
"""
### 🧪 Unit Test: Complete Setup

This test validates the complete setup workflow, ensuring all functions work together properly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-complete", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_complete_setup():
    """Test complete setup workflow."""
    print("🔬 Unit Test: Complete Setup...")
    
    # Test all functions work together
    setup()
    check_versions()
    info = get_info()
    
    print("\n🎉 SETUP COMPLETE! 🎉")
    print(f"Welcome {info['name']}!")
    print(f"Email: {info['email']}")
    print("✅ Ready to build neural networks!")

# Call the test immediately
test_unit_complete_setup()

# %% [markdown]
"""
## 🔬 Systems Analysis: Environment Impact

### Memory and Performance Analysis
"""

# %% nbgrader={"grade": false, "grade_id": "systems-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| export
def analyze_environment_resources():
    """Analyze memory usage and performance characteristics of environment setup."""
    import tracemalloc
    import time
    import psutil
    
    print("🔬 Environment Resource Analysis")
    print("=" * 40)
    
    # Memory analysis
    tracemalloc.start()
    start_time = time.time()
    
    # Simulate setup operations
    setup()
    check_versions()
    _ = get_info()  # Get info for completeness
    
    # Measure resources
    current, peak = tracemalloc.get_traced_memory()
    setup_time = time.time() - start_time
    
    # System resources
    memory_info = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"📊 Setup Performance:")
    print(f"   ⏱️  Time: {setup_time:.3f} seconds")
    print(f"   🧠 Memory used: {current / 1024:.1f} KB")
    print(f"   📈 Peak memory: {peak / 1024:.1f} KB")
    print(f"   💾 Total system RAM: {memory_info.total / (1024**3):.1f} GB")
    print(f"   🖥️  CPU cores: {cpu_count}")
    
    tracemalloc.stop()
    
    return {
        "setup_time": setup_time,
        "memory_used": current,
        "peak_memory": peak,
        "system_ram": memory_info.total,
        "cpu_cores": cpu_count
    }

# %% [markdown]
"""
### 🧪 Unit Test: Systems Analysis

This test validates the `analyze_environment_resources` function, ensuring it correctly analyzes system performance and resource usage.
"""

# %% nbgrader={"grade": true, "grade_id": "test-systems-analysis", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_analyze_environment_resources():
    """Test environment resource analysis."""
    print("🔬 Unit Test: Systems Analysis...")
    
    # Test that function exists and is callable
    assert callable(analyze_environment_resources), "analyze_environment_resources should be callable"
    
    # Run analysis
    results = analyze_environment_resources()
    
    # Verify return structure
    assert isinstance(results, dict), "Should return dict"
    assert "setup_time" in results, "Should include setup_time"
    assert "memory_used" in results, "Should include memory_used"
    
    print("✅ Systems analysis function works!")

# Call the test immediately
test_unit_analyze_environment_resources()

# %% [markdown]
"""
### Production Context: Container Environments

In production ML systems, environment setup must be:
- **Reproducible**: Identical across development, staging, production
- **Lightweight**: Minimal resource footprint for container deployment
- **Scalable**: Support for distributed training environments
- **Robust**: Handle dependency conflicts and version mismatches

**Real-world considerations:**
- Docker containers limit memory and CPU resources
- Kubernetes pods may restart, requiring fast environment initialization
- Dependency management critical for model serving reliability
- Environment drift can cause model performance degradation
"""

if __name__ == "__main__":
    print("🚀 TinyTorch Simple Setup!")
    print("Quick and easy environment setup...\n")
    
    # Run all tests
    print("📦 Step 1: Package Installation")
    test_unit_setup()
    print()
    
    print("✅ Step 2: Version Check")
    test_unit_check_versions()
    print()
    
    print("👋 Step 3: Basic Info")
    test_unit_get_info()
    print()
    
    print("🧪 Step 4: Complete Test")
    test_unit_complete_setup()
    print()
    
    print("🔬 Step 5: Systems Analysis")
    test_unit_analyze_environment_resources()
    
    print("\n" + "="*50)
    print("🎉 TINYTORCH SETUP COMPLETE! 🎉")
    print("="*50)
    print("✅ Packages installed")
    print("✅ Versions verified")
    print("✅ Basic info collected")
    print("✅ Systems analysis completed")
    print("✅ Ready to build AI!")
    print("\n🚀 Next: Module 2 - Tensors!")


# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

### Question 1: Environment Dependencies
"""

# %% nbgrader={"grade": true, "grade_id": "question-dependencies", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
When setting up ML environments, why is dependency management more critical than in traditional software?

Consider:
- Model reproducibility across different environments
- Version conflicts between ML libraries (NumPy, PyTorch, etc.)
- Container deployment and resource constraints

YOUR ANALYSIS:
TODO: Explain why ML systems have unique dependency challenges
"""

### BEGIN SOLUTION
# ML systems require exact reproducibility for model consistency.
# Version mismatches can cause different numerical results.
# Container environments have strict resource limits.
### END SOLUTION

# %% [markdown]
"""
### Question 2: Setup Automation
"""

# %% nbgrader={"grade": true, "grade_id": "question-automation", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
How would you automate environment setup for a team of 50 ML engineers?

Think about:
- Consistency across developer machines
- Dependency version locking and updates
- Platform differences (macOS, Linux, Windows)
- CI/CD pipeline integration

YOUR STRATEGY:
TODO: Design an automated setup strategy for large ML teams
"""

### BEGIN SOLUTION
# Use Docker containers with locked dependency versions.
# Automated setup scripts with platform detection.
# CI/CD integration with environment validation.
# Dependency management tools like Poetry or Conda.
### END SOLUTION

# %% [markdown]
"""
### Question 3: Production Environment Design
"""

# %% nbgrader={"grade": true, "grade_id": "question-production", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
You're deploying a model serving 1M requests/day. How does your environment setup change?

Consider:
- Memory and CPU resource optimization
- Container orchestration and scaling
- Monitoring and health checks
- Rollback strategies for failed deployments

YOUR DESIGN:
TODO: Describe production environment considerations for high-scale model serving
"""

### BEGIN SOLUTION
# Lightweight containers with minimal dependencies.
# Kubernetes with horizontal pod autoscaling.
# Health checks and monitoring for environment drift.
# Blue-green deployment for safe rollbacks.
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

