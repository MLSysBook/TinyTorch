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
# Setup - TinyTorch System Configuration

Welcome to TinyTorch! This module configures your development environment and establishes professional ML engineering practices.

## Learning Goals
- Configure personal developer identification for your TinyTorch installation
- Query system information for hardware-aware ML development
- Master the NBGrader workflow: implement → test → export
- Build functions that integrate into your tinytorch package

## Why Configuration Matters in ML Systems
Every production ML system needs proper configuration:
- **Developer attribution**: Professional identification and contact info
- **System awareness**: Understanding hardware limitations and capabilities
- **Reproducibility**: Documenting exact environment for experiment tracking
- **Debugging support**: System specs help troubleshoot performance issues

You'll learn to build ML systems that understand their environment and identify their creators.
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

#| export
import sys
import platform
import psutil
import os
from typing import Dict, Any

# %% nbgrader={"grade": false, "grade_id": "setup-verification", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Setup Module")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Platform: {platform.system()}")
print("Ready to configure your TinyTorch installation!\n")

# Display configuration workflow
print("Configuration Workflow:")
print("1.1 Personal Information → 1.2 System Information → Complete")
print("")

# %% [markdown]
"""
## 1.1 Personal Information Configuration

### The 5 C's Framework
Before we implement, let's understand what we're building through our 5 C's approach:

**Concept:** Developer Identity Configuration
Personal information identifies you as the creator of ML systems. Every professional system needs proper attribution - just like Git commits have author info, your TinyTorch installation needs your identity.

**Code Structure:** Building Developer Identity
```python
def personal_info() -> Dict[str, str]:     # Returns developer identity
    return {                               # Dictionary with required fields
        'developer': 'Your Name',         # Your actual name
        'email': 'your@domain.com',       # Contact information
        'institution': 'Your Place',      # Affiliation
        'system_name': 'YourName-Dev',    # Unique system identifier
        'version': '1.0.0'                # Configuration version
    }
```

**Connections:** Real-World Parallels
- **Git commits**: Author name and email in every commit
- **Docker images**: Maintainer information in container metadata
- **Python packages**: Author info in setup.py and pyproject.toml
- **ML model cards**: Creator information for model attribution

**Constraints:** Implementation Requirements
- Use your actual information (not placeholder text)
- Email must contain @ and domain
- System name should be unique and descriptive
- All values must be strings, keep version as '1.0.0'

**Context:** Why This Matters
Professional ML development requires clear attribution:
- **Model ownership**: Who built this neural network?
- **Collaboration**: Others can contact you about issues
- **Professional standards**: Industry practice for all software
- **System customization**: Makes your TinyTorch installation unique

**You're establishing your professional identity in the ML systems world.**
"""

# %% nbgrader={"grade": false, "grade_id": "personal-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def personal_info() -> Dict[str, str]:
    """
    Return personal information for this TinyTorch installation.
    
    This function configures your personal TinyTorch installation with your identity.
    It's the foundation of proper ML engineering practices - every system needs
    to know who built it and how to contact them.
    
    TODO: Implement personal information configuration.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Create a dictionary with your personal details
    2. Include all required keys: developer, email, institution, system_name, version
    3. Use your actual information (not placeholder text)
    4. Make system_name unique and descriptive
    5. Keep version as '1.0.0' for now
    
    Returns:
        Dict[str, str]: Personal configuration with developer identity
    """
    ### BEGIN SOLUTION
    return {
        'developer': 'Student Name',
        'email': 'student@university.edu',
        'institution': 'University Name',
        'system_name': 'StudentName-TinyTorch-Dev',
        'version': '1.0.0'
    }
    ### END SOLUTION

# Test and validate the personal_info function
def test_personal_info_comprehensive():
    """Comprehensive test for personal_info function."""
    print("🔬 Testing Personal Information Configuration...")
    
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
    
    print("✅ All personal info tests passed!")
    print(f"✅ TinyTorch configured for: {personal['developer']}")
    print(f"✅ Contact: {personal['email']}")
    print(f"✅ System: {personal['system_name']}")
    return personal

# Run comprehensive test and display results
personal_config = test_personal_info_comprehensive()
print("\n" + "="*50)
print("✅ 1.1 Personal Information Configuration COMPLETE")
print("="*50)

# %% [markdown]
"""
## 1.2 System Information Collection

### The 5 C's Framework
Before we implement, let's understand what we're building through our 5 C's approach:

**Concept:** Hardware-Aware ML Systems
System information detection provides hardware and software specs that ML systems need for performance optimization. Think computer specifications for gaming - ML needs to know what resources are available.

**Code Structure:** Building System Awareness
```python
def system_info() -> Dict[str, Any]:       # Queries system specs
    return {                               # Hardware/software details
        'python_version': '3.9.7',        # Python compatibility
        'platform': 'Darwin',             # Operating system
        'architecture': 'arm64',          # CPU architecture
        'cpu_count': 8,                   # Parallel processing cores
        'memory_gb': 16.0                 # Available RAM in GB
    }
```

**Connections:** Real-World Applications
- **PyTorch**: `torch.get_num_threads()` uses CPU count for optimization
- **TensorFlow**: `tf.config.list_physical_devices()` queries hardware
- **Scikit-learn**: `n_jobs=-1` uses all available CPU cores
- **MLflow**: Documents system environment for experiment reproducibility

**Constraints:** Implementation Requirements
- Use actual system queries (not hardcoded values)
- Convert memory from bytes to GB for readability
- Round memory to 1 decimal place for clean output
- Return proper data types (strings, int, float)

**Context:** Why Hardware Awareness Matters
ML systems need to understand their environment:
- **Performance**: CPU cores determine parallel processing capability
- **Memory limits**: RAM affects maximum model and batch sizes
- **Debugging**: System specs help troubleshoot performance issues
- **Reproducibility**: Document exact environment for experiment tracking

**You're building ML systems that adapt intelligently to their hardware environment.**
"""

# %% nbgrader={"grade": false, "grade_id": "system-info", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def system_info() -> Dict[str, Any]:
    """
    Query and return system information for this TinyTorch installation.
    
    This function gathers crucial hardware and software information that affects
    ML performance, compatibility, and debugging. It's the foundation of 
    hardware-aware ML systems.
    
    TODO: Implement system information queries.
    
    STEP-BY-STEP IMPLEMENTATION:
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
    
    IMPLEMENTATION HINTS:
    - Use f-string formatting for Python version: f"{major}.{minor}.{micro}"
    - Memory conversion: bytes / (1024^3) = GB
    - Round memory to 1 decimal place for readability
    - Make sure data types are correct (strings for text, int for cpu_count, float for memory_gb)
    
    LEARNING CONNECTIONS:
    - This is like `torch.cuda.is_available()` in PyTorch
    - Similar to system info in MLflow experiment tracking
    - Parallels hardware detection in TensorFlow
    - Foundation for performance optimization in ML systems
    
    PERFORMANCE IMPLICATIONS:
    - cpu_count affects parallel processing capabilities
    - memory_gb determines maximum model and batch sizes
    - platform affects file system and process management
    - architecture influences numerical precision and optimization
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
### 🧪 Unit Test: System Information Query

This test validates your `system_info()` function implementation, ensuring it accurately detects and reports hardware and software specifications for performance optimization and debugging.
"""

# %% nbgrader={"grade": true, "grade_id": "test-system-info-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_system_info_basic():
    """Test system_info function implementation."""
    print("🔬 Unit Test: System Information...")
    
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

# Run the test
test_unit_system_info_basic()


# %% [markdown]
"""
## Module Summary: TinyTorch Setup Complete

Congratulations! You've successfully configured your TinyTorch development environment and established professional ML engineering practices.

### What You've Accomplished
✅ **1.1 Personal Configuration**: Established developer identity and system attribution  
✅ **1.2 System Information**: Built hardware-aware ML system foundation  
✅ **Testing Integration**: Implemented comprehensive validation for both functions  
✅ **Professional Workflow**: Mastered NBGrader solution blocks and testing  

Your TinyTorch installation is now properly configured with:
- **Developer attribution** for professional collaboration
- **System awareness** for performance optimization
- **Tested functions** ready for package integration

### Key ML Systems Concepts Learned
- **Configuration management**: Professional setup and attribution standards
- **Hardware awareness**: System specs affect ML performance and capabilities
- **Testing practices**: Comprehensive validation ensures reliability
- **Package development**: Functions become part of production codebase

### Next Steps
1. **Export your work**: Use `tito module export 01_setup` to integrate with TinyTorch
2. **Verify integration**: Test that your functions work in the tinytorch package
3. **Ready for tensors**: Move on to building the fundamental ML data structure

**You've built the foundation - now let's construct the ML system on top of it!**
""" 
