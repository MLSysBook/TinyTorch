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

## The Big Picture: Why Configuration Matters in ML Systems
Configuration is the foundation of any production ML system. In this module, you'll learn:

### 1. **System Awareness**
Real ML systems need to understand their environment:
- **Hardware constraints**: Memory, CPU cores, GPU availability
- **Software dependencies**: Python version, library compatibility
- **Platform differences**: Linux servers, macOS development, Windows deployment

### 2. **Reproducibility**
Configuration enables reproducible ML:
- **Environment documentation**: Exactly what system was used
- **Dependency management**: Precise versions and requirements
- **Debugging support**: System info helps troubleshoot issues

### 3. **Professional Development**
Proper configuration shows engineering maturity:
- **Attribution**: Your work is properly credited
- **Collaboration**: Others can understand and extend your setup
- **Maintenance**: Systems can be updated and maintained

### 4. **ML Systems Context**
This connects to broader ML engineering:
- **Model deployment**: Different environments need different configs
- **Monitoring**: System metrics help track performance
- **Scaling**: Understanding hardware helps optimize training

Let's build the foundation of your ML systems engineering skills!
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

#| export
import sys
import platform
import psutil
import os
from typing import Dict, Any

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Setup Module")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Platform: {platform.system()}")
print("Ready to configure your TinyTorch installation!")

# %% [markdown]
"""
## 🏗️ The Architecture of ML Systems Configuration

### Configuration Layers in Production ML
Real ML systems have multiple configuration layers:

```
┌─────────────────────────────────────┐
│        Application Config           │  ← Your personal info
├─────────────────────────────────────┤
│        System Environment           │  ← Hardware specs
├─────────────────────────────────────┤
│        Runtime Configuration        │  ← Python, libraries
├─────────────────────────────────────┤
│        Infrastructure Config        │  ← Cloud, containers
└─────────────────────────────────────┘
```

### Why Each Layer Matters
- **Application**: Identifies who built what and when
- **System**: Determines performance characteristics and limitations
- **Runtime**: Affects compatibility and feature availability
- **Infrastructure**: Enables scaling and deployment strategies

### Connection to Real ML Frameworks
Every major ML framework has configuration:
- **PyTorch**: `torch.cuda.is_available()`, `torch.get_num_threads()`
- **TensorFlow**: `tf.config.list_physical_devices()`, `tf.sysconfig.get_build_info()`
- **Hugging Face**: Model cards with system requirements and performance metrics
- **MLflow**: Experiment tracking with system context and reproducibility

### TinyTorch's Approach
We'll build configuration that's:
- **Educational**: Teaches system awareness
- **Practical**: Actually useful for debugging
- **Professional**: Follows industry standards
- **Extensible**: Ready for future ML systems features
"""

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

#### 1. **Reproducibility** 
Your setup can be documented and shared:
```python
# Someone else can recreate your environment
config = {
    'developer': 'Your Name',
    'python_version': '3.9.7',
    'platform': 'Darwin',
    'memory_gb': 16.0
}
```

#### 2. **Debugging**
System info helps troubleshoot ML performance issues:
- **Memory errors**: "Do I have enough RAM for this model?"
- **Performance issues**: "How many CPU cores can I use?"
- **Compatibility problems**: "What Python version am I running?"

#### 3. **Professional Development**
Shows proper engineering practices:
- **Attribution**: Your work is properly credited
- **Collaboration**: Others can contact you about your code
- **Documentation**: System context is preserved

#### 4. **ML Systems Integration**
Connects to broader ML engineering:
- **Model cards**: Document system requirements
- **Experiment tracking**: Record hardware context
- **Deployment**: Match development to production environments

### Real-World Examples
- **Google Colab**: Shows GPU type, RAM, disk space
- **Kaggle**: Displays system specs for reproducibility
- **MLflow**: Tracks system context with experiments
- **Docker**: Containerizes entire system configuration

Let's start configuring your TinyTorch system!
"""

# %% [markdown]
"""
## Step 2: Personal Information Configuration

### The Concept: Identity in ML Systems
Your **personal information** identifies you as the developer and configures your TinyTorch installation. This isn't just administrative - it's foundational to professional ML development.

### Why Personal Info Matters in ML Engineering

#### 1. **Attribution and Accountability**
- **Model ownership**: Who built this model?
- **Responsibility**: Who should be contacted about issues?
- **Credit**: Proper recognition for your work

#### 2. **Collaboration and Communication**
- **Team coordination**: Multiple developers on ML projects
- **Knowledge sharing**: Others can learn from your work
- **Bug reports**: Contact info for issues and improvements

#### 3. **Professional Standards**
- **Industry practice**: All professional software has attribution
- **Open source**: Proper credit in shared code
- **Academic integrity**: Clear authorship in research

#### 4. **System Customization**
- **Personalized experience**: Your TinyTorch installation
- **Unique identification**: Distinguish your work from others
- **Development tracking**: Link code to developer

### Real-World Parallels
- **Git commits**: Author name and email in every commit
- **Docker images**: Maintainer information in container metadata
- **Python packages**: Author info in `setup.py` and `pyproject.toml`
- **Model cards**: Creator information for ML models

### Best Practices for Personal Configuration
- **Use real information**: Not placeholders or fake data
- **Professional email**: Accessible and appropriate
- **Descriptive system name**: Unique and meaningful
- **Consistent formatting**: Follow established conventions

Now let's implement your personal configuration!
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
    
    EXAMPLE OUTPUT:
    {
        'developer': 'Vijay Janapa Reddi',
        'email': 'vj@eecs.harvard.edu', 
        'institution': 'Harvard University',
        'system_name': 'VJ-TinyTorch-Dev',
        'version': '1.0.0'
    }
    
    IMPLEMENTATION HINTS:
    - Replace the example with your real information
    - Use a descriptive system_name (e.g., 'YourName-TinyTorch-Dev')
    - Keep email format valid (contains @ and domain)
    - Make sure all values are strings
    - Consider how this info will be used in debugging and collaboration
    
    LEARNING CONNECTIONS:
    - This is like the 'author' field in Git commits
    - Similar to maintainer info in Docker images
    - Parallels author info in Python packages
    - Foundation for professional ML development
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

### The Concept: Hardware-Aware ML Systems
**System information** provides details about your hardware and software environment. This is crucial for ML development because machine learning is fundamentally about computation, and computation depends on hardware.

### Why System Information Matters in ML Engineering

#### 1. **Performance Optimization**
- **CPU cores**: Determines parallelization strategies
- **Memory**: Limits batch size and model size
- **Architecture**: Affects numerical precision and optimization

#### 2. **Compatibility and Debugging**
- **Python version**: Determines available features and libraries
- **Platform**: Affects file paths, process management, and system calls
- **Architecture**: Influences numerical behavior and optimization

#### 3. **Resource Planning**
- **Training time estimation**: More cores = faster training
- **Memory requirements**: Avoid out-of-memory errors
- **Deployment matching**: Development should match production

#### 4. **Reproducibility**
- **Environment documentation**: Exact system specifications
- **Performance comparison**: Same code, different hardware
- **Bug reproduction**: System-specific issues

### The Python System Query Toolkit
You'll learn to use these essential Python modules:

#### `sys.version_info` - Python Version
```python
version_info = sys.version_info
python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
# Example: "3.9.7"
```

#### `platform.system()` - Operating System
```python
platform_name = platform.system()
# Examples: "Darwin" (macOS), "Linux", "Windows"
```

#### `platform.machine()` - CPU Architecture
```python
architecture = platform.machine()
# Examples: "x86_64", "arm64", "aarch64"
```

#### `psutil.cpu_count()` - CPU Cores
```python
cpu_count = psutil.cpu_count()
# Example: 8 (cores available for parallel processing)
```

#### `psutil.virtual_memory().total` - Total RAM
```python
memory_bytes = psutil.virtual_memory().total
memory_gb = round(memory_bytes / (1024**3), 1)
# Example: 16.0 GB
```

### Real-World Applications
- **PyTorch**: `torch.get_num_threads()` uses CPU count
- **TensorFlow**: `tf.config.list_physical_devices()` queries hardware
- **Scikit-learn**: `n_jobs=-1` uses all available cores
- **Dask**: Automatically configures workers based on CPU count

### ML Systems Performance Considerations
- **Memory-bound operations**: Matrix multiplication, large model loading
- **CPU-bound operations**: Data preprocessing, feature engineering
- **I/O-bound operations**: Data loading, model saving
- **Platform-specific optimizations**: SIMD instructions, memory management

Now let's implement system information queries!
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
## 🧪 Testing Your Configuration Functions

### The Importance of Testing in ML Systems
Before we test your implementation, let's understand why testing is crucial in ML systems:

#### 1. **Reliability**
- **Function correctness**: Does your code do what it's supposed to?
- **Edge case handling**: What happens with unexpected inputs?
- **Error detection**: Catch bugs before they cause problems

#### 2. **Reproducibility**
- **Consistent behavior**: Same inputs always produce same outputs
- **Environment validation**: Ensure setup works across different systems
- **Regression prevention**: New changes don't break existing functionality

#### 3. **Professional Development**
- **Code quality**: Well-tested code is maintainable code
- **Collaboration**: Others can trust and extend your work
- **Documentation**: Tests serve as executable documentation

#### 4. **ML-Specific Concerns**
- **Data validation**: Ensure data types and shapes are correct
- **Performance verification**: Check that optimizations work
- **System compatibility**: Verify cross-platform behavior

### Testing Strategy
We'll use comprehensive testing that checks:
- **Return types**: Are outputs the correct data types?
- **Required fields**: Are all expected keys present?
- **Data validation**: Are values reasonable and properly formatted?
- **System accuracy**: Do queries match actual system state?

Now let's test your configuration functions!
"""

# %% [markdown]
"""
### 🧪 Test Your Configuration Functions

Once you implement both functions above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-personal-info", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test personal information configuration
print("🔬 Unit Test: Personal Information...")

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
print(f"✅ Hardware: {sys_info['cpu_count']} cores, {sys_info['memory_gb']} GB RAM")

# %% [markdown]
"""
## 🎯 Module Summary: Foundation of ML Systems Engineering

Congratulations! You've successfully configured your TinyTorch installation and learned the foundations of ML systems engineering:

### What You've Accomplished
✅ **Personal Configuration**: Set up your identity and custom system name  
✅ **System Queries**: Learned to gather hardware and software information  
✅ **NBGrader Workflow**: Mastered solution blocks and automated testing  
✅ **Code Export**: Created functions that become part of your tinytorch package  
✅ **Professional Setup**: Established proper development practices  

### Key Concepts You've Learned

#### 1. **System Awareness**
- **Hardware constraints**: Understanding CPU, memory, and architecture limitations
- **Software dependencies**: Python version and platform compatibility
- **Performance implications**: How system specs affect ML workloads

#### 2. **Configuration Management**
- **Personal identification**: Professional attribution and contact information
- **Environment documentation**: Reproducible system specifications
- **Professional standards**: Industry-standard development practices

#### 3. **ML Systems Foundations**
- **Reproducibility**: System context for experiment tracking
- **Debugging**: Hardware info for performance troubleshooting
- **Collaboration**: Proper attribution and contact information

#### 4. **Development Workflow**
- **NBGrader integration**: Automated testing and grading
- **Code export**: Functions become part of production package
- **Testing practices**: Comprehensive validation of functionality

### Connections to Real ML Systems

This module connects to broader ML engineering practices:

#### **Industry Parallels**
- **Docker containers**: System configuration and reproducibility
- **MLflow tracking**: Experiment context and system metadata
- **Model cards**: Documentation of system requirements and performance
- **CI/CD pipelines**: Automated testing and environment validation

#### **Production Considerations**
- **Deployment matching**: Development environment should match production
- **Resource planning**: Understanding hardware constraints for scaling
- **Monitoring**: System metrics for performance optimization
- **Debugging**: System context for troubleshooting issues

### Next Steps in Your ML Systems Journey

#### **Immediate Actions**
1. **Export your code**: `tito module export 00_setup`
2. **Test your installation**: 
   ```python
   from tinytorch.core.setup import personal_info, system_info
   print(personal_info())  # Your personal details
   print(system_info())    # System information
   ```
3. **Verify package integration**: Ensure your functions work in the tinytorch package

#### **Looking Ahead**
- **Module 1 (Tensor)**: Build the fundamental data structure for ML
- **Module 2 (Activations)**: Add nonlinearity for complex learning
- **Module 3 (Layers)**: Create the building blocks of neural networks
- **Module 4 (Networks)**: Compose layers into powerful architectures

#### **Course Progression**
You're now ready to build a complete ML system from scratch:
```
Setup → Tensor → Activations → Layers → Networks → CNN → DataLoader → 
Autograd → Optimizers → Training → Compression → Kernels → Benchmarking → MLOps
```

### Professional Development Milestone

You've taken your first step in ML systems engineering! This module taught you:
- **System thinking**: Understanding hardware and software constraints
- **Professional practices**: Proper attribution, testing, and documentation
- **Tool mastery**: NBGrader workflow and package development
- **Foundation building**: Creating reusable, tested, documented code

**Ready for the next challenge?** Let's build the foundation of ML systems with tensors!
""" 