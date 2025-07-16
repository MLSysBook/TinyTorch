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
print(f"✅ Memory: {sys_info['memory_gb']} GB, CPUs: {sys_info['cpu_count']}")

# %% [markdown]
"""
### 🧪 Inline Test Functions

These test functions provide immediate feedback when developing your solutions:
"""

# %%
def test_personal_info():
    """Test personal_info function implementation."""
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

# %%
def test_system_info():
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

# %% [markdown]
"""
## 🎯 Professional ML Engineering Skills

You've successfully configured your TinyTorch installation and learned the foundations of ML systems engineering:

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
1. **Export your code**: `tito module export 01_setup`
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

# %% [markdown]
"""
## Step 4: Environment Validation

### The Concept: Dependency Management in ML Systems
**Environment validation** ensures your system has the necessary packages and versions for ML development. This is crucial because ML systems have complex dependency chains that can break in subtle ways.

### Why Environment Validation Matters

#### 1. **Compatibility Assurance**
- **Version conflicts**: Different packages may require incompatible versions
- **API changes**: New versions might break existing code
- **Feature availability**: Some features require specific versions

#### 2. **Reproducibility**
- **Environment documentation**: Exact package versions for reproduction
- **Dependency tracking**: Understanding what's installed and why
- **Debugging support**: Version info helps troubleshoot issues

#### 3. **Professional Development**
- **Deployment safety**: Ensure development matches production
- **Collaboration**: Team members need compatible environments
- **Quality assurance**: Validate setup before beginning work

### Essential ML Dependencies
We'll check for core packages that ML systems depend on:
- **numpy**: Fundamental numerical computing
- **matplotlib**: Visualization and plotting
- **psutil**: System information and monitoring
- **jupyter**: Interactive development environment
- **nbdev**: Package development tools
- **pytest**: Testing framework

### Real-World Applications
- **Docker**: Container images include dependency validation
- **CI/CD**: Automated testing validates environment setup
- **MLflow**: Tracks package versions with experiment metadata
- **Kaggle**: Validates package availability in competition environments

Let's implement environment validation!
"""

# %% nbgrader={"grade": false, "grade_id": "environment-validation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import importlib
import pkg_resources
from typing import Dict, List, Optional

def validate_environment() -> Dict[str, Any]:
    """
    Validate ML development environment and check essential dependencies.
    
    This function checks that your system has the necessary packages for ML development.
    It's like a pre-flight check before you start building ML systems.
    
    TODO: Implement environment validation.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Define list of essential ML packages to check
    2. For each package, try to import it and get version
    3. Track which packages are available vs missing
    4. Calculate environment health score
    5. Return comprehensive environment report
    
    ESSENTIAL PACKAGES TO CHECK:
    - numpy: Numerical computing foundation
    - matplotlib: Visualization and plotting
    - psutil: System monitoring
    - jupyter: Interactive development
    - nbdev: Package development
    - pytest: Testing framework
    
    IMPLEMENTATION HINTS:
    - Use try/except to handle missing packages gracefully
    - Use pkg_resources.get_distribution(package).version for versions
    - Calculate health_score as (available_packages / total_packages) * 100
    - Round health_score to 1 decimal place
    """
    ### BEGIN SOLUTION
    essential_packages = [
        'numpy', 'matplotlib', 'psutil', 'jupyter', 'nbdev', 'pytest'
    ]
    
    available = {}
    missing = []
    
    for package in essential_packages:
        try:
            # Try to import the package
            importlib.import_module(package)
            # Get version information
            version = pkg_resources.get_distribution(package).version
            available[package] = version
        except (ImportError, pkg_resources.DistributionNotFound):
            missing.append(package)
    
    # Calculate health score
    total_packages = len(essential_packages)
    available_packages = len(available)
    health_score = round((available_packages / total_packages) * 100, 1)
    
    return {
        'available_packages': available,
        'missing_packages': missing,
        'health_score': health_score,
        'total_checked': total_packages,
        'status': 'healthy' if health_score >= 80 else 'needs_attention'
    }
    ### END SOLUTION

# %% [markdown]
"""
## Step 5: Performance Benchmarking

### The Concept: Hardware Performance Profiling
**Performance benchmarking** measures your system's computational capabilities for ML workloads. This helps you understand your hardware limits and optimize your development workflow.

### Why Performance Benchmarking Matters

#### 1. **Resource Planning**
- **Training time estimation**: How long will model training take?
- **Memory allocation**: What's the maximum batch size you can handle?
- **Parallelization**: How many cores can you effectively use?

#### 2. **Optimization Guidance**
- **Bottleneck identification**: Is your system CPU-bound or memory-bound?
- **Hardware upgrades**: What would improve performance most?
- **Algorithm selection**: Which algorithms suit your hardware?

#### 3. **Performance Comparison**
- **Baseline establishment**: Track performance over time
- **System comparison**: Compare different development environments
- **Deployment planning**: Match development to production performance

### Benchmarking Strategy
We'll test key ML operations:
- **CPU computation**: Matrix operations that stress the processor
- **Memory bandwidth**: Large data transfers that test memory speed
- **Overall system**: Combined CPU and memory performance

### Real-World Applications
- **MLPerf**: Industry-standard ML benchmarks
- **Cloud providers**: Performance metrics for instance selection
- **Hardware vendors**: Benchmark comparisons for purchasing decisions

Let's implement performance benchmarking!
"""

# %% nbgrader={"grade": false, "grade_id": "performance-benchmark", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time
import random

def benchmark_performance() -> Dict[str, Any]:
    """
    Benchmark system performance for ML workloads.
    
    This function measures computational performance to help you understand
    your system's capabilities and optimize your ML development workflow.
    
    TODO: Implement performance benchmarking.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. CPU Test: Time a computationally intensive operation
    2. Memory Test: Time a memory-intensive operation
    3. Calculate performance scores based on execution time
    4. Determine overall system performance rating
    5. Return comprehensive benchmark results
    
    BENCHMARK TESTS:
    - CPU: Nested loop calculation (computational intensity)
    - Memory: Large list operations (memory bandwidth)
    - Combined: Overall system performance score
    
    IMPLEMENTATION HINTS:
    - Use time.time() to measure execution time
    - CPU test: nested loops with mathematical operations
    - Memory test: large list creation and manipulation
    - Lower execution time = better performance
    - Calculate scores as inverse of time (e.g., 1/time * 1000)
    """
    ### BEGIN SOLUTION
    benchmarks = {}
    
    # CPU Performance Test
    print("⚡ Running CPU benchmark...")
    start_time = time.time()
    
    # CPU-intensive calculation
    result = 0
    for i in range(100000):
        result += i * i + i / 2
    
    cpu_time = time.time() - start_time
    benchmarks['cpu_time'] = round(cpu_time, 3)
    benchmarks['cpu_score'] = round(1000 / cpu_time, 1)
    
    # Memory Performance Test
    print("🧠 Running memory benchmark...")
    start_time = time.time()
    
    # Memory-intensive operations
    large_list = list(range(1000000))
    large_list.reverse()
    large_list.sort()
    
    memory_time = time.time() - start_time
    benchmarks['memory_time'] = round(memory_time, 3)
    benchmarks['memory_score'] = round(1000 / memory_time, 1)
    
    # Overall Performance Score
    overall_score = round((benchmarks['cpu_score'] + benchmarks['memory_score']) / 2, 1)
    benchmarks['overall_score'] = overall_score
    
    # Performance Rating
    if overall_score >= 80:
        rating = 'excellent'
    elif overall_score >= 60:
        rating = 'good'
    elif overall_score >= 40:
        rating = 'fair'
    else:
        rating = 'needs_optimization'
    
    benchmarks['performance_rating'] = rating
    
    return benchmarks
    ### END SOLUTION

# %% [markdown]
"""
## Step 6: Development Environment Setup

### The Concept: Professional Development Configuration
**Development environment setup** configures essential tools and settings for professional ML development. This includes Git configuration, Jupyter settings, and other tools that make development more efficient.

### Why Development Setup Matters

#### 1. **Professional Standards**
- **Version control**: Proper Git configuration for collaboration
- **Code quality**: Consistent formatting and style
- **Documentation**: Automatic documentation generation

#### 2. **Productivity Optimization**
- **Tool configuration**: Optimized settings for efficiency
- **Workflow automation**: Reduce repetitive tasks
- **Error prevention**: Catch issues before they become problems

#### 3. **Collaboration Readiness**
- **Team compatibility**: Consistent development environment
- **Code sharing**: Proper attribution and commit messages
- **Project standards**: Follow established conventions

### Essential Development Tools
We'll configure key tools for ML development:
- **Git**: Version control and collaboration
- **Jupyter**: Interactive development environment
- **Python**: Code formatting and quality tools

Let's implement development environment setup!
"""

# %% nbgrader={"grade": false, "grade_id": "development-setup", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import subprocess
import json
from pathlib import Path

def setup_development_environment() -> Dict[str, Any]:
    """
    Configure development environment for professional ML development.
    
    This function sets up essential tools and configurations to make your
    development workflow more efficient and professional.
    
    TODO: Implement development environment setup.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Check if Git is installed and configured
    2. Verify Jupyter installation and configuration
    3. Check Python development tools
    4. Configure any missing tools
    5. Return setup status and recommendations
    
    DEVELOPMENT TOOLS TO CHECK:
    - Git: Version control system
    - Jupyter: Interactive development
    - Python tools: Code quality and formatting
    
    IMPLEMENTATION HINTS:
    - Use subprocess.run() to check tool availability
    - Use try/except to handle missing tools gracefully
    - Provide helpful recommendations for missing tools
    - Focus on tools that improve ML development workflow
    """
    ### BEGIN SOLUTION
    setup_status = {}
    recommendations = []
    
    # Check Git installation and configuration
    try:
        git_version = subprocess.run(['git', '--version'], 
                                   capture_output=True, text=True, check=True)
        setup_status['git_installed'] = True
        setup_status['git_version'] = git_version.stdout.strip()
        
        # Check Git configuration
        try:
            git_name = subprocess.run(['git', 'config', 'user.name'], 
                                    capture_output=True, text=True, check=True)
            git_email = subprocess.run(['git', 'config', 'user.email'], 
                                     capture_output=True, text=True, check=True)
            setup_status['git_configured'] = True
            setup_status['git_name'] = git_name.stdout.strip()
            setup_status['git_email'] = git_email.stdout.strip()
        except subprocess.CalledProcessError:
            setup_status['git_configured'] = False
            recommendations.append("Configure Git: git config --global user.name 'Your Name'")
            recommendations.append("Configure Git: git config --global user.email 'your.email@domain.com'")
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        setup_status['git_installed'] = False
        recommendations.append("Install Git: https://git-scm.com/downloads")
    
    # Check Jupyter installation
    try:
        jupyter_version = subprocess.run(['jupyter', '--version'], 
                                       capture_output=True, text=True, check=True)
        setup_status['jupyter_installed'] = True
        setup_status['jupyter_version'] = jupyter_version.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        setup_status['jupyter_installed'] = False
        recommendations.append("Install Jupyter: pip install jupyter")
    
    # Check Python tools
    python_tools = ['pip', 'python']
    for tool in python_tools:
        try:
            tool_version = subprocess.run([tool, '--version'], 
                                        capture_output=True, text=True, check=True)
            setup_status[f'{tool}_installed'] = True
            setup_status[f'{tool}_version'] = tool_version.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            setup_status[f'{tool}_installed'] = False
            recommendations.append(f"Install {tool}: Check Python installation")
    
    # Calculate setup health
    total_tools = 4  # git, jupyter, pip, python
    installed_tools = sum([
        setup_status.get('git_installed', False),
        setup_status.get('jupyter_installed', False),
        setup_status.get('pip_installed', False),
        setup_status.get('python_installed', False)
    ])
    
    setup_score = round((installed_tools / total_tools) * 100, 1)
    
    return {
        'setup_status': setup_status,
        'recommendations': recommendations,
        'setup_score': setup_score,
        'status': 'ready' if setup_score >= 75 else 'needs_configuration'
    }
    ### END SOLUTION

# %% [markdown]
"""
## Step 7: Comprehensive System Report

### The Concept: Integrated System Analysis
**Comprehensive system reporting** combines all your configuration and diagnostic information into a single, actionable report. This is like a "health check" for your ML development environment.

### Why Comprehensive Reporting Matters

#### 1. **Holistic View**
- **Complete picture**: All system information in one place
- **Dependency analysis**: How different components interact
- **Performance context**: Understanding system capabilities

#### 2. **Troubleshooting Support**
- **Debugging aid**: Complete environment information for issue resolution
- **Performance analysis**: Identify bottlenecks and optimization opportunities
- **Compatibility checking**: Ensure all components work together

#### 3. **Professional Documentation**
- **Environment documentation**: Complete system specification
- **Reproducibility**: All information needed to recreate environment
- **Sharing**: Easy to share system information with collaborators

Let's create a comprehensive system report!
"""

# %% nbgrader={"grade": false, "grade_id": "system-report", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
from datetime import datetime

def generate_system_report() -> Dict[str, Any]:
    """
    Generate comprehensive system report for ML development.
    
    This function combines all configuration and diagnostic information
    into a single, actionable report for your ML development environment.
    
    TODO: Implement comprehensive system reporting.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Gather personal information
    2. Collect system information
    3. Validate environment
    4. Run performance benchmarks
    5. Check development setup
    6. Generate overall health score
    7. Create comprehensive report with recommendations
    
    REPORT SECTIONS:
    - Personal configuration
    - System specifications
    - Environment validation
    - Performance benchmarks
    - Development setup
    - Overall health assessment
    - Recommendations for improvement
    
    IMPLEMENTATION HINTS:
    - Call all previously implemented functions
    - Combine results into comprehensive report
    - Calculate overall health score from all components
    - Provide actionable recommendations
    """
    ### BEGIN SOLUTION
    print("📊 Generating comprehensive system report...")
    
    # Gather all information
    personal = personal_info()
    system = system_info()
    environment = validate_environment()
    performance = benchmark_performance()
    development = setup_development_environment()
    
    # Calculate overall health score (normalize performance score to 0-100 range)
    normalized_performance = min(performance['overall_score'], 100)  # Cap at 100
    
    health_components = [
        environment['health_score'],
        normalized_performance,
        development['setup_score']
    ]
    
    overall_health = round(sum(health_components) / len(health_components), 1)
    
    # Generate status
    if overall_health >= 85:
        status = 'excellent'
    elif overall_health >= 70:
        status = 'good'
    elif overall_health >= 50:
        status = 'fair'
    else:
        status = 'needs_attention'
    
    # Compile recommendations
    recommendations = []
    
    if environment['health_score'] < 80:
        recommendations.extend([f"Install missing package: {pkg}" for pkg in environment['missing_packages']])
    
    if performance['overall_score'] < 50:
        recommendations.append("Consider hardware upgrade for better ML performance")
    
    recommendations.extend(development['recommendations'])
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'personal_info': personal,
        'system_info': system,
        'environment_validation': environment,
        'performance_benchmarks': performance,
        'development_setup': development,
        'overall_health': overall_health,
        'status': status,
        'recommendations': recommendations,
        'report_version': '1.0.0'
    }
    
    return report
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Enhanced Setup Functions

Test all the new enhanced setup functions:
"""

# Old function removed - using shared test runner pattern

# %%
def test_performance_benchmark():
    """Test performance benchmarking function."""
    print("🔬 Unit Test: Performance Benchmarking...")
    
    benchmark_report = benchmark_performance()
    
    # Test return type and structure
    assert isinstance(benchmark_report, dict), "benchmark_performance should return a dictionary"
    
    # Test required keys
    required_keys = ['cpu_time', 'cpu_score', 'memory_time', 'memory_score', 'overall_score', 'performance_rating']
    for key in required_keys:
        assert key in benchmark_report, f"Report should have '{key}' key"
    
    # Test data types
    assert isinstance(benchmark_report['cpu_time'], (int, float)), "cpu_time should be number"
    assert isinstance(benchmark_report['cpu_score'], (int, float)), "cpu_score should be number"
    assert isinstance(benchmark_report['memory_time'], (int, float)), "memory_time should be number"
    assert isinstance(benchmark_report['memory_score'], (int, float)), "memory_score should be number"
    assert isinstance(benchmark_report['overall_score'], (int, float)), "overall_score should be number"
    assert isinstance(benchmark_report['performance_rating'], str), "performance_rating should be string"
    
    # Test reasonable values
    assert benchmark_report['cpu_time'] > 0, "cpu_time should be positive"
    assert benchmark_report['memory_time'] > 0, "memory_time should be positive"
    assert benchmark_report['cpu_score'] > 0, "cpu_score should be positive"
    assert benchmark_report['memory_score'] > 0, "memory_score should be positive"
    assert benchmark_report['overall_score'] > 0, "overall_score should be positive"
    
    valid_ratings = ['excellent', 'good', 'fair', 'needs_optimization']
    assert benchmark_report['performance_rating'] in valid_ratings, "performance_rating should be valid"
    
    print("✅ Performance benchmark tests passed!")
    print(f"✅ Performance rating: {benchmark_report['performance_rating']}")

# %%
def test_development_setup():
    """Test development environment setup function."""
    print("🔬 Unit Test: Development Environment Setup...")
    
    setup_report = setup_development_environment()
    
    # Test return type and structure
    assert isinstance(setup_report, dict), "setup_development_environment should return a dictionary"
    
    # Test required keys
    required_keys = ['setup_status', 'recommendations', 'setup_score', 'status']
    for key in required_keys:
        assert key in setup_report, f"Report should have '{key}' key"
    
    # Test data types
    assert isinstance(setup_report['setup_status'], dict), "setup_status should be dict"
    assert isinstance(setup_report['recommendations'], list), "recommendations should be list"
    assert isinstance(setup_report['setup_score'], (int, float)), "setup_score should be number"
    assert isinstance(setup_report['status'], str), "status should be string"
    
    # Test reasonable values
    assert 0 <= setup_report['setup_score'] <= 100, "setup_score should be between 0 and 100"
    assert setup_report['status'] in ['ready', 'needs_configuration'], "status should be valid"
    
    print("✅ Development setup tests passed!")
    print(f"✅ Setup score: {setup_report['setup_score']}%")

# %%
def test_system_report():
    """Test comprehensive system report function."""
    print("🔬 Unit Test: System Report Generation...")
    
    report = generate_system_report()
    
    # Test return type and structure
    assert isinstance(report, dict), "generate_system_report should return a dictionary"
    
    # Test required keys
    required_keys = ['timestamp', 'personal_info', 'system_info', 'environment_validation', 
                    'performance_benchmarks', 'development_setup', 'overall_health', 
                    'status', 'recommendations', 'report_version']
    for key in required_keys:
        assert key in report, f"Report should have '{key}' key"
    
    # Test data types
    assert isinstance(report['timestamp'], str), "timestamp should be string"
    assert isinstance(report['personal_info'], dict), "personal_info should be dict"
    assert isinstance(report['system_info'], dict), "system_info should be dict"
    assert isinstance(report['environment_validation'], dict), "environment_validation should be dict"
    assert isinstance(report['performance_benchmarks'], dict), "performance_benchmarks should be dict"
    assert isinstance(report['development_setup'], dict), "development_setup should be dict"
    assert isinstance(report['overall_health'], (int, float)), "overall_health should be number"
    assert isinstance(report['status'], str), "status should be string"
    assert isinstance(report['recommendations'], list), "recommendations should be list"
    assert isinstance(report['report_version'], str), "report_version should be string"
    
    # Test reasonable values
    assert 0 <= report['overall_health'] <= 100, "overall_health should be between 0 and 100"
    valid_statuses = ['excellent', 'good', 'fair', 'needs_attention']
    assert report['status'] in valid_statuses, "status should be valid"
    
    print("✅ System report tests passed!")
    print(f"✅ Overall system health: {report['overall_health']}%")



# %%
def test_personal_info():
    """Test personal information function comprehensively."""
    personal = personal_info()
    assert isinstance(personal, dict), "personal_info should return a dictionary"
    assert 'developer' in personal, "Dictionary should have 'developer' key"
    assert '@' in personal['email'], "Email should contain @ symbol"
    print("✅ Personal information function works")

def test_system_info():
    """Test system information function comprehensively."""
    system = system_info()
    assert isinstance(system, dict), "system_info should return a dictionary"
    assert 'python_version' in system, "Dictionary should have 'python_version' key"
    assert system['memory_gb'] > 0, "Memory should be positive"
    print("✅ System information function works")

def test_environment_validation():
    """Test environment validation function comprehensively."""
    env = validate_environment()
    assert isinstance(env, dict), "validate_environment should return a dictionary"
    assert 'health_score' in env, "Dictionary should have 'health_score' key"
    print("✅ Environment validation function works")

def test_performance_benchmark():
    """Test performance benchmarking function comprehensively."""
    perf = benchmark_performance()
    assert isinstance(perf, dict), "benchmark_performance should return a dictionary"
    assert 'cpu_score' in perf, "Dictionary should have 'cpu_score' key"
    print("✅ Performance benchmarking function works")

def test_development_setup():
    """Test development setup function comprehensively."""
    dev = setup_development_environment()
    assert isinstance(dev, dict), "setup_development_environment should return a dictionary"
    assert 'setup_score' in dev, "Dictionary should have 'setup_score' key"
    print("✅ Development setup function works")

def test_system_report():
    """Test system report comprehensive function."""
    report = generate_system_report()
    assert isinstance(report, dict), "generate_system_report should return a dictionary"
    assert 'overall_health' in report, "Dictionary should have 'overall_health' key"
    print("✅ System report function works")

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Setup")

# %% [markdown]
"""
## 🎯 Module Summary: Development Environment Setup Complete!

Congratulations! You've successfully set up your TinyTorch development environment:

### What You've Accomplished
✅ **Personal Configuration**: Developer information and preferences
✅ **System Analysis**: Hardware and software environment validation
✅ **Environment Validation**: Python packages and dependencies
✅ **Performance Benchmarking**: CPU and memory performance testing
✅ **Development Setup**: IDE configuration and tooling
✅ **Comprehensive Reporting**: System health and recommendations

### Key Concepts You've Learned
- **Environment Management**: How to validate and configure development environments
- **Performance Analysis**: Benchmarking system capabilities for ML workloads
- **System Diagnostics**: Comprehensive health checking and reporting
- **Development Best Practices**: Professional setup for ML development

### Next Steps
1. **Export your code**: `tito package nbdev --export 00_setup`
2. **Test your implementation**: `tito test 00_setup`
3. **Use your environment**: Start building with confidence in a validated setup
4. **Move to Module 1**: Begin implementing the core tensor system!

**Ready for the ML journey?** Your development environment is now optimized for building neural networks from scratch!
""" 