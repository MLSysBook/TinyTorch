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
# Setup - TinyTorch Development Environment Configuration

Welcome to TinyTorch! This is your first module where you'll verify your environment is ready and set up your personal TinyTorch installation.

## Learning Goals
- **Environment Verification**: Confirm your Python environment is working correctly
- **Personal Configuration**: Set up your identity for TinyTorch development
- **System Information**: Learn to query basic system specs
- **Testing Introduction**: Experience automated testing with your first functions
- **Ready to Build**: Confirm you're prepared to start building ML systems

## Build → Use → Reflect
1. **Build**: Simple environment verification and personal configuration functions
2. **Use**: Configure your TinyTorch installation and verify everything works
3. **Reflect**: Why is proper environment setup important for any programming project?

## What You'll Achieve
By the end of this module, you'll have:
- ✅ **Working Environment**: Confirmed Python and required packages are installed
- ✅ **Personal Setup**: Your name and information configured in TinyTorch
- ✅ **System Awareness**: Basic understanding of your hardware capabilities
- ✅ **First Functions**: Successfully implemented and tested your first TinyTorch code
- ✅ **Ready to Code**: Confidence that your environment is prepared for building ML systems

## First Day Success
💡 **Goal**: Get your environment working and feel confident about building TinyTorch
🎯 **Outcome**: You'll know your setup is correct and be excited to start building ML systems

Let's make sure you're ready to build amazing things!
"""

# %% nbgrader={"grade": false, "grade_id": "setup-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.setup

#| export
import sys
import platform
import psutil
from typing import Dict, Any

# %% nbgrader={"grade": false, "grade_id": "setup-verification", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Setup Module")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Platform: {platform.system()}")
print("Ready to configure your TinyTorch installation!")

# %% [markdown]
"""
### Before We Code: Environment Verification Function

```python
# CONCEPT: What is Environment Verification?
# A simple function that checks if your development environment is ready.
# Think of it like a "systems check" before takeoff - making sure everything
# works before you start building complex ML systems.

# CODE STRUCTURE: What We're Building  
def verify_environment() -> Dict[str, bool]:   # Returns success/failure for each check
    checks = {}                                # Dictionary to store results
    
    # Test basic Python operations
    try:
        result = 2 + 2                        # Simple math test
        checks['basic_math'] = (result == 4)   # Should be True
    except:
        checks['basic_math'] = False           # Something's wrong
    
    # Test package imports (more tests here...)
    return checks                              # Return all check results

# CONNECTIONS: Real-World Equivalents
# Docker health checks - verify containers are working
# pytest setup/teardown - ensure test environment is ready
# CI/CD pipeline checks - validate environment before deployment
# Software installation verification - confirm everything installed correctly

# CONSTRAINTS: Keep It Simple
# - Only test essential functionality (math, imports, version)
# - Return boolean results (True = success, False = failure)
# - Use try/except to handle any errors gracefully
# - Make it quick - this should run in under a second

# CONTEXT: Why This Matters
# Environment verification prevents frustration:
# - Catch setup problems early, not during complex implementations
# - Build confidence that your environment is ready
# - Provide clear success/failure feedback
# - Create a foundation for more advanced TinyTorch modules
```

**You're building your first TinyTorch function - a simple environment checker!**
"""

# %% nbgrader={"grade": false, "grade_id": "verify-environment", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def verify_environment() -> Dict[str, bool]:
    """
    Simple verification that environment is ready for TinyTorch.
    
    This function performs basic checks to ensure your Python environment
    is working correctly and ready for TinyTorch development. It's your
    first TinyTorch function!
    
    TODO: Implement environment verification checks.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Create empty dictionary to store check results
    2. Test basic Python math operations (2 + 2 = 4)
    3. Test that required packages can be imported
    4. Test that Python version is compatible (>= 3.8)
    5. Return dictionary with results
    
    EXAMPLE USAGE:
    ```python
    # Check if environment is ready
    checks = verify_environment()
    print(checks)  # Expected: {'basic_math': True, 'required_packages': True, 'python_version_ok': True}
    
    # Check individual results
    if checks['basic_math']:
        print("✅ Python math works!")
    if checks['required_packages']:
        print("✅ Required packages available!")
    if checks['python_version_ok']:
        print("✅ Python version compatible!")
    ```
    
    IMPLEMENTATION HINTS:
    - Use try/except blocks to handle any import or operation errors
    - Test simple math: result = 2 + 2, then check if result == 4
    - Import sys, platform, psutil in a try block
    - Check sys.version_info >= (3, 8) for Python version
    - Return False for any check that fails
    
    LEARNING CONNECTIONS:
    - This is like pytest fixtures that verify test environment
    - Similar to Docker health checks that verify container status
    - Parallels CI/CD pipeline verification steps
    - Foundation for more complex TinyTorch functionality
    """
    ### BEGIN SOLUTION
    checks = {}
    
    # Test basic Python operations
    try:
        result = 2 + 2
        checks['basic_math'] = (result == 4)
    except:
        checks['basic_math'] = False
    
    # Test package imports
    try:
        import sys, platform, psutil
        checks['required_packages'] = True
    except ImportError:
        checks['required_packages'] = False
    
    # Test Python version compatibility
    checks['python_version_ok'] = sys.version_info >= (3, 8)
    
    return checks
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Environment Verification

This test validates your `verify_environment()` function implementation, ensuring it correctly checks that your development environment is ready for TinyTorch.
"""

# %% nbgrader={"grade": true, "grade_id": "test-verify-environment-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_verify_environment():
    """Test verify_environment function implementation."""
    print("🔬 Unit Test: Environment Verification...")
    
    # Test verify_environment function
    checks = verify_environment()
    
    # Test return type
    assert isinstance(checks, dict), "verify_environment should return a dictionary"
    
    # Test required keys
    required_keys = ['basic_math', 'required_packages', 'python_version_ok']
    for key in required_keys:
        assert key in checks, f"Dictionary should have '{key}' key"
    
    # Test data types (all should be boolean)
    for key, value in checks.items():
        assert isinstance(value, bool), f"Value for '{key}' should be boolean (True/False)"
    
    # Test that basic functionality works (these should be True on a working system)
    assert checks['basic_math'] == True, "Basic math should work (2 + 2 = 4)"
    assert checks['required_packages'] == True, "Required packages should be importable"
    assert checks['python_version_ok'] == True, "Python version should be >= 3.8"
    
    print("✅ Environment verification function tests passed!")
    print("✅ Your environment is ready for TinyTorch development!")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 1: Environment Verification

### What We're Doing
Before we start building TinyTorch, let's make sure your environment is working correctly. We'll create a simple function that checks:

- **Basic Python**: Can Python do simple math operations?
- **Required Packages**: Are the packages we need installed?
- **Python Version**: Is your Python version compatible?

### Why This Matters
- **Confidence**: Know your setup works before diving into complex code
- **Debugging**: If something goes wrong later, we know it's not a basic setup issue
- **Learning**: This is your first TinyTorch function - a gentle introduction to coding

### What We're Building
```python
def verify_environment() -> Dict[str, bool]:
    # Test that basic Python operations work
    # Test that required packages are available  
    # Test that Python version is compatible
    return {"basic_math": True, "required_packages": True, "python_version_ok": True}
```

Let's implement this together!
"""

# %% [markdown]
"""
## Step 2: Personal Information Configuration

### What We're Doing
Now that we've verified your environment works, let's set up your personal TinyTorch installation. This means adding your name, email, and other information so TinyTorch knows who you are.

### Why Personal Information Matters
- **Identification**: Your work should be properly attributed to you
- **Contact**: Others can reach you if they have questions about your code
- **Professional Practice**: All software should know who built it
- **Customization**: Makes your TinyTorch installation uniquely yours

### What We're Building
```python
def personal_info() -> Dict[str, str]:
    return {
        'developer': 'Your Name',
        'email': 'your@email.com',
        'institution': 'Your School',
        'system_name': 'YourName-TinyTorch-Dev',
        'version': '1.0.0'
    }
```

### Real-World Examples
- **Git commits**: Every commit has author name and email
- **Python packages**: setup.py includes author information
- **Software licenses**: Show who created the software

Let's set up your personal TinyTorch configuration!
"""



# %% [markdown]
"""
### Personal Information Configuration

Now let's set up your personal TinyTorch installation with your identity information. This function configures your personal details so TinyTorch knows who you are.

This is similar to setting up Git with your name and email - professional software development always includes proper attribution.
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
    
    EXAMPLE USAGE:
    ```python
    # Get your personal configuration
    info = personal_info()
    print(info['developer'])     # Expected: "Your Name" (not placeholder)
    print(info['email'])         # Expected: "you@domain.com" (valid email)
    print(info['system_name'])   # Expected: "YourName-Dev" (unique identifier)
    print(info)                  # Expected: Complete dict with 5 fields
    # Output: {
    #     'developer': 'Your Name',
    #     'email': 'you@domain.com',
    #     'institution': 'Your Institution',
    #     'system_name': 'YourName-TinyTorch-Dev',
    #     'version': '1.0.0'
    # }
    ```
    
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
        'developer': 'Student Name',
        'email': 'student@university.edu',
        'institution': 'University Name',
        'system_name': 'StudentName-TinyTorch-Dev',
        'version': '1.0.0'
    }
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Personal Information Configuration

This test validates your `personal_info()` function implementation, ensuring it returns properly formatted developer information for system attribution and collaboration.
"""

# %% nbgrader={"grade": true, "grade_id": "test-personal-info-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_unit_personal_info_basic():
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

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 3: System Information Queries

### What We're Doing
Next, let's gather some basic information about your computer. This helps us understand what hardware we're working with.

### Why System Information Matters
- **Debugging**: If something runs slowly, we can check if it's a hardware limitation
- **Compatibility**: Different operating systems sometimes behave differently
- **Learning**: Understanding your development environment
- **Future Planning**: Knowing your specs helps with more advanced modules

### What We're Building
```python
def system_info() -> Dict[str, Any]:
    return {
        'python_version': '3.9.7',      # What Python version you're using
        'platform': 'Darwin',           # Your operating system (Mac/Windows/Linux)
        'architecture': 'arm64',        # Your CPU type
        'cpu_count': 8,                 # How many CPU cores you have
        'memory_gb': 16.0               # How much RAM you have
    }
```

### Python Tools We'll Use
- **`sys.version_info`**: Gets your Python version
- **`platform.system()`**: Gets your operating system
- **`platform.machine()`**: Gets your CPU architecture
- **`psutil.cpu_count()`**: Counts your CPU cores
- **`psutil.virtual_memory()`**: Gets your RAM amount

Let's implement system information queries!
"""

# %% [markdown]
"""
### System Information Queries

Next, let's gather basic information about your computer hardware and software. This helps us understand what resources we're working with and can be useful for debugging performance issues.

We'll query your Python version, operating system, CPU cores, and available memory.
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
    
    EXAMPLE USAGE:
    ```python
    # Query system information
    sys_info = system_info()
    print(f"Python: {sys_info['python_version']}")  # Expected: "3.x.x"
    print(f"Platform: {sys_info['platform']}")      # Expected: "Darwin"/"Linux"/"Windows"
    print(f"CPUs: {sys_info['cpu_count']}")         # Expected: 4, 8, 16, etc.
    print(f"Memory: {sys_info['memory_gb']} GB")    # Expected: 8.0, 16.0, 32.0, etc.
    
    # Full output example:
    print(sys_info)
    # Expected: {
    #     'python_version': '3.9.7',
    #     'platform': 'Darwin',
    #     'architecture': 'arm64', 
    #     'cpu_count': 8,
    #     'memory_gb': 16.0
    # }
    ```
    
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

# %% [markdown]
"""
### 🎯 Additional Comprehensive Tests

These comprehensive tests validate that your configuration functions work together and integrate properly with the TinyTorch system.
"""

# Test function defined (called in main block)

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

if __name__ == "__main__":
    # Run all unit tests
    test_unit_verify_environment()
    test_unit_personal_info_basic()
    test_unit_system_info_basic()
    
    print("\n🎉 All tests passed!")
    print("✅ Your environment is verified and ready")
    print("✅ Personal configuration is set up")
    print("✅ System information is available")
    print("\n🚀 Setup module complete! Ready to start building TinyTorch!")


# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've set up your TinyTorch environment, let's think about why proper setup matters and how it connects to larger programming and ML challenges.

Take time to think about each question - your insights will help you understand why these basic setup steps are important for any programming project.
"""

# %% [markdown]
"""
### Question 1: Environment Problems

**Context**: You've implemented environment verification that checks if basic functionality works. In real programming projects, environment setup is often the first source of frustration for new developers.

**Reflection Question**: Think about a time when you had trouble getting software to work on your computer (maybe installing a game, app, or programming tool). What went wrong? How could a verification function like the one you built have helped identify the problem faster? What additional checks would you add to catch common setup issues?

Think about: common installation problems, version conflicts, missing dependencies, and helpful error messages.

*Target length: 100-200 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-team-config", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TEAM DEVELOPMENT CONFIGURATION:

TODO: Replace this text with your thoughtful response about designing configuration systems for team-based ML development.

Consider addressing:
- How would you maintain individual identity while enabling team coordination?
- What information would be shared vs. personal in a team configuration system?
- How would you handle resource conflicts when multiple developers need GPU access?
- What role would configuration play in debugging issues across team environments?
- How might team configuration differ from individual configuration?

Write a thoughtful analysis connecting your setup module experience to real team challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of individual vs team configuration needs (3 points)
- Addresses resource sharing and conflict resolution challenges (3 points)  
- Connects setup module concepts to real team scenarios (2 points)
- Shows systems thinking about scalability and coordination (2 points)
- Clear writing and practical insights (bonus points for exceptional responses)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring thoughtful analysis of team configuration challenges
# Students should demonstrate understanding of balancing individual accountability with team coordination
### END SOLUTION

# %% [markdown]
"""
### Question 2: System Information Usage

**Context**: Your system_info() function detects CPU cores, memory, and other hardware details. This information can be useful for understanding why programs run fast or slow.

**Reflection Question**: Imagine you're working on a group project and your code runs much slower on your teammate's computer than on yours. How could the system information you're collecting help debug this problem? What hardware differences might cause performance issues? How would you use this information to help your teammate?

Think about: memory limitations, CPU differences, operating system variations, and how to explain technical issues to others.

*Target length: 100-200 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-hardware-aware", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON HARDWARE-AWARE SYSTEM DESIGN:

TODO: Replace this text with your thoughtful response about hardware-aware configuration systems.

Consider addressing:
- How would your system automatically adapt training parameters based on detected hardware?
- What would you do differently for development vs. training vs. inference environments?
- How would you handle memory constraints and batch size optimization automatically?
- What configuration decisions should be automated vs. left to manual tuning?
- How would your system gracefully handle resource limitations?

Write a practical design connecting your system_info() implementation to real performance challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of hardware constraints impact on ML performance (3 points)
- Designs practical automated adaptation strategies (3 points)
- Addresses different deployment environments appropriately (2 points)
- Demonstrates systems thinking about resource optimization (2 points)
- Clear design reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup  
# This is a manually graded question requiring design thinking about hardware-aware systems
# Students should demonstrate understanding of automatic adaptation based on hardware detection
### END SOLUTION

# %% [markdown]
"""
### Question 3: Professional Attribution

**Context**: Your personal_info() function records your name, email, and other identifying information. In professional software development, proper attribution is important for collaboration and accountability.

**Reflection Question**: Why do you think it's important for software to know who created it? Think about situations where you might need to contact the person who wrote some code, or when you might want credit for your work. How does personal attribution help in team projects, open source software, or professional development?

Think about: collaboration benefits, getting help with code, career development, and responsibility in software projects.

*Target length: 100-200 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-production-config", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON PRODUCTION ML PIPELINE CONFIGURATION:

TODO: Replace this text with your thoughtful response about configuration in production ML systems.

Consider addressing:
- How would configuration information become part of model lineage and audit trails?
- What configuration would be bundled with the model vs. environment-specific?
- How would your configuration system support A/B testing and deployment strategies?
- What role would configuration play in monitoring and debugging production issues?
- How would compliance requirements (data governance, model validation) affect configuration design?

Write an analysis connecting your setup module concepts to MLOps and production deployment challenges.

GRADING RUBRIC (Instructor Use):
- Understands configuration role in model lineage and compliance (3 points)
- Addresses model artifact vs. environment configuration separation (3 points)
- Shows awareness of MLOps integration challenges (2 points)
- Demonstrates production systems thinking (2 points)
- Clear analysis with practical MLOps insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of production ML configuration challenges  
# Students should demonstrate understanding of configuration in MLOps and production deployment contexts
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Setup Configuration

Congratulations! You've successfully set up your TinyTorch development environment and you're ready to start building.

### What You've Accomplished
✅ **Environment Verification**: Confirmed your Python setup works correctly
✅ **Personal Configuration**: Set up your identity in the TinyTorch system
✅ **System Information**: Learned to query your computer's hardware specs
✅ **First Functions**: Successfully implemented and tested your first TinyTorch code
✅ **Testing Experience**: Experienced automated testing and saw how it validates your work

### Key Skills You've Learned

#### 1. **Environment Setup**
- **Verification**: How to check if your development environment is working
- **Troubleshooting**: Understanding what can go wrong with setup
- **Confidence Building**: Knowing your environment is ready before starting

#### 2. **Basic Programming**
- **Function Implementation**: Writing functions that return structured data
- **Error Handling**: Using try/except to handle potential problems
- **Data Types**: Working with dictionaries, strings, booleans, and numbers

#### 3. **System Awareness**
- **Hardware Information**: Understanding your computer's capabilities
- **Software Versions**: Knowing what Python version you're using
- **Cross-Platform Thinking**: Awareness that different computers behave differently

#### 4. **Professional Practices**
- **Attribution**: Proper identification of who wrote code
- **Testing**: Automated validation that code works correctly
- **Documentation**: Clear explanations of what code does

### Next Steps in Your TinyTorch Journey

#### **Immediate Actions**
1. **Export your code**: Use the tito command to make your functions available
2. **Test your installation**: Verify everything works in the full TinyTorch package
3. **Celebrate**: You've completed your first TinyTorch module!

#### **Looking Ahead**
You're now ready to start building the core components of machine learning systems:
- **Module 2 (Tensor)**: The fundamental data structure for ML
- **Module 3 (Activations)**: Adding the intelligence to neural networks
- **Module 4 (Layers)**: Building blocks of neural networks
- **And beyond**: Training, optimization, and complete ML systems

### Development Milestone

You've taken your first step in building machine learning systems from scratch! This module taught you:
- **Environment confidence**: Your setup works and you're ready to code
- **Basic implementation**: How to write and test functions
- **Professional practices**: Proper attribution and testing
- **Foundation building**: Creating reliable, tested code

**You're ready to build something amazing!** Let's start building the core of machine learning systems with tensors!
"""
