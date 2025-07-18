# 🔥 Module: Setup

## 📊 Module Info
- **Difficulty**: ⭐ Beginner
- **Time Estimate**: 1-2 hours
- **Prerequisites**: Basic Python knowledge
- **Next Steps**: Tensor module

Welcome to TinyTorch! This foundational module introduces the complete development workflow that powers every subsequent module. You'll master the nbdev notebook-to-Python workflow, implement your first TinyTorch components, and establish the coding patterns used throughout the entire course.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Master the nbdev workflow**: Write code with `#| export` directives and understand the notebook-to-package pipeline
- **Implement system utilities**: Build functions for system information collection and developer profiles
- **Use TinyTorch CLI tools**: Run tests, sync modules, and check development progress
- **Write production-ready code**: Follow professional patterns for error handling, testing, and documentation
- **Establish development rhythm**: Understand the build → test → iterate cycle that drives all TinyTorch modules

## 🧠 Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement core utilities (hello functions, system info, developer profiles)
2. **Use**: Apply these components in real development workflows and testing scenarios  
3. **Master**: Understand how this foundation supports all advanced TinyTorch modules and establish professional development habits

## 📚 What You'll Build

### Core Functions
```python
# Welcome and basic operations
hello_tinytorch()           # ASCII art and course introduction
add_numbers(2, 3)          # Foundation arithmetic operations

# System information utilities
info = SystemInfo()
print(f"Platform: {info.platform}")
print(f"Compatible: {info.is_compatible()}")

# Developer profile management
profile = DeveloperProfile(name="Your Name", affiliation="University")
print(profile.get_full_profile())
```

### System Information Class
- **Platform detection**: Python version, operating system, machine architecture
- **Compatibility checking**: Verify minimum requirements for TinyTorch development
- **Environment validation**: Ensure proper setup for course progression

### Developer Profile Class
- **Personalized signatures**: Professional code attribution and contact information
- **ASCII art integration**: Custom flame art loading with graceful fallbacks
- **Educational customization**: Personalize your TinyTorch learning experience

## 🚀 Getting Started

### Prerequisites
Ensure you have completed the TinyTorch installation and environment setup:

   ```bash
# Activate TinyTorch environment
   source bin/activate-tinytorch.sh

# Verify installation
tito doctor
```

### Development Workflow
1. **Open the development notebook**: `modules/source/01_setup/setup_dev.py`
2. **Follow the guided implementation**: Complete TODO sections with provided scaffolding
3. **Export your code**: `tito export --module setup`
4. **Test your implementation**: `tito test --module setup`
5. **Verify integration**: `tito nbdev build` to ensure package compatibility

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify your implementation:

   ```bash
# TinyTorch CLI (recommended)
   tito test --module setup

# Direct pytest execution
python -m pytest tests/ -k setup -v
```

### Test Coverage (20 Tests)
- ✅ **Function execution**: All functions run without errors
- ✅ **Output validation**: Correct content and formatting  
- ✅ **Arithmetic operations**: Basic, negative, and floating-point math
- ✅ **System information**: Platform detection and compatibility
- ✅ **Developer profiles**: Default and custom configurations
- ✅ **ASCII art handling**: File loading and fallback behavior
- ✅ **Error recovery**: Graceful handling of missing files
- ✅ **Integration testing**: All components work together

### Inline Testing
The module includes educational inline tests that run during development:
```python
# Example inline test output
🔬 Unit Test: SystemInfo functionality...
✅ System detection works
✅ Compatibility checking works
📈 Progress: SystemInfo ✓
```

## 🎯 Key Concepts

### Real-World Applications
- **Development Environment Management**: Like PyTorch's system compatibility checking
- **Professional Code Attribution**: Similar to open-source project contributor systems
- **Educational Scaffolding**: Mirrors industry onboarding and training workflows
- **System Validation**: Foundation for deployment compatibility (used in modules 12-14)

### Core Programming Patterns
- **NBDev Integration**: Write once in notebooks, deploy everywhere as Python packages
- **Export Directives**: Strategic use of `#| export` for clean package structure
- **Error Handling**: Graceful fallbacks for missing resources and system incompatibilities
- **Object-Oriented Design**: Classes with clear responsibilities and professional interfaces
- **Testing Philosophy**: Comprehensive coverage with both unit and integration approaches

### TinyTorch Foundation
This module establishes patterns used throughout the course:
- **Module → Package Mapping**: `setup_dev.py` → `tinytorch.core.setup`
- **Development Workflow**: Edit → Export → Test → Iterate cycle
- **Educational Structure**: Guided implementation with instructor solutions
- **Professional Standards**: Production-ready code with full test coverage

## 🎉 Ready to Build?

You're about to establish the foundation that will power your entire TinyTorch journey! This module teaches the development workflow mastery that professional ML engineers use daily. 

Every advanced concept you'll learn - from tensors to optimizers to MLOps - builds on the solid patterns you're about to implement here. Take your time, test thoroughly, and enjoy building something that really works! 

```{grid} 3
:gutter: 3
:margin: 2

{grid-item-card} 🚀 Launch Builder
:link: https://mybinder.org/v2/gh/VJProductions/TinyTorch/main?filepath=modules/source/01_setup/setup_dev.py
:class-title: text-center
:class-body: text-center

Interactive development environment

{grid-item-card} 📓 Open in Colab  
:link: https://colab.research.google.com/github/VJProductions/TinyTorch/blob/main/modules/source/01_setup/setup_dev.ipynb
:class-title: text-center
:class-body: text-center

Google Colab notebook

{grid-item-card} 👀 View Source
:link: https://github.com/VJProductions/TinyTorch/blob/main/modules/source/01_setup/setup_dev.py  
:class-title: text-center
:class-body: text-center

Browse the code on GitHub
``` 