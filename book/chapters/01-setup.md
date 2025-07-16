---
title: "Setup & Environment"
description: "Development environment setup and basic TinyTorch functionality"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: ['Understand the nbdev notebook-to-Python workflow', 'Write your first TinyTorch code with `#| export` directives', 'Implement system information collection and developer profiles', 'Run tests and use the CLI tools', 'Get comfortable with the development rhythm']
---

# Setup
---
**Course Navigation:** [Home](../intro.html) → [Module 1: 01 Setup](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐ | <strong>Time:</strong> 1-2 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐ Beginner
- **Time Estimate**: 1-2 hours
- **Prerequisites**: Basic Python knowledge
- **Next Steps**: Tensor module

Welcome to TinyTorch! This is your first module in the Machine Learning Systems course.

## Overview

The setup module teaches you the complete TinyTorch development workflow while introducing fundamental programming concepts. You'll learn to write code with NBDev directives, implement classes and functions, and understand the module-to-package export system.

## Learning Goals

- Understand the nbdev notebook-to-Python workflow
- Write your first TinyTorch code with `#| export` directives
- Implement system information collection and developer profiles
- Run tests and use the CLI tools
- Get comfortable with the development rhythm

## Files

- `setup_dev.py` - Main development file (Jupytext format with full educational content)
- `setup_dev.ipynb` - Jupyter notebook version (auto-generated and executed)
- `tinytorch_flame.txt` - ASCII art file containing the TinyTorch flame design
- `tests/test_setup.py` - Comprehensive pytest test suite
- `README.md` - This file

## What You'll Implement

### 1. Basic Functions
- `hello_tinytorch()` - Display ASCII art and welcome message
- `add_numbers()` - Basic arithmetic (foundation of ML operations)

### 2. System Information Class
- `SystemInfo` - Collect and display Python version, platform, and machine info
- Compatibility checking for minimum requirements

### 3. Developer Profile Class
- `DeveloperProfile` - Personalized developer information and signatures
- ASCII art customization and file loading
- Professional code attribution system

## Usage

### Python Script
```python
from setup_dev import hello_tinytorch, add_numbers, SystemInfo, DeveloperProfile

# Display welcome message
hello_tinytorch()

# Basic arithmetic
result = add_numbers(2, 3)

# System information
info = SystemInfo()
print(f"System: {info}")
print(f"Compatible: {info.is_compatible()}")

# Developer profile
profile = DeveloperProfile()
print(profile.get_full_profile())
```

### Jupyter Notebook
Open `setup_dev.ipynb` and work through the educational content step by step.

## Testing

Run the comprehensive test suite using pytest:

```bash
# Using the TinyTorch CLI (recommended)
tito test --module setup

# Or directly with pytest
python -m pytest tests/test_setup.py -v
```

### Test Coverage

The test suite includes **20 comprehensive tests** covering:
- ✅ **Function execution** - All functions run without errors
- ✅ **Output validation** - Correct content and formatting
- ✅ **Arithmetic operations** - Basic, negative, and floating-point math
- ✅ **System information** - Platform detection and compatibility
- ✅ **Developer profiles** - Default and custom configurations
- ✅ **ASCII art handling** - File loading and fallback behavior
- ✅ **Error recovery** - Graceful handling of missing files
- ✅ **Integration testing** - All components work together

## Getting Started

### Prerequisites

1. **Activate the virtual environment**:
   ```bash
   source bin/activate-tinytorch.sh
   ```

2. **Test the setup module**:
   ```bash
   tito test --module setup
   ```

## Development Workflow

This module teaches the core TinyTorch development cycle:

1. **Write code** in the notebook using `#| export` directives
2. **Export code** with `tito sync --module setup`
3. **Run tests** with `tito test --module setup`
4. **Check progress** with `tito info`

## Key Concepts

- **NBDev workflow** - Write in notebooks, export to Python packages
- **Export directives** - Use `#| export` to mark code for export
- **Module → Package mapping** - This module exports to `tinytorch/core/utils.py`
- **Teaching vs. Building** - Learn by modules, build by function
- **Student implementation** - TODO sections with instructor solutions hidden

## Personalization Features

### ASCII Art Customization
The ASCII art is loaded from `tinytorch_flame.txt`. You can customize it by:

1. **Edit the file directly** - Modify `tinytorch_flame.txt` with your own ASCII art
2. **Custom parameter** - Pass your own ASCII art to `DeveloperProfile`
3. **Create your own design** - Your initials, logo, or motivational art

### Developer Profile Customization
```python
my_profile = DeveloperProfile(
    name="Your Name",
    affiliation="Your University",
    email="your.email@example.com",
    github_username="yourgithub",
    ascii_art="Your custom ASCII art here!"
)
```

## What You'll Learn

This comprehensive module introduces:
- **NBDev educational patterns** - `#| export` directives and NBGrader solution markers
- **File I/O operations** - Loading ASCII art with error handling
- **Object-oriented programming** - Classes, methods, and properties
- **System programming** - Platform detection and compatibility
- **Testing with pytest** - Professional test structure and assertions
- **Code organization** - Module structure and package exports
- **The TinyTorch development workflow** - Complete cycle from code to tests

## Next Steps

Once you've completed this module and all tests pass, you're ready to move on to the **tensor module** where you'll build the core data structures that power TinyTorch neural networks!

The skills you learn here - the development workflow, testing patterns, and code organization - will be used throughout every module in TinyTorch. 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/01_setup/setup_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/01_setup/setup_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/01_setup/setup_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="right-next" href="../chapters/02_tensor.html" title="next page">Next Module →</a>
</div>
