# 📖 TinyTorch Module Development Guide

**Complete methodology for creating educational modules with real-world ML engineering practices.**

## 🎯 Philosophy

**"Build → Use → Understand → Repeat"** with real data and immediate feedback.

Create complete, working implementations that automatically generate student exercise versions while maintaining production-quality exports.

## 🔑 Core Principles

### **Real Data, Real Systems**
- **Use production datasets**: No mock/fake data - students work with CIFAR-10, not synthetic data
- **Show progress feedback**: Downloads, training need visual progress indicators  
- **Cache for efficiency**: Download once, use repeatedly
- **Real-world scale**: Use actual dataset sizes, not toy examples

### **Immediate Visual Feedback**
- **Visual confirmation**: Students see their code working (images, plots, results)
- **Development vs. Export separation**: Rich feedback in `_dev.py`, clean exports to package
- **Progress indicators**: Status messages, progress bars for long operations
- **Real-time validation**: Students can verify each step immediately

### **Educational Excellence**
- **Progressive complexity**: Easy → Medium → Hard with clear difficulty indicators
- **Comprehensive guidance**: TODO sections with approach, examples, hints, systems thinking
- **Real-world connections**: Connect every concept to production ML engineering
- **Immediate testing**: Test each component with real inputs as you build

## 🏗️ Development Workflow

### Step 1: Choose the Learning Pattern
- **Select engagement pattern**: Reflect, Analyze, or Optimize?
- **Use the Pattern Selection Guide** from [Pedagogical Principles](../pedagogy/pedagogical-principles.md):
  - **Build → Use → Reflect**: Early modules, design decisions, systems thinking
  - **Build → Use → Analyze**: Middle modules, technical depth, performance
  - **Build → Use → Optimize**: Advanced modules, iteration, production focus
- **Document your choice** with clear rationale

### Step 2: Plan the Learning Journey
- **Define learning objectives**: What should students implement vs. receive?
- **Choose real data**: What production dataset will they use?
- **Design progression**: How does complexity build through the module?
- **Map to production**: How does this connect to real ML systems?
- **Design pattern-specific activities**: Questions, exercises, or challenges

### Step 3: Write Complete Implementation
Create `modules/{module}/{module}_dev.py` with NBDev structure:

```python
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
# Module: {Title} - {Purpose}

## 🎯 Learning Pattern: Build → Use → [Pattern]

**Pattern Choice**: [Reflect/Analyze/Optimize]
**Rationale**: [Why this pattern fits the learning objectives]

**Key Activities**:
- [Pattern-specific activity 1]
- [Pattern-specific activity 2]
- [Pattern-specific activity 3]

## Learning Objectives
- ✅ Build {core_concept} from scratch
- ✅ Use it with real data ({dataset_name})
- ✅ [Engage] through {pattern_specific_activities}
- ✅ Connect to production ML systems

## What You'll Build
{description_of_what_students_build}
"""

# %%
#| default_exp core.{module}
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional

# %%
#| export
class MainClass:
    """
    {Description of the class}
    
    TODO: {What students need to implement}
    
    APPROACH:
    1. {Step 1 with specific guidance}
    2. {Step 2 with specific guidance}
    3. {Step 3 with specific guidance}
    
    EXAMPLE:
    Input: {concrete_example}
    Expected: {expected_output}
    
    HINTS:
    - {Helpful hint about approach}
    - {Systems thinking hint}
    - {Real-world connection}
    """
    def __init__(self, params):
        raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
class MainClass:
    """Complete implementation (hidden from students)."""
    def __init__(self, params):
        # Actual working implementation
        pass

# %% [markdown]
"""
## 🧪 Test Your Implementation
"""

# %%
# Test with real data
try:
    # Test student implementation
    result = MainClass(real_data_example)
    print(f"✅ Success: {result}")
except NotImplementedError:
    print("⚠️ Implement the class above first!")

# Visual feedback (development only - not exported)
def show_results(data):
    """Show visual confirmation of working code."""
    plt.figure(figsize=(10, 6))
    # Visualization code
    plt.show()

if _should_show_plots():
    show_results(real_data)
```

### Step 4: Create Tests with Real Data
Create `modules/{module}/tests/test_{module}.py`:

```python
import pytest
import numpy as np
from {module}_dev import MainClass

def test_with_real_data():
    """Test with actual production data."""
    # Use real datasets, not mocks
    real_data = load_real_dataset()
    
    instance = MainClass(real_data)
    result = instance.process()
    
    # Test real properties
    assert result.shape == expected_real_shape
    assert result.dtype == expected_real_dtype
    # Test with actual data characteristics
```

### Step 5: Convert and Export
```bash
# Convert to notebook (using Jupytext)
tito module notebooks --module {module}

# Export to package
python bin/tito.py sync --module {module}

# Test everything
python bin/tito.py test --module {module}
```

## 🏷️ NBDev Directives

### Core Directives
- `#| default_exp core.{module}` - Sets export destination
- `#| export` - Marks code for export to package
- `#| hide` + `#| export` - Hidden implementation (instructor solution)
- `# %% [markdown]` - Markdown cells for explanations
- `# %%` - Code cells

### Educational Structure
- **Concept explanation** → **Implementation guidance** → **Hidden solution** → **Testing** → **Visual feedback**

## 🎨 Difficulty System

- **🟢 Easy (5-10 min)**: Constructor, properties, basic operations
- **🟡 Medium (10-20 min)**: Conditional logic, data processing, error handling  
- **🔴 Hard (20+ min)**: Complex algorithms, system integration, optimization

## 📋 Implementation Guidelines

### Students Implement (Core Learning)
- **Main functionality**: Core algorithms and data structures
- **Data processing**: Loading, preprocessing, batching
- **Error handling**: Input validation, type checking
- **Basic operations**: Mathematical operations, transformations

### Students Receive (Focus on Learning Goals)
- **Complex setup**: Download progress bars, caching systems
- **Utility functions**: Visualization, debugging helpers
- **Advanced features**: Optimization, GPU support
- **Infrastructure**: Test frameworks, import management

### TODO Guidance Quality
```python
"""
TODO: {Clear, specific task}

APPROACH:
1. {Concrete first step}
2. {Concrete second step}  
3. {Concrete third step}

EXAMPLE:
Input: {actual_data_example}
Expected: {concrete_expected_output}

HINTS:
- {Helpful guidance without giving code}
- {Systems thinking consideration}
- {Real-world connection}

SYSTEMS THINKING:
- {Performance consideration}
- {Scalability question}
- {User experience aspect}
"""
```

## 🗂️ Module Structure

```
modules/{module}/
├── {module}_dev.py              # 🔧 Complete implementation
├── {module}_dev.ipynb           # 📓 Generated notebook
├── tests/
│   └── test_{module}.py         # 🧪 Real data tests
├── README.md                    # 📖 Module guide
└── data/                        # 📊 Cached datasets (if needed)
```

## ✅ Quality Standards

### Before Release
- [ ] Uses real data, not synthetic/mock data
- [ ] Includes progress feedback for long operations
- [ ] Visual feedback functions (development only)
- [ ] Tests use actual datasets at realistic scales
- [ ] TODO guidance includes systems thinking
- [ ] Clean separation between development and exports
- [ ] Follows "Build → Use → Understand" progression

### Integration Requirements
- [ ] Exports correctly to `tinytorch.core.{module}`
- [ ] No circular dependencies
- [ ] Consistent with existing module patterns
- [ ] Compatible with TinyTorch CLI tools

## 💡 Best Practices

### Development Process
1. **Start with real data**: Choose production dataset first
2. **Write complete implementation**: Get it working before adding markers
3. **Add rich feedback**: Visual confirmation, progress indicators
4. **Test the student path**: Follow your own TODO guidance
5. **Optimize user experience**: Consider performance, caching, error messages

### Systems Thinking
- **Performance**: How does this scale with larger datasets?
- **Caching**: How do we avoid repeated expensive operations?
- **User Experience**: How do students know the code is working?
- **Production Relevance**: How does this connect to real ML systems?

### Educational Design
- **Immediate gratification**: Students see results quickly
- **Progressive complexity**: Build understanding step by step
- **Real-world connections**: Connect every concept to production
- **Visual confirmation**: Students see their code working

## 🔄 Continuous Improvement

After teaching with a module:
1. **Monitor student experience**: Where do they get stuck?
2. **Improve guidance**: Better TODO instructions, clearer hints
3. **Enhance feedback**: More visual confirmation, better progress indicators
4. **Optimize performance**: Faster data loading, better caching
5. **Update documentation**: Share learnings with other developers

## 🎯 Success Metrics

**Students should be able to:**
- Explain what they built in simple terms
- Modify code to solve related problems
- Connect module concepts to real ML systems
- Debug issues by understanding the system

**Modules should achieve:**
- High student engagement and completion rates
- Smooth progression to next modules
- Real-world relevance and production quality
- Consistent patterns across the curriculum

---

**Remember**: We're teaching ML systems engineering, not just algorithms. Every module should reflect real-world practices and challenges while maintaining the "Build → Use → Understand" educational cycle. 