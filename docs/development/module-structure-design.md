# TinyTorch Module Structure Design Document

## Overview

This document defines the standard structure for TinyTorch educational modules, ensuring consistency, educational effectiveness, and maintainability across all components.

## Module Architecture Philosophy

### Core Principles

1. **Educational First**: Every module is designed for learning, not just functionality
2. **Progressive Complexity**: Start simple, build complexity step by step
3. **Real-World Connection**: Connect concepts to practical ML applications
4. **Standalone Learning**: Each module should be self-contained
5. **Professional Standards**: Use industry-standard patterns and practices

### "Build → Use → Understand" Framework

Each module follows this pedagogical pattern:
- **Build**: Implement the component from scratch
- **Use**: Apply it to real data and problems
- **Understand**: Analyze behavior, trade-offs, and connections

## Standard Module Structure

### File Organization

```
modules/source/{module_name}/
├── {module_name}_dev.py           # Main development file (Jupytext format)
├── README.md                      # Module documentation and guide
├── tests/                         # Module-specific tests (if needed)
│   └── test_{module_name}.py      # Comprehensive test suite
├── data/                          # Module-specific data files (if needed)
│   └── sample_data.npy
└── assets/                        # Images, diagrams, etc. (if needed)
    └── architecture_diagram.png
```

### Development File Structure (`*_dev.py`)

Every module development file follows this standardized structure:

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
# Module {N}: {Title} - {Brief Description}

## 🎯 Learning Objectives
- ✅ Build {core_concept} from scratch
- ✅ Use it with real data ({specific_dataset})
- ✅ Understand {key_insight}
- ✅ Connect to {next_module} and production systems

## 📚 What You'll Learn
- **Conceptual**: {concept_explanation}
- **Technical**: {implementation_details}
- **Practical**: {real_world_applications}

## 🛠️ What You'll Build
- **Core Component**: {main_class_or_function}
- **Supporting Functions**: {helper_functions}
- **Integration Points**: {connections_to_other_modules}

## 📊 Module Info
- **Difficulty**: {⭐⭐⭐} (1-5 stars)
- **Time Estimate**: {X-Y hours}
- **Prerequisites**: {previous_modules}
- **Next Steps**: {next_modules}
"""

# %%
#| default_exp core.{module_name}

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Module-specific imports
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# %% [markdown]
"""
## Step 1: Conceptual Foundation

### What is {Concept}?

**Definition**: {Clear, simple definition with examples}

**Why it matters**: {Real-world motivation and ML context}

**How it works**: {Intuitive explanation before math}

**Visual examples**: {Concrete examples, diagrams, analogies}

**Connection**: {How it builds on previous modules}

### Mathematical Foundation

{Mathematical concepts explained intuitively}

### Real-World Applications

{Specific examples in ML and AI}
"""

# %% [markdown]
"""
## Step 2: Implementation Planning

### Design Decisions

Before we implement, let's think about:
1. **Interface Design**: How should users interact with this component?
2. **Data Structures**: What internal representation makes sense?
3. **Error Handling**: What can go wrong and how do we handle it?
4. **Performance**: What are the computational considerations?
5. **Integration**: How does this connect to other modules?

### Implementation Strategy

We'll build this component in stages:
1. **Core Functionality**: {basic_implementation}
2. **Enhanced Features**: {advanced_features}
3. **Integration Points**: {connections}
4. **Optimization**: {performance_improvements}
"""

# %% [markdown]
"""
## Step 3: Core Implementation

### {Component Name}

Let's implement the core component step by step.
"""

# %%
#| export
class {ComponentName}:
    """
    {Component description and purpose}
    
    This class implements {specific_functionality} for the TinyTorch framework.
    
    Args:
        {parameter_descriptions}
    
    Example:
        >>> {usage_example}
    
    Note:
        {important_notes_or_warnings}
    """
    
    def __init__(self, {parameters}):
        """
        Initialize the {component_name}.
        
        TODO: Implement initialization logic
        
        APPROACH:
        1. {step_1_description}
        2. {step_2_description}
        3. {step_3_description}
        
        EXAMPLE:
        Input: {input_example}
        Expected: {expected_behavior}
        
        HINTS:
        - {hint_1}
        - {hint_2}
        - {hint_3}
        """
        ### BEGIN SOLUTION
        {instructor_implementation}
        ### END SOLUTION
    
    def {method_name}(self, {parameters}) -> {return_type}:
        """
        {Method description}
        
        TODO: Implement {method_functionality}
        
        APPROACH:
        1. {implementation_step_1}
        2. {implementation_step_2}
        3. {implementation_step_3}
        
        EXAMPLE:
        Input: {concrete_input_example}
        Expected output: {concrete_output_example}
        Your code should: {specific_behavior_description}
        
        HINTS:
        - {specific_hint_1}
        - {specific_hint_2}
        - {specific_hint_3}
        """
        ### BEGIN SOLUTION
        {instructor_implementation}
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Comprehensive Test: {Component Name}

Let's test our implementation thoroughly to make sure it works correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-{component}-comprehensive", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
import pytest

class Test{ComponentName}:
    """Comprehensive test suite for {ComponentName}."""
    
    def test_initialization(self):
        """Test component initialization."""
        # Test basic initialization
        component = {ComponentName}({basic_params})
        assert {basic_assertion}
        
        # Test with different parameters
        component2 = {ComponentName}({different_params})
        assert {different_assertion}
    
    def test_core_functionality(self):
        """Test core component functionality."""
        component = {ComponentName}({params})
        
        # Test basic operation
        result = component.{method_name}({input_data})
        expected = {expected_result}
        assert {assertion}, f"Expected {expected}, got {result}"
        
        # Test with different inputs
        result2 = component.{method_name}({different_input})
        assert {different_assertion}
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        component = {ComponentName}({params})
        
        # Test empty input
        {edge_case_tests}
        
        # Test large input
        {large_input_tests}
        
        # Test invalid input
        with pytest.raises({ExpectedException}):
            component.{method_name}({invalid_input})
    
    def test_integration(self):
        """Test integration with other components."""
        {integration_tests}

def run_comprehensive_tests():
    """Run all tests with educational feedback."""
    print("🔬 Running comprehensive {component_name} tests...")
    
    test_class = Test{ComponentName}()
    tests = [
        ('Initialization', test_class.test_initialization),
        ('Core Functionality', test_class.test_core_functionality),
        ('Edge Cases', test_class.test_edge_cases),
        ('Integration', test_class.test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All {component_name} tests passed!")
        print("📈 Progress: {ComponentName} ✓")
        return True
    else:
        print("⚠️  Some tests failed - check your implementation")
        return False

# Execute tests
success = run_comprehensive_tests()

# %% [markdown]
"""
## Step 4: Real-World Application

### Using {ComponentName} with Real Data

Let's see how our component works with actual data from {dataset_name}.
"""

# %%
# Load real data for demonstration
{real_data_loading_code}

# Apply our component
print("🔬 Testing with real data...")
component = {ComponentName}({real_params})
result = component.{method_name}(real_data)

print(f"✅ Real data processing successful!")
print(f"Input shape: {real_data.shape}")
print(f"Output shape: {result.shape}")
print(f"Sample output: {result[:5]}")  # Show first 5 elements

# %% [markdown]
"""
### Visualization and Analysis

Let's visualize what our component does to understand it better.
"""

# %%
# Create visualization
plt.figure(figsize=(12, 4))

# Input visualization
plt.subplot(1, 3, 1)
{input_visualization_code}
plt.title('Input Data')

# Process visualization
plt.subplot(1, 3, 2)
{process_visualization_code}
plt.title('{Component} Processing')

# Output visualization
plt.subplot(1, 3, 3)
{output_visualization_code}
plt.title('Output Data')

plt.tight_layout()
plt.show()

print("📊 Visualization shows how {component_name} transforms the data")

# %% [markdown]
"""
## Step 5: Integration and Next Steps

### Connection to Other Modules

This {component_name} connects to the broader TinyTorch ecosystem:

- **Previous modules**: {previous_connections}
- **Next modules**: {next_connections}
- **Production use**: {production_applications}

### Performance Considerations

{performance_analysis}

### Advanced Features (Optional)

{advanced_features_description}
"""

# %% [markdown]
"""
## 🎯 Module Summary

### What You've Built
- ✅ **{ComponentName}**: {achievement_1}
- ✅ **Real Data Integration**: {achievement_2}
- ✅ **Comprehensive Testing**: {achievement_3}
- ✅ **Visualization**: {achievement_4}

### Key Insights
- **Technical**: {technical_insight}
- **Practical**: {practical_insight}
- **Conceptual**: {conceptual_insight}

### Next Steps
- **Immediate**: {next_immediate_step}
- **Advanced**: {next_advanced_step}
- **Integration**: {next_integration_step}

### Success Criteria
Your module is complete when:
1. **All tests pass**: Comprehensive testing shows everything works
2. **Real data works**: Component processes actual ML data correctly
3. **Integration ready**: Component exports to `tinytorch.core.{module_name}`
4. **Understanding**: You can explain how and why it works

Ready to move to the next module? Let's go! 🚀
"""
```

## README Structure

Every module should have a comprehensive README following this template:

```markdown
# {Module Name} Module

## 📊 Module Info
- **Difficulty**: {⭐⭐⭐} (1-5 stars)
- **Time Estimate**: {X-Y hours}
- **Prerequisites**: {previous_modules}
- **Next Steps**: {next_modules}

## Overview

{Brief description of what this module teaches and why it matters}

## Learning Goals

{Specific learning objectives}

## What You'll Implement

{Detailed description of components to build}

## Files

{Description of all files in the module}

## Usage

{Code examples showing how to use the module}

## Testing

{Instructions for running tests}

## Development Workflow

{Step-by-step development process}

## Key Concepts

{Important concepts and takeaways}

## Troubleshooting

{Common issues and solutions}
```

## Testing Integration

### Comprehensive Notebook Testing

Each module includes comprehensive tests within the notebook:

1. **Immediate Feedback**: Tests run as students implement
2. **Educational Context**: Tests explain what they're checking
3. **Professional Structure**: Uses pytest patterns
4. **Visual Feedback**: Clear pass/fail indicators
5. **Progress Tracking**: Shows completion status

### Test Categories

1. **Initialization Tests**: Component creation and setup
2. **Functionality Tests**: Core operations and methods
3. **Edge Case Tests**: Boundary conditions and error handling
4. **Integration Tests**: Connections to other modules
5. **Real Data Tests**: Performance with actual datasets

## Visual Design Guidelines

### Progress Indicators
- 🔬 Testing phase
- ✅ Success indicators
- ❌ Failure indicators
- 📊 Results summary
- 🎉 Completion celebration
- 📈 Progress tracking

### Educational Formatting
- **Bold** for key concepts
- `Code` for technical terms
- > Quotes for important notes
- Lists for step-by-step processes
- Tables for comparisons

## Data Integration Standards

### Real Data Requirements
- Use production datasets (CIFAR-10, ImageNet, etc.)
- Include data loading and preprocessing
- Show performance with realistic scales
- Demonstrate practical applications

### Visualization Standards
- Input/Process/Output flow diagrams
- Before/after comparisons
- Performance metrics
- Error analysis plots

## Export and Integration

### NBDev Integration
- `#| default_exp core.{module_name}` for package destination
- `#| export` for production code
- `#| hide` for instructor solutions
- Proper imports and dependencies

### Package Structure
```
tinytorch/
├── core/
│   ├── {module_name}.py    # Exported module code
│   └── __init__.py         # Package initialization
└── __init__.py             # Main package init
```

## Quality Checklist

### Before Module Completion
- [ ] All learning objectives addressed
- [ ] Comprehensive tests implemented and passing
- [ ] Real data integration working
- [ ] Visualization and analysis included
- [ ] README documentation complete
- [ ] Code exports to package correctly
- [ ] Integration with other modules tested
- [ ] Performance considerations addressed

### Educational Quality
- [ ] Concepts explained clearly
- [ ] Step-by-step implementation guidance
- [ ] Real-world connections made
- [ ] Visual learning aids included
- [ ] Progressive complexity maintained
- [ ] Student success criteria defined

## Examples

### Tensor Module Structure
```python
# Core tensor operations with comprehensive testing
# Real data integration with NumPy arrays
# Visual demonstrations of tensor operations
# Integration with activation functions
```

### Activation Module Structure
```python
# Mathematical foundations explained
# Multiple activation functions implemented
# Real neural network data processing
# Visualization of activation behaviors
```

### Layer Module Structure
```python
# Linear algebra foundations
# Dense layer implementation
# Real image classification example
# Integration with tensor and activation modules
```

## Conclusion

This standardized module structure ensures:
- **Consistency** across all TinyTorch modules
- **Educational effectiveness** through proven patterns
- **Professional quality** with industry standards
- **Maintainability** through clear organization
- **Scalability** for future module additions

Every module following this structure provides students with a complete, professional learning experience that builds both understanding and practical skills. 