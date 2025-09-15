# TinyTorch Module Standard Template
**For Agent Implementation**

## Module Structure - FOLLOW THIS EXACTLY

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
# Module XX: [Name] - [Descriptive Subtitle]

[Welcome message - 1-2 sentences that excite students]

## Learning Goals
- [Technical skill they'll learn]
- [Implementation they'll complete]
- [Concept they'll understand]
- [Integration with other modules]
- Master the NBGrader workflow with comprehensive testing

## Build → Use → Understand
1. **Build**: [What we're creating]
2. **Use**: [How we'll apply it]
3. **Understand**: [Deep insight we'll gain]
"""

# %% nbgrader={"grade": false, "grade_id": "[module]-imports", "locked": false, "schema_version": 3, "solution": false}
#| default_exp core.[module_name]

#| export
import numpy as np
import sys
# [Other imports as needed]

# %% nbgrader={"grade": false, "grade_id": "[module]-setup", "locked": false, "schema_version": 3, "solution": false}
print("🔥 TinyTorch [Module Name] Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to [action]!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/XX_name/name_dev.py`  
**Building Side:** Code exports to `tinytorch.core.name`

```python
# Final package structure:
from tinytorch.core.name import [MainClass]
# [Show how it relates to other modules]
```

**Why this matters:**
- **Learning:** [Educational organization benefit]
- **Production:** [How real frameworks organize this]
- **Consistency:** [Why grouping makes sense]
- **Foundation/Integration:** [How other modules depend on/use this]
"""

# %% [markdown]
"""
## Part 1: Concept - What is [Topic]?

### Definition
[Core explanation of what we're building]

### The Problem / Why We Need This
[What problem this solves in ML]

### Core Principles
[Key ideas students must understand]
"""

# %% [markdown]
"""
## Part 2: Foundations - Mathematical & Theoretical Background

### Mathematical Foundation
[Build from simple to complex]
- **Level 1**: [Basic concept]
- **Level 2**: [Intermediate]
- **Level 3**: [Advanced]

### How This Works in ML
[Connection to neural networks/ML systems]
"""

# %% [markdown]
"""
## Part 3: Context - Why This Matters

### In Neural Networks
[Specific role in NNs]

### In Real ML Systems
- **PyTorch**: [How PyTorch does it]
- **TensorFlow**: [How TF does it]
- **Industry**: [Real-world usage]

### Performance Implications
[Why efficiency matters here]
"""

# %% [markdown]
"""
## Part 4: Connections - Real-World Examples

### Computer Vision
[If applicable - how it's used in CV]

### Natural Language Processing
[If applicable - how it's used in NLP]

### [Other Domain]
[Other relevant applications]
"""

# %% [markdown]
"""
## Part 5: Design - Why Build From Scratch?

### Why Not Just Use [NumPy/PyTorch/etc]?

#### Educational Value
- [What we learn by building]
- [Concepts we understand deeply]

#### Control & Customization
- [What control we gain]
- [Future extensions possible]

#### Understanding Production Systems
- [How this helps understand real frameworks]
"""

# %% [markdown]
"""
## Part 6: Architecture - Design Decisions

### Requirements
Our implementation must:
1. [Requirement 1 with reason]
2. [Requirement 2 with reason]
3. [Requirement 3 with reason]

### Design Choices
- **Choice 1**: [Decision and why]
- **Choice 2**: [Decision and why]

### Trade-offs
- **Performance vs Simplicity**: [Our choice]
- **Memory vs Speed**: [Our choice]
"""

# %% [markdown]
"""
## Part 7: Implementation - Building [Module Name]

We'll implement this in steps:
1. [First component]
2. [Second component]
3. [Integration and testing]
"""

# For EACH implementation section:

# %% [markdown]
"""
### Step N: [Component Name]

#### What We're Building
[Explanation of this specific component]

#### Key Concepts
[Important ideas for this implementation]
"""

# %% nbgrader={"grade": false, "grade_id": "[component]-implementation", "locked": false, "schema_version": 3, "solution": true}
#| export
class ComponentName:
    """
    [Class description]
    
    [What it does in the system]
    """
    
    def method_name(self, params):
        """
        [Method description]
        
        TODO: [Specific implementation task]
        
        APPROACH:
        1. [Step 1 with reasoning]
        2. [Step 2 with reasoning]
        3. [Step 3 with reasoning]
        
        EXAMPLE:
        ```python
        # Show concrete usage
        obj = ComponentName()
        result = obj.method_name(input)
        # Expected: [output]
        ```
        
        HINTS:
        - [Implementation hint 1]
        - [Implementation hint 2]
        - [Watch out for edge case]
        """
        ### BEGIN SOLUTION
        # Implementation here
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Immediate Test: [Component Name]

Let's verify your implementation works correctly:
"""

# %% nbgrader={"grade": true, "grade_id": "test-[component]-immediate", "locked": true, "points": 5, "schema_version": 3}
# Immediate test with educational feedback
def test_immediate_component():
    """Test [component] immediately after implementation."""
    print("🔬 Testing [component]...")
    
    # Test code
    # assert statements
    
    print("✅ [Component] works perfectly!")
    print("🎯 Key insight: [What this test validates]")

test_immediate_component()

# [REPEAT Part 7 pattern for each component]

# %% [markdown]
"""
## Part 8: Integration - Bringing It Together

### How Components Work Together
[Explain integration]

### Complete Example
[Show full usage example]
"""

# %% [markdown]
"""
## Part 9: Testing - Comprehensive Validation

### Test Categories
1. **Unit Tests**: Individual components
2. **Integration Tests**: Components working together
3. **Edge Cases**: Boundary conditions
4. **Performance Tests**: Efficiency validation
"""

# %% nbgrader={"grade": true, "grade_id": "test-comprehensive-1", "locked": true, "points": 10, "schema_version": 3}
def test_comprehensive_component():
    """Comprehensive test of all functionality."""
    print("🔬 Running comprehensive tests...")
    
    # Multiple test cases
    # Various assertions
    
    print("✅ All comprehensive tests passed!")

test_comprehensive_component()

# %% [markdown]
"""
## Part 10: Module Summary

### ✅ What You've Built
- **[Component 1]**: [What it does]
- **[Component 2]**: [What it does]
- **[Integration]**: [How they work together]

### ✅ Key Learning Outcomes
- **Understood**: [Concept mastered]
- **Implemented**: [What you built]
- **Applied**: [How you used it]

### ✅ Foundations Mastered
- **Mathematical**: [Math concepts learned]
- **Algorithmic**: [Algorithms understood]
- **Systems**: [Systems concepts grasped]

### 🔗 Connection to Production ML
Your implementation mirrors:
- **PyTorch**: `torch.nn.[equivalent]`
- **TensorFlow**: `tf.[equivalent]`
- **Industry Standard**: [How it's used in production]

### 🚀 What's Next
[Preview of next module and connection]

You've mastered [concept]! Next, we'll [what comes next].
"""
```

## CRITICAL RULES FOR AGENTS

### 1. ALWAYS Include These Sections
- Module header with Learning Goals and Build→Use→Understand
- 📦 Where This Code Lives 
- All 10 Parts in order
- Immediate tests after EACH implementation
- Comprehensive tests at the end
- Module summary

### 2. Maintain 1:1 Markdown to Code Ratio
- EVERY code cell must have a preceding markdown cell
- Markdown explains what's about to happen
- Code implements it
- Test validates it

### 3. Use Exact Part Numbers and Names
```
Part 1: Concept
Part 2: Foundations  
Part 3: Context
Part 4: Connections
Part 5: Design
Part 6: Architecture
Part 7: Implementation
Part 8: Integration
Part 9: Testing
Part 10: Module Summary
```

### 4. NBGrader Metadata Pattern
- Implementation cells: `{"grade": false, "solution": true, "locked": false}`
- Test cells: `{"grade": true, "locked": true, "points": N, "solution": false}`
- grade_id format: `"[module]-[component]-[type]"`

### 5. Educational Patterns
- Build from simple to complex
- Test immediately for feedback
- Use emojis for success (✅, 🔥, 🎯, 🚀)
- Connect to real frameworks (PyTorch, TensorFlow)
- Explain WHY, not just HOW

### 6. Flexibility Within Structure
- Parts can be expanded for complex topics
- Parts can be brief for simple topics
- But ALL parts must be present
- Add subsections as needed within parts

## Quick Reference for Module Numbers

```
01_setup         - System Configuration
02_tensor        - Core Data Structure
03_activations   - Nonlinearity Functions
04_layers        - Neural Network Layers
05_networks      - Complete Networks
06_cnn          - Convolutional Networks
07_dataloader   - Data Loading & Batching
08_autograd     - Automatic Differentiation
09_optimizers   - Training Optimizers
10_training     - Training Loop
11_compression  - Model Compression
12_kernels      - Custom Kernels
13_benchmarking - Performance Testing
14_mlops        - Production Systems
```

## Example Section Headers to Copy

```markdown
## Part 1: Concept - What is [Topic]?
## Part 2: Foundations - Mathematical & Theoretical Background
## Part 3: Context - Why This Matters
## Part 4: Connections - Real-World Examples
## Part 5: Design - Why Build From Scratch?
## Part 6: Architecture - Design Decisions
## Part 7: Implementation - Building [Module Name]
## Part 8: Integration - Bringing It Together
## Part 9: Testing - Comprehensive Validation
## Part 10: Module Summary
```

Use these EXACTLY for consistency across all modules.