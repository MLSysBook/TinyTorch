# Unified Module Template for TinyTorch

## Philosophy
We combine deep educational content with consistent structure. The 5 C's framework is used as an optional quick reference, not as a replacement for comprehensive explanations.

## The Module Stencil

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
# Module XX: [Name] - [One-line Description]

[Welcome message and module overview]

## Learning Goals
- [Specific learning objective 1]
- [Specific learning objective 2]
- [Specific learning objective 3]

## Build → Use → Understand
1. **Build**: [What we're building]
2. **Use**: [How we'll use it]
3. **Understand**: [What we'll understand]
"""

# %% nbgrader={"grade": false, "locked": false, "schema_version": 3, "solution": false}
# Imports and setup

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/XX_name/name_dev.py`  
**Building Side:** Code exports to `tinytorch.core.name`

```python
# Final package structure example
from tinytorch.core.name import MainClass
```

**Why this matters:**
- **Learning:** [Educational benefit]
- **Production:** [How it relates to real systems]
- **Consistency:** [Package organization]
- **Foundation:** [How other modules depend on this]
"""

# %% [markdown]
"""
## Part 1: Understanding [Concept Name]

### [Theoretical/Mathematical Foundation]
[Deep dive into the theory - as detailed as needed]

### Why This Matters in ML/Neural Networks
[Connection to practical ML applications]

### Real-World Examples
[Domain-specific examples: Computer Vision, NLP, etc.]
"""

# %% [markdown]
"""
## Part 2: Design & Architecture

### Why Not Just Use [Alternative Library]?
[Justification for building from scratch]

### Design Requirements
[What our implementation needs to handle]

### Performance Considerations
[Memory, speed, optimization notes]

### Implementation Strategy
[How we'll approach building this]
"""

# %% [markdown]
"""
## Part 3: Implementation

### Implementation Checkpoint (Optional Quick Reference)
If helpful as a summary before coding:
- **What**: [Core concept in one line]
- **How**: [Code structure overview]
- **Like**: [Similar to PyTorch's X, TensorFlow's Y]
- **Must**: [Key constraints to remember]
- **Why**: [Impact on ML systems]
"""

# %% nbgrader={"grade": false, "locked": false, "schema_version": 3, "solution": true}
# Core Implementation with:
# - TODO: Clear task description
# - APPROACH: Step-by-step guide
# - EXAMPLE: Concrete example
# - HINTS: Implementation tips
# - BEGIN SOLUTION / END SOLUTION blocks

# %% [markdown]
"""
## Part 4: Testing Your Implementation

### [Test Category 1]
[Explanation of what we're testing and why]
"""

# %% nbgrader={"grade": true, "locked": true, "points": N, "schema_version": 3}
# Test implementation

# %% [markdown]
"""
## Module Summary

### ✅ What You've Built
[Summary of implementations]

### ✅ Key Learning Outcomes
[What students learned]

### ✅ Real-World Connection
[How this relates to production systems]

### 🚀 What's Next
[Preview of next module]
"""
```

## Key Principles

### 1. Educational Depth First
- Don't sacrifice explanation for structure
- It's okay to be verbose if it aids understanding
- Real-world examples are crucial

### 2. Avoid Duplication
- If content fits naturally in existing sections, don't repeat it
- The 5 C's is optional - use only if it adds value as a checkpoint
- Map concepts to where they naturally belong

### 3. Consistent Flow
```
Theory/Understanding → Design/Architecture → Implementation → Testing → Summary
```

### 4. Unique TinyTorch Elements to Preserve
- **"Where This Code Lives"**: Shows the learning-to-production path
- **Build → Use → Understand**: Our pedagogical framework
- **Real-world connections**: Links to PyTorch, TensorFlow, etc.
- **Performance considerations**: Systems thinking

### 5. When to Use the 5 C's
Use the condensed 5 C's checkpoint when:
- Starting a complex implementation
- Students need a quick reference
- It provides a helpful summary without duplication

Skip it when:
- The content is already covered in detail above
- It would be redundant
- The implementation is straightforward

## Module Categories

### Foundation Modules (01-05)
- More mathematical foundation
- Detailed explanations of core concepts
- Extensive "Why not just use X?" sections

### System Modules (06-10)
- Focus on architecture and design patterns
- Performance becomes more critical
- Integration between components

### Advanced Modules (11-14)
- Real-world optimizations
- Production considerations
- Industry best practices

## Flexibility Within Consistency

While maintaining this structure, each module can:
- Expand sections that are most relevant
- Add module-specific sections as needed
- Adjust depth based on complexity
- Include additional examples for difficult concepts

## The Goal

Create modules that:
1. Teach deeply, not just functionally
2. Connect theory to practice
3. Build systematic understanding
4. Prepare students for real ML engineering
5. Maintain consistent structure without sacrificing content

## Remember

**We're not trying to make modules shorter or more streamlined at the cost of education.**

We're organizing rich educational content in a consistent, learnable structure. If a module needs 1000 lines to properly teach a concept, that's perfectly fine. The structure helps students navigate, not limit content.