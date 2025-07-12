# 🚀 Quick Module Reference

**Fast reference for module development - commands, patterns, and essential workflows.**

## 🔥 **Essential Commands**

### **System Commands**
```bash
tito system info              # System information and course navigation
tito system doctor            # Environment diagnosis
tito system jupyter           # Start Jupyter Lab
```

### **Module Commands**
```bash
tito module status            # Check all module status
tito module status --details  # Detailed file structure
tito module test --module X   # Test specific module
tito module test --all        # Test all modules
tito module notebooks --module X  # Convert Python to notebook (Jupytext)
```

### **Package Commands**
```bash
tito package sync            # Export all notebooks to package
tito package sync --module X # Export specific module
tito package reset           # Reset package to clean state
tito package nbdev --export  # Run nbdev export
```

## 🎯 **Development Workflow**

### **1. Module Planning**
- [ ] Choose real dataset (CIFAR-10, ImageNet, etc.)
- [ ] Define learning objectives and progression
- [ ] Identify production ML connections
- [ ] Plan visual feedback and progress indicators
- [ ] Decide what to provide vs. what students implement

### **2. Write Complete Implementation**
Create `modules/{module}/{module}_dev.py`:
```python
# %% [markdown]
# # Module: {Title}
# 
# ## 🎯 Learning Pattern: Build → Use → [Pattern]
# **Pattern Choice**: [Reflect/Analyze/Optimize]
# **Rationale**: [Why this pattern fits]
# 
# Learning objectives and overview

# %%
#| default_exp core.{module}

import numpy as np

# %%
#| export
class YourClass:
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
class YourClass:
    """Complete implementation (hidden from students)."""
    def __init__(self, params):
        # Actual working implementation
        pass
```

### **3. Design Pattern-Specific Activities**
- **Reflect**: Add reflection questions and trade-off analysis
- **Analyze**: Include profiling tools and debugging exercises
- **Optimize**: Create performance challenges and iteration tasks

### **4. Convert and Generate**
```bash
# Convert Python to notebook (using Jupytext)
tito module notebooks --module {module}

# Generate student version
python3 bin/generate_student_notebooks.py --module {module}
```

### **5. Test and Verify**
```bash
# Test both versions work
jupyter lab modules/{module}/{module}_dev.ipynb
jupyter lab modules/{module}/{module}_dev_student.ipynb

# Test integration
tito package sync --module {module}
tito module test --module {module}
```

## 🏷️ **Essential NBDev Directives**

| Directive | Purpose | Example |
|-----------|---------|---------|
| `#| default_exp core.{module}` | Set export destination | Top of file |
| `#| export` | Export to package | Classes/functions |
| `#| hide` | Hide from students | Instructor solutions |
| `#| hide #| export` | Export but hide | Complete implementations |

## 📁 **Module Structure**

```
modules/{module}/
├── {module}_dev.py          # Main development file (Jupytext format)
├── {module}_dev.ipynb       # Jupyter notebook (auto-generated)
├── module.yaml              # Simple metadata (name, title, description, etc.)
├── tests/
│   └── test_{module}.py     # Comprehensive pytest tests
└── README.md                # Module overview and usage
```

## ⚡ **Common Patterns**

### **Real Data Loading**
```python
# ✅ Good: Real data with progress feedback
def load_cifar10():
    """Load CIFAR-10 with progress bar."""
    from tqdm import tqdm
    # Show download progress
    # Cache for performance
    # Handle errors gracefully
```

### **Visual Feedback (Development Only)**
```python
# Development visualization (not exported)
def _show_results(data):
    """Show visual confirmation (development only)."""
    if not _in_development():
        return
    plt.figure(figsize=(10, 6))
    # Rich visualization
    plt.show()
```

### **Student Implementation Guidance**
```python
def method_to_implement(self):
    """
    TODO: Implement this method
    
    APPROACH:
    1. Parse input data and validate shapes
    2. Apply the core algorithm step by step
    3. Return results in expected format
    
    EXAMPLE:
    Input: tensor([1, 2, 3])
    Expected: tensor([2, 4, 6])
    
    HINTS:
    - Start with the simple case first
    - Think about edge cases (empty input, wrong shapes)
    - Use vectorized operations for performance
    """
    raise NotImplementedError("Student implementation required")
```

## 🧪 **Testing Patterns**

### **Test with Real Data**
```python
def test_with_real_data():
    """Test with actual production data."""
    # Load real dataset
    data = load_real_cifar10_sample()
    
    # Test with realistic parameters
    model = YourClass(realistic_params)
    result = model.process(data)
    
    # Verify real properties
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
```

### **Performance Testing**
```python
def test_performance():
    """Ensure reasonable performance."""
    import time
    
    large_data = create_realistic_large_dataset()
    start = time.time()
    result = process(large_data)
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    assert elapsed < 5.0  # 5 seconds max
```

## 🎯 **Quality Checklist**

### **Before Release**
- [ ] Uses real data throughout (no synthetic/mock data)
- [ ] Includes progress feedback for long operations
- [ ] Provides visual confirmation of working code
- [ ] Tests with realistic data scales
- [ ] Follows "Build → Use → [Pattern]" progression
- [ ] Comprehensive TODO guidance with examples
- [ ] Clean separation: rich development, clean exports

### **Integration Testing**
- [ ] Module exports correctly to `tinytorch.core.{module}`
- [ ] No circular import issues
- [ ] Compatible with existing modules
- [ ] Works with TinyTorch CLI tools
- [ ] Consistent with established patterns

---

**💡 Pro Tip**: Start with real data and production concerns first. Educational structure and TODO guidance come after you have working, realistic code. 