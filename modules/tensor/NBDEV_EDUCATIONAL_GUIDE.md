# NBDev Educational Module Guide for TinyTorch

## 🎓 Overview

This guide demonstrates how to use **NBDev's built-in educational features** instead of custom generators. NBDev already has powerful, mature capabilities for educational content - we just needed to discover them!

## 📁 Files Created

- **`tensor_nbdev_educational.py`** - Source Python file with `# %%` cell markers
- **`tensor_nbdev_educational.ipynb`** - Generated Jupyter notebook  
- **`NBDEV_EDUCATIONAL_GUIDE.md`** - This guide

## 🎯 NBDev Educational Directives Demonstrated

### Content Visibility Control

| Directive | Purpose | Student View | Instructor View |
|-----------|---------|--------------|-----------------|
| `#|hide` | Complete solutions | Hidden | Visible |
| `#|code-fold: show` | Collapsible code | Collapsed but expandable | Same |
| `#|code-fold: true` | Hidden by default | Hidden but expandable | Same |
| `#|filter_stream <keywords>` | Clean output | Warnings filtered | Same |
| `#|hide_line` | Hide specific lines | Line hidden | Visible |

### Export Control

| Directive | Purpose | Documentation | Package Export |
|-----------|---------|---------------|----------------|
| `#|export` | Standard export | Shows in docs | Goes to package |
| `#|exports` | Export + source | Shows code + docs | Goes to package |
| `#|exporti` | Internal export | Hidden from docs | Goes to package |

## 🔄 Development Workflow

### 1. Write Python First
```bash
# Create module in Python with # %% cell markers
vim modules/tensor/tensor_nbdev_educational.py
```

### 2. Convert to Notebook
```bash
# Use Jupytext (industry standard)
jupytext --to notebook tensor_nbdev_educational.py
```

### 3. Generate Documentation
```bash
# Use NBDev to build docs
nbdev_docs
```

### 4. Export to Package
```bash
# Export code to tinytorch package
nbdev_export
```

## 🎨 Educational Patterns

### Pattern 1: Progressive Revelation
```python
# %% [markdown]
"""
### 🎯 Your Task
Implement the Tensor class initialization.
"""

# %%
#| code-fold: show
def __init__(self, data):
    """
    TODO: Implement initialization
    """
    pass  # Student implements this

# %%
#| hide
def __init__(self, data):
    """Complete solution - hidden from students"""
    # Full implementation here
```

### Pattern 2: Clean Demos
```python
# %%
#| filter_stream FutureWarning DeprecationWarning
import numpy as np
# Students see clean output without warnings
```

### Pattern 3: Instructor Notes
```python
# %%
#| hide_line
print("This line only visible to instructors")
print("Students see this line")
```

## ✨ Advantages Over Custom Generator

### ✅ **Industry Standard**
- NBDev is used by fast.ai, Hugging Face, and many educational institutions
- Proven patterns from years of ML education

### ✅ **Rich Built-in Features**
- Interactive toggle buttons
- Beautiful syntax highlighting
- Automatic table of contents
- Cross-references and links

### ✅ **No Maintenance**
- No custom code to maintain
- Updates come from NBDev team
- Battle-tested and reliable

### ✅ **Single Source of Truth**
- One file with all content
- Directives control what's visible when
- No need to sync multiple versions

### ✅ **Beautiful Output**
- Professional documentation site
- Responsive design
- Mobile-friendly
- Search functionality

## 🎓 Educational Use Cases

### Beginner Mode
- Hide complex implementations with `#|hide`
- Show only essential concepts
- Clean output with `#|filter_stream`

### Intermediate Mode  
- Use `#|code-fold: show` for optional details
- Progressive complexity revelation
- Students choose their depth

### Instructor Mode
- All content visible
- Hidden solutions accessible
- Additional teaching notes

### Assessment Mode
- Hide solutions completely
- Provide clear TODOs
- Include test cells for verification

## 🚀 Getting Started

1. **Install NBDev**
   ```bash
   pip install nbdev
   ```

2. **Copy the Template**
   ```bash
   cp modules/tensor/tensor_nbdev_educational.py modules/[module]/[module]_dev.py
   ```

3. **Customize Content**
   - Replace tensor-specific content
   - Add your educational directives
   - Structure for your learning goals

4. **Convert and Build**
   ```bash
   jupytext --to notebook [module]_dev.py
   nbdev_docs
   ```

## 📚 Further Reading

- [NBDev Directives Documentation](https://nbdev.fast.ai/explanations/directives.html)
- [Quarto Educational Features](https://quarto.org/docs/authoring/notebook-filters.html)
- [Jupyter Book vs NBDev Comparison](https://fastai.github.io/jb-nbdev/)

## 🎯 Next Steps for TinyTorch

1. **Convert Existing Modules**
   - Apply this pattern to autograd, mlp, cnn modules
   - Use appropriate educational directives for each

2. **Build Course Structure**
   - Use NBDev's documentation features
   - Create learning paths with cross-references
   - Add interactive examples

3. **Advanced Features**
   - Cell tags for different difficulty levels
   - Conditional execution based on student level
   - Integration with autograders

## 💡 Key Insight

**We don't need to build custom educational tools when NBDev already provides mature, powerful capabilities specifically designed for educational content.**

The beauty is in leveraging existing, proven patterns rather than reinventing the wheel! 