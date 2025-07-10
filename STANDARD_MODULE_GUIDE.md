# 📖 TinyTorch Module Development - Standard Approach

**Using industry-standard tools instead of custom solutions.**

## 🎯 Why Standards Matter

Instead of inventing our own system, we use:
- **Jupytext**: Industry standard for Python ↔ Notebook conversion
- **NBDev-style directives**: Proven educational notebook patterns  
- **Percent format**: Supported by VSCode, PyCharm, Spyder, Hydrogen

## 🏗️ Standard Development Workflow

### Step 1: Write Python with Standard Cell Markers
Create `modules/{module}/{module}_dev.py`:

```python
# %% [markdown]
# # Module: Tensor Fundamentals
# 
# Build the core Tensor class that powers TinyTorch neural networks.

# %%
#| default_exp core.tensor
import numpy as np
from typing import Union, List, Optional

# %%
class Tensor:
    """TinyTorch Tensor: N-dimensional array with ML operations."""
    
    def __init__(self, data: Union[int, float, List, np.ndarray], dtype: Optional[str] = None):
        """
        Create a new tensor from data.
        
        Args:
            data: Input data (scalar, list, or numpy array)
            dtype: Data type ('float32', 'int32', etc.). Defaults to float32.
        """
        #| exercise_start
        # TODO: Convert input to numpy array
        # HINT: Use isinstance() to check input types  
        # HINT: Default to 'float32' for ML compatibility
        if isinstance(data, (int, float)):
            self._data = np.array(data, dtype=dtype or 'float32')
        elif isinstance(data, list):
            self._data = np.array(data, dtype=dtype or 'float32')
        elif isinstance(data, np.ndarray):
            self._data = data.astype(dtype or data.dtype)
        else:
            raise TypeError(f"Cannot create tensor from {type(data)}")
        #| exercise_end

# %% [markdown]
# ## Testing
# Let's verify our implementation works:

# %%
# Test tensor creation
scalar = Tensor(5.0)
vector = Tensor([1, 2, 3])
matrix = Tensor([[1, 2], [3, 4]])
print(f"Scalar: {scalar.shape}")
print(f"Vector: {vector.shape}")  
print(f"Matrix: {matrix.shape}")
```

### Step 2: Convert to Notebook (Standard Tool)
```bash
# Use Jupytext (industry standard)
jupytext --to ipynb modules/tensor/tensor_dev.py
```

### Step 3: Generate Student Version (Standard Directives)
```bash
# Use NBDev-style processing
nbdev_process --module tensor
```

## 🏷️ Standard Directives (NBDev Compatible)

### Educational Markers
```python
#| exercise_start
# Student implements this section
#| exercise_end

#| hide
# Hidden from students (instructor notes)

#| export  
# Exports to package (standard NBDev)

#| default_exp core.tensor
# Sets export target (standard NBDev)
```

### Cell Types (Jupytext Standard)
```python
# %% [markdown]
# Markdown content here

# %%
# Code cell (default)

# %% [raw]
# Raw cell content
```

## 🛠️ Using Standard Tools

### Install Standard Tools
```bash
pip install jupytext nbdev
```

### Standard Conversion
```bash
# Python → Notebook (Jupytext)
jupytext --to ipynb tensor_dev.py

# Generate docs (NBDev)  
nbdev_docs

# Export to package (NBDev)
nbdev_export
```

### Standard Configuration
Add to `pyproject.toml`:
```toml
[tool.jupytext]
formats = "ipynb,py:percent"

[tool.nbdev]
lib_name = "tinytorch"
lib_path = "tinytorch"
```

## 📋 Standard Format Example

```python
# %% [markdown]
# # Tensor Implementation
# Core data structure for TinyTorch

# %%
#| default_exp core.tensor
#| export
import numpy as np
from typing import Union, List

# %%
#| export
class Tensor:
    def __init__(self, data):
        #| exercise_start
        # TODO: Students implement this
        self._data = np.array(data)
        #| exercise_end
    
    #| hide
    def _internal_method(self):
        # Hidden from students
        pass

# %% [markdown]
# ## Testing

# %%
# Test our implementation
t = Tensor([1, 2, 3])
assert t._data.shape == (3,)
```

## ✅ Benefits of Standards

### For Instructors
- **No custom tools to maintain** - use battle-tested solutions
- **Wide IDE support** - works in VSCode, PyCharm, Spyder
- **Community compatibility** - follows established patterns  
- **Rich ecosystem** - plugins, extensions, integrations

### For Students  
- **Industry-standard workflow** - learns real development practices
- **Better tooling** - syntax highlighting, debugging, refactoring
- **Transferable skills** - applies to other projects
- **Community resources** - tutorials, documentation, help

## 🔧 Migration from Custom System

If you have existing custom markers:

```python
# Old custom approach
#| exercise_start
#| difficulty: easy
#| hint: Use numpy arrays
# ... implementation

# New standard approach  
#| exercise_start
# TODO: Use numpy arrays
# ... implementation
#| exercise_end
```

## 📚 Resources

- **Jupytext docs**: https://jupytext.readthedocs.io/
- **NBDev docs**: https://nbdev.fast.ai/
- **VS Code Jupyter**: Built-in support for `# %%` cells
- **PyCharm**: Native percent format support

## 🎉 Result

By using standards:
- ✅ **Less maintenance** - no custom tools to maintain
- ✅ **Better support** - works with existing IDE features
- ✅ **Future-proof** - follows community direction
- ✅ **Easier onboarding** - instructors know these tools

**Use proven tools instead of reinventing the wheel!** 