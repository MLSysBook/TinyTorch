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
# Module 0: Setup - Tiny🔥Torch Development Workflow

Welcome to TinyTorch! This module teaches you the development workflow you'll use throughout the course.

## Learning Goals
- Understand the nbdev notebook-to-Python workflow
- Write your first TinyTorch code
- Run tests and use the CLI tools
- Get comfortable with the development rhythm

## The TinyTorch Development Cycle

1. **Write code** in this notebook using `#| export` 
2. **Export code** with `python bin/tito.py sync --module setup`
3. **Run tests** with `python bin/tito.py test --module setup`
4. **Check progress** with `python bin/tito.py info`

Let's get started!
"""

# %%
#| default_exp core.utils

# Setup imports and environment
import sys
import platform
from datetime import datetime

print("🔥 TinyTorch Development Environment")
print(f"Python {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
"""
## Step 1: Understanding the Module → Package Structure

**🎓 Teaching vs. 🔧 Building**: This course has two sides:
- **Teaching side**: You work in `modules/setup/setup_dev.ipynb` (learning-focused)
- **Building side**: Your code exports to `tinytorch/core/utils.py` (production package)

**Key Concept**: The `#| default_exp core.utils` directive at the top tells nbdev to export all `#| export` cells to `tinytorch/core/utils.py`.

This separation allows us to:
- Organize learning by **concepts** (modules)  
- Organize code by **function** (package structure)
- Build a real ML framework while learning systematically

Let's write a simple "Hello World" function with the `#| export` directive:
"""

# %%
#| export
def hello_tinytorch():
    """
    A simple hello world function for TinyTorch.
    
    TODO: Make this return a more welcoming message about TinyTorch.
    """
    return "Hello from TinyTorch! 🔥"

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

# %%
#| hide
#| export
def hello_tinytorch():
    """A simple hello world function for TinyTorch."""
    return "🔥 Welcome to TinyTorch! Ready to build ML systems from scratch? Let's go! 🔥"

# %%
# Test the functions in the notebook
print(hello_tinytorch())
print(f"2 + 3 = {add_numbers(2, 3)}")

# %% [markdown]
"""
## Step 2: A Simple Class

Let's create a simple class that will help us understand system information. This is still basic, but shows how to structure classes in TinyTorch.
"""

# %%
#| export
class SystemInfo:
    """Simple system information class."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.machine = platform.machine()
    
    def __str__(self):
        return f"Python {self.python_version.major}.{self.python_version.minor} on {self.platform} ({self.machine})"
    
    def is_compatible(self):
        """Check if system meets minimum requirements."""
        return self.python_version >= (3, 8)

# %%
# Test the SystemInfo class
info = SystemInfo()
print(f"System: {info}")
print(f"Compatible: {info.is_compatible()}")

# %% [markdown]
"""
## Step 3: Try the Export Process

Now let's export our code! In your terminal, run:

```bash
python bin/tito.py sync --module setup
```

This will export the code marked with `#| export` to `tinytorch/core/utils.py`.

**What happens during export:**
1. nbdev scans this notebook for `#| export` cells
2. Extracts the Python code  
3. Writes it to `tinytorch/core/utils.py` (because of `#| default_exp core.utils`)
4. Handles imports and dependencies automatically

**🔍 Verification**: After export, check `tinytorch/core/utils.py` - you'll see your functions there with auto-generated headers pointing back to this notebook!
"""

# %% [markdown]
"""
## Step 4: Run Tests

After exporting, run the tests:

```bash
python bin/tito.py test --module setup
```

This will run all tests for the setup module and verify your implementation works correctly.

## Step 5: Check Your Progress

See your overall progress:

```bash
python bin/tito.py info
```

This shows which modules are complete and which are pending.
"""

# %% [markdown]
"""
## 🎉 Congratulations!

You've learned the TinyTorch development workflow:

1. ✅ Write code in notebooks with `#| export`
2. ✅ Export with `tito sync --module setup`  
3. ✅ Test with `tito test --module setup`
4. ✅ Check progress with `tito info`

**This is the rhythm you'll use for every module in TinyTorch.**

### Next Steps

Ready for the real work? Head to **Module 1: Tensor** where you'll build the core data structures that power everything else in TinyTorch.

**Development Tips:**
- Always test your code in the notebook first
- Export frequently to catch issues early  
- Read error messages carefully - they're designed to help
- When stuck, check if your code exports cleanly first

Happy building! 🔥
"""
