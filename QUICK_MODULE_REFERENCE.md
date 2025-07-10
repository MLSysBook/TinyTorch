# 🚀 TinyTorch Module Development - Quick Reference

## 📋 Development Checklist

### 1. Plan Module
- [ ] Define learning objectives (what students will implement)
- [ ] Choose difficulty levels (🟢 easy → 🟡 medium → 🔴 hard)
- [ ] Decide what to provide vs. what students implement

### 2. Write Complete Implementation
Create `modules/{module}/{module}_dev.py`:
```python
# %% [markdown]
# # Module: {Title}
# Learning objectives and overview

# %%
#| keep_imports
import numpy as np

# %%
class YourClass:
    #| exercise_start
    #| difficulty: easy
    #| hint: Clear guidance without giving away code
    #| solution_test: How students verify their work
    def method_to_implement(self):
        """Full signature and docstring."""
        # Complete working implementation
        pass
    #| exercise_end
```

### 3. Convert and Generate
```bash
# Convert Python to notebook
python3 tools/py_to_notebook.py modules/{module}/{module}_dev.py

# Generate student version
python3 bin/generate_student_notebooks.py --module {module}
```

### 4. Test and Verify
```bash
# Test both versions work
jupyter lab modules/{module}/{module}_dev.ipynb
jupyter lab modules/{module}/{module}_dev_student.ipynb

# Test integration
python bin/tito.py sync --module {module}
python bin/tito.py test --module {module}
```

## 🏷️ Essential Markers

| Marker | Purpose | Example |
|--------|---------|---------|
| `#| exercise_start/end` | Mark student implementation | Method body |
| `#| difficulty: easy\|medium\|hard` | Visual indicator | 🟢🟡🔴 |
| `#| hint:` | Guide student thinking | Multiple allowed |
| `#| solution_test:` | Verification guidance | Expected behavior |
| `#| keep_imports` | Preserve imports | Setup code |
| `#| keep_complete` | Keep full implementation | Utilities |
| `#| remove_cell` | Remove from student version | Instructor notes |

## 🎨 Difficulty Guidelines

- **🟢 Easy (5-10 min)**: Constructor, properties, basic operations
- **🟡 Medium (10-20 min)**: Conditional logic, shape manipulation  
- **🔴 Hard (20+ min)**: Complex algorithms, multiple concepts

## ✅ Quality Check

**Before release:**
- [ ] Complete version works and passes tests
- [ ] Student version preserves signatures and docstrings
- [ ] Hints are helpful but not prescriptive
- [ ] Tests provide clear verification guidance
- [ ] Exports correctly to tinytorch package

## 🔄 File Structure

```
modules/{module}/
├── {module}_dev.py              # 🔧 Write this first
├── {module}_dev.ipynb           # 📓 Generated 
├── {module}_dev_student.ipynb   # 🎓 Auto-generated
├── test_{module}.py             # 🧪 Test suite
└── README.md                    # 📖 Module guide
```

## 💡 Pro Tips

1. **Write complete implementation first** - Get it working before adding markers
2. **Test the student path** - Follow your own hints to verify they work
3. **Be generous with hints** - Better too helpful than too cryptic
4. **Preserve all signatures** - Students need to know the interface
5. **Progressive difficulty** - Start easy, build complexity

## 🛠️ Common Commands

```bash
# Create new module
mkdir modules/{module}
cp modules/example/example_dev.py modules/{module}/{module}_dev.py

# Full workflow
python3 tools/py_to_notebook.py modules/{module}/{module}_dev.py
python3 bin/generate_student_notebooks.py --module {module}

# Test everything
python bin/tito.py test --module {module}
```

*See `MODULE_DEVELOPMENT_GUIDE.md` for complete details.* 