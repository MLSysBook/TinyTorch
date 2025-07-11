# 🚀 TinyTorch Module Development - Quick Reference

## 🧠 Choose Your Learning Pattern First

**Before writing any code, decide which engagement pattern fits your module:**

### **🤔 Build → Use → Reflect** (Design & Systems Thinking)
- **Best for**: Early modules, foundational concepts, design decisions
- **Focus**: "Why did we make this choice? What are the trade-offs?"
- **Activities**: Design comparisons, trade-off analysis, systems implications
- **Examples**: Setup, Tensor, Layers, Data

### **🔍 Build → Use → Analyze** (Technical Depth)
- **Best for**: Middle modules, performance-critical components
- **Focus**: "How does this actually work? Where are the bottlenecks?"
- **Activities**: Profiling, debugging, measurement, technical investigation
- **Examples**: Training, Autograd, Profiling, Benchmarking

### **⚡ Build → Use → Optimize** (Systems Iteration)
- **Best for**: Advanced modules, production-focused components
- **Focus**: "How can we make this better? How does it scale?"
- **Activities**: Performance improvement, scaling, iteration, optimization
- **Examples**: MLOps, Kernels, Compression, Distributed

### **Decision Questions**
1. **What type of learning is most important?** (Conceptual/Technical/Optimization)
2. **Where does this fit in progression?** (Early/Middle/Advanced)
3. **What skills do students need most?** (Design thinking/Debugging/Performance)

### **Document Your Choice**
```markdown
## 🎯 Learning Pattern: Build → Use → [Pattern]

**Pattern Choice**: [Reflect/Analyze/Optimize]
**Rationale**: [Why this pattern fits the learning objectives]
**Key Activities**:
- [Specific activity 1]
- [Specific activity 2]
- [Specific activity 3]
```

---

## 📋 Development Checklist

### 1. Plan Module
- [ ] **Choose engagement pattern** (Reflect/Analyze/Optimize)
- [ ] Define learning objectives (what students will implement)
- [ ] Choose difficulty levels (🟢 easy → 🟡 medium → 🔴 hard)
- [ ] Decide what to provide vs. what students implement

### 2. Write Complete Implementation
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

### 3. Design Pattern-Specific Activities
- **Reflect**: Add reflection questions and trade-off analysis
- **Analyze**: Include profiling tools and debugging exercises
- **Optimize**: Create performance challenges and iteration tasks

### 4. Convert and Generate
```bash
# Convert Python to notebook
python bin/tito.py notebooks --module {module}

# Generate student version
python3 bin/generate_student_notebooks.py --module {module}
```

### 5. Test and Verify
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

## 🎯 Pattern-Specific Activities

### **Reflection Activities**
- Design comparison questions
- Trade-off analysis exercises
- "What if we had different constraints?" scenarios
- Connection to production systems

### **Analysis Activities**
- Profiling memory usage and performance
- Debugging exercises with real issues
- Measurement and interpretation tasks
- Technical investigation challenges

### **Optimization Activities**
- Performance improvement challenges
- Scaling exercises with larger datasets
- Iteration on system design
- Benchmarking and comparison tasks

## ✅ Quality Check

**Before release:**
- [ ] **Pattern choice documented** with clear rationale
- [ ] **Pattern-specific activities** included and tested
- [ ] Complete version works and passes tests
- [ ] Student version preserves signatures and docstrings
- [ ] Hints are helpful but not prescriptive
- [ ] Tests provide clear verification guidance
- [ ] Exports correctly to tinytorch package

## 🔄 File Structure

```
modules/{module}/
├── {module}_dev.py              # 🔧 Write this first (with pattern choice)
├── {module}_dev.ipynb           # 📓 Generated 
├── {module}_dev_student.ipynb   # 🎓 Auto-generated
├── test_{module}.py             # 🧪 Test suite
└── README.md                    # 📖 Module guide (include pattern)
```

## 💡 Pro Tips

1. **Choose pattern first** - Everything else flows from this decision
2. **Write complete implementation first** - Get it working before adding markers
3. **Design pattern-specific activities** - Make them engaging and educational
4. **Test the student path** - Follow your own hints to verify they work
5. **Be generous with hints** - Better too helpful than too cryptic
6. **Document your pattern choice** - Help future developers understand your reasoning

## 🛠️ Common Commands

```bash
# Create new module
mkdir modules/{module}
cp modules/example/example_dev.py modules/{module}/{module}_dev.py

# Full workflow
python bin/tito.py notebooks --module {module}
python bin/generate_student_notebooks.py --module {module}

# Test everything
python bin/tito.py test --module {module}
```

*See [Module Development Guide](module-development-guide.md) for complete details and [Pedagogical Principles](../pedagogy/pedagogical-principles.md) for the full framework.* 