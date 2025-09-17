# 🎓 TinyTorch Student Quickstart Guide

Welcome to TinyTorch! You're about to build an ML framework from scratch and understand ML systems engineering.

## 🚀 Getting Started (5 minutes)

### 1️⃣ **Setup Your Environment**
```bash
# Clone the repository
git clone https://github.com/MLSysBook/TinyTorch.git
cd TinyTorch

# Create virtual environment (MANDATORY)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
tito system doctor
```

✅ **You should see all green checkmarks!**

### 2️⃣ **Start Your First Module**
```bash
# View the first module
tito module view 01_setup

# Or open the notebook directly
jupyter notebook modules/source/01_setup/setup_dev.ipynb
```

## 📚 Learning Path

### **Module Progression**
Each module builds on the previous one:

| Module | What You'll Build | Capability Unlocked |
|--------|------------------|---------------------|
| 01 Setup | Development environment | Configure TinyTorch |
| 02 Tensor | Core data structure | Manipulate ML building blocks |
| 03 Activations | Non-linearity functions | Add intelligence to networks |
| 04 Layers | Neural network layers | Build network components |
| 05 Dense | Complete networks | Create multi-layer networks |
| 06 Spatial | Convolution operations | Process images |
| 07 Attention | Attention mechanisms | Understand sequences |
| 08 Dataloader | Data pipelines | Efficient data loading |
| 09 Autograd | Automatic differentiation | Compute gradients |
| 10 Optimizers | Training algorithms | Optimize networks |
| 11 Training | Complete training loops | End-to-end learning |
| 12 Compression | Model optimization | Deploy efficiently |
| 13 Kernels | Custom operations | Hardware acceleration |
| 14 Benchmarking | Performance analysis | Find bottlenecks |
| 15 MLOps | Production systems | Deploy and monitor |
| 16 TinyGPT | Language models | Build transformers |

## 📝 Interactive Learning

### **Answer ML Systems Questions**
Each module has 3 interactive questions where you write 150-300 word responses:

```python
# %% nbgrader={"grade": true, "grade_id": "ml_systems_q1", ...}
"""
YOUR RESPONSE HERE

[Write your analysis about memory usage, scaling, or system design]
"""
```

**Tips for Good Responses:**
- Reference the actual code you implemented
- Discuss memory/performance implications
- Compare to production systems (PyTorch, TensorFlow)
- Think about scaling to larger models

## 🎯 Track Your Progress

### **Check Your Capabilities**
```bash
# See overall progress
tito checkpoint status

# Visual timeline
tito checkpoint timeline

# Test specific capability
tito checkpoint test 01
```

### **Complete Modules**
```bash
# When you finish a module, validate it
tito module complete 02_tensor

# This will:
# 1. Export your code to the package
# 2. Run integration tests
# 3. Test your capability checkpoint
```

## 💡 Learning Tips

### **1. Build First, Understand Through Building**
Don't just read - type the code, run it, break it, fix it!

### **2. Test Immediately**
After each implementation, run the test right away:
```python
# Implementation
def my_function():
    return result

# Test immediately
assert my_function() == expected
print("✅ Test passed!")
```

### **3. Think About Systems**
For every operation, ask:
- How much memory does this use?
- What's the time complexity?
- How would this scale to 1B parameters?
- What would break in production?

### **4. Use the Profiler**
Many modules include profiling code:
```python
with MemoryProfiler() as prof:
    result = operation()
print(prof.report())
```

## 🆘 Getting Help

### **Module Issues**
```bash
# Check module status
tito module status

# Run tests for debugging
tito module test 02_tensor

# View detailed errors
tito module test 02_tensor --verbose
```

### **Environment Issues**
```bash
# Full system check
tito system doctor

# Reset if needed
tito system reset
```

### **Community**
- GitHub Issues: [Report problems](https://github.com/MLSysBook/TinyTorch/issues)
- Discussions: Ask questions and share insights

## 🏆 Challenge Yourself

### **After Each Module**
1. Run all tests successfully
2. Answer all ML Systems questions thoughtfully
3. Pass the capability checkpoint
4. Try modifying the code and see what breaks

### **Final Goal**
Complete all 16 modules and build TinyGPT - a working language model using the framework you built!

## 📊 Submission (For Courses)

If you're taking this as a course:

### **Submit Your Work**
```bash
# Your instructor will provide submission instructions
# Typically involves pushing to a specific repository
git add .
git commit -m "Complete module 02_tensor"
git push origin main
```

### **Grading**
Your work is graded on:
1. **Code Implementation** (auto-graded)
2. **Test Passing** (auto-graded)
3. **ML Systems Questions** (manually graded)
4. **Checkpoint Achievements** (auto-validated)

---

**Ready to build ML systems from scratch? Start with Module 01! 🚀**

```bash
tito module view 01_setup
```