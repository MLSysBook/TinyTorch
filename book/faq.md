# Frequently Asked Questions

## 🤔 Getting Started Questions

### **Installation & Setup**

**Q: I'm getting "tito: command not found" - what's wrong?**

A: This usually means your virtual environment isn't activated or TinyTorch isn't installed:

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install TinyTorch
pip install -e .

# 3. Verify installation
tito system doctor
```

**Q: What Python version do I need?**

A: Python 3.8 or higher. Check with:
```bash
python --version  # Should show 3.8+
```

**Q: Can I use conda instead of venv?**

A: Yes! Replace the venv setup with:
```bash
conda create -n tinytorch python=3.9
conda activate tinytorch
pip install -r requirements.txt && pip install -e .
```

**Q: The installation is taking forever - is this normal?**

A: Initial setup typically takes 2-5 minutes depending on your connection. The main time is downloading NumPy, Jupyter, and other scientific packages.

---

## 📚 Learning Questions

### **Course Structure**

**Q: How long does TinyTorch take to complete?**

A: Depends on your goals and pace:

| **Goal** | **Time** | **Coverage** | **What You'll Build** |
|----------|----------|--------------|----------------------|
| **Quick Taste** | 15 minutes | Demo + overview | See framework in action |
| **Weekend Project** | 8-12 hours | Modules 1-6 | Neural network solver |
| **Neural Networks** | 4 weeks | Modules 1-8 | MNIST classifier |
| **Computer Vision** | 6 weeks | Modules 1-10 | CIFAR-10 CNN |
| **Language Models** | 8 weeks | Modules 1-14 | TinyGPT generator |
| **Full Framework** | 12 weeks | All 20 modules | Production-ready system |

**Q: Do I need machine learning experience to start?**

A: **No!** TinyTorch teaches ML systems from fundamentals. You need:

**✅ Required:**
- Basic Python (functions, classes, imports)
- High school math (multiplication, basic algebra)
- Curiosity about how things work

**❌ Not Required:**
- Previous ML experience
- Deep learning knowledge  
- Advanced mathematics
- PyTorch/TensorFlow experience

**Q: Can I skip modules or do them out of order?**

A: **No** - the progression is carefully designed:
- Each module builds on previous implementations
- Later modules import code from earlier ones
- Checkpoints verify prerequisites are met
- Skipping creates import errors and broken functionality

**Example:** Module 6 (Autograd) requires your Tensor class from Module 2. Skipping Module 2 breaks everything that follows.

**Q: What if I get stuck on a difficult concept?**

A: Multiple support options:

1. **Interactive Help**: `tito help --interactive` for personalized guidance
2. **Module README**: Each module has detailed explanations
3. **Community Support**: Join leaderboard for peer help
4. **Troubleshooting**: `tito help troubleshooting` for common issues
5. **Office Hours**: If taking as a course, use instructor support

### **Learning Methods**

**Q: Should I read everything before coding, or jump right into coding?**

A: **Jump into coding!** TinyTorch uses active learning:
- Read just enough to understand the task
- Start implementing immediately
- Learn through building and testing
- Explanations become clearer after you've tried the code

**Q: How much time should I spend on each module?**

A: Varies by module and experience:

| **Module Type** | **Typical Time** | **Examples** |
|----------------|------------------|--------------|
| **Foundation** | 2-4 hours | Tensors, Activations |
| **Architecture** | 3-5 hours | Layers, Training |
| **Advanced** | 4-6 hours | Attention, Transformers |
| **Optimization** | 2-3 hours | Profiling, Benchmarking |

**Don't rush!** Deep understanding matters more than speed.

**Q: What's the difference between modules and checkpoints?**

A: **Modules** = Building, **Checkpoints** = Validating

| **Modules** | **Checkpoints** |
|-------------|-----------------|
| 20 hands-on coding sessions | 16 capability assessments |
| You build implementations | Tests verify understanding |
| `tito module complete 05` | `tito checkpoint test 05` |
| Export code to framework | Validate you achieved capability |

**Workflow:** Complete module → Export implementation → Checkpoint test validates learning

---

## 🛠️ Technical Questions

### **Development Workflow**

**Q: Why can't I edit files in the `tinytorch/` directory?**

A: Those files are **auto-generated** from your source modules:

**✅ Edit Here:**
```
modules/source/02_tensor/tensor_dev.py  ← Your source code
```

**❌ Don't Edit:**
```
tinytorch/core/tensor.py  ← Generated from source
```

**Workflow:**
1. Edit source: `modules/source/0X_name/name_dev.py`
2. Export: `tito module complete 0X_name`
3. Uses your code: `from tinytorch.core.name import Component`

**Q: What's the difference between .py and .ipynb files?**

A: **TinyTorch uses .py files only** for all development:

- **Source**: `tensor_dev.py` (edit this)
- **Generated**: `tensor_dev.ipynb` (auto-created from .py)
- **Never edit**: `.ipynb` files directly

**Why .py only?**
- Clean version control (no JSON metadata)
- Professional development practices
- Consistent environment across contributors
- Easy code review and collaboration

**Q: My tests are failing after implementing a function - what's wrong?**

A: Common debugging steps:

1. **Check syntax**: Run the module file directly
   ```bash
   python modules/source/03_activations/activations_dev.py
   ```

2. **Verify function signature**: Make sure your function matches the expected interface
   ```python
   # Expected
   def relu(x: np.ndarray) -> np.ndarray:
   
   # Not this
   def relu(x):  # Missing type hints
   ```

3. **Test incrementally**: Run tests after each function
   ```bash
   tito checkpoint test 02 --verbose
   ```

4. **Check imports**: Ensure NumPy is imported as `np`

**Q: How do I run just one test instead of all tests?**

A: Use specific test commands:

```bash
# Test specific checkpoint
tito checkpoint test 03

# Test specific module export
tito module complete 03_activations --dry-run

# Run module file directly
python modules/source/03_activations/activations_dev.py
```

### **System Issues**

**Q: Jupyter Lab won't start - what's wrong?**

A: Common solutions:

1. **Check installation**:
   ```bash
   pip install jupyterlab jupyter
   jupyter lab --version
   ```

2. **Port conflict**:
   ```bash
   jupyter lab --port 8889  # Try different port
   ```

3. **Virtual environment**:
   ```bash
   source .venv/bin/activate  # Ensure activated
   which jupyter  # Should show .venv path
   ```

**Q: I'm getting import errors when testing - help!**

A: Import errors usually mean:

1. **Virtual environment not activated**:
   ```bash
   source .venv/bin/activate
   ```

2. **TinyTorch not installed in development mode**:
   ```bash
   pip install -e . --force-reinstall
   ```

3. **Module not exported**:
   ```bash
   tito module complete 0X_module_name
   ```

4. **Check your export directive**:
   ```python
   #| default_exp tinytorch.core.module_name  # At top of file
   ```

---

## 🌍 Community Questions

### **Leaderboard & Community**

**Q: Is the leaderboard competitive or supportive?**

A: **Both!** We designed it to be inclusive and encouraging:

**🏆 Multiple Ways to Excel:**
- **Progress**: Checkpoint completion (everyone can achieve)
- **Speed**: Fast learners (if that's your style)
- **Innovation**: Creative optimizations (for advanced users)
- **Community**: Helping others (valuable contribution)

**🤝 Supportive Culture:**
- Celebrate all achievements, not just "first place"
- Anonymous participation options available
- Community helps each other learn
- No shame in taking time to understand concepts

**Q: Do I have to share my progress publicly?**

A: **No!** Participation is entirely optional:

- All learning features work without leaderboard
- Checkpoint system tracks progress locally
- Join community only when/if you want to
- Privacy controls let you share what you're comfortable with

**Q: What information is shared when I join the leaderboard?**

A: You control what's shared:

**Always Shared:**
- Display name (you choose - can be pseudonymous)
- Checkpoint completion status
- Module completion dates

**Optionally Shared:**
- Real name (if you choose)
- Institution/company
- Achievement celebrations
- Optimization benchmarks

**Never Shared:**
- Personal information
- Email addresses
- Code implementations
- Detailed progress metrics (unless you opt in)

### **Competition & Olympics**

**Q: What are the Olympics and how are they different from the leaderboard?**

A: **Leaderboard** = Learning Progress, **Olympics** = Performance Competition

| **Leaderboard** | **Olympics** |
|-----------------|--------------|
| Track learning progress | Compete on optimization |
| Checkpoint completion | Benchmark performance |
| Supportive community | Competitive challenges |
| All experience levels | Advanced optimization |

**Olympics Events:**
- **MLP Sprint**: Fastest matrix operations
- **CNN Marathon**: Memory-efficient convolutions  
- **Transformer Decathlon**: Complete language model optimization

**Q: Do I need to be an expert to participate in Olympics?**

A: **No!** Olympics have multiple categories:

- **Beginner**: Just-working implementations compete
- **Intermediate**: Solid optimizations
- **Advanced**: Cutting-edge techniques
- **Innovation**: Novel approaches

**Everyone can contribute and learn from others' solutions.**

---

## 🎓 Instructor Questions

### **Classroom Setup**

**Q: How much setup is required to use TinyTorch in my class?**

A: **Minimal!** TinyTorch includes complete teaching infrastructure:

**One-time Setup (30 minutes):**
```bash
tito nbgrader setup-instructor
tito grade setup-course
```

**Per-semester Setup (10 minutes):**
```bash
tito nbgrader create-student-repos
tito grade release-module 01_setup
```

**Everything Included:**
- NBGrader integration works out-of-the-box
- Student progress tracking built-in
- Automated grading workflow
- Assignment release/collection system

**Q: Can I customize the curriculum for my specific course?**

A: **Absolutely!** TinyTorch is designed for flexibility:

**Duration Options:**
- **4 weeks**: Neural network foundations (Modules 1-8)
- **8 weeks**: Add computer vision (Modules 1-10)  
- **12 weeks**: Include language models (Modules 1-14)
- **16 weeks**: Complete system optimization (All 20)

**Difficulty Customization:**
- **Beginner**: Additional scaffolding and explanations
- **Advanced**: Extra optimization challenges
- **Research**: Custom project integration

**Q: How do I track student progress across the class?**

A: Multiple tracking tools built-in:

```bash
# Class overview
tito grade class-overview

# Individual student
tito grade student-progress student_name

# Checkpoint statistics
tito checkpoint class-stats

# Module completion rates
tito grade module-stats 05_losses
```

**Visual dashboards show:**
- Who's completed which modules
- Where students are getting stuck
- Average completion times
- Achievement distributions

### **Grading & Assessment**

**Q: How does automated grading work?**

A: **Three-layer validation system:**

1. **Functional Tests**: Does the code work correctly?
2. **Interface Tests**: Does it match expected function signatures?
3. **Checkpoint Tests**: Can student use their implementation?

```bash
# Grade student submission
tito nbgrader autograde 05_losses student_name

# Results show:
# ✅ Function implementation (40 points)
# ✅ Interface compliance (30 points)  
# ✅ Integration test (30 points)
# Total: 100/100
```

**Q: What if a student's implementation works but doesn't match the test exactly?**

A: **Flexible grading system:**

- **Core functionality**: Must work correctly (non-negotiable)
- **Implementation details**: Multiple valid approaches accepted
- **Code style**: Guidance provided, not penalized
- **Performance**: Bonus points for optimization, not required

**Manual review system** catches edge cases and provides personalized feedback.

**Q: How do I handle students working at different paces?**

A: **Built-in flexibility:**

**Self-paced Options:**
- Students can work ahead through modules
- Checkpoint system validates readiness for advanced topics
- Extra credit opportunities for early finishers

**Support for Struggling Students:**
- Extended deadlines through system configuration
- Additional scaffolding materials included
- Peer tutoring through community features
- Office hours integration with progress tracking

---

## 🔧 Troubleshooting

### **Common Error Messages**

**Error: `ModuleNotFoundError: No module named 'tinytorch'`**

**Solutions:**
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install in development mode
pip install -e .

# 3. Verify installation
python -c "import tinytorch; print('Success!')"
```

**Error: `AttributeError: module 'tinytorch.core.tensor' has no attribute 'Tensor'`**

**Cause:** Module not exported or export failed

**Solutions:**
```bash
# 1. Check export status
tito module status 02_tensor

# 2. Re-export module
tito module complete 02_tensor

# 3. Verify export worked
python -c "from tinytorch.core.tensor import Tensor; print('Success!')"
```

**Error: Tests pass individually but fail in checkpoint**

**Cause:** Integration issues between modules

**Solutions:**
```bash
# 1. Test integration
tito checkpoint test 05 --verbose

# 2. Check all dependencies exported
tito module status --all

# 3. Re-export dependency chain
tito module complete 02_tensor
tito module complete 03_activations
# ... up to current module
```

### **Performance Issues**

**Q: Training is really slow - is this normal?**

A: **Some slowness is expected** (you're building from scratch!), but here's how to optimize:

**Expected Performance:**
- **Pure NumPy**: 10-100x slower than PyTorch
- **Simple examples**: Should complete in seconds
- **CIFAR-10 training**: 5-10 minutes per epoch

**Optimization Tips:**
```python
# Use vectorized operations
result = np.dot(x, weights)  # Fast

# Avoid Python loops
for i in range(len(x)):      # Slow
    result[i] = x[i] * weights[i]
```

**Q: My computer is running out of memory during training**

A: **Memory management strategies:**

1. **Reduce batch size**:
   ```python
   batch_size = 32  # Instead of 256
   ```

2. **Use gradient accumulation**:
   ```python
   # Accumulate gradients over mini-batches
   optimizer.step_every_n_batches(4)
   ```

3. **Profile memory usage**:
   ```bash
   tito checkpoint test 10 --profile-memory
   ```

---

## 💡 Best Practices

### **Learning Effectively**

**Q: What's the best way to approach each module?**

A: **Follow the Build → Use → Reflect pattern:**

**1. Build (Implementation)**
- Read the introduction to understand the goal
- Implement functions one at a time
- Test each function immediately after writing it

**2. Use (Integration)**  
- Export your module: `tito module complete 0X_name`
- Test the integration with checkpoint
- Use your component in examples

**3. Reflect (Understanding)**
- Answer the ML Systems Thinking questions
- Consider memory usage and performance
- Connect to production ML systems

**Q: How do I know if I really understand a concept?**

A: **True understanding means you can:**

1. **Implement from memory**: Re-write the function without looking
2. **Explain to others**: Describe how and why it works  
3. **Debug problems**: Fix issues when something breaks
4. **Optimize performance**: Improve memory or speed
5. **Connect to production**: Relate to PyTorch/TensorFlow internals

**Checkpoint tests verify some of this, but self-reflection is crucial.**

### **Time Management**

**Q: I'm spending too much time on implementation details - should I move on?**

A: **Balance depth with progress:**

**When to Push Through:**
- Core concepts not clicking yet
- Function doesn't work correctly
- Tests are failing

**When to Move On:**
- Function works and passes tests
- You understand the main concept
- You're optimizing minor details

**Remember:** You can always return to optimize later. The goal is understanding systems, not perfect code.

**Q: Should I complete all modules before starting real projects?**

A: **No!** Start projects as soon as you have the basics:

- **After Module 6**: Build XOR solver
- **After Module 8**: Train MNIST classifier  
- **After Module 10**: CIFAR-10 CNN
- **After Module 14**: TinyGPT language model

**Real projects reinforce learning and show practical applications.**

---

## 🚀 Getting More Help

### **When These FAQs Don't Help**

**1. Interactive CLI Help**
```bash
tito help --interactive  # Personalized guidance
tito help troubleshooting  # Common technical issues
```

**2. System Diagnostics**
```bash
tito system doctor  # Comprehensive system check
```

**3. Community Support**
- Join leaderboard for peer help and discussion
- Share specific error messages for targeted assistance
- Learn from others' solutions and approaches

**4. Documentation Resources**
- **Module README files**: Detailed explanations for each topic
- **User Manual**: Comprehensive guide to all features
- **Instructor Guide**: Teaching resources and classroom management

**5. Course Support (if applicable)**
- Office hours with instructor
- Class discussion forums
- Teaching assistant support

### **Reporting Issues**

**Found a bug or unclear documentation?**

Please include:
- **System info**: Output of `tito system doctor`
- **Error message**: Complete traceback if available
- **Steps to reproduce**: What commands led to the issue
- **Expected vs actual**: What you expected to happen

**Contact through:**
- Course instructor (if taking as class)
- Community leaderboard (for peer support)
- GitHub issues (for bug reports)

---

**Still have questions? Try `tito help --interactive` for personalized guidance! 🚀**