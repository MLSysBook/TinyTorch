# 👩‍🏫 TinyTorch Instructor Guide

Complete guide for teaching ML Systems Engineering with TinyTorch.

## 🎯 Course Overview

TinyTorch teaches ML systems engineering through building, not just using. Students construct a complete ML framework from tensors to transformers, understanding memory, performance, and scaling at each step.

## 🛠️ Instructor Setup

### **1. Initial Setup**
```bash
# Clone and setup
git clone https://github.com/MLSysBook/TinyTorch.git
cd TinyTorch

# Virtual environment (MANDATORY)
python -m venv .venv
source .venv/bin/activate

# Install with instructor tools
pip install -r requirements.txt
pip install nbgrader

# Setup grading infrastructure
tito grade setup
```

### **2. Verify Installation**
```bash
tito system doctor
# Should show all green checkmarks

tito grade
# Should show available grade commands
```

## 📝 Assignment Workflow

### **Simplified with Tito CLI**
We've wrapped NBGrader behind simple `tito grade` commands so you don't need to learn NBGrader's complex interface.

### **1. Prepare Assignments**
```bash
# Generate instructor version (with solutions)
tito grade generate 01_setup

# Create student version (solutions removed)
tito grade release 01_setup

# Student version will be in: release/tinytorch/01_setup/
```

### **2. Distribute to Students**
```bash
# Option A: GitHub Classroom (recommended)
# 1. Create assignment repository from TinyTorch
# 2. Remove solutions from modules
# 3. Students clone and work

# Option B: Direct distribution
# Share the release/ directory contents
```

### **3. Collect Submissions**
```bash
# Collect all students
tito grade collect 01_setup

# Or specific student
tito grade collect 01_setup --student student_id
```

### **4. Auto-Grade**
```bash
# Grade all submissions
tito grade autograde 01_setup

# Grade specific student
tito grade autograde 01_setup --student student_id
```

### **5. Manual Review**
```bash
# Open grading interface (browser-based)
tito grade manual 01_setup

# This launches a web interface for:
# - Reviewing ML Systems question responses
# - Adding feedback comments
# - Adjusting auto-grades
```

### **6. Generate Feedback**
```bash
# Create feedback files for students
tito grade feedback 01_setup
```

### **7. Export Grades**
```bash
# Export all grades to CSV
tito grade export

# Or specific module
tito grade export --module 01_setup --output grades_module01.csv
```

## 📊 Grading Components

### **Auto-Graded (70%)**
- Code implementation correctness
- Test passing
- Function signatures
- Output validation

### **Manually Graded (30%)**
- ML Systems Thinking questions (3 per module)
- Each question: 10 points
- Focus on understanding, not perfection

### **Grading Rubric for ML Systems Questions**

| Points | Criteria |
|--------|----------|
| 9-10 | Demonstrates deep understanding, references specific code, discusses systems implications |
| 7-8 | Good understanding, some code references, basic systems thinking |
| 5-6 | Surface understanding, generic response, limited systems perspective |
| 3-4 | Attempted but misses key concepts |
| 0-2 | No attempt or completely off-topic |

**What to Look For:**
- References to actual implemented code
- Memory/performance analysis
- Scaling considerations
- Production system comparisons
- Understanding of trade-offs

## 📚 Module Teaching Notes

### **Module 01: Setup**
- **Focus**: Environment configuration, systems thinking mindset
- **Key Concept**: Development environments matter for ML systems
- **Common Issues**: Virtual environment confusion

### **Module 02: Tensor**
- **Focus**: Memory layout, data structures
- **Key Concept**: Understanding memory is crucial for ML performance
- **Demo**: Show memory profiling, copying behavior

### **Module 03: Activations**
- **Focus**: Vectorization, numerical stability
- **Key Concept**: Small details matter at scale
- **Demo**: Gradient vanishing/exploding

### **Module 04-05: Layers & Networks**
- **Focus**: Composition, parameter management
- **Key Concept**: Building blocks combine into complex systems
- **Project**: Build a small CNN

### **Module 06-07: Spatial & Attention**
- **Focus**: Algorithmic complexity, memory patterns
- **Key Concept**: O(N²) operations become bottlenecks
- **Demo**: Profile attention memory usage

### **Module 08-11: Training Pipeline**
- **Focus**: End-to-end system integration
- **Key Concept**: Many components must work together
- **Project**: Train a real model

### **Module 12-15: Production**
- **Focus**: Deployment, optimization, monitoring
- **Key Concept**: Academic vs production requirements
- **Demo**: Model compression, deployment

### **Module 16: TinyGPT**
- **Focus**: Framework generalization
- **Key Concept**: 70% component reuse from vision to language
- **Capstone**: Build a working language model

## 🎯 Learning Objectives

By course end, students should be able to:

1. **Build** complete ML systems from scratch
2. **Analyze** memory usage and computational complexity
3. **Debug** performance bottlenecks
4. **Optimize** for production deployment
5. **Understand** framework design decisions
6. **Apply** systems thinking to ML problems

## 📈 Tracking Progress

### **Individual Progress**
```bash
# Check specific student progress
tito checkpoint status --student student_id
```

### **Class Overview**
```bash
# Export all checkpoint achievements
tito checkpoint export --output class_progress.csv
```

### **Identify Struggling Students**
Look for:
- Missing checkpoint achievements
- Low scores on ML Systems questions
- Incomplete module submissions

## 💡 Teaching Tips

### **1. Emphasize Building Over Theory**
- Have students type every line of code
- Run tests immediately after implementation
- Break and fix things intentionally

### **2. Connect to Production Systems**
- Show PyTorch/TensorFlow equivalents
- Discuss real-world bottlenecks
- Share production war stories

### **3. Make Performance Visible**
```python
# Use profilers liberally
with TimeProfiler("operation"):
    result = expensive_operation()
    
# Show memory usage
print(f"Memory: {get_memory_usage():.2f} MB")
```

### **4. Encourage Systems Questions**
- "What would break at 1B parameters?"
- "How would you distributed this?"
- "What's the bottleneck here?"

## 🔧 Troubleshooting

### **Common Student Issues**

**Environment Problems**
```bash
# Student fix:
tito system doctor
tito system reset
```

**Module Import Errors**
```bash
# Rebuild package
tito export --all
```

**Test Failures**
```bash
# Detailed test output
tito module test MODULE --verbose
```

### **NBGrader Issues**

**Database Locked**
```bash
# Clear NBGrader database
rm gradebook.db
tito grade setup
```

**Missing Submissions**
```bash
# Check submission directory
ls submitted/*/MODULE/
```

## 📊 Sample Schedule (16 Weeks)

| Week | Module | Focus |
|------|--------|-------|
| 1 | 01 Setup | Environment, Tools |
| 2 | 02 Tensor | Data Structures |
| 3 | 03 Activations | Functions |
| 4 | 04 Layers | Components |
| 5 | 05 Dense | Networks |
| 6 | 06 Spatial | Convolutions |
| 7 | 07 Attention | Transformers |
| 8 | Midterm Project | Build CNN |
| 9 | 08 Dataloader | Data Pipeline |
| 10 | 09 Autograd | Differentiation |
| 11 | 10 Optimizers | Training Algorithms |
| 12 | 11 Training | Complete Pipeline |
| 13 | 12 Compression | Optimization |
| 14 | 13 Kernels | Performance |
| 15 | 14-15 MLOps | Production |
| 16 | 16 TinyGPT | Capstone |

## 🎓 Assessment Strategy

### **Continuous Assessment (70%)**
- Module completion: 4% each × 16 = 64%
- Checkpoint achievements: 6%

### **Projects (30%)**
- Midterm: Build and train CNN (15%)
- Final: Extend TinyGPT (15%)

## 📚 Additional Resources

- [MLSys Book](https://mlsysbook.ai) - Companion textbook
- [Course Discussions](https://github.com/MLSysBook/TinyTorch/discussions)
- [Issue Tracker](https://github.com/MLSysBook/TinyTorch/issues)

---

**Need help? Open an issue or contact the TinyTorch team!**