# 🎓 TinyTorch Instructor Guide

**Complete instructor guide for teaching the TinyTorch ML Systems course.**

## 🎯 **Course Overview**

TinyTorch is a **Machine Learning Systems** course where students build every component from scratch. Students implement their own ML framework, then use their implementation to solve real problems.

### **Core Philosophy: Build → Use → Understand → Repeat**
- **Build**: Students implement functions like `ReLU()` from scratch
- **Use**: Students immediately use `from tinytorch.core.activations import ReLU` with their own code
- **Understand**: Students see how their implementations work in real systems
- **Repeat**: Each module builds on previous work

## 🚀 **Quick Start for Instructors**

### **Verify Your System** (5 minutes)
```bash
cd TinyTorch
tito system info         # Check overall status
tito system doctor       # Verify installation
tito module status       # See all module progress
```

### **Test Drive the Course** (10 minutes)
```bash
cd modules/00_setup
jupyter lab setup_dev.py                    # Open first module
python -m pytest tests/ -v                  # Run tests
python bin/tito module export 00_setup      # Export to package
```

### **Verify Student Experience** (5 minutes)
```bash
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
python -c "from tinytorch.core.activations import ReLU; print(ReLU()([-1, 0, 1]))"
```

## 📚 **Current Module Status** (Ready for Students)

### **✅ Fully Working Modules**
1. **00_setup** (20/20 tests) - Development workflow, CLI tools
2. **02_activations** (24/24 tests) - ReLU, Sigmoid, Tanh implementations
3. **06_dataloader** (15/15 tests) - CIFAR-10 loading, batch processing

### **✅ Core Working Modules**
4. **03_layers** (17/22 tests) - Dense layers, network building blocks
5. **04_networks** (20/25 tests) - Sequential networks, MLP creation
6. **05_cnn** (2/2 tests) - Basic convolution operations

### **⚠️ Partially Working**
7. **01_tensor** (22/33 tests) - Missing arithmetic operators

### **🚧 Not Yet Implemented**
- Modules 07-13 (autograd, optimizers, training, MLOps)

## 🎯 **Recommended Teaching Sequence**

### **Week 1-2: Foundation**
- **00_setup**: Students learn development workflow
- **02_activations**: Core ML math (ReLU, Sigmoid, Tanh)

### **Week 3-4: Building Blocks**
- **03_layers**: Neural network layers (Dense, parameter management)
- **04_networks**: Complete network composition (Sequential, MLP)

### **Week 5-6: Real Data**
- **06_dataloader**: Production data handling (CIFAR-10)
- **05_cnn**: Convolution operations

### **Week 7+: Advanced Features**
- **01_tensor**: Tensor arithmetic (when remaining tests pass)
- **Future modules**: Autograd, training, optimization

## 🛠️ **Instructor Workflow** (Python-First Development)

### **🐍 Python-First Philosophy**
- **Always work in raw Python files** (`modules/XX/XX_dev.py`)
- **Generate Jupyter notebooks on demand** using Jupytext
- **NBGrader compliance** through automated cell metadata
- **nbdev for package building** and exports

### **Step 1: Create/Edit Solution (Python File)**
```bash
cd modules/00_setup
# Edit the raw Python file (source of truth)
code setup_dev.py                # or vim/nano/your editor
```

### **Step 2: Test Solution**
```bash
python -m pytest modules/00_setup/tests/ -v  # Verify solution works
```

### **Step 3: Export to Package (nbdev)**
```bash
python bin/tito module export 00_setup  # Export to tinytorch package
```

### **Step 4: Generate Student Assignment (NBGrader)**
```bash
tito nbgrader generate 00_setup   # Python → Jupyter → NBGrader
```

### **Step 5: Release to Students**
```bash
tito nbgrader release 00_setup    # Deploy to student environment
```

### **Step 6: Collect & Grade**
```bash
tito nbgrader collect 00_setup    # Collect submissions
tito nbgrader autograde 00_setup  # Auto-grade with pytest
```

### **🔄 Complete Workflow Diagram**
```
modules/XX/XX_dev.py    (Source of Truth)
        ↓
    [nbdev export]      (Package Building)
        ↓
  tinytorch/core/       (Production Package)
        ↓
  [jupytext convert]    (On Demand)
        ↓
    XX_dev.ipynb        (Instructor Notebook)
        ↓
  [NBGrader process]    (Student Generation)
        ↓
assignments/source/XX/  (Student Assignments)
```

## 🛠️ **Student Workflow** (5 Simple Steps)

### **Step 1: Open Module**
```bash
cd modules/00_setup
jupyter lab setup_dev.py
```

### **Step 2: Learn & Implement**
- Read markdown explanations
- Complete TODO sections
- Test understanding incrementally

### **Step 3: Export Code**
```bash
python bin/tito module export 00_setup
```

### **Step 4: Test Work**
```bash
python -m pytest modules/00_setup/tests/ -v
```

### **Step 5: Use Their Code**
```bash
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
```

## 📋 **Class Session Structure** (50 minutes)

### **Preparation** (5 minutes)
```bash
tito system info                    # Check system
tito module status                  # Review progress
cd modules/[current_module]         # Navigate to module
```

### **Opening** (10 minutes)
- Demo end goal: Show working code
- Connect to previous module
- Explain real-world relevance

### **Guided Implementation** (30 minutes)
- Students work on TODO sections
- Instructor provides hints/guidance
- Students test their implementations

### **Wrap-up** (5 minutes)
- Celebrate successful exports
- Preview next module
- Assign homework/practice

## 🔧 **Essential Instructor Commands**

### **System Status**
```bash
tito system info              # Overall system status
tito system doctor            # Environment verification
tito module status            # All module progress
```

### **Module Management**
```bash
tito module test XX           # Test specific module
tito module export XX         # Export module to package
jupyter lab XX_dev.py         # Open module for development
```

### **NBGrader Workflow (Instructor → Student)**
```bash
tito nbgrader generate XX     # Generate student version from instructor solution
tito nbgrader generate --all  # Generate all student assignments
tito nbgrader status          # Check assignment status
tito nbgrader release XX      # Release assignment to students
tito nbgrader collect XX      # Collect student submissions
tito nbgrader autograde XX    # Auto-grade submissions
```

### **Student Help**
```bash
cat modules/XX/README.md      # Module overview
tito package reset            # Reset package state
```

### **Troubleshooting**
```bash
tito system doctor            # Diagnose issues
tito package sync             # Sync all modules
pytest modules/XX/tests/ -v  # Detailed test output
```

## 👥 **Student Support**

### **Common Issues & Solutions**

**"Tests are failing"**
```bash
cd modules/XX
python -m pytest tests/ -v    # See detailed failures
```

**"Can't import my code"**
```bash
python bin/tito module export XX    # Re-export module
tito package sync                    # Sync all modules
```

**"Environment issues"**
```bash
tito system doctor                   # Check environment
source .venv/bin/activate            # Activate virtual env
```

## 🎓 **Assessment Strategy**

### **Formative Assessment**
- **Daily**: Test output shows working implementations
- **Weekly**: Module completion with passing tests
- **Progressive**: Each module builds on previous work

### **Summative Assessment**
- **Portfolio**: Complete working TinyTorch implementation
- **Reflection**: Understanding of design decisions
- **Application**: Use their framework for real problems

## 🌟 **Key Educational Principles**

### **Real Data, Real Systems**
- Use CIFAR-10 (not toy datasets)
- Production-style code organization
- Performance considerations matter

### **Immediate Feedback**
- Tests provide instant verification
- Students see their code working quickly
- Progress is visible and measurable

### **Progressive Complexity**
- Start simple (activation functions)
- Build complexity gradually (layers → networks)
- Connect to real ML engineering

## 💡 **Teaching Tips**

### **Start Each Module**
1. **Demo the end goal**: Show working code
2. **Connect to previous work**: "Remember how we built..."
3. **Explain the why**: Real-world motivation
4. **Guide the how**: Implementation strategy

### **During Implementation**
- Encourage testing small pieces
- Use `print()` statements for debugging
- Celebrate small victories
- Connect to broader ML concepts

### **After Completion**
- Show how their code exports to package
- Demonstrate using their implementation
- Preview next module connections
- Assign practice problems

## 🔄 **Course Maintenance**

### **Before Each Semester**
```bash
git pull origin main          # Update to latest
tito system doctor            # Verify environment
tito module status            # Check all modules
```

### **Regular Updates**
- Monitor test results across modules
- Update datasets/examples as needed
- Collect student feedback for improvements

---

## 🚀 **Ready to Start Teaching?**

### **First Class Preparation**
1. **Clone repository**: `git clone [repo_url]`
2. **Test your setup**: `tito system doctor`
3. **Review module 00**: `cat modules/00_setup/README.md`
4. **Open in Jupyter**: `jupyter lab modules/00_setup/setup_dev.py`

### **Your First Class**
```bash
cd modules/00_setup
jupyter lab setup_dev.py
# Guide students through setup_dev.py
# Watch them implement TODO sections
# Celebrate when tests pass!
```

**🎉 You're ready to teach TinyTorch! Start with module 00_setup and watch your students build their own ML framework.** 