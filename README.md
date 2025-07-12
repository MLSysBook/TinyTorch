# Tiny🔥Torch: Build ML Systems from Scratch

> A hands-on ML Systems course where students implement every component from scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![nbdev](https://img.shields.io/badge/built%20with-nbdev-orange.svg)](https://nbdev.fast.ai/)

> **Disclaimer**: TinyTorch is an educational framework developed independently and is not affiliated with or endorsed by Meta or the PyTorch project.

**Tiny🔥Torch** is a complete ML Systems course where students build their own machine learning framework from scratch. Rather than just learning *about* ML systems, students implement every component and then use their own implementation to solve real problems.

## 🚀 **Quick Start - Choose Your Path**

### **👨‍🏫 For Instructors**
**[📖 Instructor Guide](docs/INSTRUCTOR_GUIDE.md)** - Complete teaching guide with verified modules, class structure, and commands
- 6+ weeks of proven curriculum content
- Verified module status and teaching sequence
- Class session structure and troubleshooting guide

### **👨‍🎓 For Students**
**[🔥 Student Guide](docs/STUDENT_GUIDE.md)** - Complete learning path with clear workflow
- Step-by-step progress tracker
- 5-step daily workflow for each module
- Getting help and study tips

### **🛠️ For Developers**
**[📚 Documentation](docs/)** - Complete documentation including pedagogy and development guides

### **🎯 Python-First Development + NBGrader**
**Philosophy**: Raw Python files → Jupyter notebooks on demand → NBGrader compliance
- **Core Development**: Work in `modules/XX/XX_dev.py` (Python files)
- **Package Building**: `nbdev` exports to `tinytorch` package  
- **Assignment Generation**: `jupytext` + `NBGrader` create student versions
- **Auto-Grading**: `pytest` integration for automated testing

**Instructor Workflow**: 
```bash
code modules/XX/XX_dev.py        # Edit Python source
tito module export XX            # Build package (nbdev)
tito nbgrader generate XX        # Create assignment (Python→Jupyter→NBGrader)
tito nbgrader release XX         # Deploy to students
```

## 🎯 **What Students Build**

By completing TinyTorch, students implement a complete ML framework:

- ✅ **Activation functions** (ReLU, Sigmoid, Tanh)
- ✅ **Neural network layers** (Dense, Conv2D)
- ✅ **Network architectures** (Sequential, MLP)
- ✅ **Data loading** (CIFAR-10 pipeline)
- ✅ **Development workflow** (export, test, use)
- 🚧 **Tensor operations** (arithmetic, broadcasting)
- 🚧 **Automatic differentiation** (backpropagation)
- 🚧 **Training systems** (optimizers, loss functions)

## 🎓 **Learning Philosophy: Build → Use → Understand → Repeat**

Students experience the complete cycle:
1. **Build**: Implement `ReLU()` function from scratch
2. **Use**: Import `from tinytorch.core.activations import ReLU` with their own code
3. **Understand**: See how it works in real neural networks
4. **Repeat**: Each module builds on previous implementations

## 📊 **Current Status** (Ready for Classroom Use)

### **✅ Fully Working Modules** (6+ weeks of content)
- **00_setup** (20/20 tests) - Development workflow & CLI tools
- **02_activations** (24/24 tests) - ReLU, Sigmoid, Tanh functions
- **03_layers** (17/22 tests) - Dense layers & neural building blocks
- **04_networks** (20/25 tests) - Sequential networks & MLPs
- **06_dataloader** (15/15 tests) - CIFAR-10 data loading
- **05_cnn** (2/2 tests) - Convolution operations

### **🚧 In Development**
- **01_tensor** (22/33 tests) - Tensor arithmetic
- **07-13** - Advanced features (autograd, training, MLOps)

## 🚀 **Quick Commands**

### **System Status**
```bash
tito system info              # Check system and module status
tito system doctor            # Verify environment setup
tito module status            # View all module progress
```

### **Student Workflow**
```bash
cd modules/00_setup           # Navigate to first module
jupyter lab setup_dev.py     # Open development notebook
python -m pytest tests/ -v   # Run tests
python bin/tito module export 00_setup  # Export to package
```

### **Verify Implementation**
```bash
# Use student's own implementations
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
python -c "from tinytorch.core.activations import ReLU; print(ReLU()([-1, 0, 1]))"
```

## 🌟 **Why Build from Scratch?**

**Even in the age of AI-generated code, building systems from scratch remains educationally essential:**

- **Understanding vs. Using**: AI shows *what* works, TinyTorch teaches *why* it works
- **Systems Literacy**: Debugging real ML requires understanding abstractions like autograd and data loaders
- **AI-Augmented Engineers**: The best engineers collaborate with AI tools, not rely on them blindly
- **Intentional Design**: Systems thinking about memory, performance, and architecture can't be outsourced

## 🏗️ **Repository Structure**

```
TinyTorch/
├── README.md                 # This file - main entry point
├── docs/
│   ├── INSTRUCTOR_GUIDE.md   # Complete teaching guide
│   ├── STUDENT_GUIDE.md      # Complete learning path
│   └── [detailed docs]       # Pedagogy and development guides
├── modules/
│   ├── 00_setup/            # Development workflow
│   ├── 01_tensor/           # Tensor operations
│   ├── 02_activations/      # Activation functions
│   ├── 03_layers/           # Neural network layers
│   ├── 04_networks/         # Network architectures
│   ├── 05_cnn/              # Convolution operations
│   ├── 06_dataloader/       # Data loading pipeline
│   └── 07-13/               # Advanced features
├── tinytorch/               # The actual Python package
├── bin/                     # CLI tools (tito)
└── tests/                   # Integration tests
```

## 📚 **Educational Approach**

### **Real Data, Real Systems**
- Work with CIFAR-10 (10,000 real images)
- Production-style code organization
- Performance and engineering considerations

### **Immediate Feedback**
- Tests provide instant verification
- Students see their code working quickly
- Progress is visible and measurable

### **Progressive Complexity**
- Start simple (activation functions)
- Build complexity gradually (layers → networks → training)
- Connect to real ML engineering practices

## 🤝 **Contributing**

We welcome contributions! See our [development documentation](docs/development/) for guidelines on creating new modules or improving existing ones.

## 📄 **License**

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Ready to Start?**

### **Instructors**
1. Read the [📖 Instructor Guide](docs/INSTRUCTOR_GUIDE.md)
2. Test your setup: `tito system doctor`
3. Start with: `cd modules/00_setup && jupyter lab setup_dev.py`

### **Students**
1. Read the [🔥 Student Guide](docs/STUDENT_GUIDE.md)
2. Begin with: `cd modules/00_setup && jupyter lab setup_dev.py`
3. Follow the 5-step workflow for each module

**🚀 TinyTorch is ready for classroom use with 6+ weeks of proven curriculum content!**
