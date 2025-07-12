# 🔥 TinyTorch: Build ML Systems from Scratch

**A complete Machine Learning Systems course where students build their own ML framework.**

## 🎯 What You'll Build

- **Complete ML Framework**: Build your own PyTorch-style framework from scratch
- **Real Applications**: Use your framework to classify CIFAR-10 images  
- **Production Skills**: Learn ML systems engineering, not just algorithms
- **Immediate Feedback**: See your code working at every step

## 🚀 Quick Start (2 minutes)

### **Students**
```bash
git clone https://github.com/your-org/tinytorch.git
cd TinyTorch
make install                                    # Install dependencies
tito system doctor                              # Verify setup
cd assignments/source/00_setup                  # Start with setup
jupyter lab setup_dev.py                       # Open first assignment
```

### **Instructors**
```bash
# System check
tito system info                                # Check course status
tito system doctor                              # Verify environment  

# Assignment management
tito nbgrader generate 00_setup                 # Create student assignments
tito nbgrader release 00_setup                  # Release to students
tito nbgrader autograde 00_setup                # Auto-grade submissions
```

## 📚 Course Structure

### **Core Assignments** (6+ weeks of proven content)
- **00_setup** (20/20 tests) - Development workflow & CLI tools
- **02_activations** (24/24 tests) - ReLU, Sigmoid, Tanh functions
- **03_layers** (17/22 tests) - Dense layers & neural building blocks
- **04_networks** (20/25 tests) - Sequential networks & MLPs
- **06_dataloader** (15/15 tests) - CIFAR-10 data loading
- **05_cnn** (2/2 tests) - Convolution operations

### **Advanced Features** (in development)
- **01_tensor** (22/33 tests) - Tensor arithmetic
- **07-13** - Autograd, optimizers, training, MLOps

## 🛠️ Development Workflow

### **NBGrader** (Assignment Creation & Testing)
```bash
tito nbgrader generate 00_setup     # Create student assignments
tito nbgrader release 00_setup      # Release to students
tito nbgrader collect 00_setup      # Collect submissions
tito nbgrader autograde 00_setup    # Auto-grade with pytest
```

### **nbdev** (Package Export & Building)
```bash
tito module export 00_setup         # Export to tinytorch package
tito module test 00_setup           # Test package integration
```

## 📈 Student Success Path

### **Build → Use → Understand → Repeat**
1. **Build**: Implement `ReLU()` function from scratch
2. **Use**: `from tinytorch.core.activations import ReLU` - your own code!
3. **Understand**: See how it works in real neural networks
4. **Repeat**: Each assignment builds on previous work

### **Example: First Assignment**
```python
# You implement this:
def hello_tinytorch():
    print("Welcome to TinyTorch!")

# Then immediately use it:
from tinytorch.core.utils import hello_tinytorch
hello_tinytorch()  # Your code working!
```

## 🎓 Educational Philosophy

### **Real Data, Real Systems**
- Work with CIFAR-10 (not toy datasets)
- Production-style code organization
- Performance and engineering considerations
- Immediate visual feedback

### **Build Everything from Scratch**
- No black boxes or "magic" functions
- Understanding through implementation
- Connect every concept to production systems
- See your code working immediately

## 📁 Repository Structure

```
TinyTorch/
├── assignments/source/XX/          # Assignment source files
│   ├── XX_dev.py                   # Development assignment
│   └── tests/                      # Assignment tests
├── tinytorch/                      # Your built framework
│   └── core/                       # Exported student code
├── tito/                           # CLI tools
└── docs/                           # Documentation
```

## 🔧 Technical Requirements

- **Python 3.8+**
- **Jupyter Lab** for development
- **PyTorch** for comparison and final projects
- **NBGrader** for assignment management
- **nbdev** for package building

## 🎯 Getting Started

### **Students**
1. **System Check**: `tito system doctor`
2. **First Assignment**: `cd assignments/source/00_setup && jupyter lab setup_dev.py`
3. **Build & Test**: Follow the notebook, export when complete
4. **Use Your Code**: `from tinytorch.core.utils import hello_tinytorch`

### **Instructors** 
1. **Course Status**: `tito system info`
2. **Assignment Management**: `tito nbgrader generate 00_setup`
3. **Student Release**: `tito nbgrader release 00_setup`
4. **Auto-grading**: `tito nbgrader autograde 00_setup`

## 📊 Success Metrics

**Students can currently:**
- Build and test multi-layer perceptrons
- Implement custom activation functions  
- Load and process CIFAR-10 data
- Create basic convolution operations
- Export their code to a working package

**Verified workflows:**
- ✅ **Student Journey**: receive assignment → implement → export → use
- ✅ **Instructor Journey**: create → release → collect → grade
- ✅ **Package Integration**: All core imports work correctly

---

**🎉 TinyTorch is ready for classroom use with 6+ weeks of proven curriculum content!**
