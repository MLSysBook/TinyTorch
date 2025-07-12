# 📚 TinyTorch Documentation

**Complete documentation for the TinyTorch ML Systems course.**

## 🎯 **Quick Start Navigation**

### **For Instructors** 👨‍🏫
- **[📖 Instructor Guide](INSTRUCTOR_GUIDE.md)** - Complete teaching guide with verified modules, commands, and class structure
- **[🎓 Pedagogy](pedagogy/)** - Educational principles and course philosophy

### **For Students** 👨‍🎓
- **[🔥 Student Guide](STUDENT_GUIDE.md)** - Complete course navigation and learning path
- **[📚 Module README](../modules/)** - Individual module instructions and status

### **For Developers** 👨‍💻
- **[🛠️ Development](development/)** - Module creation and contribution guidelines

## 📊 **Current Course Status**

### **✅ Ready for Students** (6+ weeks of content)
- **00_setup** (20/20 tests) - Development workflow & CLI tools
- **02_activations** (24/24 tests) - ReLU, Sigmoid, Tanh functions
- **03_layers** (17/22 tests) - Dense layers & neural building blocks
- **04_networks** (20/25 tests) - Sequential networks & MLPs
- **06_dataloader** (15/15 tests) - CIFAR-10 data loading
- **05_cnn** (2/2 tests) - Convolution operations

### **🚧 In Development**
- **01_tensor** (22/33 tests) - Tensor arithmetic (partially working)
- **07-13** - Advanced features (autograd, training, MLOps)

## 🚀 **Quick Commands**

### **System Status**
```bash
tito system info              # Check system and module status
tito system doctor            # Verify environment setup
tito module status            # View all module progress
```

### **Module Development**
```bash
cd modules/00_setup           # Navigate to module
jupyter lab setup_dev.py     # Open development notebook
python -m pytest tests/ -v   # Run module tests
python bin/tito module export 00_setup  # Export to package
```

### **Package Usage**
```bash
# Use student implementations
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
python -c "from tinytorch.core.activations import ReLU; print(ReLU()([-1, 0, 1]))"
```

## 🎓 **Educational Philosophy**

### **Build → Use → Understand → Repeat**
Students implement ML components from scratch, then immediately use their implementations:
1. **Build**: Implement `ReLU()` function
2. **Use**: Import `from tinytorch.core.activations import ReLU`
3. **Understand**: See how it works in real networks
4. **Repeat**: Each module builds on previous work

### **Real Data, Real Systems**
- Work with CIFAR-10 (not toy datasets)
- Production-style code organization
- Performance and engineering considerations

### **Immediate Feedback**
- Tests provide instant verification
- Students see their code working quickly
- Progress is visible and measurable

## 📁 **Documentation Structure**

### **Quick Reference**
- **[INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md)** - Complete teaching guide
- **[STUDENT_GUIDE.md](STUDENT_GUIDE.md)** - Complete learning path

### **Detailed Guides**
- **[pedagogy/](pedagogy/)** - Educational principles and course philosophy
- **[development/](development/)** - Module creation and development guidelines

### **Legacy Documentation**
The `development/` directory contains detailed module creation guides that were used to build the current working modules. This documentation is preserved for reference but the main teaching workflow is now covered in the Instructor and Student guides.

## 🌟 **Success Metrics**

### **Working Capabilities**
Students can currently:
- Build and test multi-layer perceptrons
- Implement custom activation functions
- Load and process CIFAR-10 data
- Create basic convolution operations
- Export their code to a working package

### **Verified Workflows**
- ✅ **Instructor Journey**: develop → export → test → package
- ✅ **Student Journey**: import → use → build → understand
- ✅ **Package Integration**: All core imports work correctly

## 🔧 **Technical Details**

### **Module Structure**
Each module follows this pattern:
- `modules/XX_name/` - Module directory
- `XX_name_dev.py` - Development notebook (Jupytext format)
- `tests/` - Comprehensive test suite
- `README.md` - Module-specific instructions

### **Export System**
- Students develop in `XX_name_dev.py`
- Export to `tinytorch.core.XX_name` package
- Import and use their implementations immediately

---

## 🚀 **Getting Started**

### **Instructors**
1. Read the [Instructor Guide](INSTRUCTOR_GUIDE.md)
2. Verify your system: `tito system doctor`
3. Test the first assignment: `cd assignments/source/00_setup && jupyter lab setup_dev.py`

### **Students**
1. Read the [Student Guide](STUDENT_GUIDE.md)
2. Start with: `cd assignments/source/00_setup && jupyter lab setup_dev.py`
3. Follow the 5-step workflow for each module

### **Developers**
1. Review the [development/](development/) directory
2. Follow existing module patterns
3. Test thoroughly before contributing

**🎉 TinyTorch is ready for classroom use with 6+ weeks of proven curriculum content!** 