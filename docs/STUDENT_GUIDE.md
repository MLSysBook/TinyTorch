# 🔥 TinyTorch Student Guide

**Build Your Own Machine Learning Framework from Scratch**

Welcome to TinyTorch! You're about to build a complete ML framework from the ground up. By the end of this course, you'll have implemented your own neural networks, data loaders, and training systems - then used them to solve real problems.

## 🎯 **What You'll Build**

- **Your own ML framework** that you can `pip install` and use
- **Neural networks** that classify real images (CIFAR-10)
- **Data loading systems** that handle production datasets
- **Complete understanding** of how ML systems actually work

## 🚀 **Quick Start** (First 10 Minutes)

### **1. Get Started**
```bash
cd TinyTorch
tito system info         # Check your system
tito system doctor       # Verify everything works
```

### **2. Start Your First Assignment**
```bash
cd assignments/source/00_setup
jupyter lab setup_dev.py
```

### **3. Complete the Setup**
- Follow the notebook instructions
- Complete the TODO sections
- Run the tests to verify your work

## 📚 **Course Progress Tracker**

Track your journey through TinyTorch:

### **✅ Ready to Start** (6+ weeks of content)
- [ ] **00_setup** - Development workflow & CLI tools
- [ ] **02_activations** - ReLU, Sigmoid, Tanh functions
- [ ] **03_layers** - Dense layers & neural building blocks
- [ ] **04_networks** - Sequential networks & MLPs
- [ ] **06_dataloader** - CIFAR-10 data loading
- [ ] **05_cnn** - Convolution operations

### **🚧 Coming Soon**
- [ ] **01_tensor** - Tensor arithmetic (partially working)
- [ ] **07_autograd** - Automatic differentiation
- [ ] **08_optimizers** - SGD, Adam optimizers
- [ ] **09_training** - Complete training loops
- [ ] **Future modules** - Advanced ML systems

## 🛠️ **Your Daily Workflow** (5 Simple Steps)

This is your rhythm for every module:

### **Step 1: Open Assignment**
```bash
cd assignments/source/00_setup
jupyter lab setup_dev.py
```

### **Step 2: Learn & Implement**
- Read the markdown explanations
- Complete each TODO section
- Test your understanding as you go

### **Step 3: Export Your Code**
```bash
python bin/tito module export 00_setup
```

### **Step 4: Test Your Work**
```bash
python -m pytest modules/00_setup/tests/ -v
```

### **Step 5: Use Your Code**
```bash
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
```

**🎉 When all tests pass, you're ready for the next module!**

## 📋 **Module Details**

### **🔧 Module 00: Setup**
**Goal**: Learn the development workflow
**Time**: 30-45 minutes
**Tests**: 20/20 must pass

**What you'll build**:
- Development environment setup
- CLI tool familiarity
- Your first TinyTorch function

**Verification**:
```bash
tito system doctor           # Should show all ✅
python -c "from tinytorch.core.utils import hello_tinytorch; hello_tinytorch()"
```

---

### **🧠 Module 02: Activations**
**Goal**: Build core ML math functions
**Prerequisites**: Module 00 complete
**Tests**: 24/24 must pass

**What you'll build**:
- ReLU activation function
- Sigmoid activation function  
- Tanh activation function
- Understanding of neural network math

**Verification**:
```bash
python -c "from tinytorch.core.activations import ReLU; print(ReLU()([-1, 0, 1]))"
```

---

### **🏗️ Module 03: Layers**
**Goal**: Build neural network layers
**Prerequisites**: Module 02 complete
**Tests**: 17/22 passing (core features work)

**What you'll build**:
- Dense/Linear layers
- Parameter management
- Forward pass implementation
- Layer composition

**Verification**:
```bash
python -c "from tinytorch.core.layers import Dense; layer = Dense(10, 5); print(layer)"
```

---

### **🖼️ Module 04: Networks**
**Goal**: Build complete neural networks
**Prerequisites**: Module 03 complete
**Tests**: 20/25 passing (core features work)

**What you'll build**:
- Sequential network container
- Multi-layer perceptron (MLP)
- Network composition
- Model architecture design

**Verification**:
```bash
python -c "from tinytorch.core.networks import Sequential, create_mlp; net = create_mlp([10, 5, 1]); print(net)"
```

---

### **📊 Module 06: DataLoader**
**Goal**: Handle real production data
**Prerequisites**: Module 04 complete
**Tests**: 15/15 must pass

**What you'll build**:
- CIFAR-10 dataset loading
- Batch processing
- Data transformations
- Production data pipeline

**Verification**:
```bash
python -c "from tinytorch.core.dataloader import DataLoader; loader = DataLoader('cifar10'); print(next(loader))"
```

---

### **🔍 Module 05: CNN**
**Goal**: Basic convolution operations
**Prerequisites**: Module 06 complete
**Tests**: 2/2 must pass

**What you'll build**:
- Conv2D layer implementation
- Basic convolution math
- Image processing foundations

**Verification**:
```bash
python -c "from tinytorch.core.cnn import Conv2D; conv = Conv2D(3, 16, 3); print(conv)"
```

## 🆘 **Getting Help**

### **Tests Are Failing?**
```bash
cd modules/XX
python -m pytest tests/ -v    # See detailed error messages
```

### **Can't Import Your Code?**
```bash
python bin/tito module export XX    # Re-export your module
tito package sync                    # Sync all modules
```

### **Environment Issues?**
```bash
tito system doctor                   # Check your environment
source .venv/bin/activate            # Activate virtual environment
```

### **General Debugging**
```bash
tito system info                     # Check system status
tito module status                   # See all module progress
```

## 💡 **Study Tips**

### **Start Small**
- Complete one TODO section at a time
- Test your code frequently
- Use `print()` statements to debug

### **Build Understanding**
- Read the markdown explanations carefully
- Connect new concepts to previous modules
- Try variations of the examples

### **Use Real Data**
- Work with CIFAR-10 (not toy datasets)
- See how your code handles realistic problems
- Understand performance implications

### **Celebrate Progress**
- Each passing test is a victory
- You're building real ML systems
- Your code becomes part of a working framework

## 🏆 **Success Milestones**

### **Week 1-2: Foundation**
- [ ] Complete setup and activations
- [ ] Understand development workflow
- [ ] See your first neural network math working

### **Week 3-4: Building Blocks**
- [ ] Build complete neural network layers
- [ ] Compose layers into networks
- [ ] Create your first MLP

### **Week 5-6: Real Systems**
- [ ] Load and process CIFAR-10 data
- [ ] Build basic convolution operations
- [ ] Train networks on real images

### **Beyond: Advanced Features**
- [ ] Complete tensor arithmetic
- [ ] Implement automatic differentiation
- [ ] Build training loops and optimizers

## 🎓 **Learning Philosophy**

### **Build → Use → Understand → Repeat**
1. **Build**: You implement `ReLU()` from scratch
2. **Use**: You immediately use `from tinytorch.core.activations import ReLU`
3. **Understand**: You see how it works in real networks
4. **Repeat**: Each module builds on this foundation

### **Real Data, Real Systems**
- Work with CIFAR-10 (10,000 real images)
- Handle production-scale datasets
- Build systems that actually work

### **Immediate Feedback**
- Tests show you're on the right track
- Your code exports to a real package
- You can use your implementations immediately

---

## 🚀 **Ready to Start?**

### **Your First Session**
1. **Open terminal**: `cd TinyTorch`
2. **Check system**: `tito system doctor`
3. **Start assignment**: `cd assignments/source/00_setup && jupyter lab setup_dev.py`
4. **Follow instructions**: Complete the TODO sections
5. **Test your work**: `python -m pytest tests/ -v`

### **When You're Stuck**
- Read the error messages carefully
- Check the module README: `cat modules/XX/README.md`
- Ask your instructor for help
- Remember: everyone gets stuck sometimes!

**🎉 You're about to build your own ML framework. Let's start with module 00_setup!** 