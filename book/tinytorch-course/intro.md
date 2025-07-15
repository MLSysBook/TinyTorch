# 🔥 TinyTorch: Build Machine Learning Systems from Scratch

**Learn ML by building your own PyTorch-style framework from the ground up.**

---

## 🎯 What You'll Build

By the end of this course, you'll have built:
- **Your own ML framework** with tensors, layers, networks, and optimizers
- **Real applications** that classify CIFAR-10 images using your code
- **Production skills** in ML systems engineering, not just algorithms
- **Deep understanding** of how modern ML frameworks actually work

---

## 🚀 How to Use TinyTorch

### 🔬 Option 1: Quick Exploration (5 minutes)
**Perfect for:** "I want to see what this is about"

```{admonition} Try Immediately - No Setup Required
:class: tip
Click the 🚀 **Launch Binder** button at the top of any chapter to:
- Get immediate hands-on experience
- Implement real ML components (ReLU, tensors, neural networks)
- See your code working instantly in the browser
- No installation, no account needed

**Note:** Binder sessions are temporary (timeout after ~20 minutes idle). Great for exploration, but work isn't saved permanently.
```

**What you'll experience:**
1. Click "Launch Binder" on any chapter
2. Get a live Jupyter environment with incomplete code
3. Fill in the blanks to implement ReLU, Tensor operations, etc.
4. Run tests to see your implementations working
5. Immediately understand how ML frameworks work under the hood

---

### 🏗️ Option 2: Build Your Own TinyTorch (Serious Learners)
**Perfect for:** "I want to build this myself" or "This is a class assignment"

```{admonition} Full Development Environment
:class: important
For persistent work, multi-session projects, and building your complete ML framework:

**Step 1: Get the Code**
```bash
git clone https://github.com/your-org/tinytorch.git
cd TinyTorch
```

**Step 2: Setup Environment**
```bash
make install              # Install dependencies
tito system doctor        # Verify everything works
```

**Step 3: Start Building**
```bash
cd modules/source/00_setup
jupyter lab setup_dev.py  # Open first assignment
```

**Step 4: Build → Test → Export → Use**
```bash
# After implementing code in the notebook:
tito export               # Export your code to tinytorch package
tito test setup          # Test your implementation

# Now use your own code:
python -c "from tinytorch.core.setup import hello_tinytorch; hello_tinytorch()"
```
```

**What you'll build:**
- **Module 00-02**: Development workflow, tensors, activation functions  
- **Module 03-05**: Neural layers, networks, convolutions
- **Module 06-08**: Data loading, autograd, optimizers
- **Module 09-13**: Training systems, compression, production MLOps

---

### 👨‍🏫 Option 3: Classroom Use (Instructors)
**Perfect for:** Teaching ML systems in a structured course

```{admonition} Complete Course Infrastructure
:class: note
TinyTorch includes full NBGrader integration for classroom management:

**Assignment Creation:**
```bash
tito nbgrader generate 00_setup    # Create student assignments
tito nbgrader release 00_setup     # Release to students
```

**Grading & Feedback:**
```bash
tito nbgrader collect 00_setup     # Collect submissions
tito nbgrader autograde 00_setup   # Auto-grade with tests
tito nbgrader feedback 00_setup    # Generate feedback
```

**Course Status:**
```bash
tito system info                   # Check all module status
tito system doctor                 # Verify environment
```
```

---

## 🎓 Learning Philosophy: Build → Use → Understand

### Example: How You'll Learn Activation Functions

**🔧 Build:** Implement ReLU from scratch
```python
def relu(x):
    # YOU implement this function
    return ???  # What should this be?
```

**🚀 Use:** Immediately use your own code
```python
from tinytorch.core.activations import ReLU  # YOUR implementation!
layer = ReLU()
output = layer.forward(input_tensor)  # Your code working!
```

**💡 Understand:** See it working in real networks
```python
# Your ReLU is now part of a real neural network
model = Sequential([
    Dense(784, 128),
    ReLU(),           # <-- Your implementation
    Dense(128, 10)
])
```

---

## 📊 What Students Build (Proven Results)

```{admonition} Real Student Outcomes
:class: success
**After 6 weeks, students can:**
- ✅ Build multi-layer perceptrons from scratch
- ✅ Implement custom activation functions (ReLU, Sigmoid, Tanh)
- ✅ Load and process real datasets (CIFAR-10)
- ✅ Create convolution operations
- ✅ Export their code to a working Python package
- ✅ Use their own ML framework for real image classification

**Test Coverage:** 100+ automated tests ensure student implementations work correctly
```

---

## 🛣️ Your Learning Path

### Phase 1: Foundation (Modules 0-2)
- **Setup & CLI**: Professional development workflow
- **Tensors**: Multi-dimensional arrays and operations  
- **Activations**: ReLU, Sigmoid, Tanh functions

### Phase 2: Building Blocks (Modules 3-5)
- **Layers**: Dense layers and neural building blocks
- **Networks**: Sequential models and MLPs
- **CNNs**: Convolutional operations

### Phase 3: Training Systems (Modules 6-9)
- **DataLoader**: Real dataset handling (CIFAR-10)
- **Autograd**: Automatic differentiation
- **Optimizers**: SGD, Adam, learning schedules
- **Training**: Complete training loops

### Phase 4: Production (Modules 10-13)
- **Compression**: Model optimization techniques
- **Kernels**: High-performance operations
- **Benchmarking**: Performance measurement
- **MLOps**: Production deployment

---

## 🚀 Start Your Journey

```{admonition} Choose Your Path
:class: tip
**Just Exploring?** → Click 🚀 **Launch Binder** on any chapter below

**Ready to Build?** → Follow the setup instructions above

**Teaching a Class?** → Check out the instructor documentation
```

**Next Steps:**
1. **Read the course overview** in the chapters below
2. **Try Chapter 1** (Setup) to understand the development workflow
3. **Experience Chapter 2** (Tensors) to implement your first ML component
4. **Build progressively** through layers, networks, and complete systems

---

*Built with ❤️ for hands-on ML systems education. Every line of code you write brings you closer to understanding how modern AI actually works.*
