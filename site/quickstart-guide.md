# Quick Start Guide

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">From Zero to Building Neural Networks</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Complete setup + first module in 15 minutes</p>
</div>

**Purpose**: Get hands-on experience building ML systems in 15 minutes. Complete setup verification and build your first neural network component from scratch.

## ⚡ 2-Minute Setup

Let's get you ready to build ML systems:

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">
<h4 style="margin: 0 0 1rem 0; color: #1976d2;">Step 1: One-Command Setup</h4>

```bash
# Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# Automated setup (handles everything!)
./setup-environment.sh

# Activate environment
source activate.sh
```

**What this does:**
- ✅ Creates optimized virtual environment (arm64 on Apple Silicon)
- ✅ Installs all dependencies (NumPy, Jupyter, Rich, PyTorch for validation)
- ✅ Configures TinyTorch in development mode
- ✅ Verifies installation

**📖 See [Essential Commands](tito-essentials.html)** for detailed workflow and troubleshooting.

</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin: 1.5rem 0;">
<h4 style="margin: 0 0 1rem 0; color: #15803d;">Step 2: Verify Setup</h4>

```bash
# Run system diagnostics
tito system doctor
```

You should see all green checkmarks! This confirms your environment is ready for hands-on ML systems building.

**📖 See [Essential Commands](tito-essentials.html)** for verification commands and troubleshooting.

</div>

## 🏗️ 15-Minute First Module Walkthrough

Let's build your first neural network component following the **TinyTorch workflow**:

```
1. Edit modules → 2. Export to package → 3. Validate with milestones
```

**📖 See [Student Workflow](student-workflow.html)** for the complete development cycle.

### Module 01: Tensor Foundations

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

**🎯 Learning Goal:** Build N-dimensional arrays - the foundation of all neural networks

**⏱️ Time:** 15 minutes

**💻 Action:** Start with Module 01 to build tensor operations from scratch.

```bash
# Step 1: Edit the module source
cd modules/source/01_tensor
jupyter lab 01_tensor_dev.py
```

You'll implement core tensor operations:
- N-dimensional array creation
- Basic mathematical operations (add, multiply, matmul)
- Shape manipulation (reshape, transpose)
- Memory layout understanding

**Key Implementation:** Build the `Tensor` class that forms the foundation of all neural networks

```bash
# Step 2: Export to package when ready
tito module complete 01
```

This makes your implementation importable: `from tinytorch import Tensor`

**📖 See [Student Workflow](student-workflow.html)** for the complete edit → export → validate cycle.

**✅ Achievement Unlocked:** Foundation capability - "Can I create and manipulate the building blocks of ML?"

</div>

### Next Step: Module 02 - Activations

<div style="background: #fdf2f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ec4899; margin: 1.5rem 0;">

**🎯 Learning Goal:** Add nonlinearity - the key to neural network intelligence

**⏱️ Time:** 10 minutes

**💻 Action:** Continue with Module 02 to add activation functions.

```bash
# Step 1: Edit the module
cd modules/source/02_activations
jupyter lab 02_activations_dev.py
```

You'll implement essential activation functions:
- ReLU (Rectified Linear Unit) - the workhorse of deep learning
- Softmax - for probability distributions
- Understand gradient flow and numerical stability
- Learn why nonlinearity enables learning

**Key Implementation:** Build activation functions that allow neural networks to learn complex patterns

```bash
# Step 2: Export when ready
tito module complete 02
```

**📖 See [Student Workflow](student-workflow.html)** for the complete edit → export → validate cycle.

**✅ Achievement Unlocked:** Intelligence capability - "Can I add nonlinearity to enable learning?"

</div>

## 📊 Track Your Progress

After completing your first modules:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Check your new capabilities:** Use the optional checkpoint system to track your progress:

```bash
tito checkpoint status  # View your completion tracking
```

This is helpful for self-assessment but not required for the core workflow.

**📖 See [Student Workflow](student-workflow.html)** for the essential edit → export → validate cycle, and [Track Your Progress](learning-progress.html)** for detailed capability tracking.

</div>

## 🏆 Validate with Historical Milestones

After exporting your modules, **prove what you've built** by running milestone scripts:

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 0.5rem; margin: 1.5rem 0; color: white;">

**After Module 07**: Build **Rosenblatt's 1957 Perceptron** - the first trainable neural network
**After Module 07**: Solve the **1969 XOR Crisis** with multi-layer networks
**After Module 08**: Achieve **95%+ accuracy on MNIST** with 1986 backpropagation
**After Module 09**: Hit **75%+ on CIFAR-10** with 1998 CNNs
**After Module 13**: Generate text with **2017 Transformers**
**After Module 18**: Optimize for production with **2018 MLPerf**

**📖 See [Journey Through ML History](chapters/milestones.html)** for complete timeline, requirements, and expected results.

</div>

**The Workflow**: Edit modules → Export with `tito module complete N` → Run milestone scripts to validate

**📖 See [Student Workflow](student-workflow.html)** for the complete cycle.

## 🎯 What You Just Accomplished

In 15 minutes, you've:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div style="background: #e6fffa; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #26d0ce;">
<h4 style="margin: 0 0 0.5rem 0; color: #0d9488;">🔧 Setup Complete</h4>
<p style="margin: 0; font-size: 0.9rem;">Installed TinyTorch and verified your environment</p>
</div>

<div style="background: #f0f9ff; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #3b82f6;">
<h4 style="margin: 0 0 0.5rem 0; color: #1d4ed8;">🧱 Created Foundation</h4>
<p style="margin: 0; font-size: 0.9rem;">Implemented core tensor operations from scratch</p>
</div>

<div style="background: #fefce8; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #eab308;">
<h4 style="margin: 0 0 0.5rem 0; color: #a16207;">🏆 First Capability</h4>
<p style="margin: 0; font-size: 0.9rem;">Earned your first ML systems capability checkpoint</p>
</div>

</div>

## 🚀 Your Next Steps

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0;">

### Immediate Next Actions (Choose One):

**🔥 Continue Building (Recommended):** Begin Module 03 to add layers to your network.

**📚 Master the Workflow:**
- **📖 See [Student Workflow](student-workflow.html)** for the complete edit → export → validate cycle
- **📖 See [Essential Commands](tito-essentials.html)** for complete TITO command reference
- **📖 See [Track Your Progress](learning-progress.html)** for the full learning path

**🎓 For Instructors:**
- **📖 See [Classroom Setup Guide](usage-paths/classroom-use.html)** for NBGrader integration (coming soon)

</div>

## 💡 Pro Tips for Continued Success

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**The TinyTorch Development Cycle:**
1. Edit module sources in `modules/source/`
2. Export with `tito module complete N`
3. Validate by running milestone scripts

**📖 See [Student Workflow](student-workflow.html)** for detailed workflow guide and best practices.

</div>

## 🌟 You're Now a TinyTorch Builder!

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Ready to Build Production ML Systems</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">You've proven you can build ML components from scratch. Time to keep going!</p>
<a href="chapters/03-activations.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Continue Building →</a>
<a href="tito-essentials.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Master Commands →</a>
</div>

---

**What makes TinyTorch different:** You're not just learning *about* neural networks—you're building them from fundamental mathematical operations. Every line of code you write builds toward complete ML systems mastery.

**Next milestone:** After Module 08, you'll train real neural networks on actual datasets using 100% your own code!