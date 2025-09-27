# TinyTorch User Manual

## Welcome to Your ML Systems Engineering Journey

This comprehensive user manual will guide you from installation to mastery, whether you're spending 15 minutes exploring or 15 weeks building complete systems.

## 🧭 Navigation Guide

### **For New Users**
- **[🚀 Quick Start](#quick-start)** - Get running in 5 minutes
- **[🎯 Choose Your Path](#learning-paths)** - Find your ideal learning journey
- **[📱 First Commands](#essential-commands)** - Master the basics

### **For Active Learners**
- **[📚 Module Guide](#module-system)** - Understand the learning structure
- **[✅ Checkpoint System](#checkpoint-system)** - Track and validate progress
- **[🌍 Community Features](#community-features)** - Connect with fellow learners

### **For Instructors**
- **[🎓 Teaching Guide](#instructor-resources)** - Classroom setup and management
- **[📊 Progress Tracking](#student-progress)** - Monitor student achievements
- **[📝 Grading System](#nbgrader-integration)** - Automated assessment workflow

---

## 🚀 Quick Start

### **Installation (3 minutes)**

```bash
# 1. Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Verify installation
tito system doctor
```

### **First Experience (2 minutes)**

```bash
# See what you'll build
tito demo quick

# Check your learning path
tito checkpoint status

# Start your journey
tito help --interactive
```

**✅ Success Indicators:**
- `tito system doctor` shows all green checkmarks
- `tito checkpoint status` displays 16 learning checkpoints
- `tito demo quick` runs without errors

---

## 🎯 Learning Paths

Choose your journey based on your goals and available time:

### 🔬 **Explorer Path** (15 minutes - 2 hours)
**Goal**: Understand what TinyTorch is and see it in action

```bash
# Quick demonstration
tito demo quick

# See the big picture
tito checkpoint timeline --horizontal

# Try building something small
cd modules/source/01_setup
jupyter lab setup_dev.py
```

**You'll Experience:**
- How neural networks work at the code level
- What building ML systems from scratch looks like
- Whether you want to go deeper

---

### 🎯 **Builder Path** (Weekend - 4 weeks)
**Goal**: Build substantial ML components and understand systems

```bash
# Start structured learning
tito checkpoint status
cd modules/source/01_setup

# Work through foundation modules
# Complete 1-2 modules per session
# Goal: Build working neural network (Modules 1-6)
```

**Milestones:**
- **Week 1**: Tensors and basic operations (Modules 1-2)
- **Week 2**: Neural network components (Modules 3-5)  
- **Week 3**: Training systems (Modules 6-8)
- **Week 4**: First real project - CIFAR-10 CNN

---

### 🚀 **Engineer Path** (8-12 weeks)
**Goal**: Complete framework capable of modern ML applications

```bash
# Full curriculum with community participation
tito leaderboard join
tito checkpoint status

# Systematic progression through all modules
# Regular community engagement
# Optimization and competition participation
```

**Journey Stages:**
1. **Foundation** (Weeks 1-4): Neural networks from scratch
2. **Architecture** (Weeks 5-7): Computer vision and language models
3. **Training** (Weeks 8-10): Complete training systems
4. **Optimization** (Weeks 11-12): Performance and deployment
5. **Mastery** (Ongoing): TinyMLPerf competition and community

---

### 🎓 **Instructor Path**
**Goal**: Teach TinyTorch to students with full classroom support

```bash
# Instructor setup
tito nbgrader setup-instructor
tito grade setup-course

# Student progress tracking
tito leaderboard instructor-dashboard
```

**Resources:**
- **[Classroom Setup Guide](usage-paths/classroom-use.html)** - Complete NBGrader workflow
- **[Student Progress Tracking](#student-progress)** - Monitor achievements
- **[Assessment Resources](#instructor-resources)** - Grading and feedback tools

---

## 📱 Essential Commands

### **Daily Learning Workflow**

```bash
# Check your progress
tito checkpoint status

# Work on current module
cd modules/source/0X_module_name
jupyter lab module_name_dev.py

# Complete module when done
tito module complete 0X_module_name

# Celebrate achievement
tito checkpoint test XX
```

### **Getting Help**

```bash
# Interactive guidance
tito help --interactive

# Quick reference
tito help --quick

# Specific help topics
tito help getting-started
tito help workflow
tito help troubleshooting
```

### **Community Engagement**

```bash
# Join the global community
tito leaderboard join

# Submit your progress
tito leaderboard submit

# See your ranking
tito leaderboard status

# Compete in Olympics
tito olympics register
```

### **System Management**

```bash
# Check system health
tito system doctor

# Clean up generated files
tito clean all

# Reset progress (careful!)
tito reset --confirm
```

---

## 📚 Module System

### **Understanding the Structure**

TinyTorch organizes learning into **20 progressive modules**, each building essential ML systems capabilities:

```
modules/source/
├── 01_setup/           📦 Development environment
├── 02_tensor/          🔢 N-dimensional arrays + operations  
├── 03_activations/     📈 ReLU, Sigmoid, Softmax
├── 04_layers/          🧱 Linear layers + parameters
├── 05_losses/          📉 CrossEntropy, MSE + gradients
├── 06_autograd/        🔄 Automatic differentiation
├── 07_optimizers/      📊 SGD, Adam + learning schedules
├── 08_training/        🎯 Complete training loops
├── 09_spatial/         🖼️  Conv2d, MaxPool2d + CNNs
├── 10_dataloader/      📂 Efficient data pipelines
├── 11_tokenization/    📝 Text processing + vocabularies
├── 12_embeddings/      🎭 Token + positional embeddings
├── 13_attention/       👁️  Multi-head attention
├── 14_transformers/    🤖 Complete transformer blocks
├── 15_profiling/       🔍 Performance analysis
├── 16_acceleration/    ⚡ Hardware optimization
├── 17_quantization/    📦 Model compression
├── 18_compression/     🗜️  Pruning + distillation
├── 19_caching/         💾 Memory optimization
└── 20_capstone/        🏆 Complete ML systems
```

### **Module Workflow**

Each module follows the proven **Build → Use → Reflect** pattern:

#### **1. Build Implementation**
```python
# In module_name_dev.py
def your_implementation():
    """Build component from scratch using only NumPy."""
    return result
```

#### **2. Use Immediately**
```python
# Test your implementation
from tinytorch.core.module_name import YourComponent
component = YourComponent()
result = component(data)
```

#### **3. Reflect on Systems**
- **Memory Analysis**: How much RAM does this use?
- **Performance Profile**: Where are the bottlenecks?
- **Scaling Behavior**: What breaks with larger inputs?
- **Production Context**: How do real systems handle this?

#### **4. Export and Validate**
```bash
# Export your implementation to the framework
tito module complete 0X_module_name

# Automatically runs checkpoint test
# Celebrates achievement
# Shows next steps
```

---

## ✅ Checkpoint System

### **16 Capability Checkpoints**

The checkpoint system validates your learning through **capability-based assessment**:

```bash
# See all checkpoints
tito checkpoint timeline

# Check current progress  
tito checkpoint status

# Test specific capability
tito checkpoint test 05
```

### **Checkpoint Progression**

| Checkpoint | Capability Question | Prerequisites |
|------------|-------------------|---------------|
| 00 | Can I configure my development environment? | Setup complete |
| 01 | Can I create and manipulate ML building blocks? | Module 02 |
| 02 | Can I add nonlinearity for intelligence? | Module 03 |
| 03 | Can I build neural network components? | Module 04 |
| 04 | Can I build complete multi-layer networks? | Module 05 |
| 05 | Can I process spatial data with convolutions? | Module 09 |
| 06 | Can I build attention mechanisms? | Module 13 |
| 07 | Can I stabilize training with normalization? | Module 08 |
| 08 | Can I compute gradients automatically? | Module 06 |
| 09 | Can I optimize with sophisticated algorithms? | Module 07 |
| 10 | Can I build complete training loops? | Module 08 |
| 11 | Can I prevent overfitting? | Module 11 |
| 12 | Can I implement high-performance kernels? | Module 16 |
| 13 | Can I analyze and optimize performance? | Module 15 |
| 14 | Can I deploy ML systems in production? | Module 20 |
| 15 | Can I build complete end-to-end systems? | Capstone |

### **Checkpoint Achievement Flow**

```bash
# Automatic flow when completing modules
tito module complete 02_tensor
# ↓ Automatically triggers
# ↓ Export to tinytorch package  
# ↓ Run checkpoint_01_foundation test
# ↓ Show achievement celebration
# ↓ Display next steps
```

---

## 🌍 Community Features

### **Global Learning Community**

Join thousands of learners worldwide building ML systems:

```bash
# Join the community
tito leaderboard join

# Submit your progress
tito leaderboard submit

# See global rankings
tito leaderboard view
```

### **Leaderboard Categories**

**🏃‍♂️ Progress Leaderboard**
- **Checkpoint completion**: How many capabilities achieved?
- **Module completion**: Which modules finished?
- **Achievement dates**: When did you reach milestones?

**🏆 Performance Olympics**
- **Speed competitions**: Fastest training times
- **Memory challenges**: Most memory-efficient implementations
- **Accuracy contests**: Highest model performance
- **Innovation showcases**: Novel optimization techniques

### **Community Interaction**

```bash
# See your community profile
tito leaderboard profile

# View achievement feed
tito leaderboard feed

# Join competitions
tito olympics register --event cnn_marathon

# Share achievements
tito leaderboard share --milestone "First neural network!"
```

### **Privacy and Inclusion**

- **Pseudonymous participation**: Choose your display name
- **Inclusive categories**: Multiple ways to excel and contribute
- **Supportive community**: Celebration of all learning achievements
- **Privacy controls**: Share what you're comfortable sharing

---

## 🎓 Instructor Resources

### **Classroom Setup**

Complete NBGrader integration for seamless course management:

```bash
# Initial instructor setup
tito nbgrader setup-instructor
tito grade setup-course

# Student workspace preparation
tito nbgrader create-student-repos
```

### **Student Progress Tracking**

```bash
# Class overview
tito grade class-overview

# Individual student progress
tito grade student-progress <student_name>

# Checkpoint completion rates
tito checkpoint class-stats
```

### **Assignment Management**

```bash
# Release new module
tito nbgrader release 05_losses

# Collect submissions
tito nbgrader collect 05_losses

# Auto-grade submissions
tito nbgrader autograde 05_losses

# Manual grading interface
tito nbgrader formgrade 05_losses
```

### **Course Customization**

**Semester Planning:**
- **8-week intensive**: Modules 1-12 (foundations + one specialization)
- **16-week comprehensive**: All 20 modules with optimization
- **4-week bootcamp**: Modules 1-8 (neural network foundations)

**Difficulty Adjustment:**
- **Beginner**: Extended explanations and scaffolding
- **Advanced**: Additional optimization challenges
- **Research**: Custom project integration

---

## 🔧 Troubleshooting Guide

### **Common Issues and Solutions**

#### **Installation Problems**

**Issue**: `tito: command not found`
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

**Issue**: Import errors in modules
```bash
# Solution: Check system health
tito system doctor

# Fix common issues
pip install -r requirements.txt --force-reinstall
```

#### **Module Development Issues**

**Issue**: Notebook won't open
```bash
# Solution: Check Jupyter installation
pip install jupyter jupyterlab
jupyter lab --version
```

**Issue**: Tests failing after implementation
```bash
# Solution: Debug with verbose output
tito checkpoint test 03 --verbose

# Check implementation against expected interface
python modules/source/03_activations/activations_dev.py
```

#### **Export and Integration Issues**

**Issue**: `tito module complete` fails
```bash
# Solution: Check module structure
tito module validate 05_losses

# Fix export directives
# Ensure #| default_exp tinytorch.core.losses at top of file
```

**Issue**: Checkpoint tests fail after export
```bash
# Solution: Check package imports
python -c "from tinytorch.core.losses import CrossEntropyLoss; print('Success')"

# Reinstall in development mode
pip install -e . --force-reinstall
```

### **Getting More Help**

1. **Interactive CLI Help**: `tito help --interactive`
2. **System Diagnostics**: `tito system doctor`
3. **Community Support**: Join the leaderboard for peer help
4. **Documentation**: Check module README files
5. **Instructor Support**: Contact course staff through established channels

---

## 📊 Frequently Asked Questions

### **Learning Questions**

**Q: How long does it take to complete TinyTorch?**
A: Depends on your goals:
- **Quick exploration**: 15 minutes - 2 hours
- **Weekend project**: Build neural networks (8-12 hours)  
- **Complete journey**: 8-12 weeks for full framework
- **Instructor preparation**: 2-3 weeks for course setup

**Q: Do I need ML experience to start?**
A: No! TinyTorch teaches ML systems from fundamentals. You need:
- Basic Python programming (functions, classes)
- High school math (matrix multiplication)
- Curiosity about how things work internally

**Q: How is this different from PyTorch tutorials?**
A: PyTorch teaches you to USE frameworks. TinyTorch teaches you to BUILD them:
- **PyTorch**: `torch.nn.Linear(784, 128)` (black box)
- **TinyTorch**: You implement every line of Linear layer
- **Result**: Deep understanding of how frameworks actually work

### **Technical Questions**

**Q: What's the difference between modules and checkpoints?**
A: 
- **Modules**: 20 hands-on coding sessions where you build components
- **Checkpoints**: 16 capability tests that validate your learning
- **Relationship**: Modules provide code, checkpoints verify understanding

**Q: Can I skip modules or do them out of order?**
A: No, the progression is carefully designed:
- Each module builds on previous ones
- Checkpoints verify prerequisites  
- Skipping breaks the learning flow and later modules won't work

**Q: What if I get stuck on a module?**
A: Multiple support options:
- `tito help troubleshooting` for common issues
- `tito system doctor` for technical problems
- Community leaderboard for peer support
- Module README files for detailed guidance

### **Community Questions**

**Q: Is the leaderboard competitive or collaborative?**
A: Both! We celebrate all achievements:
- **Multiple categories**: Progress, speed, memory efficiency, innovation
- **Inclusive design**: Many ways to excel and contribute
- **Supportive community**: Everyone's learning journey matters
- **Privacy controls**: Share only what you're comfortable sharing

**Q: Can I participate without sharing my progress publicly?**
A: Yes! Leaderboard participation is optional:
- All learning features work independently
- Checkpoint system tracks your progress locally
- You can join community later if you change your mind

### **Instructor Questions**

**Q: How much setup is required for classroom use?**
A: Minimal - TinyTorch includes complete teaching infrastructure:
- NBGrader integration works out-of-the-box
- Student repositories auto-generated
- Progress tracking built-in
- Grading workflow automated

**Q: Can I customize the curriculum for my class?**
A: Absolutely:
- **Flexible duration**: 4-16 weeks depending on depth
- **Difficulty adjustment**: Extra scaffolding or advanced challenges
- **Custom projects**: Integration with existing coursework
- **Modular design**: Focus on specific topics as needed

---

## 🚀 Next Steps

### **Ready to Start?**

Choose your path and begin your ML systems engineering journey:

🔬 **[Explorer (15 minutes)](#explorer-path)**: Quick taste with `tito demo quick`

🎯 **[Builder (Weekend)](#builder-path)**: Build neural networks from scratch

🚀 **[Engineer (8-12 weeks)](#engineer-path)**: Complete framework development

🎓 **[Instructor](#instructor-path)**: Teach TinyTorch to your students

### **Essential First Commands**

```bash
# System check
tito system doctor

# Interactive guidance  
tito help --interactive

# See the journey ahead
tito checkpoint timeline

# Start building
cd modules/source/01_setup
jupyter lab setup_dev.py
```

### **Join the Community**

```bash
# Connect with learners worldwide
tito leaderboard join

# Share your progress
tito leaderboard submit

# Compete and learn
tito olympics explore
```

---

**You're about to build everything from tensors to transformers. Let's start your journey! 🚀**