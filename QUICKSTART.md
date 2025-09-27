# TinyTorch Quick Start Guide

## 🚀 5-Minute Start: From Zero to Running

Get TinyTorch running and see your future ML framework in action.

### **Step 1: Install (2 minutes)**

```bash
# Clone and enter directory
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install everything
pip install -r requirements.txt && pip install -e .
```

### **Step 2: Verify (30 seconds)**

```bash
# Check installation
tito system doctor
```
**✅ Expected:** All green checkmarks and "System ready!"

### **Step 3: Experience (2 minutes)**

```bash
# See your future framework in action
tito demo quick

# See your learning journey
tito checkpoint status

# Get personalized guidance
tito help --interactive
```

### **Step 4: Start Building (30 seconds)**

```bash
# Enter first module
cd modules/source/01_setup
jupyter lab setup_dev.py
```

**🎉 Success!** You're now building ML systems from scratch.

---

## 🎯 What Happens Next?

### **Your Learning Path**

```
Today: Setup & First Module     →  Week 1: Tensors & Operations
Week 2: Neural Networks         →  Week 4: Computer Vision  
Week 8: Language Models         →  Week 12: System Optimization
```

### **Essential Commands to Remember**

```bash
tito checkpoint status          # See your progress
tito module complete 0X         # Finish a module  
tito help --quick              # Quick reference
tito leaderboard join          # Join global community
```

### **When You Get Stuck**

```bash
tito system doctor             # Check technical issues
tito help troubleshooting      # Common problems
tito help --interactive        # Get guidance
```

---

## 📚 Choose Your Commitment Level

### 🔬 **Explorer (15 minutes)**
**Just want to see what this is about?**

```bash
tito demo quick                # See framework in action
tito checkpoint timeline       # View learning journey
```

### 🎯 **Weekend Builder (8-12 hours)**  
**Want to build something real?**

**Goal:** Build neural network that solves XOR problem
- Modules 1-6: Foundation components
- Test with: `python examples/xor_1969/minsky_xor_problem.py`

### 🚀 **Systems Engineer (8-12 weeks)**
**Ready for the full transformation?**

**Goal:** Complete ML framework with community participation
- All 20 modules with systematic progression
- Community leaderboard participation
- TinyMLPerf optimization competition

### 🎓 **Instructor (2-3 weeks setup)**
**Want to teach this course?**

```bash
tito nbgrader setup-instructor  # Classroom configuration
```
**Resources:** [Instructor Guide](book/usage-paths/classroom-use.html)

---

## 🌍 Join the Global Community

**Connect with learners worldwide building ML systems:**

```bash
# Join community leaderboard
tito leaderboard join

# See global progress
tito leaderboard view

# Compete in optimization challenges
tito olympics explore
```

**Why Join?**
- **Learn from peers** building the same systems
- **Celebrate milestones** with supportive community  
- **Compete in Olympics** for optimization mastery
- **Share achievements** and inspire others

---

## ❓ Quick FAQ

**Q: Do I need ML experience?**
A: No! Start with basic Python knowledge - we teach the rest.

**Q: How long does this take?**
A: Your choice:
- 15 minutes: Quick exploration
- Weekend: Build neural networks
- 8-12 weeks: Complete framework

**Q: What will I build?**
A: Your own ML framework capable of:
- Training CNNs on CIFAR-10 to 75%+ accuracy
- Building GPT-style language models
- Optimizing for production deployment

**Q: How is this different from PyTorch tutorials?**
A: PyTorch teaches you to USE frameworks. TinyTorch teaches you to BUILD them.

---

## 🎯 Ready to Start?

**Choose your first command:**

```bash
# 🔬 Quick exploration
tito demo quick

# 🎯 Structured learning  
tito help --interactive

# 🚀 Jump right in
cd modules/source/01_setup && jupyter lab setup_dev.py
```

**Next:** Follow the [Complete User Manual](book/user-manual.html) for detailed guidance.

---

**You're about to understand how ML frameworks really work. Let's build! 🚀**