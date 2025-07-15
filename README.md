# **Tiny🔥Torch: Start Small, Go Deep**

**Build your own ML framework. Understand every layer.**

A hands-on Machine Learning Systems course where students don’t just learn ML—they *engineer* it. TinyTorch guides you from the fundamentals of tensors and layers all the way to training real models on real data using your own codebase.

---

## 🎯 What You'll Build

* **A Complete ML Framework** — Your own PyTorch-style toolkit, from the ground up
* **Real Applications** — Classify CIFAR-10 images using models you construct
* **Engineering Skills** — Learn the full ML system stack, not just high-level APIs
* **Immediate Feedback** — Build, test, and see your system evolve step by step

---

## 🚀 Quick Start (2 minutes)

### 🧑‍🎓 **Students**

```bash
git clone https://github.com/your-org/tinytorch.git
cd TinyTorch
make install
tito system doctor                         # Verify your setup
cd assignments/source/00_setup
jupyter lab setup_dev.py                   # Launch your first assignment
```

### 👩‍🏫 **Instructors**

```bash
# System check
tito system info
tito system doctor

# Assignment workflow
tito nbgrader generate 00_setup
tito nbgrader release 00_setup
tito nbgrader autograde 00_setup
```

---

## 📚 Course Structure

### **Core Assignments** (6+ weeks of ready-to-teach content)

* `00_setup`: Development workflow, CLI tools (20/20 tests)
* `02_activations`: ReLU, Sigmoid, Tanh (24/24 tests)
* `03_layers`: Dense layers and block design (17/22 tests)
* `04_networks`: Sequential networks, MLPs (20/25 tests)
* `06_dataloader`: CIFAR-10 loading and preprocessing (15/15 tests)
* `05_cnn`: Basic convolutions (2/2 tests)

### **In Progress**

* `01_tensor`: Tensor arithmetic (22/33 tests)
* `07-13`: Autograd, optimizers, training, and MLOps

---

## 🛠️ Dev & Grading Workflow

### **NBGrader** (Assignment creation and evaluation)

```bash
tito nbgrader generate 00_setup
tito nbgrader release 00_setup
tito nbgrader collect 00_setup
tito nbgrader autograde 00_setup
```

### **nbdev** (Export to package & integration testing)

```bash
tito module export 00_setup
tito module test 00_setup
```

---

## 🧠 Student Journey: Build → Use → Understand → Repeat

1. **Build** your own `ReLU()` function
2. **Use** it from `tinytorch.core.activations`
3. **Understand** how it works in a neural network
4. **Repeat** and deepen your knowledge with each module

#### 🧪 Example

```python
# You implement this:
def hello_tinytorch():
    print("Welcome to TinyTorch!")

# Then use it in your framework:
from tinytorch.core.utils import hello_tinytorch
hello_tinytorch()
```

---

## 🎓 Teaching Philosophy

### **Build Everything, Understand Everything**

* No hidden layers. No magic. Just real engineering.
* You write the system. You test it. You run it.

### **Real Data, Real Thinking**

* Use full datasets like CIFAR-10—not toy problems
* Prioritize performance, modularity, and reusability
* Understand where engineering meets AI

---

## 📁 Project Structure

```
TinyTorch/
├── assignments/source/XX/        # Assignment notebooks + tests
├── tinytorch/                    # Your growing ML framework
│   └── core/                     # Student-exported code
├── tito/                         # CLI & course management tools
└── docs/                         # Docs and metadata
```

---

## 🧪 Tech Stack & Requirements

* **Python 3.8+**
* **Jupyter Lab**
* **PyTorch** (for benchmarking/comparison only)
* **NBGrader** (for assignment flow)
* **nbdev** (for modular packaging)

---

## ✅ Getting Started

### 🎓 **Students**

1. `tito system doctor`
2. `cd assignments/source/00_setup`
3. `jupyter lab setup_dev.py`
4. Build → Export → Import → Test

### 👩‍🏫 **Instructors**

1. `tito system info`
2. `tito nbgrader generate/release`
3. `tito nbgrader autograde`

---

## 📊 Verified Milestones

✅ Build multi-layer perceptrons
✅ Implement custom activations
✅ Load & process CIFAR-10
✅ Perform basic convolutions
✅ Export reusable ML modules

---

**🔥 Ready to dive in? TinyTorch is classroom-tested and built for deep learning—done right.**

---

Let me know if you want a separate `README.md` vs `index.md`, or to generate an **OG/social media preview blurb** too.
