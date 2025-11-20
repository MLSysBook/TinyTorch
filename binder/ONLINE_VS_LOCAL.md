# Online Notebooks vs Local Setup

## Important Distinction

### Online Notebooks (Binder, Colab, Marimo)
**Purpose**: Viewing, learning, exploration

**What you CAN do:**
- ✅ View notebook content
- ✅ Read code and explanations
- ✅ Run basic code cells
- ✅ Learn from examples

**What you CANNOT do:**
- ❌ Import from `tinytorch.*` package (not installed)
- ❌ Run milestone validation scripts
- ❌ Use `tito` CLI commands
- ❌ Execute full experiments
- ❌ Export modules to package
- ❌ Complete the full development workflow

### Local Setup (Required)
**Purpose**: Full package, experiments, milestone validation

**What you CAN do:**
- ✅ Full `tinytorch.*` package available
- ✅ Run milestone validation scripts
- ✅ Use `tito` CLI commands (`tito module complete`, `tito milestone validate`)
- ✅ Execute complete experiments
- ✅ Export modules to package
- ✅ Full development workflow

## When to Use What

### Use Online Notebooks When:
- 📖 **Learning**: Reading through modules to understand concepts
- 🔍 **Exploration**: Quick look at code examples
- 💡 **Inspiration**: Seeing how things work before implementing
- 🚀 **Quick Start**: Getting familiar with the structure

### Use Local Setup When:
- 🏗️ **Building**: Actually implementing modules
- ✅ **Validating**: Running milestone checks
- 🧪 **Experimenting**: Running full experiments
- 📦 **Exporting**: Completing modules and exporting to package
- 🎯 **Serious Work**: Doing the actual coursework

## Setup Instructions

### Local Setup (Required for Full Package)

```bash
# 1. Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install TinyTorch package in editable mode
pip install -e .

# 5. Verify installation
tito system doctor
```

Now you have:
- ✅ Full `tinytorch.*` package available
- ✅ `tito` CLI commands working
- ✅ Milestone scripts executable
- ✅ Complete development environment

## Student Workflow

**Recommended approach:**

1. **Start Online**: Use Binder/Colab/Marimo to explore and understand modules
2. **Switch to Local**: When ready to build, set up local environment
3. **Work Locally**: Implement modules, run milestones, use CLI tools
4. **Submit**: Export and submit `.ipynb` files for grading

## Common Questions

**Q: Can I do everything online?**
A: No. Online notebooks are for viewing/learning. You need local setup for the full package and experiments.

**Q: Do I need both?**
A: Not required, but recommended. Use online for learning, local for building.

**Q: Can I use online notebooks for assignments?**
A: You can view notebooks online, but you'll need local setup to actually complete modules and run milestone validations.

**Q: What if I only have online access?**
A: You can learn from online notebooks, but you won't be able to complete the full coursework without local installation.

## Summary

- **Online Notebooks**: Great for learning and exploration
- **Local Setup**: Required for building, validating, and completing modules
- **Best Practice**: Use online to learn, local to build

