# 👨‍🏫 Instructor Guide: NBGrader + TinyTorch

<div style="background: #f0fff4; border: 1px solid #22c55e; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
<strong>📖 Technical Setup & Workflow:</strong> This page provides step-by-step NBGrader setup and daily semester management.<br>
<strong>📖 For Course Overview & Benefits:</strong> See <a href="usage-paths/classroom-use.html">TinyTorch for Instructors</a> for educational philosophy and course structure.
</div>

**Complete workflow for instructors and TAs using TinyTorch with automated grading**

---

## 🎯 The Complete Instructor Journey

This guide walks you through everything you need to know to successfully run a TinyTorch course with automated grading, from initial setup to semester completion.

---

## 📋 Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed on your system
- **Git** for version control
- **Terminal/Command Line** access
- **Basic familiarity** with Jupyter notebooks

**Time Investment:** ~30 minutes for initial setup, then 5-10 minutes per assignment

---

## 🚀 Phase 1: Initial Setup (One-Time)

### Step 1: Clone and Setup Repository

```bash
# Clone the TinyTorch repository
git clone https://github.com/your-org/TinyTorch.git
cd TinyTorch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install nbgrader jupyter jupytext rich
```

### Step 2: Verify Installation

```bash
# Test that everything is working
./bin/tito system doctor

# Expected output:
# ✅ Python 3.x.x
# ✅ Virtual environment active
# ✅ All dependencies installed
# ✅ TinyTorch CLI ready
```

### Step 3: Initialize NBGrader Environment

```bash
# Initialize the grading infrastructure
./bin/tito nbgrader init

# Expected output:
# ✅ NBGrader version: 0.9.5
# 📁 Created directory: assignments
# ✅ NBGrader database initialized
# 🎉 NBGrader environment initialized successfully!
```

### Step 4: Verify Complete Setup

```bash
# Check system status
./bin/tito module status --comprehensive

# Should show:
# 🐍 Environment Health: All ✅ green
# 📊 Module Status: Overview of 17 modules
# 🎯 Priority Actions: Any setup issues to fix
```

---

## 🎓 Phase 2: Course Preparation

### Understanding the TinyTorch Module Structure

TinyTorch has **17 progressive modules**:

**Foundation (Modules 00-02):**
- `00_introduction` - Visual system overview and dependencies
- `01_setup` - Development environment and CLI workflow  
- `02_tensor` - Multi-dimensional arrays and operations

**Building Blocks (Modules 03-07):**
- `03_activations` - Mathematical functions and nonlinearity
- `04_layers` - Neural network layer abstractions
- `05_dense` - Fully connected layers and matrix operations
- `06_spatial` - Convolutional operations and computer vision
- `07_attention` - Self-attention and transformer mechanisms

**Training Systems (Modules 08-11):**
- `08_dataloader` - Data pipeline and CIFAR-10 integration
- `09_autograd` - Automatic differentiation engine
- `10_optimizers` - SGD, Adam, and learning rate scheduling
- `11_training` - Training loops, loss functions, and metrics

**Production & Performance (Modules 12-16):**
- `12_compression` - Model pruning and quantization
- `13_kernels` - Custom operations and hardware optimization
- `14_benchmarking` - MLPerf-style evaluation and profiling
- `15_mlops` - Production deployment and monitoring
- `16_capstone` - Final integration project

### Course Planning Recommendations

**📅 Semester Planning (14-16 weeks):**
```
Week 1: 00_introduction + 01_setup
Week 2: 02_tensor  
Week 3: 03_activations
Week 4: 04_layers
Week 5: 05_dense
Week 6: 06_spatial
Week 7: 07_attention
Week 8: Midterm / Review
Week 9: 08_dataloader
Week 10: 09_autograd
Week 11: 10_optimizers  
Week 12: 11_training
Week 13: 12_compression + 13_kernels
Week 14: 14_benchmarking + 15_mlops
Week 15: 16_capstone
Week 16: Final presentations
```

---

## 📝 Phase 3: Assignment Management

### Creating Student Assignments

**For Individual Modules:**
```bash
# Generate assignment from TinyTorch module
./bin/tito nbgrader generate 01_setup

# This creates:
# assignments/source/01_setup/01_setup.ipynb (instructor version)
```

**For Multiple Modules:**
```bash
# Generate first 4 modules
./bin/tito nbgrader generate --range 01-04

# Or generate all modules at once (start of semester)
./bin/tito nbgrader generate --all
```

**What happens during generation:**
1. Reads the module's `.py` file from `modules/source/XX_module/`
2. Converts to Jupyter notebook using jupytext
3. Processes with NBGrader to create student version
4. Removes instructor solutions, adds `# YOUR CODE HERE` stubs
5. Creates assignments in `assignments/source/XX_module/`

### Releasing Assignments to Students

```bash
# Release individual assignment
./bin/tito nbgrader release 01_setup

# Release multiple assignments
./bin/tito nbgrader release --range 01-04

# This creates:
# assignments/release/01_setup/01_setup.ipynb (student version)
```

**Distribution to Students:**
- Upload `assignments/release/XX_module/XX_module.ipynb` to your LMS
- Or provide direct access to the `assignments/release/` directory
- Students download and work on their local copies

### Monitoring Assignment Status

```bash
# Check what assignments exist
./bin/tito nbgrader status

# Example output:
# 📚 Source assignments: 4
#   - 01_setup, 02_tensor, 03_activations, 04_layers
# 🚀 Released assignments: 2  
#   - 01_setup, 02_tensor
# 📥 Submitted assignments: 1
#   - 01_setup
# 🎯 Graded assignments: 0
```

---

## 📊 Phase 4: Grading Workflow

### Collecting Student Submissions

**Manual Collection (Most Common):**
```bash
# Students submit via LMS, you download to:
mkdir -p assignments/submitted/01_setup/student_name/
# Place student notebooks in: assignments/submitted/01_setup/student_name/01_setup.ipynb
```

**NBGrader Exchange (If Using Shared Server):**
```bash
# Collect all submissions for an assignment
./bin/tito nbgrader collect 01_setup
```

### Auto-Grading Process

```bash
# Auto-grade specific assignment
./bin/tito nbgrader autograde 01_setup

# Auto-grade all collected assignments
./bin/tito nbgrader autograde --all

# What happens:
# - Executes all student code
# - Runs hidden test cells  
# - Checks assert statements
# - Records pass/fail for each test
# - Creates detailed grading reports
```

### Generating Student Feedback

```bash
# Generate feedback for specific assignment
./bin/tito nbgrader feedback 01_setup

# Generate feedback for all assignments
./bin/tito nbgrader feedback --all

# Creates:
# assignments/feedback/01_setup/student_name/01_setup.html
```

### Exporting Grades

```bash
# Export grades to CSV
./bin/tito nbgrader report --format csv

# Creates: grades.csv with all student scores
```

---

## 🔧 Phase 5: Common Workflows

### Weekly Assignment Routine

```bash
# Monday: Generate and release new assignment
./bin/tito nbgrader generate 03_activations
./bin/tito nbgrader release 03_activations

# Upload assignments/release/03_activations/03_activations.ipynb to LMS
# Announce assignment to students

# Friday: Collect submissions and grade
# (Download student submissions from LMS to assignments/submitted/)
./bin/tito nbgrader autograde 03_activations
./bin/tito nbgrader feedback 03_activations

# Monday: Return graded assignments and feedback to students
```

### Mid-Semester Status Check

```bash
# Comprehensive system status
./bin/tito module status --comprehensive

# Assignment analytics
./bin/tito nbgrader analytics 03_activations

# Export current gradebook
./bin/tito nbgrader report --format csv
```

### End-of-Semester Workflow

```bash
# Generate final gradebook
./bin/tito nbgrader report --format csv

# Archive all assignments and submissions
tar -czf course_archive_fall2024.tar.gz assignments/ gradebook.db

# Clean up for next semester
./bin/tito clean
./bin/tito nbgrader init  # Fresh start
```

---

## 🛠️ Phase 6: Troubleshooting & Tips

### Common Issues

**"Module not found" when generating:**
```bash
# Check available modules
ls modules/source/

# Use exact directory name
./bin/tito nbgrader generate 02_tensor  # Not just "tensor"
```

**"NBGrader validation failed":**
```bash
# This is expected for student notebooks (they have unimplemented functions)
# Validation failure = students need to implement the code
```

**Environment issues:**
```bash
# Always activate virtual environment first
source .venv/bin/activate

# Check environment health
./bin/tito system doctor
```

### Best Practices

**📋 Assignment Preparation:**
- Generate all assignments at start of semester
- Test each assignment yourself before releasing
- Provide clear due dates and submission instructions

**⏰ Grading Efficiency:**
- Set up consistent folder structure for submissions
- Use batch grading commands (`--all` flags)
- Review auto-graded results before finalizing

**💡 Student Support:**
- Share `./bin/tito module status` command with students
- Encourage testing with provided test functions
- Provide clear error message interpretation

### Advanced Configuration

**Customize point values in `nbgrader_config.py`:**
```python
# Adjust timeout for long-running assignments
c.ExecutePreprocessor.timeout = 300  # 5 minutes per cell

# Customize solution stubs
c.ClearSolutions.code_stub = {
    "python": "# YOUR IMPLEMENTATION HERE\nraise NotImplementedError()"
}
```

---

## 📚 Phase 7: Student Guidance

### What to Tell Your Students

**Setup Instructions for Students:**
```bash
# Students should run:
git clone [your-course-repo]
cd TinyTorch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Test their setup:
./bin/tito system doctor
```

**Working on Assignments:**
```markdown
1. Download the assignment notebook from [LMS]
2. Open in Jupyter: `jupyter lab assignment.ipynb`
3. Look for `# YOUR CODE HERE` markers
4. Implement the required functions
5. Test your work with provided test cells
6. Submit the completed notebook
```

**Debugging Help:**
```bash
# Students can check their module status
./bin/tito module status

# Get help with specific modules
./bin/tito module info 02_tensor
```

---

## 🎯 Quick Reference Commands

### Essential Daily Commands
```bash
# Check overall system status
./bin/tito module status --comprehensive

# Assignment lifecycle
./bin/tito nbgrader generate MODULE_NAME
./bin/tito nbgrader release MODULE_NAME  
./bin/tito nbgrader autograde MODULE_NAME
./bin/tito nbgrader feedback MODULE_NAME

# Monitor progress
./bin/tito nbgrader status
./bin/tito nbgrader analytics MODULE_NAME
```

### Batch Operations
```bash
# Work with multiple modules
./bin/tito nbgrader generate --range 01-04
./bin/tito nbgrader release --all
./bin/tito nbgrader autograde --all
./bin/tito nbgrader feedback --all
```

### System Maintenance
```bash
# Environment health
./bin/tito system doctor

# Clean temporary files  
./bin/tito clean

# Export final grades
./bin/tito nbgrader report --format csv
```

---

## 📞 Getting Help

**If you encounter issues:**

1. **Check system status**: `./bin/tito system doctor`
2. **Review logs**: Check output messages for specific errors
3. **Consult documentation**: This guide covers 95% of common scenarios
4. **Community support**: [GitHub Issues](https://github.com/your-org/TinyTorch/issues)

**For urgent instructor support:**
- Create detailed issue with error messages
- Include output of `./bin/tito module status --comprehensive`
- Specify which assignment and step is failing

---

## 🎉 Success Metrics

**You'll know you're successful when:**
- ✅ Students can download and run assignments without setup issues  
- ✅ Auto-grading provides consistent, fair evaluation
- ✅ Weekly assignment workflow takes <10 minutes
- ✅ Students build a complete ML framework by semester end
- ✅ You have detailed analytics on student progress and common issues

**Ready to run the most comprehensive ML systems course your students will ever take!** 🚀

---

*This guide covers the complete instructor journey from setup to course completion. For specific technical details, see the individual command documentation with `./bin/tito --help`.*