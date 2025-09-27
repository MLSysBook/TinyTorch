# TinyTorch Command Reference

## 🚀 Quick Reference

Essential commands for daily TinyTorch usage:

```bash
# Getting Started
tito help --interactive         # Personalized onboarding wizard
tito system doctor             # Check installation and environment
tito checkpoint status         # See your learning progress

# Learning Workflow  
cd modules/source/0X_name      # Enter module directory
jupyter lab name_dev.py        # Work on module implementation
tito module complete 0X_name   # Export and test your implementation
tito checkpoint test XX        # Validate your learning

# Community & Competition
tito leaderboard join          # Join global learning community
tito leaderboard submit        # Share your progress
tito olympics explore          # See competition events
```

---

## 📚 Complete Command Reference

### **Help & Guidance Commands**

#### `tito help`
Interactive help system with contextual guidance.

**Usage:**
```bash
tito help [topic] [options]
```

**Examples:**
```bash
# Interactive onboarding wizard
tito help --interactive

# Quick reference card
tito help --quick

# Specific help topics
tito help getting-started
tito help workflow
tito help commands
tito help troubleshooting
```

**Options:**
- `--interactive, -i`: Launch interactive onboarding wizard
- `--quick, -q`: Show quick reference card

**Available Topics:**
- `getting-started`: Installation and first steps
- `commands`: Command overview and usage
- `workflow`: Common development patterns
- `modules`: Module system explanation
- `checkpoints`: Progress tracking system
- `community`: Leaderboard and Olympics features
- `troubleshooting`: Common issues and solutions

---

### **System Management Commands**

#### `tito system`
System health checking and environment management.

**Subcommands:**

##### `tito system doctor`
Comprehensive system health check.

**Usage:**
```bash
tito system doctor [options]
```

**Examples:**
```bash
# Basic health check
tito system doctor

# Detailed diagnostics
tito system doctor --verbose

# Check specific components
tito system doctor --check python,jupyter,git
```

**What It Checks:**
- ✅ Python version compatibility
- ✅ Virtual environment activation
- ✅ Required package installation
- ✅ Jupyter Lab availability
- ✅ Git repository status
- ✅ TinyTorch package installation
- ✅ Module structure integrity

##### `tito system info`
Display system and project information.

```bash
tito system info                # Basic system info
tito system info --detailed     # Comprehensive details
```

---

### **Learning Progress Commands**

#### `tito checkpoint`
Track and validate learning progress through capability checkpoints.

**Subcommands:**

##### `tito checkpoint status`
View your current learning progress.

**Usage:**
```bash
tito checkpoint status [options]
```

**Examples:**
```bash
# Overview of all checkpoints
tito checkpoint status

# Detailed progress with module mapping
tito checkpoint status --detailed

# Progress for specific checkpoint
tito checkpoint status 05
```

**Options:**
- `--detailed`: Show module-to-checkpoint mapping
- `--json`: Output in JSON format for scripting

##### `tito checkpoint timeline`
Visual representation of learning journey.

**Usage:**
```bash
tito checkpoint timeline [options]
```

**Examples:**
```bash
# Vertical tree view with connecting lines
tito checkpoint timeline

# Horizontal progress bar
tito checkpoint timeline --horizontal

# Focus on specific range
tito checkpoint timeline --range 1-10
```

**Options:**
- `--horizontal`: Show as horizontal progress bar
- `--range START-END`: Show specific checkpoint range

##### `tito checkpoint test`
Test specific capability checkpoint.

**Usage:**
```bash
tito checkpoint test <checkpoint_number> [options]
```

**Examples:**
```bash
# Test checkpoint 5 (spatial convolution capability)
tito checkpoint test 05

# Run with detailed output
tito checkpoint test 05 --verbose

# Test without stopping on first failure
tito checkpoint test 05 --continue-on-failure
```

**Options:**
- `--verbose, -v`: Show detailed test output
- `--continue-on-failure`: Don't stop on first test failure
- `--profile-memory`: Include memory usage profiling

##### `tito checkpoint run`
Run checkpoint implementation interactively.

**Usage:**
```bash
tito checkpoint run <checkpoint_number> [options]
```

**Examples:**
```bash
# Run checkpoint 3 interactively
tito checkpoint run 03

# Run with verbose debugging
tito checkpoint run 03 --verbose --debug
```

---

### **Module Development Commands**

#### `tito module`
Module development, completion, and management.

**Subcommands:**

##### `tito module complete`
Complete a module by exporting implementation and running checkpoint test.

**Usage:**
```bash
tito module complete <module_identifier> [options]
```

**Examples:**
```bash
# Complete module by number
tito module complete 02_tensor

# Complete by name only
tito module complete tensor

# Complete without running checkpoint test
tito module complete 02_tensor --skip-test

# Dry run to see what would happen
tito module complete 02_tensor --dry-run
```

**Options:**
- `--skip-test`: Skip automatic checkpoint testing
- `--dry-run`: Show what would be done without executing
- `--force`: Force re-export even if already completed

**What It Does:**
1. Exports your implementation to `tinytorch` package
2. Maps module to appropriate checkpoint
3. Runs capability test automatically
4. Shows achievement celebration
5. Displays next recommended steps

##### `tito module status`
Check status of modules and their exports.

**Usage:**
```bash
tito module status [module_identifier] [options]
```

**Examples:**
```bash
# Status of all modules
tito module status

# Status of specific module
tito module status 02_tensor

# Detailed export information
tito module status --detailed
```

##### `tito module validate`
Validate module structure and implementation.

**Usage:**
```bash
tito module validate <module_identifier> [options]
```

**Examples:**
```bash
# Validate module structure
tito module validate 05_losses

# Check specific validation rules
tito module validate 05_losses --check exports,tests,structure
```

##### `tito module export`
Export module implementation to package (lower-level than complete).

**Usage:**
```bash
tito module export <module_identifier> [options]
```

**Examples:**
```bash
# Export specific module
tito module export 02_tensor

# Force re-export
tito module export 02_tensor --force
```

---

### **Demo & Examples Commands**

#### `tito demo`
Demonstrations of TinyTorch capabilities.

**Subcommands:**

##### `tito demo quick`
Quick demonstration of framework capabilities.

**Usage:**
```bash
tito demo quick [options]
```

**Examples:**
```bash
# Standard quick demo
tito demo quick

# Interactive demo with explanations
tito demo quick --interactive

# Silent demo for scripting
tito demo quick --quiet
```

##### `tito demo module`
Demonstrate specific module capabilities.

**Usage:**
```bash
tito demo module <module_name> [options]
```

**Examples:**
```bash
# Demo tensor operations
tito demo module tensor

# Demo with step-by-step explanation
tito demo module activations --interactive
```

---

### **Community & Leaderboard Commands**

#### `tito leaderboard`
Global learning community participation.

**Subcommands:**

##### `tito leaderboard join`
Join the global TinyTorch learning community.

**Usage:**
```bash
tito leaderboard join [options]
```

**Examples:**
```bash
# Interactive join process
tito leaderboard join

# Join with preset preferences
tito leaderboard join --name "ML_Learner" --anonymous

# Join as instructor
tito leaderboard join --instructor
```

**Options:**
- `--name NAME`: Set display name
- `--anonymous`: Join with anonymous participation
- `--instructor`: Join as course instructor

##### `tito leaderboard submit`
Submit your progress to the community leaderboard.

**Usage:**
```bash
tito leaderboard submit [options]
```

**Examples:**
```bash
# Submit current progress
tito leaderboard submit

# Submit with celebration message
tito leaderboard submit --message "First neural network working!"

# Submit specific milestone
tito leaderboard submit --milestone checkpoint_05
```

##### `tito leaderboard view`
View community leaderboard and rankings.

**Usage:**
```bash
tito leaderboard view [category] [options]
```

**Examples:**
```bash
# View main progress leaderboard
tito leaderboard view

# View specific category
tito leaderboard view progress
tito leaderboard view innovation

# View your ranking
tito leaderboard view --my-rank
```

**Categories:**
- `progress`: Checkpoint completion progress
- `speed`: Learning velocity rankings
- `innovation`: Creative optimization solutions
- `community`: Community contribution scores

##### `tito leaderboard profile`
Manage your community profile.

**Usage:**
```bash
tito leaderboard profile [action] [options]
```

**Examples:**
```bash
# View your profile
tito leaderboard profile view

# Update profile information
tito leaderboard profile update --name "New Name"

# Privacy settings
tito leaderboard profile privacy --level anonymous
```

---

### **Olympics & Competition Commands**

#### `tito olympics`
Performance optimization competitions.

**Subcommands:**

##### `tito olympics explore`
Explore available Olympic events and competitions.

**Usage:**
```bash
tito olympics explore [options]
```

**Examples:**
```bash
# See all available events
tito olympics explore

# View specific event details
tito olympics explore --event cnn_marathon

# See your past participations
tito olympics explore --my-events
```

##### `tito olympics register`
Register for Olympic competition events.

**Usage:**
```bash
tito olympics register --event <event_name> [options]
```

**Examples:**
```bash
# Register for CNN optimization marathon
tito olympics register --event cnn_marathon

# Register with team
tito olympics register --event transformer_decathlon --team "Speed_Demons"
```

**Available Events:**
- `mlp_sprint`: Fast matrix operations optimization
- `cnn_marathon`: Memory-efficient convolution implementation
- `transformer_decathlon`: Complete language model optimization
- `innovation_showcase`: Novel optimization techniques

##### `tito olympics submit`
Submit optimization results to competition.

**Usage:**
```bash
tito olympics submit --event <event_name> [options]
```

**Examples:**
```bash
# Submit CNN optimization results
tito olympics submit --event cnn_marathon

# Submit with detailed metrics
tito olympics submit --event mlp_sprint --include-profiling
```

---

### **Testing & Validation Commands**

#### `tito test`
Comprehensive testing framework.

**Subcommands:**

##### `tito test run`
Run various test suites.

**Usage:**
```bash
tito test run [test_type] [options]
```

**Examples:**
```bash
# Run all tests
tito test run

# Run module-specific tests
tito test run --module tensor

# Run checkpoint tests
tito test run --checkpoints

# Run with coverage
tito test run --coverage
```

**Test Types:**
- `unit`: Individual function tests
- `integration`: Module interaction tests
- `checkpoint`: Capability validation tests
- `performance`: Speed and memory benchmarks

##### `tito test validate`
Validate implementations against requirements.

**Usage:**
```bash
tito test validate <target> [options]
```

**Examples:**
```bash
# Validate all modules
tito test validate modules

# Validate specific implementation
tito test validate tensor --function relu
```

---

### **Instructor & Classroom Commands**

#### `tito nbgrader`
NBGrader integration for classroom management.

**Subcommands:**

##### `tito nbgrader setup-instructor`
Setup instructor environment for course management.

**Usage:**
```bash
tito nbgrader setup-instructor [options]
```

**Examples:**
```bash
# Interactive instructor setup
tito nbgrader setup-instructor

# Automated setup
tito nbgrader setup-instructor --course-name "CS249r" --semester "Fall2024"
```

##### `tito nbgrader release`
Release assignments to students.

**Usage:**
```bash
tito nbgrader release <assignment> [options]
```

**Examples:**
```bash
# Release specific module
tito nbgrader release 05_losses

# Release with custom deadline
tito nbgrader release 05_losses --deadline "2024-10-15"
```

#### `tito grade`
Student progress tracking and grading.

**Subcommands:**

##### `tito grade class-overview`
Overview of class progress and performance.

```bash
tito grade class-overview                    # All students
tito grade class-overview --module 05       # Specific module
```

##### `tito grade student-progress`
Individual student progress tracking.

```bash
tito grade student-progress <student_name>   # Specific student
tito grade student-progress --all           # All students
```

---

### **Maintenance & Cleanup Commands**

#### `tito clean`
Clean up generated files and reset state.

**Usage:**
```bash
tito clean [target] [options]
```

**Examples:**
```bash
# Clean all generated files
tito clean all

# Clean specific components
tito clean notebooks
tito clean exports
tito clean cache

# Dry run to see what would be cleaned
tito clean all --dry-run
```

**Targets:**
- `all`: Everything except source code
- `notebooks`: Generated .ipynb files
- `exports`: Package exports in tinytorch/
- `cache`: Temporary and cache files
- `checkpoints`: Checkpoint test results

#### `tito reset`
Reset progress and start fresh (use carefully!).

**Usage:**
```bash
tito reset [options]
```

**Examples:**
```bash
# Reset with confirmation
tito reset --confirm

# Reset specific components
tito reset --checkpoints-only
tito reset --exports-only
```

---

## 🔄 Common Workflows

### **New User Onboarding**

```bash
# 1. System setup and verification
tito system doctor
tito help --interactive

# 2. See the learning journey
tito checkpoint status
tito checkpoint timeline

# 3. Start first module
cd modules/source/01_setup
jupyter lab setup_dev.py

# 4. Complete and test
tito module complete 01_setup
tito checkpoint test 00
```

### **Daily Learning Session**

```bash
# 1. Check progress and next steps
tito checkpoint status

# 2. Work on current module
cd modules/source/0X_current
jupyter lab current_dev.py

# 3. Complete when ready
tito module complete 0X_current

# 4. Celebrate and plan next
tito leaderboard submit --message "Progress update!"
```

### **Module Development Cycle**

```bash
# 1. Validate module structure
tito module validate 05_losses

# 2. Export and test implementation
tito module complete 05_losses

# 3. Run capability checkpoint
tito checkpoint test 04  # Auto-triggered by complete

# 4. Debug if needed
tito checkpoint test 04 --verbose

# 5. Move to next module
tito checkpoint status  # See next recommended step
```

### **Community Participation**

```bash
# 1. Join community
tito leaderboard join

# 2. Regular progress updates
tito leaderboard submit

# 3. View community progress
tito leaderboard view progress

# 4. Compete in Olympics
tito olympics explore
tito olympics register --event cnn_marathon
```

### **Instructor Workflow**

```bash
# 1. Course setup
tito nbgrader setup-instructor
tito grade setup-course

# 2. Assignment management
tito nbgrader release 05_losses
tito nbgrader collect 05_losses

# 3. Progress monitoring
tito grade class-overview
tito checkpoint class-stats

# 4. Grading and feedback
tito nbgrader autograde 05_losses
tito nbgrader formgrade 05_losses
```

### **Troubleshooting Workflow**

```bash
# 1. System health check
tito system doctor

# 2. Validate specific components
tito module validate 05_losses
tito test validate tensor

# 3. Re-export if needed
tito module export 05_losses --force

# 4. Test integration
tito checkpoint test 04 --verbose

# 5. Clean and retry if necessary
tito clean exports
tito module complete 05_losses
```

---

## ⚙️ Configuration & Options

### **Global Options**

Available for most commands:

- `--verbose, -v`: Detailed output and debugging information
- `--quiet, -q`: Minimal output (for scripting)
- `--no-color`: Disable colored output
- `--help, -h`: Show command-specific help

### **Environment Variables**

TinyTorch respects these environment variables:

```bash
export TINYTORCH_CONFIG_DIR="~/.tinytorch"     # Configuration directory
export TINYTORCH_CACHE_DIR="~/.tinytorch/cache" # Cache location
export TINYTORCH_LOG_LEVEL="INFO"             # Logging level
export TINYTORCH_NO_COLOR="1"                 # Disable colors
```

### **Configuration Files**

**User Config:** `~/.tinytorch/config.yaml`
```yaml
display:
  colors: true
  progress_bars: true
  
learning:
  auto_submit_progress: false
  celebration_animations: true
  
community:
  anonymous_mode: false
  share_achievements: true
```

**Project Config:** `.tinytorch/config.yaml`
```yaml
course:
  name: "ML Systems Engineering"
  semester: "Fall 2024"
  
grading:
  auto_release: false
  late_penalty: 0.1
```

---

## 🚀 Pro Tips

### **Efficient Command Usage**

**Tab Completion:** Most commands support tab completion:
```bash
tito checkpoint <TAB>           # Shows available subcommands
tito module complete 0<TAB>     # Shows available modules
```

**Command Aliases:** Set up shell aliases for common commands:
```bash
# Add to your ~/.bashrc or ~/.zshrc
alias tc="tito checkpoint"
alias tm="tito module complete"
alias ts="tito system doctor"
```

**Batch Operations:** Use command chaining for efficiency:
```bash
# Complete module and submit progress in one line
tito module complete 05_losses && tito leaderboard submit
```

### **Scripting with TinyTorch**

**JSON Output:** Many commands support JSON for scripting:
```bash
# Get checkpoint progress as JSON
tito checkpoint status --json

# Parse specific information
tito checkpoint status --json | jq '.completed_count'
```

**Exit Codes:** Commands return meaningful exit codes:
- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: System not ready

**Example Script:**
```bash
#!/bin/bash
# Check if ready for next module
if tito checkpoint test 05 --quiet; then
    echo "Ready for module 6!"
    tito module complete 06_autograd
else
    echo "Complete checkpoint 5 first"
    exit 1
fi
```

---

**For interactive help and personalized guidance, run: `tito help --interactive` 🚀**