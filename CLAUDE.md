# Claude Code Instructions for TinyTorch

## Git Workflow Standards

### 🌿 Always Use Feature Branches
**NEVER work directly on `dev` or `main`**

```bash
# Start any new work with a feature branch
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

### 📝 Branch Naming Convention
Use descriptive branch names that indicate the type of work:

- **Features**: `feature/add-tito-view-command`
- **Bug fixes**: `fix/tensor-dtype-handling`
- **Improvements**: `improve/module-documentation`
- **Experiments**: `experiment/new-testing-approach`

### 🔧 Development Workflow
1. **Activate virtual environment** - ALWAYS use `.venv` for consistent dependencies
2. **Create branch** for each logical piece of work
3. **Make focused commits** related to that branch only
4. **Test your changes** before committing
5. **Merge to dev** when feature is complete and tested
6. **Delete feature branch** after successful merge

### ✅ Commit Standards
- **One feature per branch** - don't mix unrelated changes
- **Test before committing** - ensure functionality works
- **Descriptive commit messages** that explain the "why"
- **Clean history** - squash if needed before merging
- **NO Co-Authored-By** unless explicitly requested by user
- **NO automated attribution** - keep commits clean and focused

### 🚫 What NOT to Do
- ❌ Work directly on `dev` or `main`
- ❌ Mix unrelated changes in one branch
- ❌ Commit broken code
- ❌ Merge untested changes
- ❌ Leave stale feature branches

### 📋 Merge Checklist
Before merging any feature branch:
- [ ] Virtual environment activated and dependencies installed
- [ ] Code works correctly
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if needed)
- [ ] No conflicts with dev branch
- [ ] Feature is complete and ready for use

### 🔄 Example Workflow
```bash
# 0. ALWAYS start with virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR: .venv\Scripts\activate  # On Windows
pip install -r requirements.txt

# 1. Start new feature
git checkout dev
git pull origin dev
git checkout -b feature/add-module-validation

# 2. Work on feature, make commits
git add .
git commit -m "Add module validation logic"
git add .
git commit -m "Add validation tests"

# 3. When ready, merge to dev
git checkout dev
git pull origin dev
git merge feature/add-module-validation

# 4. Clean up
git branch -d feature/add-module-validation

# 5. Push updated dev
git push origin dev
```

### 🎯 Why This Matters
- **Keeps history clean** - easy to understand what changed when
- **Enables collaboration** - multiple people can work without conflicts
- **Allows experimentation** - try things without breaking main code
- **Facilitates rollbacks** - easy to undo specific features if needed
- **Professional practice** - industry standard for software development

## 🐍 Virtual Environment Standards - MANDATORY

### 🔐 **ALWAYS Use Virtual Environments**
**NEVER work directly with system Python or globally installed packages.**

```bash
# Create virtual environment (one time setup)
python -m venv .venv

# Activate virtual environment (EVERY session)
source .venv/bin/activate     # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

# Install dependencies (after activation)
pip install -r requirements.txt

# Verify environment
which python  # Should show .venv path
pip list      # Should show only project dependencies
```

### 📋 Virtual Environment Checklist
**Before ANY development work:**
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Verification: `which python` shows `.venv/bin/python`
- [ ] Verification: `tito system doctor` shows ✅ environment checks

### 🚫 What NOT to Do
- ❌ Use system Python for development
- ❌ Install packages globally with `sudo pip install`
- ❌ Work without activating `.venv`
- ❌ Mix dependencies from different environments
- ❌ Commit with virtual environment deactivated

### 🔧 Environment Troubleshooting
If you see dependency errors:
1. **Deactivate and recreate**: `deactivate && rm -rf .venv && python -m venv .venv`
2. **Reactivate**: `source .venv/bin/activate`
3. **Reinstall**: `pip install -r requirements.txt`
4. **Verify**: `tito system doctor`

### 💡 Pro Tips
- **Add to shell profile**: `alias activate='source .venv/bin/activate'`
- **Check activation**: Your prompt should show `(.venv)` prefix
- **Architecture issues**: Use `python -m pip install --force-reinstall` for numpy/architecture conflicts

## AI Agent Workflow Standards

### 🤖 Agent Team Orchestration - Best Practices

**Managing multiple AI agents requires structured coordination. Here's how the TinyTorch agent team works together:**

### 📊 Agent Team Structure

```
Workflow Coordinator (Team Lead)
    ├── Education Architect (Strategy)
    ├── Module Developer (Implementation)
    ├── Package Manager (Integration)
    ├── Quality Assurance (Validation)
    └── Documentation Publisher (Communication)
```

### 🎯 **Checkpoint System Implementation - Agent Workflow Case Study**

**SUCCESSFUL IMPLEMENTATION:** The agent team successfully implemented a comprehensive 16-checkpoint capability assessment system with integration testing. Here's how the workflow functioned:

#### **Phase 1: Strategic Planning** (Education Architect + Workflow Coordinator)
- **Education Architect**: Designed capability-based learning progression (Foundation → Architecture → Training → Inference → Serving)
- **Workflow Coordinator**: Orchestrated agent coordination and defined implementation phases
- **Result**: 16-checkpoint structure aligned with 17 TinyTorch modules, each with clear capability statements

#### **Phase 2: Implementation** (Module Developer)
- **Implemented checkpoint test suite**: 16 individual test files (`checkpoint_00_environment.py` through `checkpoint_15_capstone.py`)
- **Built CLI integration**: Complete `tito checkpoint` command system with Rich visualizations
- **Created module completion workflow**: `tito module complete` with automatic export and testing
- **Added integration testing**: Post-module completion checkpoint validation
- **MUST call QA Agent**: Immediately after implementation completed

#### **Phase 3: Quality Assurance** (QA Agent) - **MANDATORY**
- **Tested all 16 checkpoint implementations**: Each test file executes correctly and validates capabilities
- **Verified CLI integration**: All `tito checkpoint` commands work with Rich progress tracking
- **Validated module completion workflow**: `tito module complete` correctly exports and tests checkpoints
- **Tested integration pipeline**: Module-to-checkpoint mapping functions correctly
- **Reported success to Package Manager**: All tests passed, ready for integration

#### **Phase 4: Package Integration** (Package Manager) - **MANDATORY**
- **Validated checkpoint test execution**: All checkpoint files import and run correctly
- **Verified CLI command registration**: `tito checkpoint` commands integrated into main CLI
- **Tested module-to-checkpoint mapping**: Correct checkpoint triggered for each module completion
- **Ensured complete package build**: All checkpoint functionality available in built package
- **Integration success confirmed**: Complete system works end-to-end

#### **Phase 5: Documentation** (Documentation Publisher)
- **Updated documentation**: This CLAUDE.md file, checkpoint-system.md, README.md updates
- **Documented agent workflow**: How agents successfully coordinated implementation
- **Created CLI documentation**: Usage examples and command reference
- **Explained integration testing**: How checkpoint system validates student progress

#### **Phase 6: Review and Approval** (Workflow Coordinator)
- **Verified all agents completed tasks**: QA passed, Package Manager confirmed integration
- **Confirmed capability delivery**: 16-checkpoint system with CLI and integration testing
- **Approved for commit**: Complete implementation ready for production use

### 🚀 **Implemented Checkpoint System Capabilities**

**The successful agent workflow delivered these concrete features:**

#### **16-Checkpoint Capability Assessment System**
```bash
# Checkpoint progression with capability questions:
00: Environment    - "Can I configure my TinyTorch development environment?"
01: Foundation     - "Can I create and manipulate the building blocks of ML?"
02: Intelligence   - "Can I add nonlinearity - the key to neural network intelligence?"
03: Components     - "Can I build the fundamental building blocks of neural networks?"
04: Networks       - "Can I build complete multi-layer neural networks?"
05: Learning       - "Can I process spatial data like images with convolutional operations?"
06: Attention      - "Can I build attention mechanisms for sequence understanding?"
07: Stability      - "Can I stabilize training with normalization techniques?"
08: Differentiation - "Can I automatically compute gradients for learning?"
09: Optimization   - "Can I optimize neural networks with sophisticated algorithms?"
10: Training       - "Can I build complete training loops for end-to-end learning?"
11: Regularization - "Can I prevent overfitting and build robust models?"
12: Kernels        - "Can I implement high-performance computational kernels?"
13: Benchmarking   - "Can I analyze performance and identify bottlenecks in ML systems?"
14: Deployment     - "Can I deploy and monitor ML systems in production?"
15: Capstone       - "Can I build complete end-to-end ML systems from scratch?"
```

#### **Rich CLI Progress Tracking**
```bash
# Visual progress tracking with Rich library
tito checkpoint status           # Current progress overview with capability statements
tito checkpoint status --detailed # Module-level detail with test file status
tito checkpoint timeline         # Vertical tree view with connecting lines
tito checkpoint timeline --horizontal # Linear progress bar with Rich styling
tito checkpoint test 01          # Test specific checkpoint capabilities
tito checkpoint run 00 --verbose # Run checkpoint with detailed output
```

#### **Module Completion Workflow with Integration Testing**
```bash
# Automatic export and checkpoint testing
tito module complete 02_tensor   # Exports module to package AND tests capabilities
tito module complete tensor      # Works with short names too
tito module complete 02_tensor --skip-test # Skip checkpoint test if needed

# Workflow automatically:
# 1. Exports module to tinytorch package
# 2. Maps module to appropriate checkpoint (02_tensor → checkpoint_01_foundation)
# 3. Runs capability test with Rich progress tracking
# 4. Shows achievement celebration and next steps
```

#### **Comprehensive Integration Testing**
- **Module-to-Checkpoint Mapping**: Each module automatically triggers appropriate checkpoint test
- **Capability Validation**: Tests verify actual functionality works, not just code completion
- **Progress Visualization**: Rich CLI shows achievements and suggests next steps
- **Immediate Feedback**: Students get instant validation when capabilities are achieved

### 🔄 Standard Agent Workflow Pattern

**For EVERY module update, follow this sequence:**

1. **Planning Phase** (Workflow Coordinator + Education Architect)
   - Define learning objectives
   - Plan module structure
   - Set implementation goals

2. **Implementation Phase** (Module Developer)
   - Write code following specifications
   - Add NBGrader metadata
   - Create test scaffolding
   - Add proper export directives (#| default_exp)
   - **MUST call QA Agent when done**

3. **Testing Phase** (Quality Assurance) - **MANDATORY**
   - Run comprehensive test suite
   - Verify all functionality
   - Report results to Package Manager
   - **Block progress if tests fail**

4. **Integration Phase** (Package Manager) - **MANDATORY**
   - Validate module exports correctly
   - Check integration with other modules
   - Run integration tests
   - Ensure complete package works
   - **Block progress if integration fails**

5. **Documentation Phase** (Documentation Publisher)
   - Add explanatory markdown
   - Create ML systems thinking questions
   - Ensure clarity and consistency

6. **Review Phase** (Workflow Coordinator)
   - Verify all agents completed their tasks
   - Ensure QA tests passed
   - Confirm package integration successful
   - Approve for commit

### 🎯 Agent Communication Protocol

**Agents MUST communicate through structured handoffs:**

```python
# Example workflow for module update:
workflow_coordinator.plan_update(module="tensor")
    → education_architect.design_learning_path()
    → module_developer.implement_code()
    → qa_agent.run_tests()  # MANDATORY
    → documentation_publisher.add_documentation()
    → workflow_coordinator.review_and_approve()
```

### ⚡ Parallel vs Sequential Work

**Parallel Tasks (can happen simultaneously):**
- Multiple Module Developers working on different modules
- Documentation Publisher preparing content while code is tested
- Education Architect planning next modules

**Sequential Tasks (must happen in order):**
- Virtual Environment Setup → Implementation → Testing → Commit
- Planning → Implementation → Documentation
- Test Failure → Fix → Re-test → Proceed

### 🚨 Agent Accountability Rules

1. **Module Developer**: Cannot mark task complete without QA approval
2. **QA Agent**: Must test EVERY change, no exceptions
3. **Package Manager**: Must validate integration, can block releases
4. **Workflow Coordinator**: Cannot proceed without all agent sign-offs
5. **Documentation Publisher**: Must verify code works before documenting
6. **Education Architect**: Must validate learning objectives are met

### 📝 Agent Handoff Checklist

When passing work between agents, include:
- [ ] What was completed
- [ ] What needs to be done next
- [ ] Any issues or blockers found
- [ ] Test results (if applicable)
- [ ] Recommendations for next agent

### 🔧 Conflict Resolution

If agents disagree or find conflicts:
1. QA Agent has veto power on code quality
2. Education Architect has final say on learning objectives
3. Workflow Coordinator resolves all other disputes
4. User has ultimate override authority

### 🤖 Workflow Compliance
**ALL AI agents MUST follow the Git Workflow Standards defined in `/Users/VJ/GitHub/TinyTorch/CLAUDE.md`.**

Read the complete Git Workflow Standards section in this file for all branching, commit, and merge requirements.

---

## TinyTorch Module Development Standards

### 🎯 **CRITICAL: .py Files Only**
**ALL TinyTorch development and modifications MUST be done in .py files ONLY.**

- ✅ **ALWAYS edit**: `module_name_dev.py` files
- ❌ **NEVER edit**: `.ipynb` notebook files
- ✅ **Notebooks are generated**: from .py files using jupytext
- ❌ **Direct notebook editing**: breaks the development workflow

**Why .py files only:**
- Version control friendly (clean diffs, no notebook metadata noise)
- Consistent development environment across all contributors
- Automated notebook generation ensures consistency
- Professional development practices

### 📚 Module Structure Requirements
All TinyTorch modules MUST follow the standardized structure:

1. **Module Introduction** - What we're building and why
2. **Mathematical Background** - Theory and foundations
3. **Implementation** - Building the components step by step
4. **Testing** - Immediate tests after each implementation
5. **Integration** - How components work together  
6. **Comprehensive Testing** - Full validation
7. **ML Systems Thinking** - Reflection questions
8. **Module Summary** - What was accomplished

### 🧪 Testing Pattern - MANDATORY
```
Implementation → Test Explanation (Markdown) → Test Code → Next Implementation
```

**CRITICAL RULES:**
- **EVERY test** must have a preceding markdown cell explaining what it tests and why
- **IMMEDIATE testing** after each implementation (not grouped at end)
- **Unit tests** = immediate after implementation
- **Integration tests** = Part 9 only

### 🤔 ML Systems Thinking Questions - REQUIRED
**Documentation Publisher should create thoughtful reflection questions** that connect the module to broader ML systems concepts. Keep them engaging and thought-provoking, not homework-like.

**Guidelines for Documentation Publisher:**
- **Focus on reflection**, not detailed implementation
- **Connect to real ML systems** (PyTorch, TensorFlow, industry)
- **Ask "why" and "how does this connect"** rather than "implement this"
- **Keep questions accessible** - should spark curiosity, not overwhelm
- **3-4 categories, 3-4 questions each** (total ~12 questions max)

**Question Categories:**
1. **System Design** - How does this fit into larger systems?
2. **Production ML** - How is this used in real ML workflows?
3. **Framework Design** - Why do frameworks make certain choices?
4. **Performance & Scale** - What happens when systems get large?

**Tone: Curious and Exploratory, Not Demanding**

✅ **Good**: "How might GPU memory management differ from your NumPy-based approach?"  
❌ **Too Complex**: "Implement GPU memory pooling with CUDA memory management..."

### 🎯 ML Systems Content Integration - CURRENT STATUS

**ML Systems rationale and content is ALREADY INTEGRATED** into the current TinyTorch structure:

✅ **Memory Analysis**: Optimizer modules include memory profiling (Adam = 3× parameter memory)
✅ **Performance Insights**: Production contexts in training, spatial, attention modules  
✅ **System Trade-offs**: Memory vs speed analysis in multiple modules
✅ **Production Context**: Real-world applications and deployment considerations
✅ **Comprehensive Documentation**: System architecture guide with Mermaid diagrams
✅ **NBGrader Integration**: Automated grading with instructor workflow
✅ **Updated README**: Emphasizes system-level learning and ML engineering skills

**Key ML Systems Concepts Covered:**
- **Module 02 (Tensor)**: Memory layout and performance implications
- **Module 06 (Spatial)**: Cache efficiency and memory access patterns  
- **Module 07 (Attention)**: O(N²) scaling and memory bottlenecks
- **Module 09 (Autograd)**: Graph memory management and checkpointing
- **Module 10 (Optimizers)**: Memory profiling, Adam 3× memory usage, production patterns
- **Module 11 (Training)**: Gradient accumulation and resource management
- **Module 13 (Kernels)**: Hardware acceleration and vectorization
- **Module 14 (Benchmarking)**: Performance analysis and bottleneck identification
- **Module 15 (MLOps)**: Production deployment and monitoring

**Documentation Resources:**
- `book/instructor-guide.md` - Complete NBGrader workflow for instructors
- `book/system-architecture.md` - Visual system architecture with Mermaid diagrams  
- `NBGrader_Quick_Reference.md` - Essential commands for daily use
- Module README files - Learning objectives emphasizing system concepts

### 📝 Markdown Cell Format - CRITICAL
```python
# CORRECT:
# %% [markdown]
"""
## Section Title
Content here...
"""

# WRONG (breaks notebooks):
# %% [markdown]
# ## Section Title  
# Content here...
```

### 🏗️ Agent Responsibilities for Modules

**Education Architect:**
- Learning objectives and pedagogical flow
- Build→Use→Understand compliance
- Educational strategy ONLY

**Module Developer:**
- Code implementation and NBGrader metadata
- Technical scaffolding and patterns
- Implementation ONLY
- Add export directives (#| default_exp)
- **Checkpoint system implementation**: Build checkpoint test files and CLI integration
- **Module completion workflow**: Implement `tito module complete` with export and testing
- **MUST notify QA Agent after ANY module changes**

**Package Manager:**
- Module integration and export validation
- Dependency resolution between modules
- Integration testing after exports
- **Checkpoint system integration**: Ensure checkpoint tests work with package exports
- **Module-to-checkpoint mapping**: Validate correct checkpoint triggered for each module
- **MANDATORY: Validate ALL module exports**
- **MUST ensure modules work together**
- **MUST run integration tests**
- **MUST verify complete package builds**
- **MUST block release if integration fails**

**Quality Assurance:**  
- Test coverage and functionality
- Testing infrastructure
- **Checkpoint test validation**: Test all 16 checkpoint implementations thoroughly
- **CLI integration testing**: Verify all `tito checkpoint` commands work correctly
- **Module completion workflow testing**: Validate `tito module complete` end-to-end
- **MANDATORY: Test ALL modified modules after ANY changes**
- **MUST run tests before ANY commit**
- **MUST verify module imports correctly**
- **MUST ensure all test functions work**
- **MUST report test results to Package Manager**

**Documentation Publisher:**
- Markdown prose and clarity
- **Module-specific ML systems thinking questions** (analyze actual code, reference specific implementations, build cumulative knowledge)
- **Checkpoint system documentation**: Update documentation to reflect new capabilities
- **Agent workflow documentation**: Document successful agent coordination patterns
- **CLI usage documentation**: Document new commands and workflows for users
- Writing ONLY

**Workflow Coordinator:**
- **Checkpoint system orchestration**: Coordinate complex multi-agent implementations like checkpoint system
- **Agent workflow enforcement**: Ensure proper agent handoffs and communication protocols
- **MUST enforce QA testing after EVERY module update**
- **CANNOT approve changes without QA test results**
- **MUST block commits if tests fail**

### 🧪 QA Testing Protocol - MANDATORY

**EVERY module update MUST trigger the following QA process:**

### 🎯 **Checkpoint System Testing Protocol - MANDATORY**

**When implementing checkpoint system features, follow this comprehensive testing protocol:**

#### **Checkpoint Implementation Testing**
```bash
# Test each checkpoint file individually
python tests/checkpoints/checkpoint_00_environment.py
python tests/checkpoints/checkpoint_01_foundation.py
# ... through checkpoint_15_capstone.py

# Test checkpoint CLI integration
tito checkpoint status
tito checkpoint timeline --horizontal
tito checkpoint test 01
tito checkpoint run 00 --verbose
```

#### **Module Completion Workflow Testing**
```bash
# Test module completion workflow end-to-end
tito module complete 02_tensor
tito module complete tensor --skip-test

# Verify module-to-checkpoint mapping
# 02_tensor should trigger checkpoint_01_foundation
# 03_activations should trigger checkpoint_02_intelligence
# etc.
```

#### **Integration Testing Requirements**
1. **All checkpoint tests execute without errors**
2. **CLI commands work with Rich visualizations**
3. **Module completion workflow functions end-to-end**
4. **Module-to-checkpoint mapping is correct**
5. **Progress tracking updates properly**
6. **Achievement celebrations display correctly**

1. **Immediate Testing After Changes**
   - QA Agent MUST be invoked after ANY module modification
   - Module Developer CANNOT proceed without QA approval
   - Workflow Coordinator MUST enforce this requirement

2. **Comprehensive Test Suite**
   ```python
   # QA Agent must run these tests for EVERY modified module:
   - Module imports without errors
   - All classes can be instantiated
   - All test functions execute successfully
   - No syntax errors present
   - Required profiler/classes exist
   - Tests only run when module executed directly (not on import)
   ```

3. **Test Execution Requirements**
   - Create isolated test environment with mocked dependencies
   - Test both with mocks AND actual dependencies when available
   - Verify module structure compliance
   - Check for immediate test execution issues
   - Validate all NBGrader metadata

4. **Failure Protocol**
   - If ANY test fails, QA Agent MUST:
     * Block the commit
     * Report specific failures to Module Developer
     * Require fixes before proceeding
     * Re-test after fixes applied

5. **Success Protocol**
   - Only after ALL tests pass, QA Agent:
     * Approves the changes
     * Reports success to Workflow Coordinator
     * Allows commit to proceed

6. **Test Results Documentation**
   - QA Agent MUST provide detailed test report including:
     * Module name and version
     * Tests run and results
     * Any warnings or issues found
     * Performance metrics if applicable
     * Recommendations for improvement

### ⚠️ Critical Requirements
- All module sections must be present
- Every test needs markdown explanation
- ML systems reflection is mandatory
- Maintain immediate testing pattern (test after each implementation)
- Use clear, consistent section organization
- **QA testing is MANDATORY before ANY commit**

---

**Remember**: Professional software development always uses branches AND comprehensive testing. This keeps the codebase stable, enables collaboration, and maintains a clean development history.