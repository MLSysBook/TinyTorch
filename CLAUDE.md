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
1. **Create branch** for each logical piece of work
2. **Make focused commits** related to that branch only
3. **Test your changes** before committing
4. **Merge to dev** when feature is complete and tested
5. **Delete feature branch** after successful merge

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
- [ ] Code works correctly
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if needed)
- [ ] No conflicts with dev branch
- [ ] Feature is complete and ready for use

### 🔄 Example Workflow
```bash
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
- Implementation → Testing → Commit
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
- **MUST notify QA Agent after ANY module changes**

**Package Manager:**
- Module integration and export validation
- Dependency resolution between modules
- Integration testing after exports
- **MANDATORY: Validate ALL module exports**
- **MUST ensure modules work together**
- **MUST run integration tests**
- **MUST verify complete package builds**
- **MUST block release if integration fails**

**Quality Assurance:**  
- Test coverage and functionality
- Testing infrastructure
- **MANDATORY: Test ALL modified modules after ANY changes**
- **MUST run tests before ANY commit**
- **MUST verify module imports correctly**
- **MUST ensure all test functions work**
- **MUST report test results to Package Manager**

**Documentation Publisher:**
- Markdown prose and clarity
- **Module-specific ML systems thinking questions** (analyze actual code, reference specific implementations, build cumulative knowledge)
- Writing ONLY

**Workflow Coordinator:**
- **MUST enforce QA testing after EVERY module update**
- **CANNOT approve changes without QA test results**
- **MUST block commits if tests fail**

### 🧪 QA Testing Protocol - MANDATORY

**EVERY module update MUST trigger the following QA process:**

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