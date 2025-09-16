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

### 🤖 When Working on Tasks
1. **Always create a feature branch** before making any changes
2. **Use descriptive branch names** that match the task
3. **Keep changes focused** - one logical feature per branch
4. **Test changes** before committing
5. **Merge to dev** when work is complete
6. **Clean up** by deleting merged feature branches

### 📋 Agent Responsibilities
- **NEVER** work directly on dev/main
- **ALWAYS** create branches for any code changes
- **ALWAYS** test functionality before committing
- **ALWAYS** use descriptive commit messages
- **ALWAYS** merge properly when work is complete
- **NEVER** add Co-Authored-By tags unless user explicitly requests
- **NEVER** add automated attribution to commits

### 🔍 Quality Gates
Before any merge:
- Code functionality verified
- No breaking changes introduced
- Related documentation updated
- Clean commit history maintained

---

## TinyTorch Module Development Standards

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

**Quality Assurance:**  
- Test coverage and functionality
- Testing infrastructure
- Testing ONLY

**Documentation Publisher:**
- Markdown prose and clarity
- **Module-specific ML systems thinking questions** (analyze actual code, reference specific implementations, build cumulative knowledge)
- Writing ONLY

### ⚠️ Critical Requirements
- All module sections must be present
- Every test needs markdown explanation
- ML systems reflection is mandatory
- Maintain immediate testing pattern (test after each implementation)
- Use clear, consistent section organization

---

**Remember**: Professional software development always uses branches. This keeps the codebase stable, enables collaboration, and maintains a clean development history.