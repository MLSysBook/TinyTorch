# TinyTorch Git Best Practices
## Professional Development Workflow

### 🎯 Core Principle: Clean, Trackable Development

**Every change should be intentional, tested, and traceable.**

---

## 🌿 Branch Strategy

### Main Branches
- **`main`**: Production-ready code that students use
- **`dev`**: Integration branch for tested features

### Feature Branches
**Always create a feature branch for new work:**
```bash
git checkout dev
git pull origin dev
git checkout -b feature/descriptive-name
```

### Branch Naming Convention
- **Features**: `feature/add-lstm-module`
- **Fixes**: `fix/conv2d-shape-calculation`
- **Testing**: `test/regression-suite-setup`
- **Docs**: `docs/north-star-vision`

---

## 🔄 Development Workflow

### 1. **Start Fresh**
```bash
# Always start from updated dev
git checkout dev
git pull origin dev
git checkout -b feature/your-feature
```

### 2. **Work in Small Increments**
- Make focused changes
- Commit frequently with clear messages
- Test before committing

### 3. **Write Meaningful Commit Messages**
```bash
# Good examples:
git commit -m "Add KV cache optimization for transformer inference"
git commit -m "Fix dimension mismatch in CNN to Linear layer transition"
git commit -m "Test: Add regression tests for shape compatibility"

# Bad examples:
git commit -m "Fix bug"
git commit -m "Update code"
git commit -m "Changes"
```

### 4. **Test Before Merging**
```bash
# Run tests locally
pytest tests/
python tests/regression/run_sandbox_tests.py

# Only merge if tests pass
```

### 5. **Clean Merge Process**
```bash
# Update your branch with latest dev
git checkout dev
git pull origin dev
git checkout feature/your-feature
git merge dev  # or rebase if preferred

# Test again after merge
pytest tests/

# Merge to dev
git checkout dev
git merge feature/your-feature
git push origin dev

# Clean up
git branch -d feature/your-feature
```

---

## 🧪 Testing Requirements

### Before Every Commit
1. **Run unit tests** in the module you modified
2. **Run integration tests** if you changed interfaces
3. **Run regression tests** to ensure nothing broke
4. **Test milestone examples** if core functionality changed

### Test Commands
```bash
# Quick module test
python modules/XX_module/module_dev.py

# Integration tests
pytest tests/integration/

# Regression tests (sandbox integrity)
python tests/regression/run_sandbox_tests.py

# Full test suite
pytest tests/ -v
```

---

## 📝 Commit Message Format

### Structure
```
[TYPE]: Brief description (50 chars or less)

Longer explanation if needed. Explain what and why,
not how (the code shows how).

- Bullet points for multiple changes
- Keep each point focused
- Reference issues if applicable
```

### Types
- **FEAT**: New feature
- **FIX**: Bug fix
- **TEST**: Adding tests
- **DOCS**: Documentation only
- **REFACTOR**: Code change that doesn't fix a bug or add a feature
- **PERF**: Performance improvement
- **STYLE**: Code style changes (formatting, etc.)

### Examples
```bash
# Feature
git commit -m "FEAT: Add attention mechanism with KV caching

Implements scaled dot-product attention with optional KV cache
for efficient autoregressive generation. Reduces memory usage
from O(n²) to O(n) for sequence generation."

# Fix
git commit -m "FIX: Correct convolution output size calculation

Conv2d was calculating output dimensions incorrectly when
stride > 1. Now uses formula: (input - kernel + 2*pad) // stride + 1"

# Test
git commit -m "TEST: Add regression tests for tensor reshaping

Ensures transformer 3D outputs can be properly reshaped for
Linear layer inputs. Prevents dimension mismatch errors."
```

---

## 🚫 What NOT to Do

### Never:
- ❌ Work directly on `main` or `dev`
- ❌ Commit broken code
- ❌ Merge without testing
- ❌ Mix unrelated changes in one commit
- ❌ Use generic commit messages
- ❌ Force push to shared branches
- ❌ Leave commented-out code
- ❌ Commit large binary files

---

## 🔍 Code Review Process

### Before Requesting Review
- [ ] All tests pass
- [ ] Code follows TinyTorch style
- [ ] Documentation updated if needed
- [ ] Commit history is clean
- [ ] Branch is up to date with dev

### Review Checklist
- [ ] Does it solve the stated problem?
- [ ] Is the code clear and maintainable?
- [ ] Are there tests?
- [ ] Does it maintain backward compatibility?
- [ ] Is it pedagogically sound for students?

---

## 🐛 Bug Fix Workflow

### When You Find a Bug
1. **Create issue** (if not exists)
2. **Create fix branch**: `git checkout -b fix/issue-description`
3. **Write failing test** that reproduces the bug
4. **Fix the bug** so test passes
5. **Run full test suite** to ensure no regressions
6. **Commit both** test and fix together
7. **Reference issue** in commit message

### Example
```bash
git checkout -b fix/transformer-reshape-dimensions
# Write test that fails
echo "Write failing test in tests/regression/"
# Fix the bug
echo "Fix in tinytorch/nn/transformers.py"
# Commit together
git add tests/regression/test_transformer_reshaping.py
git add tinytorch/nn/transformers.py
git commit -m "FIX: Handle 3D transformer output in Linear layers

Transformers output (batch, seq, embed) but Linear expects 2D.
Added reshaping logic to handle dimension mismatch.

Tests: tests/regression/test_transformer_reshaping.py"
```

---

## 🔄 Merge Conflict Resolution

### When Conflicts Occur
1. **Don't panic** - conflicts are normal
2. **Pull latest dev** into your branch
3. **Resolve carefully** - understand both changes
4. **Test thoroughly** after resolution
5. **Document** if resolution was non-trivial

### Resolution Process
```bash
# Update your branch
git checkout feature/your-feature
git pull origin dev  # This may cause conflicts

# Resolve conflicts in editor
# Look for <<<<<<< ======= >>>>>>>
# Choose correct resolution

# After resolving
git add .
git commit -m "Merge dev into feature/your-feature and resolve conflicts"

# Test everything still works
pytest tests/
```

---

## 📊 Git Statistics & Health

### Healthy Repository Signs
- ✅ Clear, linear history on main
- ✅ Feature branches are short-lived (< 1 week)
- ✅ Commits are atomic and focused
- ✅ Tests pass on every commit
- ✅ No long-running merge conflicts

### Commands for Repository Health
```bash
# View branch history
git log --oneline --graph --all

# Find branches that need cleanup
git branch --merged  # Can be deleted
git branch --no-merged  # Still need work

# See who's working on what
git shortlog -sn  # Commit count by author
```

---

## 🎯 TinyTorch-Specific Rules

### 1. **Student-Facing Code is Sacred**
Any change to `modules/` must:
- Maintain pedagogical clarity
- Be thoroughly tested
- Not break existing student work

### 2. **Regression Tests for Every Bug**
- Bug found = test written
- Test first, then fix
- Both committed together

### 3. **Documentation in Sync**
- Code changes require doc updates
- Examples must still work
- Module READMEs stay current

### 4. **Performance Claims Need Proof**
- Benchmark before optimization
- Show measurable improvement
- Document in commit message

---

## 🏆 Best Practice Examples

### Good Feature Development
```bash
# Start fresh
git checkout dev && git pull
git checkout -b feature/add-dropout-layer

# Develop with clear commits
git add modules/11_regularization/
git commit -m "FEAT: Add Dropout layer for regularization"

git add tests/unit/test_dropout.py
git commit -m "TEST: Add comprehensive Dropout layer tests"

git add docs/dropout-usage.md
git commit -m "DOCS: Add Dropout usage examples"

# Test and merge
pytest tests/
git checkout dev
git merge feature/add-dropout-layer
```

### Good Bug Fix
```bash
# Reproduce issue
git checkout -b fix/adam-memory-leak

# Test-driven fix
git add tests/regression/test_adam_memory.py
git add tinytorch/optimizers/adam.py
git commit -m "FIX: Prevent memory leak in Adam optimizer

Adam was accumulating gradient history indefinitely.
Now properly clears old gradients after step.

Fixes #42"
```

---

## 📚 Learning from Our Git History

Each commit tells a story:
- What problem we solved
- Why we made certain decisions
- How the framework evolved

Good git practices ensure future contributors (including students!) can understand our development journey.

---

## 🔗 Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

**Remember**: Git history is documentation. Make it clear, make it useful, make it professional.