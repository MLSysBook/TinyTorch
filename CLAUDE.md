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

**Remember**: Professional software development always uses branches. This keeps the codebase stable, enables collaboration, and maintains a clean development history.