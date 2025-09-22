# Claude Code Instructions for TinyTorch

## 📚 **START HERE: Read the Guidelines**

All development standards, principles, and workflows are documented in the `.claude/` directory.

### Quick Start
```bash
# First, read the overview
cat .claude/README.md

# Then read core guidelines in order:
cat .claude/guidelines/DESIGN_PHILOSOPHY.md    # KISS principle
cat .claude/guidelines/GIT_WORKFLOW.md        # Git standards  
cat .claude/guidelines/MODULE_DEVELOPMENT.md  # Building modules
cat .claude/guidelines/TESTING_STANDARDS.md   # Testing patterns
```

## 🎯 Core Mission

**Build an educational ML framework where students learn ML systems engineering by implementing everything from scratch.**

Key principles:
- **KISS**: Keep It Simple, Stupid
- **Build to Learn**: Implementation teaches more than reading
- **Systems Focus**: Not just algorithms, but engineering
- **Honest Claims**: Only report verified performance

## ⚡ Critical Policies

1. **ALWAYS use virtual environment** (`.venv`)
2. **ALWAYS work on feature branches** (never main/dev directly)
3. **ALWAYS test before committing**
4. **NEVER add automated attribution** to commits
5. **NEVER edit .ipynb files directly** (edit .py only)

## 🤖 Working with AI Agents

**Always start with the Technical Program Manager (TPM)**:
- TPM coordinates all other agents
- Don't invoke agents directly
- Follow the workflow in `.claude/guidelines/AGENT_COORDINATION.md`

## 📁 Key Directories

```
.claude/guidelines/     # All development standards
.claude/agents/        # AI agent definitions
modules/source/        # Module implementations (.py files)
examples/             # Working examples (keep simple)
tests/               # Test suites
```

## 🚨 Think Critically

**Don't just agree with suggestions. Always:**
1. Evaluate if it makes pedagogical sense
2. Check if there's a simpler way
3. Verify it actually works
4. Consider student perspective

## 📋 Before Any Work

1. **Read guidelines**: Start with `.claude/README.md`
2. **Create branch**: Follow `.claude/guidelines/GIT_WORKFLOW.md`
3. **Activate venv**: `source .venv/bin/activate`
4. **Use TPM agent**: For coordinated development

## 🎓 Remember

> "If students can't understand it, we've failed."

Every decision should be:
- Simple
- Verified
- Educational
- Honest

---

**For detailed instructions on any topic, see the appropriate file in `.claude/guidelines/`**