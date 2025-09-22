# TinyTorch .claude Directory Structure

This directory contains all guidelines, standards, and agent definitions for the TinyTorch project.

## 📁 Directory Structure

```
.claude/
├── README.md                    # This file
├── guidelines/                  # Development standards and principles
│   ├── DESIGN_PHILOSOPHY.md    # KISS principle and simplicity guidelines
│   ├── GIT_WORKFLOW.md         # Git branching and commit standards
│   ├── MODULE_DEVELOPMENT.md   # How to develop TinyTorch modules
│   ├── TESTING_STANDARDS.md    # Testing patterns and requirements
│   ├── PERFORMANCE_CLAIMS.md   # How to make honest performance claims
│   └── AGENT_COORDINATION.md   # How AI agents work together
├── agents/                      # AI agent definitions
│   ├── technical-program-manager.md
│   ├── education-architect.md
│   ├── module-developer.md
│   ├── package-manager.md
│   ├── quality-assurance.md
│   ├── documentation-publisher.md
│   ├── workflow-coordinator.md
│   ├── devops-engineer.md
│   └── tito-cli-developer.md
└── [legacy files to review]

```

## 🎯 Quick Start for New Development

1. **Read Core Principles First**
   - `guidelines/DESIGN_PHILOSOPHY.md` - Understand KISS principle
   - `guidelines/GIT_WORKFLOW.md` - Learn branching requirements

2. **For Module Development**
   - `guidelines/MODULE_DEVELOPMENT.md` - Module structure and patterns
   - `guidelines/TESTING_STANDARDS.md` - How to write tests
   - `guidelines/PERFORMANCE_CLAIMS.md` - How to report results

3. **For Agent Coordination**
   - `guidelines/AGENT_COORDINATION.md` - How agents work together
   - Start with Technical Program Manager (TPM) for all requests

## 📋 Key Principles Summary

### 1. Keep It Simple, Stupid (KISS)
- One file, one purpose
- Clear over clever
- Verified over theoretical
- Direct over abstract

### 2. Git Workflow
- ALWAYS work on feature branches
- NEVER commit directly to main/dev
- Test before committing
- No automated attribution in commits

### 3. Module Development
- Edit .py files only (never .ipynb)
- Test immediately after implementation
- Include systems analysis (memory, performance)
- Follow exact structure pattern

### 4. Testing Standards
- Test immediately, not at the end
- Simple assertions over complex frameworks
- Tests should educate, not just verify
- Always compare against baseline

### 5. Performance Claims
- Only claim what you've measured
- Include all relevant metrics
- Report failures honestly
- Reproducibility is key

### 6. Agent Coordination
- TPM is primary interface
- Sequential workflow with clear handoffs
- QA testing is MANDATORY
- Package integration is MANDATORY

## 🚀 Common Workflows

### Starting New Module Development
```bash
1. Create feature branch
2. Request TPM agent assistance
3. Follow MODULE_DEVELOPMENT.md structure
4. Test with TESTING_STANDARDS.md patterns
5. Verify performance per PERFORMANCE_CLAIMS.md
6. Merge following GIT_WORKFLOW.md
```

### Making Performance Claims
```bash
1. Run baseline measurements
2. Run actual measurements
3. Calculate real improvements
4. Document with all metrics
5. No unverified claims
```

### Working with Agents
```bash
1. Always start with TPM agent
2. Let TPM coordinate other agents
3. Wait for QA approval before proceeding
4. Wait for Package Manager integration
5. Only then commit
```

## 📝 Important Notes

- **Virtual Environment**: Always activate .venv before development
- **Honesty**: Report actual results, not aspirations
- **Simplicity**: When in doubt, choose the simpler option
- **Education First**: We're teaching, not impressing

## 🔗 Quick Links

- Main Instructions: `/CLAUDE.md`
- Module Source: `/modules/source/`
- Examples: `/examples/`
- Tests: `/tests/`

## 📌 Remember

> "If students can't understand it, we've failed."

Every decision should be filtered through:
1. Is it simple?
2. Is it honest?
3. Is it educational?
4. Is it verified?

If any answer is "no", reconsider.