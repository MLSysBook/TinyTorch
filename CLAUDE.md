# Claude Instructions for TinyTorch

> **🎓 TinyTorch**: A complete educational ML framework where students build every component from scratch. This is a production-quality learning system combining academic rigor with professional ML engineering practices.

## 🤖 Available Agents

TinyTorch uses a **7-agent team structure** for development:

### Primary Interface
- **🟡 Technical Program Manager** - Your main point of contact, coordinates all work

### Core Development Team
- **🟢 Module Developer** - Implements modules with educational scaffolding
- **🟣 Education Reviewer** - Validates pedagogical effectiveness
- **🔴 Quality Assurance** - Comprehensive testing and validation
- **🔵 Package Manager** - Module integration and exports
- **🌐 Website Manager** - Content strategy and documentation
- **🖥️ DevOps TITO** - CLI development and infrastructure

**Details**: See `.claude/agents/` for individual agent specifications

## 📋 Workflows

Each slash command triggers a specific workflow with defined agent sequences:

- **Website Updates**: `.claude/workflows/website.md`
- **Module Development**: `.claude/workflows/module.md`
- **Testing**: `.claude/workflows/test.md`
- **TITO CLI**: `.claude/workflows/tito.md`
- **Education**: `.claude/workflows/education.md`

**Quick Reference**: See `.claude/workflows/QUICK_REFERENCE.md`

## 💡 Project Context

### What is TinyTorch?

TinyTorch is a **Machine Learning Systems course** where students build a complete ML framework from scratch - tensors, autograd, optimizers, CNNs, transformers, and production systems.

**Key Characteristics**:
- 🎓 **Educational-first**: Every module is a learning experience
- 🔧 **Production-quality**: Real engineering, not toy examples
- 📚 **Progressive**: 9 modules from basics to advanced
- 🧪 **Test-driven**: Inline tests + integration tests
- 🎯 **Systems-focused**: Performance, memory, user experience matter

### Architecture

```
TinyTorch/
├── modules/source/        # Development modules (00-08)
├── assignments/           # NBGrader assignments (generated)
├── tinytorch/            # Exported package
├── tito/                 # Student CLI
├── tests/                # Integration tests
├── book/                 # Jupyter Book documentation
└── .claude/              # This configuration system
```

## 📚 Additional Resources

### For Development
- `.claude/guidelines/` - All development standards
- `.claude/workflows/` - Detailed workflow specifications
- `docs/development/` - Technical guides

### For Understanding
- `NORTH_STAR.md` - Project vision and goals
- `MODULE_OVERVIEW.md` - Module structure and dependencies
- `book/` - Complete documentation website

### For Reference
- `.claude/AGENT_REFERENCE.md` - Agent team structure
- `.claude/CONSOLIDATED_KNOWLEDGE_BASE.md` - All critical knowledge
- `.claude/UNIVERSAL_TEMPLATE.md` - System design patterns

## 🎯 Starting Points

### New to the Project?
1. Read `NORTH_STAR.md` for vision
2. Review `.claude/README.md` for system overview
3. Explore `modules/source/` for existing modules
4. Check `.claude/guidelines/DESIGN_PHILOSOPHY.md`

### Need to Fix Something?
1. Use `/fix [issue]` or `/module [fix request]`
2. Let the agent system coordinate the work
3. Trust the quality gates

### Want to Add Features?
1. Use `/module [feature request]` or `/tito [feature]`
2. Agents will design, implement, test, and integrate
3. Educational review ensures pedagogy is maintained

### Working on Documentation?
1. Use `/website [request]`
2. Website Manager coordinates content strategy
3. Content lives in `book/` directory

## 🚀 Best Practices

1. **Use slash commands** - They trigger proper workflows
2. **Trust the agents** - They have embedded knowledge
3. **Follow quality gates** - They prevent technical debt
4. **Test frequently** - Every change should be validated
5. **Think systems** - Performance, UX, and scale matter
6. **Stay educational** - Students are the end users

## 🤝 Working with Claude

### What Claude Knows
- Complete TinyTorch architecture and history
- All module implementations and patterns
- NBGrader integration requirements
- Testing standards and workflows
- Educational design principles
- Git workflow and best practices

### What to Provide
- Specific requests ("fix X", "add Y", "improve Z")
- Context if working on edge cases
- Student feedback if improving education
- Error messages if debugging

### What Claude Does
- Coordinates appropriate agents
- Ensures quality standards
- Tests changes thoroughly
- Maintains educational value
- Follows established patterns
- Documents decisions

## 📖 Quick Reference

| Need | Command | File |
|------|---------|------|
| System overview | - | `.claude/README.md` |
| Agent details | - | `.claude/agents/*.md` |
| Workflow specs | - | `.claude/workflows/*.md` |
| Development guide | - | `.claude/guidelines/MODULE_DEVELOPMENT.md` |
| Testing standards | - | `.claude/guidelines/TESTING_STANDARDS.md` |
| Git workflow | - | `.claude/guidelines/GIT_WORKFLOW.md` |
| Project vision | - | `NORTH_STAR.md` |
| Module overview | - | `MODULE_OVERVIEW.md` |

---

**Ready to work?** Use a slash command or describe what you need, and the appropriate agents will coordinate to help you.

**Questions?** Start with `.claude/README.md` for the complete system overview.


