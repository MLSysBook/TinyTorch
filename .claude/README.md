# 📚 TinyTorch .claude Configuration Hub

This directory contains the complete development system for TinyTorch: workflows, agents, and guidelines.

## 🚀 Quick Start: Slash Commands

Use these commands to trigger workflows:
```
/website [request]   - Update website content
/module [request]    - Module development  
/test [target]       - Run tests
/tito [request]      - CLI updates
/education [request] - Educational improvements
```

### Example Usage
```
/website add a debugging guide
/module fix tensor backward pass
/test all checkpoints
/tito add progress visualization
/education improve autograd learning
```

## 📁 Directory Structure

```
.claude/
├── README.md                           # This file (START HERE)
│
├── workflows/                          # 🎯 WORKFLOW DEFINITIONS
│   ├── README.md                      # Workflow system documentation
│   ├── website.md                     # /website command workflow
│   ├── module.md                      # /module command workflow
│   ├── test.md                        # /test command workflow
│   ├── tito.md                        # /tito command workflow
│   └── education.md                   # /education command workflow
│
├── agents/                             # AI AGENT DEFINITIONS
│   ├── website-content-strategist.md  # Website content & strategy
│   ├── website-designer.md            # Website implementation
│   ├── module-developer.md            # Module implementation
│   ├── quality-assurance.md           # Testing & validation
│   ├── package-manager.md             # Integration & packaging
│   ├── education-architect.md         # Learning design
│   ├── tito-cli-developer.md          # CLI development
│   └── technical-program-manager.md   # Orchestration
│
├── guidelines/                         # DEVELOPMENT STANDARDS
│   ├── DESIGN_PHILOSOPHY.md          # KISS principle
│   ├── GIT_WORKFLOW.md               # Git standards
│   ├── MODULE_DEVELOPMENT.md         # Module patterns
│   ├── TESTING_STANDARDS.md          # Testing requirements
│   ├── PERFORMANCE_CLAIMS.md         # Honest reporting
│   └── AGENT_COORDINATION.md         # Agent teamwork
│
└── docs/                              # Additional documentation
```

## 🎯 How the System Works

### 1. Slash Commands → Workflows
When you type `/website add content`, Claude:
1. Recognizes the `/website` command
2. Loads the `workflows/website.md` specification
3. Executes the defined agent sequence
4. Ensures quality gates are met

### 2. Workflows → Agents
Each workflow defines:
- Which agents to use
- What order to execute them
- What each agent should produce
- How to handle failures

### 3. Agents → Implementation
Each agent has:
- Specific expertise area
- Clear responsibilities
- Quality standards
- Handoff protocols

## 📖 Key Documents to Read

### For New Users
1. **Start Here:** `../CLAUDE.md` - Main instructions
2. **Workflows:** `workflows/README.md` - How to use slash commands
3. **Guidelines:** `guidelines/DESIGN_PHILOSOPHY.md` - Core principles

### For Development
- `guidelines/GIT_WORKFLOW.md` - Git practices
- `guidelines/MODULE_DEVELOPMENT.md` - Module standards
- `guidelines/TESTING_STANDARDS.md` - Testing requirements

### For Understanding Agents
- `agents/` folder - Individual agent capabilities
- `workflows/` folder - How agents work together
- `guidelines/AGENT_COORDINATION.md` - Coordination patterns

## 🚨 Important Rules

1. **Always use slash commands** when available
2. **Follow workflow sequences** - don't skip steps
3. **Respect quality gates** - fix failures before proceeding
4. **Check guidelines** before major changes
5. **Use version control** - work on feature branches

## 💡 Best Practices

### Using Slash Commands
✅ DO: `/website add debugging guide`
❌ DON'T: "Can you update the website with a debugging guide?"

### Following Workflows  
✅ DO: Let workflow complete all steps
❌ DON'T: Skip agents or change order

### Quality Standards
✅ DO: Fix issues when quality gates fail
❌ DON'T: Bypass testing or validation

## 🔧 Customization

### Adding New Workflows
1. Create `workflows/[name].md`
2. Define agent sequence
3. Add to slash commands in `CLAUDE.md`
4. Document in `workflows/README.md`

### Adding New Agents
1. Create `agents/[name].md`
2. Define capabilities and responsibilities
3. Update relevant workflows
4. Add to agent coordination guide

## 📊 System Health

### Check Configuration
```bash
ls -la .claude/          # View structure
ls .claude/workflows/    # List workflows
ls .claude/agents/       # List agents
```

### Validate Setup
- All workflows have corresponding files
- All agents referenced in workflows exist
- Guidelines are up to date
- No orphaned or duplicate files

---

**Questions?** Start with `workflows/README.md` for detailed workflow documentation.