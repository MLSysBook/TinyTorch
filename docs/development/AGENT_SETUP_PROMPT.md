# Universal Agent Setup Prompt for Any Project

Copy and paste this prompt to set up the same agent orchestration system in your other projects:

---

## 🤖 Multi-Agent Team Setup Request

I want to set up a professional multi-agent workflow system for this project. Please create a `CLAUDE.md` file in the project root that establishes the following:

### 1. **Agent Team Structure**
Create these specialized agents with clear responsibilities:

```
Workflow Coordinator (Team Lead)
    ├── Architect (Strategy & Design)
    ├── Developer (Implementation)
    ├── Quality Assurance (Testing & Validation)
    └── Documentation (Communication & Docs)
```

### 2. **Mandatory QA Testing Protocol**
- QA Agent MUST test ALL code changes before ANY commit
- QA has veto power - can block commits if tests fail
- Developer CANNOT mark tasks complete without QA approval
- Every code change triggers automatic QA review
- Create comprehensive test suites that check:
  - Code imports without errors
  - Functions/classes work correctly
  - No syntax errors present
  - All tests pass successfully

### 3. **Standard Workflow Pattern**
Enforce this sequence for EVERY code update:

1. **Planning Phase** (Coordinator + Architect)
   - Define objectives and structure
2. **Implementation Phase** (Developer)
   - Write code following specifications
   - MUST notify QA when done
3. **Testing Phase** (QA) - MANDATORY
   - Run comprehensive tests
   - Block progress if tests fail
4. **Documentation Phase** (Documentation)
   - Add clear documentation
5. **Review Phase** (Coordinator)
   - Verify all agents completed tasks
   - Ensure QA tests passed

### 4. **Agent Communication Rules**
- Structured handoffs between agents
- Each handoff includes checklist of completed/pending items
- Test results must accompany all code changes
- Clear accountability for each agent

### 5. **Git Workflow Standards**
- Always use feature branches (never commit to main)
- Test before committing
- Descriptive commit messages
- Clean commit history

### 6. **Enforcement Mechanisms**
- File should be named `CLAUDE.md` in project root
- Claude reads this automatically at session start
- Rules override default behavior
- QA testing cannot be skipped

### 7. **Project-Specific Customization**
Adapt the agents for this specific project:
- What type of project is this? [Web app, ML system, API, etc.]
- What testing frameworks should QA use?
- What documentation standards apply?
- Any special requirements?

Please create the CLAUDE.md file with these specifications, ensuring that:
1. Every future Claude session will automatically follow these rules
2. QA testing is mandatory and cannot be bypassed
3. The agent team works as a cohesive unit
4. Code quality is maintained through enforced testing gates

Also create any supporting test files or scripts needed for the QA agent to properly validate code changes.

---

## 📋 Quick Setup Instructions

1. **Copy the above prompt**
2. **Paste it in your other Claude project**
3. **Claude will create the CLAUDE.md file**
4. **The system activates automatically in future sessions**

## 🎯 Benefits You'll Get

- ✅ Automatic QA testing before commits
- ✅ Structured agent teamwork
- ✅ Consistent code quality
- ✅ Clear accountability
- ✅ Professional development workflow
- ✅ No broken code in your repository

## 💡 Pro Tips

- Customize agent names for your project type
- Add project-specific testing requirements
- Include your preferred git workflow
- Specify documentation standards
- Add any unique project rules

## 🔄 Verification

After setup, test by asking Claude:
- "What are the agent responsibilities?"
- "Can you commit without QA testing?"
- "Show me the workflow for updating code"

Claude should respond following the established protocols.