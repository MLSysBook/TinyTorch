# TinyTorch Agent Reference

## Current 7-Agent Team Structure

### Primary Interface
#### 🟡 Technical Program Manager
**Location**: `.claude/agents/technical-program-manager.md`
**Role**: Primary communication interface and project orchestration
**Responsibilities**:
- Single point of contact for all user communications
- Coordinates specialized agents and manages workflows
- Tracks progress and manages dependencies
- Ensures quality gates and timeline management

### Core Development Agents

#### 🟢 Module Developer
**Location**: `.claude/agents/module-developer.md`
**Responsibilities**:
- Implements modules with educational scaffolding
- Ensures NBGrader compatibility and proper exports
- Creates immediate tests after implementations
- Follows ML systems analysis patterns

**Key Knowledge**:
- BEGIN/END SOLUTION blocks for student releases
- Scaffolding must be OUTSIDE solution blocks
- Test-immediately pattern is non-negotiable
- Every cell needs unique grade_id
- ML systems focus with performance analysis

#### 🟣 Education Reviewer
**Location**: `.claude/agents/education-reviewer.md`
**Responsibilities**:
- Educational design, assessment, and technical validation
- Unified pedagogical and technical review
- Learning progression and systems thinking development
- Cognitive load management and production alignment

**Key Knowledge**:
- NBGrader enables dual instructor/student versions
- Educational effectiveness evaluation
- Progressive difficulty with immediate feedback
- ML systems mental model development

#### 🔴 Quality Assurance
**Location**: `.claude/agents/quality-assurance.md`
**Responsibilities**:
- Comprehensive testing and validation
- NBGrader compatibility verification
- Technical correctness and performance validation
- Integration testing and quality gates

**Key Knowledge**:
- Unique grade_ids prevent autograding failures
- Solution blocks must be properly placed
- Test cells must be locked with points
- Performance and systems analysis validation

#### 🔵 Package Manager
**Location**: `.claude/agents/package-manager.md`
**Responsibilities**:
- Module integration and export validation
- Dependency resolution between modules
- Package builds and distribution management
- Integration testing coordination

**Key Knowledge**:
- Module export directives and package structure
- Cross-module dependencies and compatibility
- Build system integration and validation
- TinyTorch package architecture

#### 🌐 Website Manager
**Location**: `.claude/agents/website-manager.md`
**Responsibilities**:
- Unified content and design strategy
- Educational framework content creation
- Multi-audience messaging and user experience
- Content deduplication and cross-reference management

**Key Knowledge**:
- Educational website architecture and navigation
- Content presentation optimization for learning
- Visual hierarchy and design guidelines
- Open source framework documentation patterns

#### 🖥️ DevOps TITO
**Location**: `.claude/agents/devops-tito.md`
**Responsibilities**:
- TITO CLI development and enhancement
- Infrastructure management and deployment
- Build and release process automation
- Development environment optimization

**Key Knowledge**:
- CLI command development and testing
- Rich library integration for visualizations
- System administration and monitoring
- User experience optimization for tools

## Workflow Coordination

### Standard Development Flow
```
User Request → TPM → Specialized Agents → Quality Gates → Delivery
```

### Agent Handoff Criteria
- Module Developer → QA Agent (after implementation complete)
- QA Agent → Package Manager (after tests pass)
- Package Manager → TPM (after integration verified)
- Education Reviewer → Module Developer (after educational specs complete)
- Website Manager → Implementation teams (after content strategy complete)

### Quality Gates
- **QA Agent can block**: No commits without test approval
- **Package Manager can block**: No releases without integration verification
- **TPM coordinates**: Final approval and user communication

## REMOVED AGENTS (Historical Reference)
These agents were consolidated into the current 7-agent structure:
- Education Architect → merged into Education Reviewer
- Website Content Strategist → merged into Website Manager
- Website Designer → merged into Website Manager
- Assessment Designer → merged into Education Reviewer
- TITO CLI Developer → became DevOps TITO
- ML Framework Advisor → capabilities distributed among agents
- Educational Review Expert → merged into Education Reviewer

## Agent Communication Protocol
All agents follow structured handoffs with:
1. What was completed
2. What needs to be done next
3. Any issues found
4. Test results (if applicable)
5. Recommendations for next agent

**Updated**: September 2024 - Reflects current 7-agent structure with streamlined coordination.