# Agent Workflow: Distinct Responsibilities

## The Problem with Overlap
Previously, agents were stepping on each other's toes. Each agent is an EXPERT in their domain and should handle ONLY their expertise.

## Clear Division of Labor

### 1. Education Architect
**ONLY handles educational strategy:**
- Learning objectives analysis
- Pedagogical flow assessment
- Build→Use→Understand compliance
- Concept progression logic
- Educational effectiveness

**Does NOT touch:**
- Code implementation
- Docstring writing
- Test writing
- Markdown polishing

### 2. Module Developer  
**ONLY handles code implementation:**
- NBGrader metadata compliance
- Solution block structure
- Implementation patterns (TODO/APPROACH/EXAMPLE/HINTS)
- Code functionality
- Technical scaffolding

**Does NOT touch:**
- Educational content writing
- Docstring polishing
- Test explanations
- Markdown prose

### 3. Quality Assurance
**ONLY handles testing validation:**
- Test coverage analysis
- Testing pattern compliance
- Immediate vs comprehensive test separation
- Test functionality verification
- Testing infrastructure

**Does NOT touch:**
- Test explanations (that's docs)
- Code implementation
- Educational strategy
- Prose writing

### 4. Documentation Publisher
**ONLY handles writing and clarity:**
- Markdown prose polishing
- Docstring enhancement
- Section header consistency
- Writing clarity and flow
- Explanation quality

**Does NOT touch:**
- Code implementation
- Test logic
- Educational strategy
- Technical patterns

## The Workflow Process

### Phase 1: Education Architect Review
**Delivers:** Educational strategy recommendations
```markdown
## Education Architect Recommendations for Module 01:
- Restructure to Build→Use→Understand flow
- Map content to 10-part structure  
- Add missing foundational concepts
- Improve learning objective clarity
```

### Phase 2: Module Developer Implementation
**Takes:** Education Architect's strategy
**Delivers:** Code and technical structure improvements
```python
# Module Developer implements:
# - 10-part structure with proper headers
# - NBGrader metadata fixes
# - Solution block corrections
# - Technical scaffolding improvements
```

### Phase 3: Quality Assurance Testing
**Takes:** Module Developer's implementation
**Delivers:** Testing validation and fixes
```python
# QA implements:
# - Missing test functions
# - Test pattern compliance
# - Comprehensive testing sections
# - Testing infrastructure
```

### Phase 4: Documentation Publisher Polish
**Takes:** All previous work
**Delivers:** Writing and clarity improvements
```markdown
# Docs Publisher polishes:
# - Markdown prose clarity
# - Docstring enhancement
# - Section flow improvement
# - Explanation quality
```

### Phase 5: Integration & Validation
**Consolidates:** All improvements into final module

## Key Principle: Experts Do Expert Work

- **Education Architect** = Pedagogical expert, not coder
- **Module Developer** = Technical expert, not writer  
- **Quality Assurance** = Testing expert, not educator
- **Documentation Publisher** = Writing expert, not implementer

Each agent stays in their lane and does what they're best at!