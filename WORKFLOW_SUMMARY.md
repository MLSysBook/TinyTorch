# TinyTorch Development Workflow

## The Complete Process

### 🎯 **Workflow Coordinator** - Your Single Point of Contact
**When you ask: "What's next?" or "Who does this?" → Talk to Workflow Coordinator**

They know:
- Complete 5-phase workflow
- Who does what when
- Current module status
- Quality gate requirements
- How to escalate issues

## The 5-Phase Workflow

### **Phase 1: DESIGN** → Education Architect
```
User Request → Workflow Coordinator → Education Architect
```
**Delivers**: Educational specifications document
- Learning objectives defined
- Content structure outlined  
- Lab sections specified
- Assessment criteria established

### **Phase 2: IMPLEMENTATION** → Module Developer  
```
Educational Spec → Workflow Coordinator → Module Developer
```
**Delivers**: Complete module with educational scaffolding
- Code implemented with BEGIN/END SOLUTION blocks
- 5 C's format applied
- Lab-style content added
- Tests working
- NBGrader metadata correct

### **Phase 3: VALIDATION** → Quality Assurance
```
Complete Module → Workflow Coordinator → Quality Assurance  
```
**Delivers**: QA-approved module
- NBGrader compatibility verified
- Educational effectiveness confirmed
- Technical correctness validated
- Integration tested

### **Phase 4: RELEASE** → DevOps Engineer
```
QA-Approved Module → Workflow Coordinator → DevOps Engineer
```
**Delivers**: Student-ready release
- Student versions generated via NBGrader
- Autograding workflow tested
- Distribution packages created
- Infrastructure updated

### **Phase 5: PUBLISHING** → Documentation Publisher
```
Released Module → Workflow Coordinator → Documentation Publisher
```
**Delivers**: Public documentation
- Jupyter Book website updated
- Instructor materials created
- API documentation generated
- Community access enabled

## Quality Gates

**Gate 1**: Educational design complete ✅
**Gate 2**: Implementation complete ✅  
**Gate 3**: Quality validation passed ✅
**Gate 4**: Release ready ✅
**Gate 5**: Publicly available ✅

## Who You Talk To

### **General workflow questions** → Workflow Coordinator
- "What's the next step?"
- "Who should do this task?"
- "What's blocking progress?"
- "When will this be done?"

### **Educational design questions** → Education Architect  
- "How should we structure learning?"
- "What lab content is needed?"
- "Are learning objectives clear?"

### **Implementation questions** → Module Developer
- "How should this be coded?"
- "Is NBGrader setup correct?"
- "Are tests sufficient?"

### **Quality concerns** → Quality Assurance
- "Does this meet standards?"
- "Will students be able to learn from this?"
- "Is integration working?"

### **Release issues** → DevOps Engineer
- "Can students access this?"
- "Is autograding working?"
- "Are packages building?"

### **Documentation needs** → Documentation Publisher
- "Is this ready for public use?"
- "Do instructors have what they need?"
- "Is the website updated?"

## The Answer to Your Question

**Q: "What's the workflow once a module is generated?"**

**A: Education Architect reviews first, then it flows through the pipeline:**

```
Module Developer creates → Education Architect reviews educational design → 
Quality Assurance validates → DevOps Engineer prepares release → 
Documentation Publisher makes it public
```

**Your dedicated workflow agent**: **Workflow Coordinator** - they know the complete flow and can answer any process questions.

## Current Module Status Example

**Module 01 & 02**: Currently in Phase 2 (Implementation) with improvements being made
**Next**: Move to Phase 3 (Quality Assurance validation)
**Who handles**: Workflow Coordinator orchestrates the handoff

**You always talk to Workflow Coordinator for "what's next" questions!**