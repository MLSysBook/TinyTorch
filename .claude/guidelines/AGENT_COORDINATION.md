# TinyTorch Agent Coordination Guidelines

## 🎯 Core Principle

**Agents work in sequence with clear handoffs, not in isolation.**

## 🤖 The Agent Team (Consolidated)

### Primary Interface: Technical Program Manager (TPM)

The TPM is your SINGLE point of communication for all development.

```
User Request → TPM → Coordinates Agents → Reports Back
```

**The TPM coordinates these core agents:**
- **Education Reviewer** - Educational design, assessment, and technical validation
- **Module Developer** - Code implementation
- **Package Manager** - Integration and builds
- **Quality Assurance** - Testing and validation
- **Website Manager** - Website content and strategy
- **DevOps TITO** - Infrastructure and CLI development

## 📋 Standard Development Workflow

### The Sequential Pattern

**For EVERY module development:**

```
1. Planning (TPM + Education Reviewer)
   ↓
2. Implementation (Module Developer)
   ↓
3. Testing (Quality Assurance) ← MANDATORY
   ↓
4. Integration (Package Manager) ← MANDATORY
   ↓
5. Documentation (Education Reviewer)
   ↓
6. Review (TPM)
```

### Critical Handoff Points

**Module Developer → QA Agent**
```python
# Module Developer completes implementation
"Implementation complete. Ready for QA testing.
Files modified: 02_tensor_dev.py
Key changes: Added reshape operation with broadcasting"

# QA MUST test before proceeding
```

**QA Agent → Package Manager**
```python
# QA completes testing
"All tests passed.
- Module imports correctly
- All functions work as expected
- Performance benchmarks met
Ready for package integration"

# Package Manager MUST verify integration
```

## 🚫 Blocking Rules

### QA Agent Can Block Progress

**If tests fail, STOP everything:**
- No commits allowed
- No integration permitted
- Must fix and re-test

### Package Manager Can Block Release

**If integration fails:**
- Module doesn't export correctly
- Breaks other modules
- Package won't build

## 📝 Agent Communication Protocol

### Structured Handoffs

Every handoff must include:
1. **What was completed**
2. **What needs to be done next**
3. **Any issues found**
4. **Test results (if applicable)**
5. **Recommendations**

**Example:**
```
From: Module Developer
To: QA Agent

Completed:
- Implemented attention mechanism in 07_attention_dev.py
- Added scaled dot-product attention
- Included positional encoding

Needs Testing:
- Attention score computation
- Mask application
- Memory usage with large sequences

Known Issues:
- Performance degrades with sequences >1000 tokens

Recommendations:
- Focus testing on edge cases with padding
```

## 🔄 Parallel vs Sequential Work

### Can Work in Parallel

✅ Different modules by different developers
✅ Documentation while code is being tested
✅ Planning next modules while current ones build

### Must Be Sequential

❌ Implementation → Testing (MUST test after implementation)
❌ Testing → Integration (MUST pass tests first)
❌ Integration → Commit (MUST integrate successfully)

## 🎯 The Checkpoint Success Story

**How agents successfully implemented the 16-checkpoint system:**

1. **Education Reviewer** designed capability progression
2. **TPM** orchestrated implementation
3. **Module Developer** built checkpoint tests + CLI
4. **QA Agent** validated all 16 checkpoints work
5. **Package Manager** ensured integration with modules
6. **Website Manager** updated all docs

**Result:** Complete working system with proper handoffs

## ⚠️ Common Coordination Failures

### Working in Isolation
❌ Module Developer implements without QA testing
❌ Documentation written before code works
❌ Integration attempted before tests pass

### Skipping Handoffs
❌ Direct commit without QA approval
❌ Missing Package Manager validation
❌ No TPM review

### Poor Communication
❌ "It's done" (no details)
❌ No test results provided
❌ Issues discovered but not reported

## 📋 Agent Checklist

### Before Module Developer Starts
- [ ] Education Reviewer defined learning objectives
- [ ] TPM approved plan
- [ ] Clear specifications provided

### Before QA Testing
- [ ] Module Developer completed ALL implementation
- [ ] Code follows standards
- [ ] Basic self-testing done

### Before Package Integration  
- [ ] QA Agent ran comprehensive tests
- [ ] All tests PASSED
- [ ] Performance acceptable

### Before Commit
- [ ] Package Manager verified integration
- [ ] Documentation complete
- [ ] TPM approved

## 🔧 Conflict Resolution

**If agents disagree:**

1. **QA has veto on quality** - If tests fail, stop
2. **Education Reviewer owns learning objectives**
3. **TPM resolves other disputes**
4. **User has final override**

## 📌 Remember

> Agents amplify capabilities when coordinated, create chaos when isolated.

**Key Success Factors:**
- Clear handoffs between agents
- Mandatory testing and integration
- Structured communication
- Sequential workflow where needed
- Parallel work where possible