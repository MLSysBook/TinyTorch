# 📦 Package Manager Agent Specification

## Overview
The Package Manager Agent is a critical specialist responsible for ensuring all student-developed modules properly integrate into the complete TinyTorch package. This agent bridges the gap between individual module development and a working, installable ML framework.

## Current System Analysis

### 🔍 What Exists Now:
1. **Module Structure**: 
   - Development files: `modules/source/XX_module/module_dev.py`
   - Package destination: `tinytorch/core/module.py`
   - Export system: Using nbdev with `#| default_exp` directives

2. **Build Tools**:
   - `tito export` - Converts .py → .ipynb → tinytorch package
   - `tito package` - Package management commands
   - nbdev integration for notebook → package conversion

3. **Testing**:
   - Integration tests in `/tests/` directory
   - Individual module tests within each module
   - No systematic package validation after export

4. **Issues Identified**:
   - No automated verification that exported modules work together
   - Integration tests in root folder (should be better organized)
   - No clear dependency resolution between modules
   - Missing validation that all module pieces "click together"

## 🎯 Package Manager Agent Responsibilities

### Primary Duties:

1. **Module Integration Validation**
   - Verify all module exports are compatible
   - Check inter-module dependencies
   - Ensure no naming conflicts or circular imports
   - Validate that all pieces form a complete system

2. **Build Pipeline Management**
   ```python
   # The agent ensures this workflow:
   Student Code (module_dev.py) 
       ↓ [Convert]
   Notebook (.ipynb)
       ↓ [Export]
   Package Module (tinytorch/core/module.py)
       ↓ [Validate]
   Working TinyTorch Package
   ```

3. **Dependency Resolution**
   - Map module dependencies (e.g., Tensor → Autograd → Training)
   - Ensure proper import order
   - Verify all required components are present
   - Check version compatibility

4. **Package Testing**
   - Run integration tests after EVERY export
   - Verify the complete package can be imported
   - Test that all modules work together
   - Validate student can use the final system

5. **Export Coordination**
   - Work with Module Developer to ensure export tags are correct
   - Verify `#| default_exp` directives match expected structure
   - Ensure consistent naming conventions
   - Manage module versioning

## 🔄 Workflow Integration

### When Package Manager Agent is Invoked:

1. **After Module Development**
   ```
   Module Developer completes work
       ↓
   QA Agent tests module
       ↓
   Package Manager validates export and integration ← YOU ARE HERE
       ↓
   Workflow Coordinator approves
   ```

2. **During Export Process**
   - Pre-export: Verify module is ready for export
   - During export: Monitor for issues
   - Post-export: Validate integration
   - Final check: Ensure complete system works

3. **For System Integration**
   - Student completes Module 1-16
   - Package Manager assembles complete TinyTorch
   - Runs comprehensive system tests
   - Validates students can use their creation

## 🛠️ Specific Implementation Tasks

### 1. Export Validation Pipeline
```bash
# Package Manager ensures this sequence:
tito export --all          # Export all modules
tito test integration      # Run integration tests
tito package validate      # Verify complete package
tito package build         # Build installable package
```

### 2. Module Dependency Map
```python
DEPENDENCY_GRAPH = {
    "tensor": [],
    "activations": ["tensor"],
    "layers": ["tensor"],
    "dense": ["tensor", "layers"],
    "spatial": ["tensor", "layers"],
    "attention": ["tensor", "layers"],
    "dataloader": ["tensor"],
    "autograd": ["tensor"],
    "optimizers": ["tensor", "autograd"],
    "training": ["tensor", "layers", "optimizers", "dataloader"],
    "compression": ["tensor", "layers"],
    "kernels": ["tensor"],
    "benchmarking": ["all"],
    "mlops": ["all"],
    "capstone": ["all"]
}
```

### 3. Integration Test Organization
```
tests/
├── unit/           # Individual module tests
├── integration/    # Inter-module tests
├── system/         # Complete system tests
└── validation/     # Package validation tests
```

### 4. Validation Checklist
- [ ] All modules export successfully
- [ ] No import errors in tinytorch package
- [ ] All integration tests pass
- [ ] Complete ML pipeline works (data → model → train → predict)
- [ ] Package can be pip installed
- [ ] Documentation is complete

## 🚨 Critical Rules for Package Manager

1. **NEVER allow broken exports to reach the package**
2. **MUST validate after EVERY module change**
3. **Block package build if integration tests fail**
4. **Ensure backward compatibility**
5. **Maintain clean module boundaries**

## 📝 Communication Protocol

### With Module Developer:
- "Module X export validation failed: [specific issue]"
- "Please add #| default_exp directive to module"
- "Module dependencies not satisfied"

### With QA Agent:
- "Running integration tests for exported modules"
- "Package validation requires these tests to pass"
- "Found integration issue between Module X and Y"

### With Workflow Coordinator:
- "Package build successful, all modules integrated"
- "Integration failed, blocking release"
- "Ready for student testing"

## 🎯 Success Metrics

The Package Manager succeeds when:
1. Students can run: `pip install -e .` and it works
2. All modules are accessible via `from tinytorch import *`
3. Complete ML pipeline runs without errors
4. Integration tests achieve 100% pass rate
5. Students can build end-to-end models using their code

## 🔧 Implementation Commands

### New tito commands needed:
```bash
tito package validate     # Validate all exports
tito package test         # Run integration tests
tito package build        # Build complete package
tito package verify       # Verify student can use it
tito package report       # Generate integration report
```

## 📋 Handoff Requirements

When Package Manager completes work:
- [ ] Export validation report
- [ ] Integration test results
- [ ] Dependency graph verification
- [ ] Package build status
- [ ] Student usability confirmation

## 🚀 Why This Matters

The Package Manager ensures that the educational journey results in a **working ML framework** that students built themselves. Without this agent, students might have great individual modules that don't work together - defeating the purpose of building a complete system.

This agent is the difference between:
- ❌ 16 separate modules that don't integrate
- ✅ One cohesive TinyTorch framework that actually works

## Next Steps

1. **Add to CLAUDE.md** agent hierarchy
2. **Implement validation commands** in tito
3. **Reorganize tests** into proper structure
4. **Create automated integration pipeline**
5. **Add to agent orchestration workflow**