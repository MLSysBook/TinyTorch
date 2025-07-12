---
description: Overview of TinyTorch's focused, actionable rule set - each rule under 500 lines and scoped to specific concerns.
---

# TinyTorch Rules Overview

This directory contains focused, actionable rules for TinyTorch development. Each rule is scoped to specific concerns and kept under 500 lines for clarity.

## Rule Organization

### Core Context & Structure
- **[ml-systems-course-context.mdc](ml-systems-course-context.mdc)** (64 lines) - Course philosophy and learning objectives
- **[tinytorch-project-structure.mdc](tinytorch-project-structure.mdc)** (62 lines) - Dual-structure architecture guide
- **[user-preferences.mdc](user-preferences.mdc)** (47 lines) - User preferences and conventions

### Development Workflow
- **[development-workflow.mdc](development-workflow.mdc)** (93 lines) - Complete tito CLI workflow
- **[git-workflow.mdc](git-workflow.mdc)** (157 lines) - Git workflow for incremental commits
- **[cli-patterns.mdc](cli-patterns.mdc)** (174 lines) - CLI development patterns

### Module Development
- **[module-development-best-practices.mdc](module-development-best-practices.mdc)** (158 lines) - Core module development principles
- **[nbdev-educational-pattern.mdc](nbdev-educational-pattern.mdc)** (187 lines) - NBDev patterns and directives
- **[real-data-principles.mdc](real-data-principles.mdc)** (112 lines) - "Real Data, Real Systems" requirements

### Testing
- **[testing-patterns.mdc](testing-patterns.mdc)** (115 lines) - Testing standards with pytest and real data

## Key Design Principles

### 1. Focused Scope
Each rule addresses a specific concern:
- **Context**: Course philosophy and project structure
- **Workflow**: Development and Git processes
- **Implementation**: Module development and NBDev patterns
- **Quality**: Testing and real data requirements

### 2. No Redundancy
- Testing content consolidated in `testing-patterns.mdc`
- Real data principles extracted to `real-data-principles.mdc`
- NBDev patterns focused on directives and structure
- Module development focused on core principles

### 3. Actionable Guidance
- Concrete examples with code snippets
- Clear DO/DON'T patterns
- Specific file references and commands
- Checklists for quality standards

### 4. Composable Rules
Rules reference each other without duplication:
- `development-workflow.mdc` references `git-workflow.mdc`
- `module-development-best-practices.mdc` references `real-data-principles.mdc`
- `testing-patterns.mdc` stands alone with focused testing guidance

## Rule Usage

Each rule is designed to be:
- **Self-contained**: Can be read independently
- **Actionable**: Provides specific guidance and examples
- **Focused**: Addresses one main concern
- **Concise**: Under 500 lines for easy consumption
- **Referenced**: Can be cited in development discussions

## Quality Standards Met

✅ **Focused**: Each rule has a single, clear purpose
✅ **Actionable**: Concrete examples and guidance
✅ **Scoped**: No rule exceeds 500 lines
✅ **Non-redundant**: No duplicate content across files
✅ **Composable**: Rules reference each other cleanly
✅ **Maintainable**: Easy to update individual concerns 