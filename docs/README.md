# 📚 TinyTorch Documentation

Welcome to the TinyTorch documentation! This directory contains guides organized by audience and purpose.

## 🎯 **Find Your Documentation**

### 🎓 **Students** (Taking the ML Systems Course)
**Start here if you're learning ML systems:**

- **[Project Guide](students/project-guide.md)** - Track your progress through all 12 modules
- **[Quickstart](../quickstart.md)** - Get up and running quickly
- **[Main README](../README.md)** - Project overview and introduction

### 👨‍🏫 **Instructors** (Teaching or Designing the Course)
**Understanding the educational philosophy:**

- **[Pedagogical Principles](pedagogy/pedagogical-principles.md)** - The "Build → Use → Understand" framework
- **[Vision](pedagogy/vision.md)** - High-level course philosophy and learning approach
- **[Testing Architecture](pedagogy/testing-architecture.md)** - Dual testing system for education

### 🔧 **Human Developers** (Building New Modules)
**Creating or contributing to TinyTorch modules:**

- **[Development Guide](development/README.md)** - Start here for complete development methodology
- **[Module Development Guide](development/module-development-guide.md)** - Educational design and best practices
- **[Module Creation Checklist](development/module-creation-checklist.md)** - Step-by-step process
- **[Quick Reference](development/quick-module-reference.md)** - Commands and common patterns

### 🤖 **AI Assistants** (Development Support)
**Coding patterns and quality enforcement:**

- **`.cursor/rules/`** - Specific implementation patterns, code examples, and anti-patterns
- **Automatic guidance** during development through cursor rules

## 📊 **Documentation Architecture**

### **Clear Audience Separation**
```
docs/
├── students/           # 🎓 Course participants
│   └── project-guide.md    # Module progression and workflow
├── pedagogy/          # 👨‍🏫 Instructors and course designers  
│   ├── pedagogical-principles.md  # Educational theory
│   ├── vision.md              # Course philosophy
│   └── testing-architecture.md   # Assessment strategy
├── development/       # 🔧 Human module developers
│   ├── README.md              # Development methodology
│   ├── module-development-guide.md  # Complete guide
│   ├── module-creation-checklist.md # Step-by-step process
│   └── quick-module-reference.md   # Commands and patterns
└── README.md          # 📍 This file - navigation hub

.cursor/rules/         # 🤖 AI assistants (not in docs/)
├── module-development-best-practices.mdc  # Coding patterns
├── testing-patterns.mdc                   # Test requirements
└── nbdev-educational-pattern.mdc          # NBDev structure
```

### **No Duplication by Design**
- **Students**: Course navigation and module progression
- **Instructors**: Educational philosophy and theory
- **Human Developers**: Methodology, workflow, and educational design
- **AI Assistants**: Specific coding patterns and implementation examples

## 🚀 **Quick Navigation**

| I am a... | I want to... | Go to... |
|-----------|--------------|----------|
| **Student** | Start the course | [Project Guide](students/project-guide.md) |
| **Student** | Get unstuck on setup | [Quickstart](../quickstart.md) |
| **Instructor** | Understand the pedagogy | [Pedagogical Principles](pedagogy/pedagogical-principles.md) |
| **Instructor** | Design a new course | [Vision](pedagogy/vision.md) |
| **Developer** | Build a new module | [Development Guide](development/README.md) |
| **Developer** | Need quick help | [Quick Reference](development/quick-module-reference.md) |
| **Contributor** | Understand the philosophy | [Pedagogical Principles](pedagogy/pedagogical-principles.md) |

## 🔑 **Core Principles** (Across All Documentation)

### **Educational Excellence**
- **"Build → Use → Understand → Repeat"** - Every module follows this cycle
- **Real-world relevance** - Connect to production ML engineering
- **Immediate feedback** - Students see their code working
- **Progressive complexity** - Build understanding step by step

### **Real Data, Real Systems**
- Use production datasets (CIFAR-10, ImageNet), not synthetic data
- Include progress feedback for downloads and long operations  
- Test with realistic scales and performance constraints
- Think about caching, user experience, and systems concerns

---

*All documentation follows the TinyTorch principle: **Build → Use → Understand → Repeat*** 