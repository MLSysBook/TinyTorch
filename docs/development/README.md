# 📚 TinyTorch Development Documentation

**Human-readable guides for module developers and contributors.**

*Note: AI assistants use `.cursor/rules/` for coding patterns. This documentation focuses on methodology, workflow, and educational design for human developers.*

## 🎯 **Quick Start** (New Human Developers)

1. **Read First**: [Module Development Guide](module-development-guide.md) - Complete methodology and philosophy
2. **Follow**: [Module Creation Checklist](module-creation-checklist.md) - Step-by-step process
3. **Reference**: [Quick Reference](quick-module-reference.md) - Commands and common patterns

## 📖 **Documentation Purpose**

### **Human Development Guides** (This Directory)
- **Why**: Educational methodology, design philosophy, workflow
- **How**: Step-by-step processes, quality standards, best practices
- **Context**: Real-world connections, systems thinking, pedagogical goals

### **AI Coding Rules** (`.cursor/rules/`)
- **What**: Specific coding patterns, implementation examples, anti-patterns
- **Enforcement**: Automatic guidance during development
- **Technical**: Code structure, testing patterns, NBDev directives

## 🔑 **Core Development Philosophy**

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

### **Development vs. Production**
- Rich visual feedback in development notebooks
- Clean, dependency-free exports to package
- Comprehensive testing with real data
- Clear separation of concerns

## 🛠️ **Development Workflow**

### **Create New Module**
```bash
# 1. Create structure
mkdir modules/{module}
mkdir modules/{module}/tests

# 2. Write complete implementation
# Follow: module-development-guide.md

# 3. Test and validate  
# Follow: module-creation-checklist.md
```

### **Quality Assurance**
All modules must:
- ✅ Use real data, not synthetic/mock data
- ✅ Include progress feedback for long operations
- ✅ Provide visual confirmation of working code
- ✅ Test with realistic data scales
- ✅ Follow "Build → Use → Understand" progression
- ✅ Include comprehensive TODO guidance
- ✅ Separate development richness from clean exports

## 📋 **Documentation Structure**

### **Core Development Process**
- **[Module Development Guide](module-development-guide.md)** - Complete methodology and best practices
- **[Module Creation Checklist](module-creation-checklist.md)** - Comprehensive step-by-step process
- **[Quick Reference](quick-module-reference.md)** - Commands, markers, and common patterns

### **Templates and Examples**
- **[Module Template](module-template.md)** - Reusable template snippets
- **[Module Template Files](module-template_files/)** - Complete file templates

## 🎓 **Educational Design Principles**

### **For Module Developers**
1. **Start with real data** - Choose production datasets first
2. **Design for immediate gratification** - Students see results quickly
3. **Build intuition before abstraction** - Concrete examples first
4. **Connect to production systems** - Real-world relevance always
5. **Provide rich feedback** - Visual confirmation, progress indicators
6. **Think systems-level** - Performance, caching, user experience

### **For Educational Content**
1. **Clear learning objectives** - What will students build and understand?
2. **Progressive complexity** - Easy → Medium → Hard with clear indicators
3. **Comprehensive guidance** - TODO sections with approach, examples, hints
4. **Real-world connections** - How does this relate to production ML?
5. **Immediate testing** - Students can verify each step
6. **Visual confirmation** - Students see their code working

## 🔄 **Relationship to Other Documentation**

### **Students** → [`docs/students/`](../students/)
- Project progression, module guides, getting started
- Course navigation and progress tracking

### **Instructors** → [`docs/pedagogy/`](../pedagogy/)
- Educational philosophy, learning theory
- Course design and assessment strategy

### **AI Assistants** → `.cursor/rules/`
- Specific coding patterns and implementation examples
- Automatic enforcement of quality standards

## 🎯 **Success Metrics**

**Human developers should be able to:**
- Understand the educational philosophy behind module design
- Create modules that follow TinyTorch principles
- Design learning experiences that build real-world skills
- Balance educational goals with production quality

**Modules should achieve:**
- High student engagement and completion rates
- Real-world relevance and production quality
- Smooth progression through the curriculum
- Clear connections to industry practices

---

**Remember**: We're teaching ML systems engineering, not just algorithms. Every module should reflect real-world practices while maintaining educational excellence. 