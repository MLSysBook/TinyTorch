# 📚 TinyTorch Documentation

**Complete documentation for the TinyTorch ML Systems course.**

## 🎯 **Quick Navigation**

### **For Students** 👨‍🎓
- **[Project Guide](students/project-guide.md)** - Complete course navigation and progress tracking
- **Start Here**: [`modules/setup/README.md`](../modules/setup/README.md) - First module setup

### **For Developers** 👨‍💻
- **[Development Guide](development/module-development-guide.md)** - Complete methodology and best practices
- **[Quick Reference](development/quick-module-reference.md)** - Commands and essential patterns
- **[Creation Checklist](development/module-creation-checklist.md)** - Step-by-step module creation

### **For Instructors** 👨‍🏫
- **[Pedagogical Principles](pedagogy/pedagogical-principles.md)** - Educational philosophy and learning theory
- **[Testing Architecture](pedagogy/testing-architecture.md)** - Assessment and verification strategy

## 📁 **Documentation Structure**

### **Development** (`development/`)
**For module developers and contributors**
- `module-development-guide.md` - Complete development methodology
- `quick-module-reference.md` - Fast reference for commands and patterns
- `module-creation-checklist.md` - Comprehensive step-by-step process
- `module-template.md` - Reusable template snippets

### **Students** (`students/`)
**For course participants**
- `project-guide.md` - Course navigation and module progression

### **Pedagogy** (`pedagogy/`)
**For instructors and educational design**
- `pedagogical-principles.md` - Educational philosophy and learning theory
- `testing-architecture.md` - Assessment strategy and testing patterns
- `vision.md` - Course vision and goals

## 🚀 **Quick Commands Reference**

### **System Commands**
```bash
tito system info              # System information and course navigation
tito system doctor            # Environment diagnosis
tito system jupyter           # Start Jupyter Lab
```

### **Module Development**
```bash
tito module status            # Check all module status
tito module test --module X   # Test specific module
tito module notebooks --module X  # Convert Python to notebook
```

### **Package Management**
```bash
tito package sync            # Export notebooks to package
tito package sync --module X # Export specific module
tito package reset           # Reset package to clean state
```

## 🎓 **Educational Philosophy**

TinyTorch follows a **"Build → Use → Understand → Repeat"** methodology where students:

1. **Build** - Implement core ML components from scratch
2. **Use** - Apply their implementations to real problems
3. **Understand** - Reflect on design decisions and trade-offs
4. **Repeat** - Apply learnings to increasingly complex systems

### **Key Principles**
- **Real Data, Real Systems** - Use production datasets and realistic constraints
- **Progressive Complexity** - Build understanding step by step
- **Systems Thinking** - Connect to production ML engineering practices
- **Immediate Feedback** - Students see their code working quickly

## 🛠️ **Development Workflow**

### **For New Modules**
1. **Plan** - Choose real datasets, define learning objectives
2. **Implement** - Write complete working version first
3. **Structure** - Add educational content and TODO guidance
4. **Test** - Comprehensive testing with real data
5. **Export** - Convert to notebooks and export to package

### **For Students**
1. **Setup** - Complete environment setup in `modules/setup/`
2. **Develop** - Work in `modules/{name}/{name}_dev.py` files
3. **Export** - Use `tito package sync` to build package
4. **Test** - Use `tito module test` to verify implementation
5. **Progress** - Use `tito module status` to track completion

## 📊 **Course Structure**

TinyTorch is organized into progressive modules:

- **Setup** - Development environment and workflow
- **Tensor** - Core data structures and operations
- **Layers** - Neural network building blocks
- **Networks** - Complete model architectures
- **Training** - Optimization and learning algorithms
- **Advanced** - Production systems and MLOps

Each module builds on previous ones, creating a complete ML systems engineering curriculum.

---

**💡 Pro Tip**: Start with the [Project Guide](students/project-guide.md) if you're a student, or the [Development Guide](development/module-development-guide.md) if you're creating modules. 