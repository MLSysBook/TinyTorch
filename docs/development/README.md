# 📚 TinyTorch Development Documentation

**Human-readable guides for module developers and contributors.**

*Note: AI assistants use `.cursor/rules/` for coding patterns. This documentation focuses on methodology, workflow, and educational design for human developers.*

## 🎯 **Current Architecture**

TinyTorch uses a **clean separation of concerns** architecture:

- **NBGrader**: Assignment creation and student workflow only
- **nbdev**: Package export and building only
- **Clear workflow**: NBGrader generates assignments → students complete them → module export builds package

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

### **Create New Assignment**
```bash
# 1. Create structure
mkdir assignments/source/{module}
mkdir assignments/source/{module}/tests

# 2. Write complete implementation with NBGrader solution delimiters
# assignments/source/{module}/{module}_dev.py

# 3. Test and validate
cd assignments/source/{module}
pytest tests/ -v
```

### **Student Workflow**
```bash
# 1. NBGrader: Generate assignments from source
tito nbgrader generate {module}

# 2. Students: Complete assignments
# Work in assignments/source/{module}/{module}.ipynb

# 3. nbdev: Export completed work to package
tito module export {module}

# 4. Test package integration
tito module test {module}
```

## 📋 **Documentation Structure**

### **Core Guides**
- **[Testing Guidelines](testing-guidelines.md)** - Testing standards and practices
- **[Template Files](module-template_files/)** - Complete file templates

### **Educational Philosophy**
- **[Pedagogy Directory](../pedagogy/)** - Learning theory and course design
- **[Vision](../pedagogy/vision.md)** - Overall educational philosophy

## 🎓 **Educational Design Principles**

### **For Assignment Developers**
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
- Understand the educational philosophy behind assignment design
- Create assignments that follow TinyTorch principles
- Design learning experiences that build real-world skills
- Balance educational goals with production quality

**Assignments should achieve:**
- High student engagement and completion rates
- Real-world relevance and production quality
- Smooth progression through the curriculum
- Clear connections to industry practices

---

**Remember**: We're teaching ML systems engineering, not just algorithms. Every assignment should reflect real-world practices while maintaining educational excellence. 