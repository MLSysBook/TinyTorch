# 📋 TinyTorch Module Creation Checklist

**Step-by-step checklist for creating high-quality educational modules following TinyTorch standards.**

*📖 For complete methodology, see [Module Development Guide](module-development-guide.md)*

## 🎯 Pre-Development

### Planning Phase
- [ ] Read [Module Development Guide](module-development-guide.md) 
- [ ] Define learning objectives: What will students implement?
- [ ] Choose production dataset: What real data will they use?
- [ ] Map to real-world: How does this connect to production ML?
- [ ] Design progression: Easy → Medium → Hard complexity

### Prerequisites Check
- [ ] Understand NBDev educational pattern
- [ ] Review existing modules (tensor, layers, activations) for patterns
- [ ] Identify module dependencies and integration points

## 📁 Module Structure Setup

### Required Files
- [ ] Create `modules/{module}/` directory
- [ ] Create `modules/{module}/tests/` directory
- [ ] Create `modules/{module}/{module}_dev.py` - Main implementation
- [ ] Create `modules/{module}/README.md` - Module guide
- [ ] Create `modules/{module}/tests/test_{module}.py` - Test suite

## 🔧 Implementation Phase

### File Header & Structure
- [ ] Add Jupytext header with correct format
- [ ] Add `#| default_exp core.{module}` directive
- [ ] Include module title and learning objectives in markdown
- [ ] Import required libraries with proper organization

### Educational Content
- [ ] **Step 1**: Concept explanation with real-world motivation
- [ ] **Step 2**: Core implementation with comprehensive TODO guidance
- [ ] **Step 3**: Advanced features and system integration
- [ ] **Step 4**: Testing with real data and visual feedback
- [ ] **Step 5**: Summary and connections to next modules

### Implementation Pattern (for each major component)
- [ ] **Student version**: Complete signature + comprehensive TODO
- [ ] **Hidden implementation**: Working code with `#| hide` + `#| export`
- [ ] **Immediate testing**: Real data tests after each component
- [ ] **Visual feedback**: Show results working (development only)

## 📝 Code Quality Standards

### Student-Facing Code
- [ ] Complete function signatures with type hints
- [ ] Comprehensive docstrings (Args, Returns, Raises)
- [ ] Detailed TODO with APPROACH, EXAMPLE, HINTS, SYSTEMS THINKING
- [ ] Clear `NotImplementedError` messages
- [ ] Real data examples, not synthetic

### Hidden Implementation
- [ ] Working, tested code that passes all tests
- [ ] Efficient implementation following best practices
- [ ] Proper error handling and edge cases
- [ ] Consistent with TinyTorch patterns

### Real Data Requirements
- [ ] Use production datasets (CIFAR-10, not mock data)
- [ ] Include progress bars for downloads/long operations
- [ ] Implement proper caching for repeated use
- [ ] Test with realistic data scales and timing

## 🧪 Testing Requirements

### Test Structure
- [ ] Import from `{module}_dev` (instructor version)
- [ ] Use real datasets, not mocks or synthetic data
- [ ] Test with actual data characteristics (shapes, types, ranges)
- [ ] Include performance tests with realistic constraints

### Test Coverage
- [ ] **Component Creation**: Object initialization with real data
- [ ] **Core Functionality**: Main methods work with production data
- [ ] **Integration**: Components work together in realistic scenarios
- [ ] **Edge Cases**: Error handling with actual edge cases
- [ ] **Performance**: Reasonable performance with real data scales

## 📖 Documentation Standards

### README Requirements
- [ ] Module title with clear learning objectives
- [ ] "What You'll Build" section with concrete examples
- [ ] Getting started steps (1-4)
- [ ] Real-world connections and applications
- [ ] Technical requirements and dependencies
- [ ] Success criteria and verification steps

### Quality Standards
- [ ] Clear, motivating language
- [ ] Specific examples using real data
- [ ] Systems thinking connections
- [ ] Appropriate difficulty progression
- [ ] Consistent formatting and style

## 🔄 Development Workflow

### Implementation Steps
- [ ] Write complete implementation first (get it working)
- [ ] Add NBDev markers and educational structure
- [ ] Create comprehensive tests with real data
- [ ] Add visual feedback and progress indicators
- [ ] Test both instructor and student paths

### Conversion & Export
- [ ] Convert to notebook: `python bin/tito.py notebooks --module {module}`
- [ ] Export to package: `python bin/tito.py sync --module {module}`
- [ ] Run tests: `python bin/tito.py test --module {module}`
- [ ] Verify imports: `from tinytorch.core.{module} import ClassName`

## ✅ Quality Assurance

### Before Release
- [ ] All automated tests pass with real data
- [ ] Student TODO guidance is comprehensive and helpful
- [ ] Visual feedback works (development only, not exported)
- [ ] Progress indicators for long operations
- [ ] Clean separation between development and exports
- [ ] Follows "Build → Use → Understand" progression

### Integration Testing
- [ ] Module exports correctly to `tinytorch.core.{module}`
- [ ] No circular import issues
- [ ] Compatible with existing modules
- [ ] Works with TinyTorch CLI tools
- [ ] Consistent with established patterns

### Student Experience
- [ ] Clear learning progression
- [ ] Immediate feedback and validation
- [ ] Real-world relevance and connections
- [ ] Motivating and engaging content
- [ ] Smooth transition to next modules

## 🎯 Final Verification

### Technical Excellence
- [ ] Code follows TinyTorch style guidelines
- [ ] Efficient implementation with proper error handling
- [ ] Real data usage throughout
- [ ] Performance optimized for student experience

### Educational Excellence
- [ ] Clear learning objectives achieved
- [ ] Progressive complexity with appropriate difficulty
- [ ] Real-world systems thinking integrated
- [ ] Immediate gratification and visual confirmation

### Documentation Complete
- [ ] README is comprehensive and motivating
- [ ] Code comments are clear and helpful
- [ ] TODO guidance includes systems thinking
- [ ] Integration with course progression documented

---

## 🚀 Ready to Release

Once all items are checked:
- [ ] Module follows all TinyTorch standards
- [ ] Students can successfully complete the module
- [ ] Real data is used throughout
- [ ] Visual feedback enhances learning
- [ ] Systems thinking is integrated
- [ ] Production relevance is clear

**Remember**: We're teaching ML systems engineering with real-world practices, not just algorithms! 