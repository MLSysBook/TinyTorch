# 🧪 Testing Framework

```{admonition} Test-Driven ML Engineering
:class: tip
TinyTorch's testing framework ensures your implementations are not just educational, but production-ready and reliable.
```

## 🎯 Testing Philosophy: Verify Understanding Through Implementation

TinyTorch testing goes beyond checking syntax - it validates that you understand ML systems engineering through working implementations.

## ⚡ Quick Start: Validate Your Implementation

### 🚀 Run Everything (Recommended)
```bash
# Complete validation suite
tito test --comprehensive

# Expected output:
# 🧪 Running 16 module tests...
# 🔗 Running integration tests...
# 📊 Running performance benchmarks...
# ✅ Overall TinyTorch Health: 100.0%
```

### 🎯 Target-Specific Testing
```bash
# Test what you just built
tito module complete 02_tensor && tito checkpoint test 01

# Quick module check
tito test --module attention --verbose

# Performance validation
tito test --performance --module training
```

## 🔬 Testing Levels: From Components to Systems

### 1. 🧩 Module-Level Testing
**Goal**: Verify individual components work correctly in isolation

```bash
# Test what you just implemented
tito test --module tensor --verbose
tito test --module attention --detailed

# Quick health check for specific module
tito module validate spatial

# Debug failing module
tito test --module autograd --debug
```

**What Gets Tested:**
- ✅ Core functionality (forward pass, backward pass)
- ✅ Memory usage patterns and leaks
- ✅ Mathematical correctness vs reference implementations
- ✅ Edge cases and error handling

### 2. 🔗 Integration Testing  
**Goal**: Ensure modules work together seamlessly

```bash
# Test module dependencies
tito test --integration --focus training

# Validate export/import chain
tito test --exports --all-modules

# Full pipeline validation
tito test --pipeline --from tensor --to training
```

**Integration Scenarios:**
- **Tensor → Autograd**: Gradient flow works correctly
- **Spatial → Training**: CNN training pipeline functions end-to-end
- **Attention → TinyGPT**: Transformer components integrate properly
- **All Modules**: Complete framework functionality

### 3. 🏆 Checkpoint Testing
**Goal**: Validate you've achieved specific learning capabilities

```bash
# Test your current capabilities
tito checkpoint test 01  # "Can I create and manipulate tensors?"
tito checkpoint test 08  # "Can I train neural networks end-to-end?"
tito checkpoint test 13  # "Can I build attention mechanisms?"

# Progressive capability validation
tito checkpoint validate --from 00 --to 15
```

**📖 See [Complete Checkpoint System Documentation](checkpoint-system.html)** for technical implementation details.

**Key Capability Categories:**
- **Foundation (00-03)**: Building blocks of neural networks
- **Training (04-08)**: End-to-end learning systems  
- **Architecture (09-14)**: Advanced model architectures
- **Optimization (15+)**: Production-ready systems

### 4. 📊 Performance & Systems Testing
**Goal**: Verify your implementation meets performance expectations

```bash
# Memory usage analysis
tito test --memory --module training --profile

# Speed benchmarking
tito test --speed --compare-baseline

# Scaling behavior validation
tito test --scaling --model-sizes 1M,5M,10M
```

**Performance Metrics:**
- **Memory efficiency**: Peak usage, gradient memory, batch scaling
- **Training speed**: Convergence time, throughput (samples/sec)
- **Inference latency**: Forward pass time, batch processing efficiency
- **Scaling behavior**: Performance vs model size, memory vs accuracy trade-offs

### 5. 🌍 Real-World Example Validation
**Goal**: Demonstrate production-ready functionality

```bash
# Train actual models
tito example train-mnist-mlp     # 95%+ accuracy target
tito example train-cifar-cnn     # 75%+ accuracy target  
tito example generate-text       # TinyGPT coherent generation

# Production scenarios
tito example benchmark-inference  # Speed/memory competitive analysis
tito example deploy-edge         # Resource-constrained deployment
```

## 🏗️ Test Architecture: Systems Engineering Approach

### 📋 Progressive Testing Pattern
Every TinyTorch module follows consistent testing standards:

```python
# Module testing template (every module follows this pattern)
class ModuleTest:
    def test_core_functionality(self):     # Basic operations work
    def test_mathematical_correctness(self): # Matches reference implementations  
    def test_memory_usage(self):          # No memory leaks, efficient usage
    def test_integration_ready(self):     # Exports correctly for other modules
    def test_real_world_usage(self):      # Works in actual ML pipelines
```

### 📁 Test Organization Structure
```bash
tests/
├── checkpoints/                    # 16 capability validation tests
│   ├── checkpoint_00_environment.py   # Development setup working
│   ├── checkpoint_01_foundation.py    # Tensor operations mastered
│   └── checkpoint_15_capstone.py      # Complete ML systems expertise
├── integration/                    # Cross-module compatibility
│   ├── test_training_pipeline.py      # End-to-end training works
│   └── test_module_exports.py         # All modules export correctly  
├── performance/                    # Systems performance validation
│   ├── memory_profiling.py           # Memory usage analysis
│   └── speed_benchmarks.py           # Computational performance
└── examples/                      # Real-world usage validation
    ├── test_mnist_training.py         # Actual MNIST training works
    └── test_cifar_cnn.py             # CNN achieves 75%+ on CIFAR-10
```

## 📊 Understanding Test Results

### 🎯 Health Status Interpretation
| Score | Status | Action Required |
|-------|--------|----------------|
| **100%** | 🟢 Excellent | All systems operational, ready for production |
| **95-99%** | 🟡 Good | Minor issues, investigate warnings |
| **90-94%** | 🟠 Caution | Some failing tests, address specific modules |
| **<90%** | 🔴 Issues | Significant problems, requires immediate attention |

### 🚦 Module Status Indicators
- ✅ **Passing**: Module implemented correctly, all tests green
- ⚠️ **Warning**: Minor issues detected, functionality mostly intact  
- ❌ **Failing**: Critical errors, module needs debugging
- 🚧 **In Progress**: Module under development, tests expected to fail
- 🎯 **Checkpoint Ready**: Module ready for capability testing

## 💡 Best Practices: Test-Driven ML Engineering

### 🔄 During Active Development
```bash
# Continuous validation workflow
tito test --module tensor         # After implementing core functionality
tito test --integration tensor    # After module completion  
tito checkpoint test 01          # After achieving milestone
```

**Development Testing Pattern:**
1. **Write minimal test first**: Define expected behavior before implementation
2. **Test each component**: Validate individual functions as you build them
3. **Integration early**: Test module interactions frequently, not just at the end
4. **Performance check**: Monitor memory and speed throughout development

### ✅ Before Code Commits
```bash
# Pre-commit validation checklist
tito test --comprehensive        # Full test suite passes
tito system doctor              # Environment is healthy
tito checkpoint status          # All achieved capabilities still work
```

**Commit Readiness Criteria:**
- ✅ All tests pass (100% health status)
- ✅ No memory leaks detected in performance tests
- ✅ Integration tests confirm module exports work
- ✅ Checkpoint tests validate learning objectives met

### 🎯 Before Module Completion
```bash
# Module completion validation
tito test --module mymodule --comprehensive
tito test --integration --focus mymodule  
tito module validate mymodule
tito module complete mymodule    # Only after all tests pass
```

## 🔧 Troubleshooting Guide

### 🚨 Common Test Failures & Solutions

#### Module Import Errors
```bash
# Problem: Module won't import
❌ ModuleNotFoundError: No module named 'tinytorch.core.tensor'

# Solution: Check module export
tito module complete tensor      # Ensure module is properly exported
tito system doctor             # Verify Python path and virtual environment
```

#### Mathematical Correctness Failures  
```bash
# Problem: Your implementation doesn't match reference
❌ AssertionError: Expected 0.5, got 0.48 (tolerance: 0.01)

# Debug process:
tito test --module tensor --debug          # Get detailed failure info
python -c "import tinytorch; help(tinytorch.tensor)"  # Check implementation
```

#### Memory Usage Issues
```bash
# Problem: Memory tests failing
❌ Memory usage: 150MB (expected: <100MB)

# Investigation:
tito test --memory --profile tensor       # Get memory profile
tito test --scaling --module tensor       # Check scaling behavior
```

#### Integration Test Failures
```bash
# Problem: Modules don't work together
❌ Integration test: tensor→autograd failed

# Debugging approach:
tito test --integration --focus autograd --verbose
tito test --exports tensor                # Check tensor exports correctly
tito test --imports autograd             # Check autograd imports correctly
```

### 🔍 Advanced Debugging Techniques

#### Verbose Test Output
```bash
# Get detailed test information
tito test --module attention --verbose --debug

# See exact error locations
tito test --traceback --module training
```

#### Performance Profiling
```bash
# Memory usage analysis
tito test --memory --profile --module spatial

# Speed profiling  
tito test --speed --profile --module training --iterations 100
```

#### Environment Validation
```bash
# Complete environment check
tito system doctor --comprehensive

# Specific dependency verification
tito system check-dependencies --module autograd
```

### 📋 Test Failure Decision Tree

```
Test Failed?
├── Import Error?
│   ├── Run `tito system doctor`
│   └── Check virtual environment activation
├── Mathematical Error?
│   ├── Compare with reference implementation
│   └── Check tensor shapes and dtypes
├── Memory Error? 
│   ├── Profile memory usage patterns
│   └── Check for memory leaks in loops
├── Integration Error?
│   ├── Test modules individually first
│   └── Verify export/import chain
└── Performance Error?
    ├── Profile bottlenecks
    └── Check algorithmic complexity
```

## 🎯 Testing Philosophy: Building Reliable ML Systems

The TinyTorch testing framework embodies professional ML engineering principles:

### 🧩 KISS Principle in Testing
- **Consistent patterns**: Every module follows identical testing structure - learn once, apply everywhere
- **Actionable feedback**: Tests provide specific error messages with exact fix suggestions
- **Essential focus**: Tests validate critical functionality without unnecessary complexity

### 🔗 Systems Engineering Mindset
- **Integration-first**: Tests verify components work together, not just in isolation
- **Real-world validation**: Examples prove your code works on actual datasets (CIFAR-10, MNIST)
- **Performance consciousness**: All tests include memory and speed awareness

### 📚 Educational Excellence
- **Understanding verification**: Tests confirm you grasp concepts, not just syntax
- **Progressive mastery**: Capabilities build systematically through checkpoint validation
- **Immediate feedback**: Know instantly if your implementation meets professional standards

### 🚀 Production Readiness
- **Professional standards**: Tests match industry-level validation practices
- **Scalability validation**: Ensure your code works at realistic data sizes
- **Reliability assurance**: Comprehensive testing prevents production failures

---

## 🏆 Success Metrics

```{admonition} Testing Success
:class: tip
A well-tested TinyTorch implementation should achieve:
- **100% test suite passing** - All functionality works correctly
- **>95% memory efficiency** - Comparable to reference implementations  
- **Real dataset success** - MNIST 95%+, CIFAR-10 75%+ accuracy targets
- **Clean integration** - All modules work together seamlessly
```

**Remember**: TinyTorch testing doesn't just verify your code works - it confirms you understand ML systems engineering well enough to build production-ready implementations. 

Your testing discipline here translates directly to building reliable ML systems in industry settings!

## 🚀 Next Steps

**Ready to start testing your implementations?**

```bash
# Begin with comprehensive health check
tito test --comprehensive

# Start building and testing your first module
tito module complete 01_setup

# Track your testing progress
tito checkpoint status
```

**Testing Integration with Your Learning Path:**
- **📖 See [Track Your Progress](learning-progress.html)** for how testing fits into capability development
- **📖 See [Track Capabilities](checkpoint-system.html)** for automated testing and progress validation
- **📖 See [Showcase Achievements](leaderboard.html)** for how testing validates the skills you can claim

<div style="background: #e3f2fd; border: 2px solid #1976d2; padding: 1.5rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h4 style="margin: 0 0 1rem 0; color: #1565c0;">🎯 Testing Excellence = ML Systems Mastery</h4>
<p style="margin: 0; color: #1976d2;">Every test you write and run builds the discipline needed for production ML engineering</p>
</div>