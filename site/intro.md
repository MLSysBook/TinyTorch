<div id="wip-banner" class="wip-banner">
  <div class="wip-banner-content">
    <div class="wip-banner-title">
      <span class="icon">🚧</span>
      <span class="icon">⚠️</span>
      <span>Under Construction - Active Development</span>
      <span class="icon">🔨</span>
      <span class="icon">🚧</span>
    </div>
    <div class="wip-banner-description">
      TinyTorch is under active construction! We're building in public and sharing our progress for early feedback. Expect frequent updates, changes, and improvements as we develop the framework together with the community.
    </div>
    <button id="wip-banner-toggle" class="wip-banner-toggle" title="Collapse banner">
      <i class="fas fa-chevron-up"></i>
    </button>
    <button id="wip-banner-close" class="wip-banner-close" title="Dismiss banner">
      ×
    </button>
  </div>
</div>

# TinyTorch: Build ML Systems from Scratch

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 3rem 0;">
Don't just import it. Build it.
</h2>

## What is TinyTorch?

TinyTorch is an educational ML systems course where you **build complete neural networks from scratch**. Instead of using PyTorch or TensorFlow as black boxes, you implement every component yourself—from tensors and gradients to optimizers and attention mechanisms—gaining deep understanding of how modern ML frameworks actually work.

**Core Learning Approach**: Build → Profile → Optimize. You'll implement each system component, measure its performance characteristics, and understand the engineering trade-offs that shape production ML systems.

## Your Learning Journey

TinyTorch organizes 20 modules through three tiers: **Foundation** (build mathematical infrastructure), **Architecture** (implement modern AI), and **Optimization** (deploy production systems).

**Browse all modules in the sidebar navigation** — organized by tier with clear learning objectives, time estimates, and implementation guides for each module.

### Foundation Tier (Modules 01-07)
Build the mathematical infrastructure: tensors, activations, layers, losses, autograd, optimizers, and training loops. By the end, you'll train neural networks achieving 95%+ accuracy on MNIST using your own implementations.

### Architecture Tier (Modules 08-13)
Implement modern AI architectures: data loading, convolutions for vision, tokenization, embeddings, attention, and transformers for language. Achieve 75%+ accuracy on CIFAR-10 with CNNs and generate coherent text with transformers.

### Optimization Tier (Modules 14-19)
Deploy production systems: profiling, quantization, compression, memoization, acceleration, and benchmarking. Transform research models into production-ready systems.

### Capstone Competition (Module 20)
Apply all optimizations in the MLPerf® Edu Competition—a standardized benchmark where you optimize models and compete fairly across different hardware platforms.

## Getting Started

Ready to build ML systems from scratch? Here's your path:

**Quick Setup** (15 minutes):
1. Clone the repository: `git clone https://github.com/mlsysbook/TinyTorch.git`
2. Run setup: `./setup-environment.sh`
3. Activate environment: `source activate.sh`
4. Verify: `tito system doctor`

**Your First Module**:
1. Start with Module 01 (Tensor) in `modules/source/01_tensor/`
2. Implement the required functionality
3. Export: `tito module complete 01`
4. Validate: Run milestone scripts to prove your implementation works

See the [Quick Start Guide](quickstart-guide.md) for detailed setup instructions and the [Student Workflow](student-workflow.md) for the complete development cycle.

## The Simple Workflow

TinyTorch follows a simple three-step cycle:

```
1. Edit modules → 2. Export to package → 3. Validate with milestones
```

**Edit**: Work on module source files in `modules/source/XX_name/`  
**Export**: Run `tito module complete XX` to make your code importable  
**Validate**: Run milestone scripts to prove your implementations work

See [Student Workflow](student-workflow.md) for the complete development cycle, best practices, and troubleshooting.

## Why Build Instead of Use?

The difference between using a library and understanding a system is the difference between being limited by tools and being empowered to create them.

When you just use PyTorch or TensorFlow, you're stuck when things break—OOM errors, NaN losses, slow training. When you build TinyTorch from scratch, you understand exactly why these issues happen and how to fix them. You know the memory layouts, gradient flows, and performance bottlenecks because you implemented them yourself.

See [FAQ](faq.md) for detailed comparisons with PyTorch, TensorFlow, micrograd, and nanoGPT, including code examples and architectural differences.

## Who Is This For?

**Perfect if you're asking these questions:**

**ML Systems Engineers**: "Why does my model training OOM at batch size 32? How do attention mechanisms scale quadratically with sequence length? When does data loading become the bottleneck?" You'll build and profile every component, understanding memory hierarchies, computational complexity, and system bottlenecks that production ML systems face daily.

**Students & Researchers**: "How does that `nn.Linear()` call actually compute gradients? Why does Adam optimizer need 3× the memory of SGD? What's actually happening during a forward pass?" You'll implement the mathematics you learned in class and discover how theoretical concepts become practical systems with real performance implications.

**Performance Engineers**: "Where are the actual bottlenecks in transformer inference? How does KV-cache reduce computation by 10-100×? Why does my CNN use 4GB of memory?" By building these systems from scratch, you'll understand memory access patterns, cache efficiency, and optimization opportunities that profilers alone can't teach.

**Academics & Educators**: "How can I teach ML systems—not just ML algorithms?" TinyTorch provides a complete pedagogical framework emphasizing systems thinking: memory profiling, performance analysis, and scaling behavior are built into every module, not added as an afterthought.

**ML Practitioners**: "Why does training slow down after epoch 10? How do I debug gradient explosions? When should I use mixed precision?" Even experienced engineers often treat frameworks as black boxes. By understanding the systems underneath, you'll debug faster, optimize better, and make informed architectural decisions.

## Learning Paths

**Three Learning Approaches**: You can **build complete tiers** (implement all 20 modules), **focus on specific tiers** (target your skill gaps), or **explore selectively** (study key concepts). Each tier builds complete, working systems.

**Quick Exploration** (2-4 weeks): Focus on Foundation Tier (Modules 01-07) to understand core ML systems  
**Complete Course** (14-18 weeks): Implement all three tiers for complete ML systems mastery  
**Focused Learning** (4-8 weeks): Pick specific tiers based on your goals

## Prove Your Mastery Through History

As you complete modules, unlock **historical milestone demonstrations** that prove what you've built works. Each milestone recreates a breakthrough using YOUR implementations—from Rosenblatt's 1957 perceptron to modern transformers and production optimization.

See [Historical Milestones](chapters/milestones.md) for complete timeline, requirements, and expected results.

## Next Steps

- **New to TinyTorch**: Start with the [Quick Start Guide](quickstart-guide.md) for immediate hands-on experience
- **Ready to Commit**: Begin Module 01: Tensor (see sidebar navigation) to start building
- **Understand the Structure**: Read [Course Structure](chapters/00-introduction.md) for detailed tier breakdown and learning outcomes
- **Teaching a Course**: Review [Instructor Guide](usage-paths/classroom-use.html) for classroom integration

TinyTorch is more than a course—it's a community of learners building together. Join thousands exploring ML systems from the ground up.
