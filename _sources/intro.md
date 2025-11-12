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

TinyTorch is an educational ML systems course where you **build complete neural networks from scratch**. Instead of blindly using PyTorch or TensorFlow as black boxes, you implement every component yourself—from tensors and gradients to optimizers and attention mechanisms—gaining deep understanding of how modern ML frameworks actually work.

**Core Learning Approach**: Build → Profile → Optimize. You'll implement each system component, measure its performance characteristics, and understand the engineering trade-offs that shape production ML systems.

## The Simple Workflow

TinyTorch follows a simple three-step cycle:

```
1. Edit modules → 2. Export to package → 3. Validate with milestones
```

**📖 See [Student Workflow](student-workflow.html)** for the complete development cycle, best practices, and troubleshooting.

## Three-Tier Learning Pathway

TinyTorch organizes 20 modules through three pedagogically-motivated tiers: **Foundation** (build mathematical infrastructure), **Architecture** (implement modern AI), and **Optimization** (deploy production systems).

**📖 See [Three-Tier Learning Structure](chapters/00-introduction.html#three-tier-learning-pathway-build-complete-ml-systems)** for detailed tier breakdown, module lists, time estimates, and learning outcomes.

## 🗺️ Understanding Your Complete Learning Journey

TinyTorch's 20 modules aren't arbitrary - they tell a carefully crafted story from building mathematical atoms to deploying production AI systems. Each module builds on previous foundations while setting up future capabilities.

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0;">

<div style="background: #f0f9ff; border: 1px solid #7dd3fc; padding: 1.5rem; border-radius: 0.5rem;">
<h4 style="margin: 0 0 1rem 0; color: #0284c7;">🏗️ Three-Tier Structure</h4>
<p style="margin: 0; font-size: 0.9rem;">Organized navigation through Foundation → Architecture → Optimization</p>
<p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;"><a href="chapters/00-introduction.html">View Course Structure →</a></p>
</div>

<div style="background: #fdf4ff; border: 1px solid #e9d5ff; padding: 1.5rem; border-radius: 0.5rem;">
<h4 style="margin: 0 0 1rem 0; color: #7c3aed;">📖 Six-Act Narrative</h4>
<p style="margin: 0; font-size: 0.9rem;">The learning story: Why modules flow from atomic components to intelligence</p>
<p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;"><a href="chapters/learning-journey.html">Read The Story →</a></p>
</div>

<div style="background: #fef3c7; border: 1px solid #fde047; padding: 1.5rem; border-radius: 0.5rem;">
<h4 style="margin: 0 0 1rem 0; color: #a16207;">🏆 Historical Milestones</h4>
<p style="margin: 0; font-size: 0.9rem;">Prove mastery by recreating ML history with YOUR implementations</p>
<p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;"><a href="chapters/milestones.html">See Timeline →</a></p>
</div>

</div>

**New to TinyTorch?** Start with the [Three-Tier Structure](chapters/00-introduction.html) to see what you'll build, then read [The Learning Journey](chapters/learning-journey.html) to understand the pedagogical progression that makes it all click.

## 🏆 Prove Your Mastery Through History

As you complete modules, unlock **historical milestone demonstrations** that prove what you've built works! Each milestone recreates a breakthrough using YOUR implementations—from Rosenblatt's 1957 perceptron to modern transformers and production optimization.

**📖 See [Journey Through ML History](chapters/milestones.html)** for complete timeline, requirements, and expected results.

## Why Build Instead of Use?

The difference between using a library and understanding a system is the difference between being limited by tools and being empowered to create them.

When you just use PyTorch or TensorFlow, you're stuck when things break—OOM errors, NaN losses, slow training. When you build TinyTorch from scratch, you understand exactly why these issues happen and how to fix them. You know the memory layouts, gradient flows, and performance bottlenecks because you implemented them yourself.

**📖 See [FAQ](faq.html)** for detailed comparisons with PyTorch, TensorFlow, micrograd, and nanoGPT, including code examples and architectural differences.

## Who Is This For?

**Perfect if you're asking these questions:**

**ML Systems Engineers**: "Why does my model training OOM at batch size 32? How do attention mechanisms scale quadratically with sequence length? When does data loading become the bottleneck?" You'll build and profile every component, understanding memory hierarchies, computational complexity, and system bottlenecks that production ML systems face daily.

**Students & Researchers**: "How does that `nn.Linear()` call actually compute gradients? Why does Adam optimizer need 3× the memory of SGD? What's actually happening during a forward pass?" You'll implement the mathematics you learned in class and discover how theoretical concepts become practical systems with real performance implications.

**Performance Engineers**: "Where are the actual bottlenecks in transformer inference? How does KV-cache reduce computation by 10-100×? Why does my CNN use 4GB of memory?" By building these systems from scratch, you'll understand memory access patterns, cache efficiency, and optimization opportunities that profilers alone can't teach.

**Academics & Educators**: "How can I teach ML systems—not just ML algorithms?" TinyTorch provides a complete pedagogical framework emphasizing systems thinking: memory profiling, performance analysis, and scaling behavior are built into every module, not added as an afterthought.

**ML Practitioners**: "Why does training slow down after epoch 10? How do I debug gradient explosions? When should I use mixed precision?" Even experienced engineers often treat frameworks as black boxes. By understanding the systems underneath, you'll debug faster, optimize better, and make informed architectural decisions.

## How to Choose Your Learning Path

**Three Learning Approaches**: You can **build complete tiers** (implement all 20 modules), **focus on specific tiers** (target your skill gaps), or **explore selectively** (study key concepts). Each tier builds complete, working systems.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 3rem 0;">

<!-- Top Row -->
<div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">🔬 Quick Start</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">15 minutes setup • Try foundational modules • Hands-on experience</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Start Building →</a>
</div>

<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">📚 Full Course</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">8+ weeks study • Complete ML framework • Systems understanding</p>
<a href="chapters/00-introduction.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Course Overview →</a>
</div>

<!-- Bottom Row -->
<div style="background: #faf5ff; border: 1px solid #b794f6; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">🎓 Instructors</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">Classroom-ready • NBGrader integration (coming soon)</p>
<a href="usage-paths/classroom-use.html" style="display: inline-block; background: #6f42c1; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Teaching Guide →</a>
</div>

<div style="background: #fff8dc; border: 1px solid #daa520; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">📊 Learning Community</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">Track progress • Join competitions • Student leaderboard</p>
<a href="leaderboard.html" style="display: inline-block; background: #b8860b; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">View Progress →</a>
</div>

</div>

## Getting Started

Ready to build ML systems from scratch? Here's how to start:

**Quick Setup** (15 minutes):
1. Clone the repository
2. Run `./setup-environment.sh`
3. Start with Module 01 (Tensors)
4. Export with `tito module complete 01`
5. Validate by running milestone scripts

**📖 See [Quick Start Guide](quickstart-guide.html)** for detailed setup instructions.

**Understanding the Workflow**:
- **📖 See [Student Workflow](student-workflow.html)** - The essential edit → export → validate cycle
- **📖 See [Essential Commands](tito-essentials.html)** - Complete TITO command reference
- **📖 See [Three-Tier Learning Structure](chapters/00-introduction.html)** - Detailed course structure

**Optional Progress Tracking**:
- **[Progress Tracking](learning-progress.html)** - Monitor your journey with capability checkpoints (optional)

TinyTorch is more than a course—it's a community of learners building together. Join thousands exploring ML systems from the ground up.