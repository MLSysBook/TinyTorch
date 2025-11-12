# TinyTorch: Build ML Systems from Scratch

<div style="background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); color: white; padding: 3rem 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">

<h2 style="color: white; font-size: 2.5rem; margin: 0 0 1rem 0; font-weight: 700;">
Don't just import it. Build it.
</h2>

<p style="font-size: 1.3rem; margin: 0 0 2rem 0; opacity: 0.95;">
Implement every component of a neural network framework yourself—from tensors to transformers to production optimization—and understand exactly how modern ML systems work.
</p>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin: 2rem 0; text-align: center;">
  <div>
    <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">95%+</div>
    <div style="opacity: 0.9;">MNIST Accuracy</div>
    <div style="font-size: 0.9rem; opacity: 0.75;">Your neural networks</div>
  </div>
  <div>
    <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">75%+</div>
    <div style="opacity: 0.9;">CIFAR-10 Accuracy</div>
    <div style="font-size: 0.9rem; opacity: 0.75;">Your CNNs</div>
  </div>
  <div>
    <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">100%</div>
    <div style="opacity: 0.9;">Your Code</div>
    <div style="font-size: 0.9rem; opacity: 0.75;">Every implementation</div>
  </div>
</div>

<div style="margin-top: 2rem;">
  <a href="quickstart-guide.html" style="display: inline-block; background: white; color: #1e3a8a; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1.1rem; margin: 0 0.5rem;">
    Start Building in 15 Minutes →
  </a>
  <a href="chapters/00-introduction.html" style="display: inline-block; background: transparent; color: white; border: 2px solid white; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1.1rem; margin: 0 0.5rem;">
    Learn More
  </a>
</div>

</div>

## Your Learning Journey

Build a complete ML systems framework through three progressive tiers—from mathematical foundations to production optimization—and prove your mastery through historically significant milestones.

```{mermaid}
graph TD
    subgraph Foundation["Foundation Tier (01-07)"]
        F["Build Mathematical Infrastructure<br/>Tensors → Autograd → Training<br/><br/>Achieve: 95%+ MNIST Accuracy"]
    end

    subgraph Architecture["Architecture Tier (08-13)"]
        A["Implement Modern AI<br/>DataLoader → CNNs → Transformers<br/><br/>Achieve: 75%+ CIFAR-10, Text Generation"]
    end

    subgraph Optimization["Optimization Tier (14-19)"]
        O["Deploy Production Systems<br/>Profile → Quantize → Benchmark<br/><br/>Achieve: Sub-100ms Inference"]
    end

    subgraph Capstone["Capstone (20)"]
        C["Complete Integration<br/>MLPerf Competition<br/><br/>Compete on Real Hardware"]
    end

    F --> A
    A --> O
    O --> C

    style Foundation fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Architecture fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Optimization fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style Capstone fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

**Browse complete module details in the sidebar navigation** — organized by tier with clear learning objectives and implementation guides.

**See [Complete Course Structure](chapters/00-introduction.html)** for detailed tier breakdowns, time estimates, and career connections.

## Why Build Instead of Use?

Understanding the difference between using a framework and building one is the difference between being limited by tools and being empowered to create them.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0;">

<div style="background: #fef2f2; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ef4444;">
<h3 style="margin: 0 0 1rem 0; color: #991b1b;">Traditional ML Education</h3>

```python
import torch
model = torch.nn.Linear(784, 10)
output = model(input)
# When this breaks, you're stuck
```

**Problem**: OOM errors, NaN losses, slow training—you can't debug what you don't understand.
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h3 style="margin: 0 0 1rem 0; color: #166534;">TinyTorch Approach</h3>

```python
from tinytorch import Linear  # YOUR code
model = Linear(784, 10)       # YOUR implementation
output = model(input)
# You know exactly how this works
```

**Advantage**: You understand memory layouts, gradient flows, and performance bottlenecks because you implemented them.
</div>

</div>

**Systems Thinking**: TinyTorch emphasizes understanding how components interact—memory hierarchies, computational complexity, and optimization trade-offs—not just isolated algorithms. Every module connects mathematical theory to production reality.

**See [Course Philosophy](chapters/00-introduction.html)** for the full origin story and pedagogical approach.

## The Build → Use → Reflect Approach

Every module follows a proven learning cycle that builds deep understanding:

```{mermaid}
graph LR
    B[Build<br/>Implement from scratch] --> U[Use<br/>Real data, real problems]
    U --> R[Reflect<br/>Systems thinking questions]
    R --> B

    style B fill:#FFC107,color:#000
    style U fill:#4CAF50,color:#fff
    style R fill:#2196F3,color:#fff
```

1. **Build**: Implement each component yourself—tensors, autograd, optimizers, attention
2. **Use**: Apply your implementations to real problems—MNIST, CIFAR-10, text generation
3. **Reflect**: Answer systems thinking questions—memory usage, scaling behavior, trade-offs

This approach develops not just coding ability, but systems engineering intuition essential for production ML.

## Is This For You?

**Perfect if you want to**:
- Debug ML systems when frameworks fail (OOM errors, gradient explosions, performance bottlenecks)
- Implement custom operations for research or production
- Understand how PyTorch, TensorFlow, and JAX actually work under the hood
- Transition from ML user to ML systems engineer

**Prerequisites**: Python programming and basic linear algebra (matrix multiplication). No prior ML framework experience required—you'll build your own.

### Start Your Path

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0;">

<div style="background: #eff6ff; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #3b82f6;">
<h4 style="margin: 0 0 1rem 0; color: #1e40af;">Quick Start</h4>
<p style="margin: 0 0 1rem 0; font-size: 0.95rem;">Get hands-on immediately</p>
<a href="quickstart-guide.html" style="color: #1e40af; font-weight: 600; text-decoration: none;">15-Minute Setup →</a>
</div>

<div style="background: #faf5ff; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #9333ea;">
<h4 style="margin: 0 0 1rem 0; color: #6b21a8;">Learn More</h4>
<p style="margin: 0 0 1rem 0; font-size: 0.95rem;">Understand the approach</p>
<a href="chapters/00-introduction.html" style="color: #6b21a8; font-weight: 600; text-decoration: none;">Course Philosophy →</a>
</div>

<div style="background: #f0fdfa; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #14b8a6;">
<h4 style="margin: 0 0 1rem 0; color: #115e59;">For Instructors</h4>
<p style="margin: 0 0 1rem 0; font-size: 0.95rem;">Classroom integration</p>
<a href="usage-paths/classroom-use.html" style="color: #115e59; font-weight: 600; text-decoration: none;">Teaching Guide →</a>
</div>

</div>

## Essential Resources

**Core Documentation**:
- **[Quick Start Guide](quickstart-guide.html)** — 15-minute setup and first module
- **[Course Structure](chapters/00-introduction.html)** — Detailed tier breakdowns and learning outcomes
- **[Student Workflow](student-workflow.md)** — Day-to-day development cycle
- **[TITO Essentials](tito-essentials.md)** — Complete CLI command reference
- **[Historical Milestones](chapters/milestones.md)** — Prove your implementations through ML history

**Learning Support**:
- **[FAQ](faq.md)** — Comparisons with PyTorch, TensorFlow, micrograd
- **[Testing Framework](testing-framework.md)** — Quality assurance and validation
- **[Community](community.md)** — Connect with other builders

---

**Ready to build?** Start with the [Quick Start Guide](quickstart-guide.html) to go from zero to building neural networks in 15 minutes.

**Want context first?** Read the [Course Introduction](chapters/00-introduction.html) to understand the origin story, philosophy, and complete learning progression.

**Teaching a course?** Review the [Instructor Guide](usage-paths/classroom-use.html) for classroom integration, automated grading, and curriculum planning.
