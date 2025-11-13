<!-- Updated main heading: Changed from "TinyTorch: Tensors to Systems" to "Build Your Own ML Framework" 
     for clearer value proposition and action-oriented messaging -->
<h1 style="text-align: center; font-size: 3rem; margin: 0rem 0 1rem 0; font-weight: 700;">
Build Your Own ML Framework
</h1>

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 2rem 0 1rem 0; font-weight: 700;">
Don't just import it. Build it.
</h2>

<!-- Enhanced description: Added "machine learning (ML)" clarification and "under the hood" 
     to emphasize deep understanding of framework internals -->
<p style="text-align: center; font-size: 1.2rem; margin: 0 auto 2rem auto; max-width: 800px; color: #374151;">
Build a complete machine learning (ML) framework from tensors to systems—understand how PyTorch, TensorFlow, and JAX really work under the hood.
</p>

<div style="text-align: center; margin: 2rem 0;">
  <a href="quickstart-guide" style="display: inline-block; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 0.875rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1rem; margin: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
    Start Building in 15 Minutes →
  </a>
</div>

## Welcome

Build ML systems understanding through progressive tiers—from mathematical foundations to competition-ready optimization.

<!-- Tier cards: Added emojis (🏗 🏛️ ⏱️ 🏅) to match _toc.yml for visual consistency 
     across navigation and content. Emojis provide quick visual recognition of tier categories. -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin: 1.5rem 0 2.5rem 0; max-width: 900px;">

<div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.75rem; border-radius: 0.5rem; border-left: 5px solid #1976d2;">
<h3 style="margin: 0 0 0.5rem 0; color: #0d47a1; font-size: 1.1rem; font-weight: 600;">🏗 Foundation (01-07)</h3>
<p style="margin: 0; color: #1565c0; font-size: 0.9rem; line-height: 1.5;">Tensors, autograd, training loops</p>
</div>

<div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.75rem; border-radius: 0.5rem; border-left: 5px solid #7b1fa2;">
<h3 style="margin: 0 0 0.5rem 0; color: #4a148c; font-size: 1.1rem; font-weight: 600;">🏛️ Architecture (08-13)</h3>
<p style="margin: 0; color: #6a1b9a; font-size: 0.9rem; line-height: 1.5;">Data loading, CNNs, transformers</p>
</div>

<div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.75rem; border-radius: 0.5rem; border-left: 5px solid #f57c00;">
<h3 style="margin: 0 0 0.5rem 0; color: #e65100; font-size: 1.1rem; font-weight: 600;">⏱️ Optimization (14-19)</h3>
<p style="margin: 0; color: #ef6c00; font-size: 0.9rem; line-height: 1.5;">Profiling, quantization, benchmarking</p>
</div>

<div style="background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); padding: 1.75rem; border-radius: 0.5rem; border-left: 5px solid #c2185b;">
<h3 style="margin: 0 0 0.5rem 0; color: #880e4f; font-size: 1.1rem; font-weight: 600;">🏅 Torch Olympics (20)</h3>
<p style="margin: 0; color: #ad1457; font-size: 0.9rem; line-height: 1.5;">Compete in ML systems challenges</p>
</div>

</div>

**[Complete course structure](chapters/00-introduction)** • **[Daily workflow guide](student-workflow)** • **[Join the community](community)**

## Validation Through Milestones

Validate your implementations with concrete benchmarks—MNIST accuracy, CIFAR-10 performance, transformer text generation. Each milestone proves your code works.

**[View milestone requirements](chapters/milestones)** to see the technical benchmarks you'll achieve.

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

**Systems Thinking**: TinyTorch emphasizes understanding how components interact—memory hierarchies, computational complexity, and optimization trade-offs—not just isolated algorithms. Every module connects mathematical theory to systems understanding.

**See [Course Philosophy](chapters/00-introduction)** for the full origin story and pedagogical approach.

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

## Essential Resources

**Core Documentation**:
- **[Quick Start Guide](quickstart-guide)** — 15-minute setup and first module
- **[Course Structure](chapters/00-introduction)** — Detailed tier breakdowns and learning outcomes
- **[Student Workflow](student-workflow.md)** — Day-to-day development cycle
- **[TITO Essentials](tito-essentials.md)** — Complete CLI command reference
- **[Historical Milestones](chapters/milestones.md)** — Prove your implementations through ML history

**Learning Support**:
- **[FAQ](faq.md)** — Comparisons with PyTorch, TensorFlow, micrograd
- **[Testing Framework](testing-framework.md)** — Quality assurance and validation
- **[Community](community.md)** — Connect with other builders
