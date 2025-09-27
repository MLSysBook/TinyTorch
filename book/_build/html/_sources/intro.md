# TinyTorch: Build ML Systems from Scratch

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 3rem 0;">
Don't just import it. Build it.
</h2>

## What is TinyTorch?

TinyTorch is an educational ML systems course where you **build complete neural networks from scratch**. Instead of blindly using PyTorch or TensorFlow as black boxes, you implement every component yourself—from tensors and gradients to optimizers and attention mechanisms—gaining deep understanding of how modern ML frameworks actually work.

**Core Learning Approach**: Build → Profile → Optimize. You'll implement each system component, measure its performance characteristics, and understand the engineering trade-offs that shape production ML systems.

## Why Build Instead of Use?

The difference between using a library and understanding a system is the difference between being limited by tools and being empowered to create them. When you build from scratch, you transform from a framework user into a systems engineer:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;">

<!-- Top Row: Using Libraries Examples -->
<div style="background: #fff5f5; border: 1px solid #feb2b2; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #c53030; font-size: 1.1rem;">❌ Using PyTorch</h3>

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(784, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Your model trains but then...
# 🔥 OOM error! Why?
# 🔥 Loss is NaN! How to debug?
# 🔥 Training is slow! What's the bottleneck?
```

<p style="color: #c53030; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
You're stuck when things break
</p>
</div>

<div style="background: #fff5f5; border: 1px solid #feb2b2; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #c53030; font-size: 1.1rem;">❌ Using TensorFlow</h3>

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Magic happens somewhere...
# 🤷 How are gradients computed?
# 🤷 Why this initialization?
# 🤷 What's happening in backward pass?
```

<p style="color: #c53030; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
Magic boxes you can't understand
</p>
</div>

<!-- Bottom Row: Building Your Own Examples -->
<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #2f855a; font-size: 1.1rem;">✅ Building TinyTorch</h3>

```python
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = randn(in_features, out_features) * 0.01
        self.bias = zeros(out_features)

    def forward(self, x):
        self.input = x  # Save for backward
        return x @ self.weight + self.bias

    def backward(self, grad):
        # You wrote this! You know exactly why:
        self.weight.grad = self.input.T @ grad
        self.bias.grad = grad.sum(axis=0)
        return grad @ self.weight.T
```

<p style="color: #2f855a; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
You can debug anything
</p>
</div>

<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #2f855a; font-size: 1.1rem;">✅ Building KV Cache</h3>

```python
class KVCache:
    def __init__(self, max_seq_len, n_heads, head_dim):
        # You understand EXACTLY the memory layout:
        self.k_cache = zeros(max_seq_len, n_heads, head_dim)
        self.v_cache = zeros(max_seq_len, n_heads, head_dim)
        # That's why GPT needs GBs of RAM!

    def update(self, k, v, pos):
        # You know why position matters:
        self.k_cache[pos:pos+len(k)] = k  # Reuse past computations
        self.v_cache[pos:pos+len(v)] = v  # O(n²) → O(n) speedup!
        # Now you understand why context windows are limited
```

<p style="color: #2f855a; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
You master modern LLM optimizations
</p>
</div>

</div>

## Who Is This For?

**Perfect if you're asking these questions:**

**Students & Researchers**: "How does that `nn.Linear()` call actually compute gradients? Why does Adam optimizer need three times the memory of my model parameters?" You'll implement the mathematics you learned in class and see how theoretical concepts become practical systems.

**Academics & Educators**: "How can I teach ML systems effectively?" TinyTorch provides a complete pedagogical framework with NBGrader integration, automated assessment, and rigorously tested foundations. Use it as a semester-long course or integrate individual modules.

**ML Practitioners**: "Why do I get stuck debugging training issues?" Even experienced engineers often treat frameworks as black boxes. By building your own, you'll debug faster, optimize better, and implement custom operations with confidence.

## How to Choose Your Learning Path

**Two Learning Approaches**: You can either **build it yourself** (work through student notebooks and implement from scratch) or **learn by reading** (study the solution notebooks to understand how ML systems work). Both approaches use the same **Build → Profile → Optimize** methodology at different scales.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 3rem 0;">

<!-- Top Row -->
<div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">🔬 Quick Start</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">15 minutes setup • Try foundational modules • Hands-on experience</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Start Building →</a>
</div>

<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">📚 Full Course</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">8+ weeks study • Complete ML framework • Systems mastery</p>
<a href="chapters/00-introduction.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Course Overview →</a>
</div>

<!-- Bottom Row -->
<div style="background: #faf5ff; border: 1px solid #b794f6; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">🎓 Instructors</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">Classroom-ready • NBGrader integration • Automated grading</p>
<a href="usage-paths/classroom-use.html" style="display: inline-block; background: #6f42c1; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Teaching Guide →</a>
</div>

<div style="background: #fff8dc; border: 1px solid #daa520; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">📊 Learning Community</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">Track progress • Join competitions • Student leaderboard</p>
<a href="leaderboard.html" style="display: inline-block; background: #b8860b; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">View Progress →</a>
</div>

</div>

## Ready to Start Building?

Transform from framework user to systems engineer. **📖 See [Essential Commands](tito-essentials.html)** for complete setup and command reference, or **📖 See [Complete Course Structure](chapters/00-introduction.html)** for detailed module descriptions.

**Additional Resources**:
- **[Progress Tracking](learning-progress.html)** - Monitor your learning journey with 16 capability checkpoints
- **[Testing Framework](testing-framework.html)** - Understand our comprehensive validation system
- **[Documentation & Guides](resources.html)** - Complete technical documentation and tutorials

TinyTorch is more than a course—it's a community of learners building together. Join thousands exploring ML systems from the ground up.