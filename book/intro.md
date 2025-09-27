# TinyTorch: Build ML Systems from Scratch

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 3rem 0;">
Don't just import it. Build it.
</h2>

## What is TinyTorch?

TinyTorch is an educational ML systems course where you build complete neural networks from scratch. Instead of using PyTorch or TensorFlow as black boxes, you implement every component yourself and gain deep understanding of how ML frameworks actually work.

<div style="text-align: center; margin: 2rem 0;">
<div style="background: linear-gradient(90deg, #E74C3C, #E67E22, #F39C12, #E67E22, #E74C3C); height: 2px; max-width: 400px; margin: 0 auto; opacity: 0.7;"></div>
</div>

## Why Build Instead of Use?

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;">

<!-- Top Row: Using Libraries Examples -->
<div style="background: #fff5f5; border: 1px solid #feb2b2; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #c53030; font-size: 1.1rem;">❌ PyTorch</h3>

```python
import torch.nn as nn

model = nn.Linear(784, 10)
# How does this work internally?
# Where are the weights stored?
```

<p style="color: #c53030; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
Black box implementation
</p>
</div>

<div style="background: #fff5f5; border: 1px solid #feb2b2; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #c53030; font-size: 1.1rem;">❌ TensorFlow</h3>

```python
import tensorflow as tf

layer = tf.keras.layers.Dense(10)
# Why does Adam use 3x memory?
# What's the actual math?
```

<p style="color: #c53030; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
Limited systems understanding
</p>
</div>

<!-- Bottom Row: Building Your Own Examples -->
<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #2f855a; font-size: 1.1rem;">✅ TinyTorch Linear</h3>

```python
class Linear:
    def forward(self, x):
        return x @ self.weight + self.bias
        # You understand exactly what happens
```

<p style="color: #2f855a; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
Complete implementation knowledge
</p>
</div>

<div style="background: #f0fff4; border: 1px solid #9ae6b4; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h3 style="margin: 0 0 1rem 0; color: #2f855a; font-size: 1.1rem;">✅ TinyTorch Optimizer</h3>

```python
class Adam:
    def step(self):
        # You know why it needs 3x memory
        # You can optimize performance
```

<p style="color: #2f855a; font-weight: 500; margin-top: 1rem; font-size: 0.9rem;">
Deep systems understanding
</p>
</div>

</div>

<div style="text-align: center; margin: 2rem 0;">
<div style="background: linear-gradient(90deg, #E74C3C, #E67E22, #F39C12, #E67E22, #E74C3C); height: 2px; max-width: 400px; margin: 0 auto; opacity: 0.7;"></div>
</div>

## 🚀 Start Building in 15 Minutes

<div style="text-align: center; margin: 3rem 0;">
<p style="font-size: 1.2rem; margin-bottom: 1.5rem; color: #2d3748;">
Ready to understand ML frameworks from the ground up? Get hands-on experience with tensor operations, neural networks, and autograd in minutes.
</p>
<a href="quickstart-guide.html" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1.2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 0.5rem;">
🚀 Quick Start Guide
</a>
<p style="font-size: 0.9rem; color: #718096; margin: 0;">
No installation required • Browser-based exploration • Real neural network training
</p>
</div>

<div style="text-align: center; margin: 2rem 0;">
<div style="background: linear-gradient(90deg, #E74C3C, #E67E22, #F39C12, #E67E22, #E74C3C); height: 2px; max-width: 400px; margin: 0 auto; opacity: 0.7;"></div>
</div>

## Three Clear Learning Paths

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 3rem 0;">

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

<div style="background: #faf5ff; border: 1px solid #b794f6; padding: 2rem; border-radius: 0.5rem; text-align: center;">
<h3 style="margin: 0 0 1rem 0; font-size: 1.2rem; color: #495057;">🎓 Instructors</h3>
<p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; color: #6c757d;">Classroom-ready • NBGrader integration • Automated grading</p>
<a href="usage-paths/classroom-use.html" style="display: inline-block; background: #6f42c1; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; font-size: 1rem;">Teaching Guide →</a>
</div>

</div>

<div style="text-align: center; margin: 2rem 0;">
<div style="background: linear-gradient(90deg, #E74C3C, #E67E22, #F39C12, #E67E22, #E74C3C); height: 2px; max-width: 400px; margin: 0 auto; opacity: 0.7;"></div>
</div>

## Learn More

<div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 0.5rem; margin: 3rem 0;">
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;">

<div>
<h4 style="margin: 0 0 1rem 0; color: #495057;">Course Details</h4>
<ul style="list-style: none; padding: 0; margin: 0;">
<li style="margin-bottom: 0.5rem;"><a href="chapters/00-introduction.html" style="color: #007bff; text-decoration: none;">Complete Course Overview</a></li>
<li style="margin-bottom: 0.5rem;"><a href="learning-progress.html" style="color: #007bff; text-decoration: none;">Progress Tracking</a></li>
<li style="margin-bottom: 0.5rem;"><a href="tito-essentials.html" style="color: #007bff; text-decoration: none;">Essential Commands</a></li>
</ul>
</div>

<div>
<h4 style="margin: 0 0 1rem 0; color: #495057;">Resources</h4>
<ul style="list-style: none; padding: 0; margin: 0;">
<li style="margin-bottom: 0.5rem;"><a href="resources.html" style="color: #007bff; text-decoration: none;">Documentation & Guides</a></li>
<li style="margin-bottom: 0.5rem;"><a href="testing-framework.html" style="color: #007bff; text-decoration: none;">Testing Framework</a></li>
<li style="margin-bottom: 0.5rem;"><a href="checkpoint-system.html" style="color: #007bff; text-decoration: none;">Checkpoint System</a></li>
</ul>
</div>

<div>
<h4 style="margin: 0 0 1rem 0; color: #495057;">Community</h4>
<ul style="list-style: none; padding: 0; margin: 0;">
<li style="margin-bottom: 0.5rem;"><a href="leaderboard.html" style="color: #007bff; text-decoration: none;">Student Leaderboard</a></li>
<li style="margin-bottom: 0.5rem;"><a href="competitions.html" style="color: #007bff; text-decoration: none;">Competitions</a></li>
<li style="margin-bottom: 0.5rem;"><a href="usage-paths/classroom-use.html" style="color: #007bff; text-decoration: none;">Teaching Resources</a></li>
</ul>
</div>

</div>
</div>