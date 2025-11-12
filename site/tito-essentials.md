# Essential TITO Commands

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Master the TinyTorch CLI in Minutes</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Everything you need to build ML systems efficiently</p>
</div>

**Purpose**: Complete command reference for the TITO CLI. Master the essential commands for development workflow, progress tracking, and system management.

## The Core Workflow

TinyTorch follows a simple three-step cycle: **Edit modules → Export to package → Validate with milestones**

**The essential command**: `tito module complete MODULE_NUMBER` - exports your code to the TinyTorch package.

See [Student Workflow](student-workflow.md) for the complete development cycle, best practices, and troubleshooting.

This page documents all available TITO commands. The checkpoint system (`tito checkpoint status`) is optional for progress tracking.

## Most Important Commands

The commands you'll use most often:

<div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">Check Your Environment</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito system doctor</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Verify your setup is ready for development</p>
</div>

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">Export Module to Package (Essential)</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module complete 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Export your module to the TinyTorch package - use this after editing modules</p>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h4 style="margin: 0 0 0.5rem 0; color: #15803d;">Track Your Progress (Optional)</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito checkpoint status</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">See which capabilities you've mastered (optional capability tracking)</p>
</div>

</div>

## Typical Development Session

Here's what a typical session looks like:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Edit modules:**
```bash
cd modules/source/03_layers
jupyter lab 03_layers_dev.py
# Make your implementation...
```

**Export to package:**
```bash
# From repository root
tito module complete 03
```

**Validate with milestones:**
```bash
cd milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Uses YOUR implementation!
```

**Optional progress tracking:**
```bash
tito checkpoint status  # See what you've completed
```

See [Student Workflow](student-workflow.md) for complete development cycle details.

</div>

## Complete Command Reference

### System & Health
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**System Check**
```bash
tito system doctor
```
*Diagnose environment issues before they block you*

**System Info**
```bash
tito system info
```
*View configuration details*

</div>

### Module Management
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Export Module to Package (Essential)**
```bash
tito module complete MODULE_NUMBER
```
*Export your implementation to the TinyTorch package - the key command in the workflow*

**Example:**
```bash
tito module complete 05  # Export Module 05 (Autograd)
```

After exporting, your code is importable:
```python
from tinytorch.autograd import backward  # YOUR implementation!
```

</div>

### Progress Tracking (Optional)
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Capability Overview**
```bash
tito checkpoint status
```
*Quick view of your capabilities (optional tracking)*

**Detailed Progress**
```bash
tito checkpoint status --detailed
```
*Module-by-module breakdown*

**Visual Timeline**
```bash
tito checkpoint timeline
```
*See your learning journey in visual format*

**Test Specific Capability**
```bash
tito checkpoint test CHECKPOINT_NUMBER
```
*Verify you've mastered a specific capability*

</div>

## Instructor Commands (Coming Soon)

<div style="background: #f3e5f5; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

TinyTorch includes NBGrader integration for classroom use. Full documentation for instructor workflows (assignment generation, autograding, etc.) will be available in future releases.

**For now, focus on the student workflow**: edit modules → export → validate with milestones.

*For current instructor capabilities, see [Classroom Use Guide](usage-paths/classroom-use.md)*

</div>

## Troubleshooting Commands

When things go wrong, these commands help:

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**Environment Issues:**
```bash
tito system doctor          # Diagnose problems
tito system info           # Show configuration details
```

**Progress Tracking (Optional):**
```bash
tito checkpoint status --detailed    # See exactly where you are
tito checkpoint timeline            # Visualize your progress
```

</div>

## Ready to Build?

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Start Your TinyTorch Journey</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Follow the 2-minute setup and begin building ML systems from scratch</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">2-Minute Setup →</a>
<a href="learning-progress.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Track Progress →</a>
</div>

---

*Master these commands and you'll build ML systems with confidence. Every command is designed to accelerate your learning and keep you focused on what matters: building production-quality ML frameworks from scratch.*
