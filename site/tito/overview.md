# TITO Command Reference

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Master the TinyTorch CLI</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Complete command reference for building ML systems efficiently</p>
</div>

**Purpose**: Quick reference for all TITO commands. Find the right command for every task in your ML systems engineering journey.

## Quick Start: Three Commands You Need

<div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">1. Check Your Environment</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito system doctor</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Verify your setup is ready for development</p>
</div>

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">2. Build & Export Modules</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module complete 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Export your module to the TinyTorch package</p>
</div>

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0;">
<h4 style="margin: 0 0 0.5rem 0; color: #7b1fa2;">3. Run Historical Milestones</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito milestone run 03</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Recreate ML history with YOUR code</p>
</div>

</div>

---

## Complete Command Reference

### System Commands

**Purpose**: Environment health and configuration

| Command | Description | Guide |
|---------|-------------|-------|
| `tito system doctor` | Diagnose environment issues | [Module Workflow](modules.md) |
| `tito system info` | Show system configuration | [Module Workflow](modules.md) |
| `tito system jupyter` | Start Jupyter Lab server | [Module Workflow](modules.md) |

### Module Commands

**Purpose**: Build-from-scratch workflow (your main development cycle)

| Command | Description | Guide |
|---------|-------------|-------|
| `tito module start XX` | Begin working on a module (first time) | [Module Workflow](modules.md) |
| `tito module resume XX` | Continue working on a module | [Module Workflow](modules.md) |
| `tito module complete XX` | Test, export, and track module completion | [Module Workflow](modules.md) |
| `tito module status` | View module completion progress | [Module Workflow](modules.md) |
| `tito module reset XX` | Reset module to clean state | [Module Workflow](modules.md) |

**See**: [Module Workflow Guide](modules.md) for complete details

### Milestone Commands

**Purpose**: Run historical ML recreations with YOUR implementations

| Command | Description | Guide |
|---------|-------------|-------|
| `tito milestone list` | Show all 6 historical milestones (1957-2018) | [Milestone System](milestones.md) |
| `tito milestone run XX` | Run milestone with prerequisite checking | [Milestone System](milestones.md) |
| `tito milestone info XX` | Get detailed milestone information | [Milestone System](milestones.md) |
| `tito milestone status` | View milestone progress and achievements | [Milestone System](milestones.md) |
| `tito milestone timeline` | Visual timeline of your journey | [Milestone System](milestones.md) |

**See**: [Milestone System Guide](milestones.md) for complete details

### Progress & Data Commands

**Purpose**: Track progress and manage user data

| Command | Description | Guide |
|---------|-------------|-------|
| `tito status` | View all progress (modules + milestones) | [Progress & Data](data.md) |
| `tito reset all` | Reset all progress and start fresh | [Progress & Data](data.md) |
| `tito reset progress` | Reset module completion only | [Progress & Data](data.md) |
| `tito reset milestones` | Reset milestone achievements only | [Progress & Data](data.md) |

**See**: [Progress & Data Management](data.md) for complete details

### Community Commands

**Purpose**: Join the global TinyTorch community and track your progress

| Command | Description | Guide |
|---------|-------------|-------|
| `tito community join` | Join the community (optional info) | [Community Guide](../community.html) |
| `tito community update` | Update your community profile | [Community Guide](../community.html) |
| `tito community profile` | View your community profile | [Community Guide](../community.html) |
| `tito community stats` | View community statistics | [Community Guide](../community.html) |
| `tito community leave` | Remove your community profile | [Community Guide](../community.html) |

**See**: [Community Guide](../community.html) for complete details

### Benchmark Commands

**Purpose**: Validate setup and measure performance

| Command | Description | Guide |
|---------|-------------|-------|
| `tito benchmark baseline` | Quick setup validation ("Hello World") | [Community Guide](../community.html) |
| `tito benchmark capstone` | Full Module 20 performance evaluation | [Community Guide](../community.html) |

**See**: [Community Guide](../community.html) for complete details

---

## Command Groups by Task

### First-Time Setup

```bash
# Clone and setup
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
./setup-environment.sh
source activate.sh

# Verify environment
tito system doctor
```

### Daily Development Workflow

```bash
# Start or continue a module
tito module start 01      # First time
tito module resume 01     # Continue later

# Export when complete
tito module complete 01

# Check progress
tito module status
```

### Achievement & Validation

```bash
# See available milestones
tito milestone list

# Get details
tito milestone info 03

# Run milestone
tito milestone run 03

# View achievements
tito milestone status
```

### Progress Management

```bash
# View all progress
tito status

# Reset if needed
tito reset all --backup
```

---

## Typical Session Flow

Here's what a typical TinyTorch session looks like:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**1. Start Session**
```bash
cd TinyTorch
source activate.sh
tito system doctor         # Verify environment
```

**2. Work on Module**
```bash
tito module start 03       # Or: tito module resume 03
# Edit in Jupyter Lab...
```

**3. Export & Test**
```bash
tito module complete 03
```

**4. Run Milestone (when prerequisites met)**
```bash
tito milestone list        # Check if ready
tito milestone run 03      # Run with YOUR code
```

**5. Track Progress**
```bash
tito status                # See everything
```

</div>

---

## Command Help

Every command has detailed help text:

```bash
# Top-level help
tito --help

# Command group help
tito module --help
tito milestone --help

# Specific command help
tito module complete --help
tito milestone run --help
```

---

## Detailed Guides

- **[Module Workflow](modules.md)** - Complete guide to building and exporting modules
- **[Milestone System](milestones.md)** - Running historical ML recreations
- **[Progress & Data](data.md)** - Managing your learning journey
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

## Related Resources

- **[Quick Start Guide](../quickstart-guide.md)** - 15-minute setup walkthrough
- **[Student Workflow](../student-workflow.md)** - Day-to-day development cycle
- **[Datasets Guide](../datasets.md)** - Understanding TinyTorch datasets

---

*Master these commands and you'll build ML systems with confidence. Every command is designed to accelerate your learning and keep you focused on what matters: building production-quality ML frameworks from scratch.*
