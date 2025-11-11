# Essential TITO Commands

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Master the TinyTorch CLI in Minutes</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Everything you need to build ML systems efficiently</p>
</div>

**Purpose**: Complete command reference for the TITO CLI. Master the essential commands for development workflow, progress tracking, and system management.

## 🚀 First 4 Commands (Start Here)

Every TinyTorch journey begins with these essential commands:

<div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">📋 Check Your Environment</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito system doctor</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Verify your setup is ready for development</p>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h4 style="margin: 0 0 0.5rem 0; color: #15803d;">🎯 Track Your Progress</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito checkpoint status</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">See which capabilities you've mastered</p>
</div>

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">🔨 Work on a Module</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module work 02_tensor</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Open and start building tensor operations</p>
</div>

<div style="background: #fdf2f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ec4899;">
<h4 style="margin: 0 0 0.5rem 0; color: #be185d;">✅ Complete Your Work</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module complete 02_tensor</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Export your code and test your capabilities</p>
</div>

</div>

## 🔄 Your Daily Learning Workflow

Follow this proven pattern for effective learning:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Morning Start:**
```bash
# 1. Check environment
tito system doctor

# 2. See your progress  
tito checkpoint status

# 3. Start working on next module
tito module work 03_activations
```

**During Development:**
```bash
# Test your understanding anytime
tito checkpoint test 02

# View your learning timeline
tito checkpoint timeline
```

**End of Session:**
```bash
# Complete and export your work
tito module complete 03_activations

# Celebrate your progress!
tito checkpoint status
```

</div>

## 💪 Most Important Commands (Top 10)

Master these commands for maximum efficiency:

### 🏥 System & Health
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**System Check**
```bash
tito system doctor
```
*Diagnose environment issues before they block you*

**Module Status**  
```bash
tito module status
```
*See all available modules and your completion status*

</div>

### 📊 Progress Tracking  
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Capability Overview**
```bash
tito checkpoint status
```
*Quick view of your 16 core capabilities*

**Detailed Progress**
```bash
tito checkpoint status --detailed
```
*Module-by-module breakdown with test status*

**Visual Timeline**
```bash
tito checkpoint timeline
```
*See your learning journey in beautiful visual format*

</div>

### 🔨 Module Development
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Start Working**
```bash
tito module work 05_dense
```
*Open module and start building*

**Export to Package**
```bash
tito module complete 05_dense
```
*Export your code to the TinyTorch package + run capability test*

**Quick Export (No Test)**
```bash
tito module export 05_dense
```
*Export without running capability tests*

</div>

### 🧪 Testing & Validation
<div style="background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Test Specific Capability**
```bash
tito checkpoint test 03
```
*Verify you've mastered a specific capability*

**Run Checkpoint with Details**
```bash
tito checkpoint run 03 --verbose
```
*See detailed output of capability validation*

</div>

## 🎓 Learning Stages & Commands

### Stage 1: Foundation (Modules 1-4)
**Key Commands:**
- `tito module work 01_setup` → `tito module complete 01_setup`
- `tito checkpoint test 00` (Environment)
- `tito checkpoint test 01` (Foundation)

### Stage 2: Core Learning (Modules 5-8)  
**Key Commands:**
- `tito checkpoint status` (Track your capabilities)
- `tito checkpoint timeline` (Visual progress)
- Complete modules 5-8 systematically

### Stage 3: Advanced Systems (Modules 9+)
**Key Commands:**
- `tito checkpoint timeline --horizontal` (Linear view)
- Focus on systems optimization modules
- Use `tito checkpoint test XX` for validation

## 👩‍🏫 Instructor Commands (NBGrader)

For instructors managing the course:

<div style="background: #f3e5f5; padding: 1rem; border-radius: 0.25rem; margin: 1rem 0;">

**Setup Course:**
```bash
tito nbgrader init              # Initialize NBGrader environment
tito nbgrader status            # Check assignment status
```

**Manage Assignments:**
```bash
tito nbgrader generate 01_setup  # Create assignment from module
tito nbgrader release 01_setup   # Release to students
tito nbgrader collect 01_setup   # Collect submissions
tito nbgrader autograde 01_setup # Automatic grading
```

**Reports & Export:**
```bash
tito nbgrader report            # Generate grade report
tito nbgrader export            # Export grades to CSV
```

*For detailed instructor workflow, see [Instructor Guide](usage-paths/classroom-use.html)*

</div>

## 🚨 Troubleshooting Commands

When things go wrong, these commands help:

<div style="background: #fff5f5; padding: 1.5rem; border: 1px solid #fed7d7; border-radius: 0.5rem; margin: 1rem 0;">

**Environment Issues:**
```bash
tito system doctor          # Diagnose problems
tito system info           # Show configuration details
```

**Module Problems:**
```bash
tito module status         # Check what's available
tito module info 02_tensor # Get specific module details
```

**Progress Confusion:**
```bash
tito checkpoint status --detailed    # See exactly where you are
tito checkpoint timeline            # Visualize your progress
```

</div>

## 🎯 Pro Tips for Efficiency

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div style="background: #e6fffa; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #26d0ce;">
<h4 style="margin: 0 0 0.5rem 0; color: #0d9488;">🔥 Hot Tip</h4>
<p style="margin: 0; font-size: 0.9rem;">Use tab completion! Type `tito mod` + TAB to auto-complete commands</p>
</div>

<div style="background: #f0f9ff; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #3b82f6;">
<h4 style="margin: 0 0 0.5rem 0; color: #1d4ed8;">⚡ Speed Boost</h4>
<p style="margin: 0; font-size: 0.9rem;">Alias common commands: `alias ts='tito checkpoint status'`</p>
</div>

<div style="background: #fefce8; padding: 1rem; border-radius: 0.5rem; border-left: 3px solid #eab308;">
<h4 style="margin: 0 0 0.5rem 0; color: #a16207;">🎯 Focus</h4>
<p style="margin: 0; font-size: 0.9rem;">Always run `tito system doctor` first when starting a new session</p>
</div>

</div>

## 🚀 Ready to Build?

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Start Your TinyTorch Journey</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Follow the 2-minute setup and begin building ML systems from scratch</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">2-Minute Setup →</a>
<a href="learning-progress.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Track Progress →</a>
</div>

---

*Master these commands and you'll build ML systems with confidence. Every command is designed to accelerate your learning and keep you focused on what matters: building production-quality ML frameworks from scratch.*