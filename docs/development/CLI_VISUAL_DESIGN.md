# TinyTorch CLI Visual Design Guidelines

> **Design Philosophy**: Professional, engaging, pedagogically sound. Every visual element should guide learning and celebrate progress.

## Core Design Principles

###

 1. **Progress Over Perfection**
Show students where they are in their journey, what they've accomplished, and what's next.

### 2. **Clear Visual Hierarchy**
- 🏆 Milestones (Epic achievements)
- ✅ Completed modules (Done!)
- 🚀 In Progress (Working on it)
- ⏳ Locked (Not yet available)
- 💡 Next Steps (What to do)

### 3. **Color Psychology**
- **Green**: Success, completion, ready states
- **Cyan/Blue**: Information, current state
- **Yellow**: Warnings, attention needed
- **Magenta/Purple**: Achievements, milestones
- **Dim**: Secondary information, hints

### 4. **Information Density**
- **Summary**: Quick glance (1-2 lines)
- **Overview**: Scannable (table format)
- **Details**: Deep dive (expandable panels)

---

## Command Visual Specifications

### `tito module status`

**Current Issues:**
- Text-heavy list format
- Hard to scan quickly
- Doesn't show progress visually

**New Design:**

```
╭─────────────────────── 📊 Your Learning Journey ────────────────────────╮
│                                                                          │
│  Progress: ████████████░░░░░░░░ 12/20 modules (60%)                    │
│  Streak: 🔥 5 days  •  Last activity: 2 hours ago                       │
│                                                                          │
╰──────────────────────────────────────────────────────────────────────────╯

┏━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ##  ┃ Module           ┃ Status     ┃ Next Action                ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 01 │ Tensor           │ ✅ Done    │ ─                          │
│ 02 │ Activations      │ ✅ Done    │ ─                          │
│ 03 │ Layers           │ 🚀 Working │ tito module complete 03    │
│ 04 │ Losses           │ ⏳ Locked  │ Complete module 03 first   │
│ 05 │ Autograd         │ ⏳ Locked  │ ─                          │
└────┴──────────────────┴────────────┴────────────────────────────┘

🏆 Milestones Unlocked: 2/6
  ✅ 01 - Perceptron (1957)
  ✅ 02 - XOR Crisis (1969)
  🎯 03 - MLP Revival (1986) [Ready when you complete module 07!]

💡 Next: tito module complete 03
```

### `tito milestone status`

**Current Issues:**
- Doesn't feel epic enough
- Missing visual timeline
- Hard to see what's unlocked vs locked

**New Design:**

```
╭─────────────────────── 🏆 Milestone Achievements ────────────────────────╮
│                                                                           │
│  You've unlocked 2 of 6 epic milestones in ML history!                   │
│  Next unlock: MLP Revival (1986) → Complete modules 01-07                │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯

                    Your Journey Through ML History

1957 ●━━━━━━━ 1969 ●━━━━━━━ 1986 ○━━━━━━━ 1998 ○━━━━━━━ 2017 ○━━━━━━━ 2024 ○
     ✅            ✅           🔒           🔒           🔒           🔒

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                          ┃
┃  ✅ 01 - Perceptron (1957)                                              ┃
┃  🧠 "I taught a computer to classify patterns!"                         ┃
┃  ───────────────────────────────────────────────────────────────        ┃
┃  Achievement: Built Rosenblatt's first trainable network                ┃
┃  Unlocked: 3 days ago                                                   ┃
┃                                                                          ┃
┃  ✅ 02 - XOR Crisis (1969)                                              ┃
┃  🔀 "I solved the problem that stalled AI research!"                    ┃
┃  ───────────────────────────────────────────────────────────────        ┃
┃  Achievement: Multi-layer networks with backprop                        ┃
┃  Unlocked: 2 days ago                                                   ┃
┃                                                                          ┃
┃  🎯 03 - MLP Revival (1986) [READY TO UNLOCK!]                          ┃
┃  🎓 "Train deep networks on real digits!"                               ┃
┃  ───────────────────────────────────────────────────────────────        ┃
┃  Requirements: Modules 01-07 ✅✅⏳⏳⏳⏳⏳                                │
┃  Next: tito module complete 03                                          ┃
┃                                                                          ┃
┃  🔒 04 - CNN Revolution (1998)                                           ┃
┃  👁️ "Computer vision with convolutional networks"                       ┃
┃  Requirements: Complete modules 01-09 first                              ┃
┃                                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

💡 Run a milestone: tito milestone run 01
```

### `tito system health`

**Current Issues:**
- Bland table format
- Doesn't prioritize critical issues
- Missing actionable fixes

**New Design:**

```
╭─────────────────────── 🔬 System Health Check ───────────────────────────╮
│                                                                           │
│  Overall Status: ✅ Healthy  •  Ready to build ML systems!               │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component              ┃ Status   ┃ Details                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 🐍 Python              │ ✅ 3.11.9 │ arm64 (Apple Silicon)       │
│ 📦 Virtual Environment │ ✅ Active │ /TinyTorch/.venv            │
│ 🔢 NumPy               │ ✅ 1.26.4 │ Core dependency             │
│ 🎨 Rich                │ ✅ 13.7.1 │ CLI framework               │
│ 🧪 Pytest              │ ✅ 8.0.0  │ Testing framework           │
│ 📓 Jupyter             │ ✅ 4.0.9  │ Interactive development     │
│ 📦 TinyTorch Package   │ ✅ 0.1.0  │ 12/20 modules exported      │
└────────────────────────┴──────────┴─────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Directory Structure    ┃ Status                                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ src/                   │ ✅ 20 module directories                 │
│ modules/               │ ✅ Ready for student work                │
│ tinytorch/             │ ✅ Package with 12 components            │
│ tests/                 │ ✅ 156 tests passing                     │
│ milestones/            │ ✅ 6 historical achievements ready       │
└────────────────────────┴──────────────────────────────────────────┘

🎉 All systems operational! Ready to start learning.

💡 Quick Start:
   tito module start 01    # Begin your journey
   tito module status      # Track your progress
```

### `tito module complete 01`

**Current Issues:**
- Minimal celebration
- Doesn't show what was accomplished
- Missing clear next steps

**New Design:**

```
╭─────────────────── 🎯 Completing Module 01: Tensor ──────────────────────╮
│                                                                           │
│  Running your tests, exporting your code, tracking your progress...      │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Step 1/3: Running Tests

   test_tensor_creation ......... ✅ PASS
   test_tensor_operations ........ ✅ PASS
   test_broadcasting ............. ✅ PASS
   test_reshape .................. ✅ PASS
   test_indexing ................. ✅ PASS

   ✅ All 5 tests passed in 0.42s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Step 2/3: Exporting to TinyTorch Package

   ✅ Exported: tinytorch/core/tensor.py (342 lines)
   ✅ Updated: tinytorch/__init__.py

   Your Tensor class is now part of the framework!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Step 3/3: Tracking Progress

   ✅ Module 01 marked complete
   📈 Progress: 1/20 modules (5%)
   🔥 Streak: 1 day

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╭───────────────────────── 🎉 Module Complete! ────────────────────────────╮
│                                                                           │
│  You didn't import Tensor. You BUILT it.                                 │
│                                                                           │
│  What you can do now:                                                    │
│    >>> from tinytorch import Tensor                                      │
│    >>> t = Tensor([1, 2, 3])                                             │
│    >>> t.reshape(3, 1)                                                   │
│                                                                           │
│  💡 Next: tito module start 02                                           │
│           Build activation functions (ReLU, Softmax)                     │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯
```

---

## Implementation Notes

### Rich Components to Use

1. **Tables**: Clean, scannable data
   - Use `rich.table.Table` with proper styling
   - Header styles: `bold blue` or `bold magenta`
   - Borders: `box.ROUNDED` or `box.SIMPLE`

2. **Panels**: Highlight important information
   - Success: `border_style="bright_green"`
   - Info: `border_style="bright_cyan"`
   - Achievements: `border_style="magenta"`
   - Warnings: `border_style="yellow"`

3. **Progress Bars**: Visual progress tracking
   - Use `rich.progress.Progress` for operations
   - Use ASCII bars (`████░░░`) for quick summaries

4. **Text Styling**:
   - Bold for emphasis: `[bold]text[/bold]`
   - Colors for status: `[green]✅[/green]`, `[yellow]⚠️[/yellow]`
   - Dim for secondary: `[dim]hint[/dim]`

### Emojis (Use Sparingly & Meaningfully)

- ✅ Success, completion
- 🚀 In progress, working
- ⏳ Locked, waiting
- 🏆 Milestones, achievements
- 💡 Tips, next steps
- 🔥 Streak, momentum
- 🎯 Goals, targets
- 📊 Statistics, data
- 🧪 Tests, validation
- 📦 Packages, exports

### Typography Hierarchy

1. **Title**: Large, bold, with emoji
2. **Section**: Bold with separator line
3. **Item**: Normal weight with status icon
4. **Detail**: Dim, smaller, indented
5. **Action**: Cyan/bold, stands out

---

## Testing Visual Output

Run these commands to see the new designs:
```bash
tito module status
tito milestone status
tito system health
tito module complete 01  # (after working on module 01)
```

Each should feel:
- **Professional**: Clean, organized, purposeful
- **Engaging**: Celebrates progress, shows growth
- **Pedagogical**: Guides learning, suggests next steps
- **Scannable**: Quick to understand at a glance
