# TinyTorch Paper - Proposed Pedagogical Figures

**Status:** Ready for Review
**Generated:** 2025-11-17
**Coordinator:** Research Coordinator (Dr. Jennifer Martinez)

---

## Quick Summary

I've identified and created **4 high-value pedagogical figures** that would significantly enhance the TinyTorch paper by visualizing concepts currently explained only through prose. All TikZ code is production-ready.

---

## The 4 Proposed Figures

### 1. Progressive Disclosure Timeline ⭐ **HIGHEST PRIORITY**

**What it shows:** How Tensor class capabilities evolve from dormant (Module 01) to active (Module 05+)

**Why it matters:** This is your **most novel contribution** but currently lacks visual support. A timeline instantly shows:
- Which features are dormant vs active at each module
- The "aha moment" when autograd activates
- How cognitive load is managed through phased complexity

**Where:** Section 3.1, after Listing 2 (line ~660)

**Visual concept:**
```
Timeline with feature layers:
- Core (.data, .shape): Always active (orange line)
- Gradients (.requires_grad, .grad, .backward()): Dormant (dashed gray) → Active (solid orange) at M05
- Activation marker showing when monkey-patching happens
```

---

### 2. Memory Hierarchy Breakdown ⭐ **HIGH PRIORITY**

**What it shows:** Stacked bar comparison of SGD vs Adam memory components

**Why it matters:** Clarifies that while "Adam uses 3× parameter memory," activations actually dominate (10-100×). Students often misunderstand this.

**Where:** Section 4.1, after Table 1 (line ~744)

**Visual concept:**
```
Side-by-side stacked bars:
SGD: Parameters (1×) + Gradients (1×) + Activations (30×) = 32× total
Adam: Parameters (1×) + Gradients (1×) + Momentum (1×) + Variance (1×) + Activations (30×) = 34× total
Key insight: Optimizer adds 2×, but activations still dominate!
```

---

### 3. Build→Use→Reflect Cycle ⭐ **HIGH PRIORITY**

**What it shows:** The three-phase pedagogical cycle with concrete Module 05 examples

**Why it matters:** This pattern structures all 20 modules but is currently only prose. Visual makes the iterative cycle explicit.

**Where:** Section 2.3, Module Structure (line ~492)

**Visual concept:**
```
Circular diagram:
BUILD (Implementation) → USE (Integration Testing) → REFLECT (Systems Analysis) → (loop back)
Each phase shows concrete examples from Module 05 (Autograd)
Center: "Repeats for all 20 modules"
```

---

### 4. Historical Milestone Timeline ⭐ **MEDIUM PRIORITY**

**What it shows:** 70-year progression from 1957 Perceptron to 2024 Production systems

**Why it matters:** The historical narrative is compelling but currently just a list. Timeline shows capability accumulation visually.

**Where:** Section 4.3, Historical Validation (line ~773)

**Visual concept:**
```
Timeline: 1957 → 1969 → 1986 → 1998 → 2017 → 2024
Milestone boxes showing which modules unlock each achievement
Arrows showing capability accumulation
Color coding by tier (Foundation/Architecture/Optimization)
```

---

## Files Created

| File | Purpose |
|------|---------|
| `proposed_figures.tex` | Complete LaTeX document with all TikZ code - **compile this to see figures** |
| `FIGURE_PROPOSALS.md` | Detailed pedagogical rationale and integration instructions |
| `FIGURE_SUMMARY.txt` | ASCII mockups for quick visualization without compiling |
| `README_FIGURES.md` | This executive summary |

---

## How to Review

### Option 1: Compile and View (Recommended)
```bash
cd /Users/VJ/GitHub/TinyTorch/paper
lualatex proposed_figures.tex
open proposed_figures.pdf
```

### Option 2: Quick Preview
Read `FIGURE_SUMMARY.txt` for ASCII mockups of all 4 figures

### Option 3: Detailed Analysis
Read `FIGURE_PROPOSALS.md` for complete pedagogical rationale

---

## Integration Recommendations

### If adding only 1 figure:
**Choose:** Progressive Disclosure Timeline (Figure 1)
**Reason:** Most novel contribution, needs strongest visual support

### If adding 2 figures:
**Choose:** Progressive Disclosure + Memory Hierarchy (Figures 1 & 2)
**Reason:** Novel contribution + systems-first clarification

### If adding 3 figures:
**Choose:** Add Build→Use→Reflect Cycle (Figure 3)
**Reason:** Complete coverage of pedagogical patterns

### If adding all 4 figures:
**Include:** Milestone Timeline (Figure 4)
**Reason:** Complete historical narrative visually

---

## Integration Steps

1. **Review figures** (compile `proposed_figures.tex`)
2. **Choose which to include** (see recommendations above)
3. **Extract or copy TikZ code** into main paper:
   - Option A: Create `figures/fig_*.tex` files and `\input{}` them
   - Option B: Copy TikZ code directly into `paper.tex` at specified locations
4. **Compile paper** and verify placement
5. **Adjust captions** if needed to match paper voice

---

## Why These Figures Matter

### Current State
- Paper has excellent code examples and comprehensive tables
- Novel contributions (progressive disclosure, systems-first) explained only in prose
- Dependency diagram (Figure 2) is good but doesn't cover pedagogical patterns

### After Adding These Figures
- **Progressive Disclosure Timeline**: Readers instantly grasp your most novel contribution
- **Memory Hierarchy**: Systems-first pedagogy becomes concrete and memorable
- **Build-Use-Reflect Cycle**: Core pedagogical pattern gets proper visual treatment
- **Milestone Timeline**: Historical narrative becomes more compelling

### Expected Impact
- **Reviewer comprehension**: Significantly improved (visual > text for complex patterns)
- **Educator adoption**: Easier (figures show "how it works" at a glance)
- **Citation value**: Higher (memorable visuals make paper more citable)
- **Page count**: +2-3 pages (but high pedagogical value per page)

---

## Figures We Deliberately Avoided

**NOT included** because they don't add unique TinyTorch insight:
- Generic computation graphs (well-known in ML education)
- Standard architecture diagrams (MLP/CNN/Transformer - available everywhere)
- Complexity curves (standard CS material)
- Decorative module icons (Figure 2 dependency diagram sufficient)

All proposed figures focus on **TinyTorch's unique pedagogical contributions** rather than generic ML concepts.

---

## Next Steps

1. **Review** the compiled `proposed_figures.pdf` (or ASCII mockups)
2. **Decide** which figures to include based on priority
3. **Provide feedback** on any modifications needed
4. **Integrate** chosen figures into `paper.tex`

All TikZ code is production-ready and matches your paper's color scheme (orange accent, blue/green/purple tier coding).

---

## Questions?

The proposed figures are designed to:
- Clarify complex pedagogical patterns currently explained in prose
- Visualize the paper's most novel contributions
- Support educator understanding and adoption
- Make the paper more memorable and citable

If you'd like modifications to any figure (different layout, simplified/expanded detail, alternative visualization), the TikZ code is modular and easy to adjust.

**Ready for your review and feedback!**
