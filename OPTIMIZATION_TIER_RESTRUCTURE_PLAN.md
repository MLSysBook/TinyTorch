# Optimization Tier Restructuring - Implementation Plan

## 🎯 Overview

**Branch:** `optimization-tier-restructure`  
**Goal:** Restructure Optimization Tier (Modules 14-19) with profiling-driven workflow

### Key Changes
1. Move Profiling from Module 15 → Module 14
2. Move KV Caching/Memoization from Module 14 → Module 15
3. Reorder subsequent optimization modules
4. Add "profiling intro" sections to each optimization module
5. Update all documentation, website, and CLI commands

---

## 📊 Current vs Target State

### Current Structure
```
Architecture Tier (08-14):
├─ 08. DataLoader
├─ 09. Spatial (CNNs)
├─ 10. Tokenization
├─ 11. Embeddings
├─ 12. Attention
├─ 13. Transformers
└─ 14. KV Caching

Optimization Tier (15-19):
├─ 15. Profiling
├─ 16. Acceleration
├─ 17. Quantization
├─ 18. Compression
└─ 19. Benchmarking
```

### Target Structure
```
Architecture Tier (08-13):
├─ 08. DataLoader
├─ 09. Convolutional Networks  ← renamed
├─ 10. Tokenization
├─ 11. Embeddings
├─ 12. Attention
└─ 13. Transformers

Optimization Tier (14-19):
├─ 14. Profiling  ← moved from 15
├─ 15. Memoization  ← moved from 14, renamed from KV Caching
├─ 16. Quantization  ← moved from 17
├─ 17. Compression  ← moved from 18
├─ 18. Acceleration  ← moved from 16
└─ 19. Benchmarking  ← stays same
```

---

## 🔍 Profiler Requirements Analysis

Each optimization module needs specific profiling capabilities:

### Module 15 (Memoization) needs:
- ✅ `measure_latency()` - to show O(n²) growth
- ✅ `profile_forward_pass()` - for inference profiling
- ✅ Sequence length scaling analysis

### Module 16 (Quantization) needs:
- ✅ `count_parameters()` - parameter count
- ✅ `measure_memory()` - FP32 memory footprint
- ✅ Memory breakdown by component

### Module 17 (Compression) needs:
- ✅ `count_parameters()` - total parameters
- ✅ Weight distribution analysis (add helper)
- ✅ Sparsity calculation (add helper)

### Module 18 (Acceleration) needs:
- ✅ `count_flops()` - computational cost
- ✅ `profile_forward_pass()` - efficiency metrics
- ✅ Bottleneck detection (compute vs memory)

**Status:** Current profiler has 95% of needed functionality. Need to add:
- Helper function for weight distribution analysis
- Helper function for quick profiling display

---

## 📋 Implementation Phases

### **PHASE 1: Profiler Enhancement** ✅
**Branch:** `optimization-tier-restructure`  
**Goal:** Ensure profiler has all needed capabilities

**Tasks:**
1. ✅ Audit current profiler (DONE - has everything)
2. Add helper functions:
   - `quick_profile()` - simplified profiling interface
   - `analyze_weight_distribution()` - for compression module
3. Test profiler exports work correctly
4. **Commit:** `"feat(profiler): Add helper functions for optimization modules"`

---

### **PHASE 2: Add Profiling Intro Sections**
**Goal:** Add profiling motivation to each optimization module

#### Task 2.1: Module 14 (Current KV Caching → Future Memoization)
- Add Section 0: "Motivation - Profile Transformer Generation"
- Shows O(n²) latency growth
- ~10 lines of code
- **Commit:** `"feat(memoization): Add profiling motivation section"`

#### Task 2.2: Module 17 (Current Quantization → Future Quantization)
- Add Section 0: "Motivation - Profile Memory Usage"
- Shows FP32 memory footprint
- ~10 lines of code
- **Commit:** `"feat(quantization): Add profiling motivation section"`

#### Task 2.3: Module 18 (Current Compression → Future Compression)
- Add Section 0: "Motivation - Profile Parameter Distribution"
- Shows weight distribution
- ~10 lines of code
- **Commit:** `"feat(compression): Add profiling motivation section"`

#### Task 2.4: Module 16 (Current Acceleration → Future Acceleration)
- Add Section 0: "Motivation - Profile CNN Bottleneck"
- Shows compute-bound bottleneck
- ~10 lines of code
- **Commit:** `"feat(acceleration): Add profiling motivation section"`

---

### **PHASE 3: Module Directory Reorganization**
**Goal:** Rename and renumber module source directories

**Tasks:**
1. Rename module directories:
   ```bash
   # Architecture Tier
   mv modules/source/09_spatial modules/source/09_convolutional_networks
   
   # Optimization Tier - careful ordering!
   mv modules/source/15_profiling modules/source/14_profiling_temp
   mv modules/source/14_kvcaching modules/source/15_memoization
   mv modules/source/17_quantization modules/source/16_quantization
   mv modules/source/18_compression modules/source/17_compression
   mv modules/source/16_acceleration modules/source/18_acceleration
   mv modules/source/14_profiling_temp modules/source/14_profiling
   ```

2. Update `*_dev.py` files in each module:
   - Module number in header
   - `#| default_exp` path (if needed)
   - Prerequisites section
   - Connection map diagrams

3. **Commit:** `"refactor(modules): Reorganize optimization tier structure"`

---

### **PHASE 4: Book Chapter Reorganization**
**Goal:** Update user-facing documentation

#### Task 4.1: Rename Chapter Files
```bash
# Architecture Tier
mv book/chapters/09-spatial.md book/chapters/09-convolutional-networks.md

# Optimization Tier
mv book/chapters/15-profiling.md book/chapters/14-profiling.md
mv book/chapters/14-kvcaching.md book/chapters/15-memoization.md
mv book/chapters/17-quantization.md book/chapters/16-quantization.md
mv book/chapters/18-compression.md book/chapters/17-compression.md
mv book/chapters/16-acceleration.md book/chapters/18-acceleration.md
```

#### Task 4.2: Update Chapter Content
For each chapter:
1. Update heading (e.g., `# 15. Memoization`)
2. Update YAML frontmatter:
   - `title`
   - `prerequisites`
   - `next_steps`
   - `difficulty` (Memoization: 3→2)
3. Update tier badge
4. Update cross-references to other modules
5. Add conceptual framing for "Memoization" vs "KV Caching"

**Commits:**
- `"docs(chapters): Reorganize optimization tier chapters"`
- `"docs(memoization): Rename from KV Caching to Memoization"`
- `"docs(convolutional-networks): Rename from Spatial"`

---

### **PHASE 5: Table of Contents Update**
**Goal:** Update `book/_toc.yml`

**Changes:**
```yaml
- caption: 🏛️ Architecture Tier (08-13)  # was 08-14
  chapters:
  - file: chapters/09-convolutional-networks  # was 09-spatial
    title: "09. Convolutional Networks"  # was "09. Spatial (CNNs)"
  # Remove 14-kvcaching from here

- caption: ⚡ Optimization Tier (14-19)  # was 15-19
  chapters:
  - file: chapters/14-profiling  # was 15-profiling
    title: "14. Profiling"
  - file: chapters/15-memoization  # was 14-kvcaching
    title: "15. Memoization"  # was "14. KV Caching"
  - file: chapters/16-quantization  # was 17-quantization
    title: "16. Quantization"
  - file: chapters/17-compression  # was 18-compression
    title: "17. Compression"
  - file: chapters/18-acceleration  # was 16-acceleration
    title: "18. Acceleration"
  - file: chapters/19-benchmarking
    title: "19. Benchmarking"
```

**Commit:** `"docs(toc): Update table of contents for new structure"`

---

### **PHASE 6: CLI (tito) Updates**
**Goal:** Ensure CLI works with new module names/numbers

**Check:**
1. Module name resolution (does `tito export 14` work?)
2. Module completion tracking
3. Any hardcoded module references

**Files to check:**
- `tito/main.py`
- `tito/commands/*.py`
- Module lookup logic

**Commit:** `"fix(cli): Update module references for new structure"`

---

### **PHASE 7: Website Documentation - Tier Structure**
**Goal:** Add conceptual documentation explaining our structure

#### Task 7.1: Create "Understanding TinyTorch Structure" Page

**File:** `book/chapters/00-course-structure.md`

**Content:**
```markdown
# Understanding TinyTorch's Structure

## Three Levels of Learning

TinyTorch is organized into **Tiers**, **Modules**, and **Milestones**.

### 📚 Modules: Building Blocks
Modules teach you to build individual components.

- **What:** Single capability (e.g., "Profiling", "Quantization")
- **How:** Step-by-step implementation with tests
- **Output:** Exported component to tinytorch package
- **Time:** 3-8 hours per module

**Example:** Module 14 (Profiling)
- Build: Profiler class with parameter/FLOP/memory counting
- Test: Unit tests validate each method
- Export: `from tinytorch.profiling.profiler import Profiler`

### 🏛️ Tiers: Pedagogical Arcs
Tiers group related modules into coherent learning narratives.

**Foundation Tier (01-07):** Build the engine
- Core abstractions: Tensors, layers, autograd, training
- Outcome: "I can train basic neural networks"

**Architecture Tier (08-13):** Build intelligence
- Modern architectures: CNNs, attention, transformers
- Outcome: "I can build state-of-the-art models"

**Optimization Tier (14-19):** Build for production
- Performance: Profiling, memoization, quantization, compression, acceleration
- Outcome: "I can deploy models efficiently"

### 🏆 Milestones: Historical Achievements
Milestones integrate multiple modules to recreate landmark achievements.

- **What:** Historically significant capability unlocked
- **How:** Combine modules to build complete systems
- **Output:** Working implementation of historical milestone
- **Time:** Variable (few hours to days)

**Examples:**
- Milestone 03: 1986 MLP (uses modules 01-07)
- Milestone 05: 2017 Transformer (uses modules 01-13)
- Milestone 06: 2018 MLPerf Era (uses modules 14-20)

### 🔄 The Learning Flow

```
Modules → Build components (horizontal learning)
   ↓
Tiers → Understand narrative arc (vertical structure)
   ↓
Milestones → Integrate & achieve (synthesis)
```

## The Optimization Tier Pattern

Starting with Module 14, each optimization module follows this workflow:

1. **Profile:** Measure to identify the problem
2. **Discover:** "Oh, THAT'S the bottleneck!"
3. **Implement:** Build the optimization technique
4. **Validate:** Re-profile to measure improvement

This mirrors professional ML engineering practice.
```

#### Task 7.2: Update Introduction/Landing Pages
- Update `book/intro.md` with tier structure explanation
- Update `book/quickstart-guide.md` if needed
- Update `book/chapters/00-introduction.md`

**Commits:**
- `"docs(structure): Add course structure explanation"`
- `"docs(intro): Update with tier/module/milestone framework"`

---

### **PHASE 8: Cross-Reference Updates**
**Goal:** Fix all broken links and references

**Search for references to old module numbers:**
```bash
# Find references to old module numbers
grep -r "Module 14" book/chapters/
grep -r "Module 15" book/chapters/
grep -r "Module 16" book/chapters/
grep -r "KV Caching" book/chapters/
grep -r "Spatial" book/chapters/
```

**Fix:**
- Module number references
- "Next module" links
- Prerequisites listings
- Cross-references in text

**Commit:** `"docs: Fix cross-references for reorganized modules"`

---

### **PHASE 9: Test & Validation**
**Goal:** Ensure everything works

#### Task 9.1: Export Tests
```bash
cd modules/source/14_profiling
tito export 14
# Verify: tinytorch/profiling/profiler.py created

cd modules/source/15_memoization
tito export 15
# Verify: exports correctly
```

#### Task 9.2: Book Build Test
```bash
cd book
source ../.venv/bin/activate
jupyter-book build .
# Check for errors/warnings
```

#### Task 9.3: Module Tests
```bash
# Run tests for reorganized modules
tito test 14  # profiling
tito test 15  # memoization
tito test 16  # quantization
```

**Commit:** `"test: Verify all modules and book build correctly"`

---

### **PHASE 10: Final Documentation**
**Goal:** Update any remaining documentation

**Files to check:**
- `README.md` (if it mentions module structure)
- `CONTRIBUTING.md`
- Any milestone documentation
- `modules/README.md` if it exists

**Commit:** `"docs: Update remaining documentation for new structure"`

---

## 🎯 Success Criteria

- [ ] All modules renamed and renumbered correctly
- [ ] Each optimization module (15-18) has profiling intro section
- [ ] Book builds without errors
- [ ] All cross-references updated
- [ ] CLI works with new module numbers
- [ ] Website explains tier/module/milestone structure
- [ ] Git history has clear, logical commits
- [ ] Easy to review/rollback individual changes

---

## 📝 Commit Strategy

### Commit Naming Convention
```
type(scope): description

Types:
- feat: New feature
- refactor: Code restructuring
- docs: Documentation updates
- fix: Bug fixes
- test: Test updates

Examples:
- feat(profiler): Add quick_profile helper function
- refactor(modules): Reorganize optimization tier structure
- docs(memoization): Rename from KV Caching
- fix(cli): Update module number references
```

### Commit Order
1. Profiler enhancements (safe, additive)
2. Add profiling intro sections (safe, additive)
3. Module reorganization (breaking, but atomic)
4. Book chapter updates (documentation)
5. TOC update (documentation)
6. CLI fixes (if needed)
7. Website documentation (documentation)
8. Cross-reference fixes (cleanup)
9. Tests (validation)
10. Final documentation (polish)

---

## ⚠️ Risks & Mitigation

### Risk 1: Breaking Module Exports
**Mitigation:** Test exports after each phase

### Risk 2: Broken Cross-References
**Mitigation:** Systematic grep + fix in dedicated phase

### Risk 3: CLI Confusion
**Mitigation:** Test CLI commands after reorganization

### Risk 4: Lost in Large Refactor
**Mitigation:** This detailed plan + small commits

---

## 🚀 Execution Plan

**Estimated Time:** 4-6 hours total
- Phase 1-2: 1 hour (profiler + intro sections)
- Phase 3-4: 1.5 hours (reorganization)
- Phase 5-6: 0.5 hours (TOC + CLI)
- Phase 7-8: 1.5 hours (documentation + cross-refs)
- Phase 9-10: 1 hour (testing + polish)

**Next Steps:**
1. Review this plan
2. Start with Phase 1
3. Commit after each completed task
4. Test frequently
5. Move to next phase only when current phase is complete

---

*Plan created: [timestamp]*  
*Branch: optimization-tier-restructure*

