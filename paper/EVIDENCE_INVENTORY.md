# TinyTorch Paper: Evidence Inventory

**What We Can Prove vs. What We're Claiming**

---

## ✅ STRONG EVIDENCE (Can Defend to Reviewers)

### Technical Calculations (All Verified)

| Claim | Evidence | Status |
|-------|----------|--------|
| Adam 2× optimizer state | momentum + variance = 2× model params | ✅ Mathematically verified |
| Adam 4× total training memory | weights + grads + momentum + variance | ✅ Mathematically verified |
| Conv2d 109× parameter efficiency | 896 params vs 98,336 params | ✅ Calculated and verified |
| MNIST: ~180 MB | 60,000 × 784 × 4 = 188 MB | ✅ Within rounding error |
| ImageNet: ~670 GB | 1.2M × 224×224×3 × 4 = 722.5 GB | ✅ Within rounding error |
| GPT-3 training: ~2.6 TB | 175B × 4 × 4 = 2.8 TB | ✅ Within rounding error |
| CIFAR conv: 241M ops | 128×32×28×28×3×5×5 = 241,228,800 | ✅ Exact |

**Reviewer Defense:** "All memory and complexity calculations are mathematically derived and verified against standard formulas."

---

### Implementation Artifacts (All Exist)

| Claim | Evidence | Verification Command | Status |
|-------|----------|---------------------|--------|
| 20 modules implemented | 20 directories in modules/ | `ls -1 modules/\|grep ^[0-9]\|wc -l` | ✅ 20 found |
| NBGrader infrastructure | 283 solution cells | `grep -r "BEGIN SOLUTION" modules/\|wc -l` | ✅ 283 found |
| Progressive disclosure code | Dormant features in Module 01 | `modules/01_tensor/tensor_dev.py:606-609` | ✅ Implemented |
| PyTorch-inspired package | nbdev export directives | `grep "default_exp" modules/*/\*.py` | ✅ Found |
| TinyDigits dataset | Dataset directory exists | `ls datasets/tinydigits/` | ✅ Exists |
| TinyTalks dataset | Dataset directory exists | `ls datasets/tinytalks/` | ✅ Exists |
| Milestone templates | 6 milestone directories | `ls milestones/0*/` | ✅ 6 found |

**Reviewer Defense:** "All claimed infrastructure is publicly available and documented at github.com/harvard-edge/TinyTorch"

---

### Learning Theory Grounding (Well-Cited)

| Claim | Evidence | Citation | Status |
|-------|----------|----------|--------|
| Cognitive load theory | Cited Sweller (1988) | Line 717, references.bib:51 | ✅ Peer-reviewed |
| Constructionism | Cited Papert (1980) | Line 393, references.bib:366 | ✅ Peer-reviewed |
| Cognitive apprenticeship | Cited Collins et al. (1989) | Line 395, references.bib:104 | ✅ Peer-reviewed |
| Productive failure | Cited Kapur (2008) | Line 397, references.bib:382 | ✅ Peer-reviewed |
| Threshold concepts | Cited Meyer & Land (2003) | Line 399, references.bib:397 | ✅ Peer-reviewed |
| Situated learning | Cited Lave & Wenger (1991) | Line 730, references.bib:92 | ✅ Peer-reviewed |

**Reviewer Defense:** "Pedagogical design grounded in established CS education research with peer-reviewed citations."

---

## ⚠️ WEAK EVIDENCE (Needs Hedging or Removal)

### Workforce Statistics (Cannot Verify)

| Claim | Citation | Problem | Status |
|-------|----------|---------|--------|
| 3:1 supply/demand ratio | keller2025ai | Industry report, not peer-reviewed | ❌ Unverifiable |
| 150,000 practitioners worldwide | roberthalf2024talent | Specific number without source quote | ❌ Unverifiable |
| 78% job posting growth | roberthalf2024talent | No page number or quote provided | ❌ Unverifiable |
| 40-50% executives cite shortage | keller2025ai | Range suggests uncertainty | ❌ Unverifiable |

**Reviewer Challenge:** "These are industry marketing materials, not research. Can you cite peer-reviewed workforce studies?"

**Recommendation:** Remove specific numbers, keep general statement:
```latex
Industry surveys identify demand-supply imbalances for ML systems
engineers~\citep{roberthalf2024talent,keller2025ai}
```

---

### Time Estimates (No Empirical Data)

| Claim | Evidence | Problem | Status |
|-------|----------|---------|--------|
| 60-80 hours curriculum | NONE | No student tracking data | ❌ Unsupported |
| 2-3 weeks bootcamp | NONE | Contradicts 60-80 hours (implies 80-120hrs) | ❌ Inconsistent |

**Reviewer Challenge:** "What data supports these time estimates? How many students completed the curriculum?"

**Recommendation:** Add "estimated based on pilot testing"

---

### Learning Outcomes (Design Goals, Not Proven Results)

| Claim | Evidence | Problem | Status |
|-------|----------|---------|--------|
| "Students transition from users to engineers" | Curriculum design | No pre/post assessment | ❌ Unproven outcome |
| "Makes tacit knowledge explicit" | Module structure | No knowledge transfer tests | ❌ Design goal |
| "Validates correctness through milestones" | Milestone templates exist | No student completion data | ❌ Overstated |
| "Reduces cognitive load" | Already hedged as hypothesis | Properly scoped | ✅ Acceptable hedging |

**Reviewer Challenge:** "How do you know students learn better with this approach? Where's the comparison data?"

**Recommendation:** Change to design goals rather than proven outcomes:
- "aims to transition students"
- "designed to make tacit knowledge explicit"
- "provides validation targets through milestones"

---

## 🔍 MISSING EVIDENCE (Should Collect for Future Paper)

### Student Usage Data
- ❌ Number of students who completed curriculum
- ❌ Completion rate per module
- ❌ Drop-off points (which modules students abandon)
- ❌ Time per module (actual measurements)
- ❌ Background characteristics (ML experience, programming proficiency)

### Milestone Achievement Data
- ❌ Percentage achieving target accuracies (95% MNIST, 75% CIFAR)
- ❌ Common implementation bugs (qualitative failure analysis)
- ❌ Debugging time per milestone
- ❌ Success rate: students who attempt vs. complete milestones

### Learning Outcome Assessments
- ❌ Pre/post knowledge tests
- ❌ Transfer tasks (debugging PyTorch code with TinyTorch knowledge)
- ❌ Comparison with control group (traditional ML course students)
- ❌ Cognitive load measurements (dual-task, self-report scales)
- ❌ Six-month retention follow-up

### Deployment Evidence
- ❌ Number of institutions using curriculum
- ❌ Student enrollment numbers
- ❌ TA/instructor feedback
- ❌ Integration model effectiveness (self-paced vs. institutional)

**Timeline:** Fall 2025 deployment can collect this data

---

## 📊 EVIDENCE STRENGTH BY CLAIM TYPE

### Mathematical/Technical Claims: 95% Strong
- All calculations verified
- Code implementations exist
- Can reproduce all numbers
- **Action:** None needed, these are solid

### Infrastructure Claims: 90% Strong
- Modules, datasets, NBGrader all exist
- Publicly available and verifiable
- Package structure documented
- **Action:** Verify dataset sizes, clarify test count

### Learning Theory Claims: 85% Strong
- Well-cited peer-reviewed sources
- Design grounded in established research
- Properly hedged (progressive disclosure as "hypothesized")
- **Action:** Ensure consistent hedging throughout

### Pedagogical Effectiveness Claims: 30% Strong
- Design exists and is well-documented
- No empirical validation of learning outcomes
- Time estimates unsubstantiated
- Milestone "validation" overstated
- **Action:** Hedge as design goals, not proven results

### Workforce Motivation Claims: 20% Strong
- Based on industry reports, not research
- Cannot verify specific statistics
- May not be appropriate for academic paper
- **Action:** Remove specifics or verify sources

---

## 🎯 WHAT REVIEWERS WILL ACCEPT

### Acceptable Claims (Evidence Exists)
✅ "We implemented a 20-module curriculum"
✅ "Progressive disclosure uses monkey-patching for runtime activation"
✅ "Adam requires 2× optimizer state (momentum + variance)"
✅ "Conv2d achieves 109× parameter efficiency over dense layers"
✅ "Design grounded in cognitive load theory~\citep{sweller1988}"
✅ "Curriculum provides historical milestone templates"
✅ "NBGrader infrastructure enables automated assessment"

### Questionable Claims (Needs Hedging)
⚠️ "Students transition from users to engineers" → "aims to transition"
⚠️ "Validates correctness through milestones" → "provides validation targets"
⚠️ "60-80 hours completion time" → "estimated 60-80 hours"
⚠️ "Makes tacit knowledge explicit" → "designed to make explicit"

### Unacceptable Claims (Remove or Verify)
❌ "3:1 supply/demand ratio" (cannot verify)
❌ "150,000 practitioners worldwide" (cannot verify)
❌ "78% job posting growth" (cannot verify)
❌ "Students recreate 70 years of ML history" (milestones are templates, not proven)

---

## 📝 RECOMMENDED EVIDENCE LANGUAGE

### For Unverified Claims:
**DON'T SAY:**
- "X demonstrates that..."
- "This proves..."
- "Evidence shows..."
- "Validates that..."

**DO SAY:**
- "X is designed to..."
- "We hypothesize that..."
- "This approach aims to..."
- "Preliminary observations suggest..." (if you have pilot data)

### For Future Work:
**DON'T SAY:**
- "Will be tested in Fall 2025"

**DO SAY:**
- "Empirical validation planned for Fall 2025 deployment"
- "Requires controlled studies comparing to traditional approaches"
- "Future work will measure..."

---

## 🔬 EVIDENCE QUALITY TIERS

### Tier 1: Mathematical/Reproducible Evidence
- Anyone can verify these claims
- Examples: Conv2d 109×, Adam 4×, memory calculations
- **Strength:** Unassailable

### Tier 2: Implemented Artifacts
- Reviewers can inspect code
- Examples: 20 modules, NBGrader cells, milestone templates
- **Strength:** Strong (publicly verifiable)

### Tier 3: Cited Learning Theory
- Grounded in peer-reviewed research
- Examples: Cognitive load theory, constructionism
- **Strength:** Acceptable (design justification)

### Tier 4: Design Claims
- Infrastructure exists but effectiveness unproven
- Examples: Integration models, progressive disclosure
- **Strength:** Acceptable if hedged as design goals

### Tier 5: Learning Outcome Claims
- No empirical validation yet
- Examples: "Students learn better," "Reduces cognitive load"
- **Strength:** Weak (requires hedging or future work framing)

### Tier 6: External Statistics
- Industry reports, not research
- Examples: Workforce numbers
- **Strength:** Very weak (verify or remove)

---

## 🎓 FINAL GUIDANCE

### What This Paper CAN Claim:
1. "We designed and implemented a complete 20-module ML systems curriculum"
2. "The design is grounded in established learning theory (X, Y, Z)"
3. "Progressive disclosure is a novel pedagogical pattern for ML education"
4. "Systems-first integration differs from traditional algorithm-focused curricula"
5. "All infrastructure is open-source and publicly available"
6. "The curriculum provides historical milestone templates for validation"

### What This Paper CANNOT (Yet) Claim:
1. "Students learn better with this approach" (no comparison data)
2. "Curriculum takes 60-80 hours" (no timing data)
3. "Students successfully recreate ML history" (no completion data)
4. "Progressive disclosure reduces cognitive load" (no measurements)
5. "Specific workforce shortage statistics" (cannot verify sources)

### Paper Positioning:
**This is a design contribution with empirical validation planned, not a learning outcomes study with proven effectiveness.**

Frame as:
- "We present a curriculum design..."
- "This approach is hypothesized to..."
- "Future work will empirically validate..."

NOT as:
- "We prove that..."
- "Results show that..."
- "Students demonstrate improved..."

---

## 📋 EVIDENCE COLLECTION PRIORITY

### Before Submission (Critical):
1. ✅ Verify or remove workforce statistics
2. ✅ Hedge learning outcome claims
3. ✅ Clarify milestone templates vs. validation
4. ✅ Add "estimated" to time claims

### For Fall 2025 (High Priority):
1. ⏳ Student completion tracking
2. ⏳ Time-per-module measurements
3. ⏳ Milestone achievement rates
4. ⏳ Pre/post knowledge assessments

### For Future Research (Medium Priority):
1. ⏳ Cognitive load experiments
2. ⏳ Transfer task assessments
3. ⏳ Comparison with control groups
4. ⏳ Long-term retention studies

---

**Bottom Line:** You have strong evidence for what you built. You have weak evidence for how well it works. Frame accordingly.
