# 20. Capstone

**TinyTorch Olympics: Compete on Systems Performance**

---

## 🎯 Overview

The TinyTorch Olympics is your **final systems engineering challenge**—a competitive capstone where you optimize your TinyTorch implementations across multiple performance dimensions. This isn't just about accuracy; it's about speed, memory efficiency, power consumption, and real-world deployment constraints.

### Why a Competitive Capstone?

**Most ML courses end with:** "Build a project that works."  
**TinyTorch ends with:** "Optimize your system and compete."

This reflects the reality of production ML engineering:
- Getting a model working is just the beginning
- Performance matters: speed, memory, power, cost
- Systems engineering skills separate good ML engineers from great ones
- Real ML teams optimize and benchmark constantly

---

## 🏆 Competition Categories

### ⚡ Speed Demon
**"Fastest inference on standard hardware"**

- **Metric**: Inferences per second
- **Skills Tested**: Kernel optimization, parallelization, caching
- **Constraint**: Must maintain ≥90% accuracy
- **Modules Applied**: 14-19 optimization techniques

### 💾 Memory Miser
**"Smallest memory footprint"**

- **Metric**: Peak memory usage during inference
- **Skills Tested**: Quantization, compression, efficient architectures
- **Constraint**: Must maintain ≥85% accuracy
- **Modules Applied**: Quantization (16), Compression (17)

### 📱 Edge Expert
**"Best performance on resource-constrained hardware"**

- **Metric**: Composite score (speed + accuracy + efficiency)
- **Skills Tested**: Complete optimization pipeline
- **Constraint**: Must run on edge devices (e.g., Raspberry Pi)
- **Modules Applied**: Full optimization suite (14-19)

### 🔋 Energy Efficient
**"Lowest power consumption"**

- **Metric**: Energy per inference (joules/prediction)
- **Skills Tested**: Model compression, efficient algorithms
- **Constraint**: Must maintain competitive accuracy
- **Modules Applied**: Profiling (14), Optimization (15-19)

### 🏃‍♂️ TinyMLPerf
**"Official MLPerf-style benchmark"**

- **Metric**: Standardized benchmark suite performance
- **Skills Tested**: Complete systems optimization
- **Constraint**: Must pass all compliance tests
- **Modules Applied**: Benchmarking (19) + All optimization

---

## 🎮 Competition Structure

### Phase 1: Baseline Submission
**"Establish your starting point"**

```bash
# Submit your best model from modules 1-13
tito olympics submit --baseline

# Get initial scores across all categories
tito olympics scores --category all
```

**What happens:**
- Your model is evaluated across all categories
- You see where you rank initially
- You identify which categories to focus on

### Phase 2: Optimization Sprint
**"Apply modules 14-19 systematically"**

```bash
# Profile your model
tito olympics profile

# Apply optimization techniques
# Module 14: Profile and identify bottlenecks
# Module 15: Implement acceleration techniques  
# Module 16: Add quantization for memory/speed
# Module 17: Apply compression for size
# Module 18: Implement caching strategies
# Module 19: Benchmark against production systems
```

**Strategy:**
1. **Week 1**: Profile and analyze bottlenecks
2. **Week 2**: Apply memory optimizations
3. **Week 3**: Implement speed improvements
4. **Week 4**: Test on edge hardware
5. **Week 5**: Final benchmarking and submission

### Phase 3: Final Submission & Rankings
**"See how you stack up"**

```bash
# Submit optimized models
tito olympics submit --final

# View live leaderboard
tito olympics leaderboard

# Generate portfolio report
tito olympics report
```

---

## 📊 Leaderboard System

### Real-Time Rankings

```
🏆 TinyTorch Olympics Leaderboard

⚡ Speed Demon Category:
1. alice_chen    847.3 inf/sec  (95.2% acc)  🥇
2. bob_smith     612.7 inf/sec  (94.8% acc)  🥈
3. carol_wong    588.1 inf/sec  (96.1% acc)  🥉

💾 Memory Miser Category:
1. dave_kim      12.4 MB        (91.7% acc)  🥇
2. eve_patel     15.8 MB        (93.2% acc)  🥈
3. frank_liu     18.2 MB        (89.9% acc)  🥉

📱 Edge Expert Category:
1. grace_lee     Score: 94.5    (Composite)  🥇
2. henry_zhao    Score: 91.2    (Composite)  🥈
3. iris_tan      Score: 88.7    (Composite)  🥉
```

### Scoring Methodology

**Primary Metrics:**
- Each category has its own performance metric
- Must meet minimum accuracy threshold to qualify
- Tie-breaker: Higher accuracy wins

**Bonus Points:**
- **Innovation Award**: Novel optimization techniques (+5%)
- **Documentation Award**: Exceptional technical writeup (+3%)
- **Teaching Award**: Best educational explanation (+3%)

**Overall Champion:**
- Best combined performance across ALL categories
- Requires competing in at least 3 categories
- Weighted by difficulty of optimization achieved

---

## 🎯 Deliverables

### Competition Submission Package

**1. Optimized Model**
```bash
my_submission/
├── model.py              # Your optimized TinyTorch model
├── requirements.txt      # Dependencies
├── README.md             # Setup instructions
└── run_benchmark.py      # Evaluation script
```

**2. Performance Report**
- Optimization techniques applied
- Before/after measurements
- Systems engineering analysis
- Trade-offs and design decisions

**3. Reproduction Guide**
- Clear setup instructions
- Hardware requirements
- Expected results
- Troubleshooting tips

### Portfolio Artifacts You Get

✅ **Leaderboard Rankings**: Proof of competitive performance  
✅ **Technical Report**: Demonstrate systems engineering skills  
✅ **Benchmark Results**: Compare your work to industry standards  
✅ **Peer Recognition**: Rankings visible to potential employers  
✅ **GitHub Portfolio**: Complete optimization case study

---

## 🔧 Technical Requirements

### Submission Requirements

**All submissions must:**
- Use ONLY TinyTorch implementations (modules 1-13)
- Run on specified reference hardware
- Include reproducible benchmarking scripts
- Meet accuracy thresholds for category
- Pass automated validation tests

**Allowed optimizations:**
- Any technique from modules 14-19
- Custom kernel implementations
- Novel architectural designs
- Creative caching strategies
- Hardware-specific optimizations

**Not allowed:**
- External ML frameworks (PyTorch, TensorFlow, etc.)
- Pre-trained models from other sources
- Hardcoded test outputs
- Breaking TinyTorch API contracts

### Evaluation Environment

**Standard Hardware:**
- CPU: AMD EPYC 7763 (or equivalent)
- Memory: 32GB RAM
- Storage: NVMe SSD
- OS: Ubuntu 22.04 LTS

**Edge Hardware (for Edge Expert category):**
- Raspberry Pi 4B (4GB RAM)
- Power monitoring equipment
- Standard cooling (no exotic setups)

---

## 📚 Educational Value

### What You Learn

**Systems Engineering:**
- Performance profiling and bottleneck analysis
- Memory optimization techniques
- Speed vs. accuracy trade-offs
- Hardware-aware algorithm design
- Production deployment constraints

**ML Engineering:**
- Real-world optimization priorities
- Benchmarking and measurement
- Competitive system design
- Documentation and reproducibility
- Community collaboration

**Career Skills:**
- Portfolio-worthy competitive performance
- Systems thinking for production ML
- Technical communication and documentation
- Performance engineering mindset

### Why This Matters

**Most ML courses teach:** Algorithm implementation  
**TinyTorch teaches:** Systems optimization

**Most projects end with:** "Does it work?"  
**TinyTorch ends with:** "How fast? How small? How efficient?"

This is what separates ML researchers from ML engineers. You learn to care about the full system, not just the algorithm.

---

## 🚀 Getting Started

### Prerequisites

**Required Modules:**
- Modules 1-13: Build your base model
- Modules 14-19: Learn optimization techniques

**Recommended Preparation:**
```bash
# Complete all modules
tito checkpoint status

# Test your optimization skills
tito module test 14  # Profiling
tito module test 15  # Acceleration
tito module test 16  # Quantization
tito module test 17  # Compression
tito module test 18  # Caching
tito module test 19  # Benchmarking
```

### Quick Start

```bash
# 1. Register for Olympics
tito olympics register

# 2. Submit baseline
tito olympics submit --baseline

# 3. View your scores
tito olympics scores

# 4. Optimize and resubmit
tito olympics submit --category speed

# 5. Check leaderboard
tito olympics leaderboard
```

---

## 🏅 Awards & Recognition

### Category Champions 🥇
- Top performer in each category
- Certificate of achievement
- Featured on leaderboard permanently
- LinkedIn-ready accomplishment

### Overall Systems Engineer 🏆
- Best combined performance across categories
- Requires competing in ≥3 categories
- Special recognition on course website
- Strong portfolio differentiator

### Special Awards

**🚀 Innovation Award**
- Most creative optimization approach
- Novel techniques or architectures
- Judged by instructors and peers

**📚 Teaching Award**
- Best documented optimization process
- Helps future students learn
- Clarity and educational value

**🎯 First Blood Award**
- First to beat instructor baseline
- In any category
- Special early-achiever recognition

---

## 💡 Strategy Tips

### Getting Started

**1. Profile First**
```bash
# Don't guess—measure!
tito olympics profile --detailed
```

**2. Pick Your Category**
- Speed Demon: Focus on compute optimization
- Memory Miser: Quantization and compression
- Edge Expert: Balanced optimization
- Energy Efficient: Algorithm efficiency

**3. Apply Systematic Optimization**
- One technique at a time
- Measure impact of each change
- Keep detailed notes
- Document trade-offs

### Advanced Strategies

**For Speed:**
- Vectorize operations (Module 15)
- Implement caching (Module 18)
- Optimize hot paths first
- Consider CPU instruction sets

**For Memory:**
- Quantization (Module 16)
- Weight pruning (Module 17)
- Efficient data structures
- Activation checkpointing

**For Edge:**
- Balance all dimensions
- Test on real hardware early
- Power profiling tools
- Thermal management

---

## 🌟 Success Stories

### What Past Participants Say

> "The Olympics forced me to actually care about performance. In previous courses, I just wanted things to work. Here, I learned to optimize." - *Alex, Spring 2024*

> "Ranking #2 in Memory Efficiency was the highlight of my portfolio. It came up in every interview." - *Jordan, Fall 2024*

> "I thought I understood optimization until the Olympics. The leaderboard competition pushed me to learn techniques I would have skipped." - *Sam, Spring 2024*

---

## 🎓 Final Thoughts

### Why Olympics > Traditional Capstone

**Traditional Capstone:**
- Build a project that works ✓
- Submit and move on
- Limited comparison with peers
- Optimization is optional

**TinyTorch Olympics:**
- Build a system that performs ⚡
- Compete and improve continuously
- Clear performance benchmarks
- Optimization is the point

### The Real Goal

The Olympics isn't just about winning. It's about:

✅ **Learning systems thinking**  
✅ **Caring about performance**  
✅ **Building portfolio-worthy projects**  
✅ **Joining a community of builders**  
✅ **Preparing for real ML engineering**

---

**Ready to compete?**

```bash
tito olympics register
```

**Build systems. Optimize relentlessly. Compete.** 🥇

