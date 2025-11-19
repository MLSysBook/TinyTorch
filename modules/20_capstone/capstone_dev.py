# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 20: TinyTorch Olympics - Competition & Submission

Welcome to the capstone module of TinyTorch! You've built an entire ML framework from scratch across 19 modules. Now it's time to compete in **TinyTorch Olympics** - demonstrating your optimization skills and generating professional competition submissions.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with benchmarking infrastructure (Module 19)
**You'll Build**: Competition workflow, submission generation, and event configuration
**You'll Enable**: Professional ML competition participation and standardized submission packaging

**Connection Map**:
```
Modules 01-19 → Benchmarking (M19) → Competition Workflow (M20)
(Foundation)    (Measurement)        (Submission)
```

## Learning Objectives
By the end of this capstone, you will:
1. **Understand** competition events and how to configure your submission
2. **Use** the benchmarking harness from Module 19 to measure performance
3. **Generate** standardized competition submissions (MLPerf-style JSON)
4. **Validate** submissions meet competition requirements
5. **Package** your work professionally for competition participation

**Key Insight**: This module teaches the workflow and packaging - you use the benchmarking tools from Module 19 and optimization techniques from Modules 14-18. The focus is on how to compete, not how to build models (that's Milestone 05).
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/20_capstone/capstone_dev.py`  
**Building Side:** Code exports to `tinytorch.competition.submit`

```python
# How to use this module:
from tinytorch.competition.submit import OlympicEvent, generate_submission
from tinytorch.benchmarking import Benchmark  # From Module 19

# Use benchmarking harness from Module 19
benchmark = Benchmark([my_model], [{"name": "my_model"}])
results = benchmark.run_latency_benchmark()

# Generate competition submission
submission = generate_submission(
    event=OlympicEvent.LATENCY_SPRINT,
    benchmark_results=results
)
```

**Why this matters:**
- **Learning:** Complete competition workflow using benchmarking tools from Module 19
- **Production:** Professional submission format following MLPerf-style standards
- **Consistency:** Standardized competition framework for fair comparison
- **Integration:** Uses benchmarking harness (Module 19) + optimization techniques (Modules 14-18)
"""

# %% nbgrader={"grade": false, "grade_id": "exports", "solution": true}
#| default_exp competition.submit
#| export

# %% [markdown]
"""
## 🔮 Introduction: From Measurement to Competition

Over the past 19 modules, you've built the complete infrastructure for modern ML:

**Foundation (Modules 01-04):** Tensors, activations, layers, and losses
**Training (Modules 05-07):** Automatic differentiation, optimizers, and training loops
**Architecture (Modules 08-09):** Spatial processing and data loading
**Language (Modules 10-14):** Text processing, embeddings, attention, transformers, and KV caching
**Optimization (Modules 15-19):** Profiling, acceleration, quantization, compression, and benchmarking

In Module 19, you built a benchmarking harness with statistical rigor. Now in Module 20, you'll use that harness to participate in **TinyTorch Olympics** - a competition framework that demonstrates professional ML systems evaluation.

```
Your Journey:
    Build Framework → Optimize → Benchmark → Compete
    (Modules 01-18)  (M14-18)   (Module 19) (Module 20)
```

This capstone teaches the workflow of professional ML competitions - how to measure, compare, and submit your work following industry standards.
"""

# %% [markdown]
"""
## 📊 Competition Workflow: From Measurement to Submission

This capstone demonstrates the complete workflow of professional ML competitions. You'll use the benchmarking harness from Module 19 to measure performance and generate standardized submissions.

### TinyTorch Olympics Competition Flow

```
                    🏅 TINYTORCH OLYMPICS COMPETITION WORKFLOW 🏅

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          STEP 1: CHOOSE YOUR EVENT                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  🏃 Latency Sprint    → Minimize inference time (accuracy ≥ 85%)                  │
│  🏋️ Memory Challenge  → Minimize model size (accuracy ≥ 85%)                      │
│  🎯 Accuracy Contest  → Maximize accuracy (latency < 100ms, memory < 10MB)        │
│  🏋️‍♂️ All-Around       → Best balanced performance                                 │
│  🚀 Extreme Push      → Most aggressive optimization (accuracy ≥ 80%)              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: MEASURE BASELINE (Module 19 Harness)                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Baseline Model → [Benchmark] → Statistical Results                                │
│                  (Module 19)                                                       │
│                                                                                     │
│  Benchmark Output:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Latency: 45.2ms ± 2.1ms (95% CI: [43.1, 47.3])                            │   │
│  │ Memory: 12.4MB                                                              │   │
│  │ Accuracy: 85.0%                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: OPTIMIZE (Modules 14-18 Techniques)                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Baseline → [Quantization] → [Pruning] → [Other Optimizations] → Optimized Model  │
│            (Module 17)     (Module 18)  (Modules 14-16)                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│              STEP 4: MEASURE OPTIMIZED (Module 19 Harness Again)                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Optimized Model → [Benchmark] → Statistical Results                               │
│                   (Module 19)                                                      │
│                                                                                     │
│  Benchmark Output:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Latency: 22.1ms ± 1.2ms (95% CI: [20.9, 23.3]) ✅ 2.0x faster            │   │
│  │ Memory: 1.24MB ✅ 10.0x smaller                                            │   │
│  │ Accuracy: 83.5% (Δ -1.5pp)                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: GENERATE SUBMISSION (Module 20)                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Benchmark Results → [generate_submission()] → submission.json                     │
│  (from Module 19)    (Module 20)                                                  │
│                                                                                     │
│  Submission JSON includes:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ • Event type (Latency Sprint, Memory Challenge, etc.)                      │   │
│  │ • Baseline metrics (from Step 2)                                           │   │
│  │ • Optimized metrics (from Step 4)                                           │   │
│  │ • Normalized scores (speedup, compression, efficiency)                      │   │
│  │ • System information (hardware, OS, Python version)                        │   │
│  │ • Validation status                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Competition Workflow Summary

**The Complete Process:**
1. **Choose Event**: Select your competition category based on optimization goals
2. **Measure Baseline**: Use Benchmark harness from Module 19 to establish baseline
3. **Optimize**: Apply techniques from Modules 14-18 (quantization, pruning, etc.)
4. **Measure Optimized**: Use Benchmark harness again to measure improvements
5. **Generate Submission**: Create standardized JSON submission file

**Key Principle**: Module 20 provides the workflow and submission format. You use:
- **Benchmarking tools** from Module 19 (measurement)
- **Optimization techniques** from Modules 14-18 (improvement)
- **Competition framework** from Module 20 (packaging)
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import competition and benchmarking modules
### BEGIN SOLUTION
# Module 19: Benchmarking harness (for measurement)
from tinytorch.benchmarking.benchmark import Benchmark, BenchmarkResult

# Module 17-18: Optimization techniques (for applying optimizations)
from tinytorch.optimization.quantization import quantize_model
from tinytorch.optimization.compression import magnitude_prune

# System information for submission metadata
import platform
import sys
### END SOLUTION

print("✅ Competition modules imported!")
print("📊 Ready to use Benchmark harness from Module 19")

# %% [markdown]
"""
## 1. Introduction: Understanding Competition Events

TinyTorch Olympics offers five different competition events, each with different optimization objectives and constraints. Understanding these events helps you choose the right strategy and configure your submission correctly.
"""

# %% nbgrader={"grade": false, "grade_id": "olympic-events", "solution": true}
#| export
from enum import Enum

class OlympicEvent(Enum):
    """
    TinyTorch Olympics event categories.
    
    Each event optimizes for different objectives with specific constraints.
    Students choose their event and compete for medals!
    """
    LATENCY_SPRINT = "latency_sprint"      # Minimize latency (accuracy >= 85%)
    MEMORY_CHALLENGE = "memory_challenge"   # Minimize memory (accuracy >= 85%)
    ACCURACY_CONTEST = "accuracy_contest"   # Maximize accuracy (latency < 100ms, memory < 10MB)
    ALL_AROUND = "all_around"               # Best balanced score across all metrics
    EXTREME_PUSH = "extreme_push"           # Most aggressive optimization (accuracy >= 80%)

# %% [markdown]
"""
## 2. Competition Workflow: Using the Benchmarking Harness

Module 19 provides the benchmarking harness. Module 20 shows you how to use it in a competition context. Let's walk through the complete workflow.
"""

# %% nbgrader={"grade": false, "grade_id": "normalized-scoring", "solution": true}
#| export
def calculate_normalized_scores(baseline_results: dict, 
                                optimized_results: dict) -> dict:
    """
    Calculate normalized performance metrics for fair competition comparison.
    
    This function converts absolute measurements into relative improvements,
    enabling fair comparison across different hardware platforms.
    
    Args:
        baseline_results: Dict with keys: 'latency', 'memory', 'accuracy'
        optimized_results: Dict with same keys as baseline_results
        
    Returns:
        Dict with normalized metrics:
        - speedup: Relative latency improvement (higher is better)
        - compression_ratio: Relative memory reduction (higher is better)
        - accuracy_delta: Absolute accuracy change (closer to 0 is better)
        - efficiency_score: Combined metric balancing all factors
        
    Example:
        >>> baseline = {'latency': 100.0, 'memory': 12.0, 'accuracy': 0.89}
        >>> optimized = {'latency': 40.0, 'memory': 3.0, 'accuracy': 0.87}
        >>> scores = calculate_normalized_scores(baseline, optimized)
        >>> print(f"Speedup: {scores['speedup']:.2f}x")
        Speedup: 2.50x
    """
    # Calculate speedup (higher is better)
    speedup = baseline_results['latency'] / optimized_results['latency']
    
    # Calculate compression ratio (higher is better)
    compression_ratio = baseline_results['memory'] / optimized_results['memory']
    
    # Calculate accuracy delta (closer to 0 is better, negative means degradation)
    accuracy_delta = optimized_results['accuracy'] - baseline_results['accuracy']
    
    # Calculate efficiency score (combined metric)
    # Penalize accuracy loss: the more accuracy you lose, the lower your score
    accuracy_penalty = max(1.0, 1.0 - accuracy_delta) if accuracy_delta < 0 else 1.0
    efficiency_score = (speedup * compression_ratio) / accuracy_penalty
    
    return {
        'speedup': speedup,
        'compression_ratio': compression_ratio,
        'accuracy_delta': accuracy_delta,
        'efficiency_score': efficiency_score,
        'baseline': baseline_results.copy(),
        'optimized': optimized_results.copy()
    }

# %% [markdown]
"""
## 3. Submission Generation: Creating Competition Submissions

Now let's build the submission generation function that uses the Benchmark harness from Module 19 and creates standardized competition submissions.
"""

# %% [markdown]
"""
## 🏗️ Stage 1: Competition Workflow - Complete Example

Let's walk through a complete competition workflow example. This demonstrates how to use the Benchmark harness from Module 19 to measure performance and generate submissions.

### Complete Competition Workflow Example

Here's a step-by-step example showing how to participate in TinyTorch Olympics:

**Step 1: Choose Your Event**
```python
from tinytorch.competition.submit import OlympicEvent

event = OlympicEvent.LATENCY_SPRINT  # Focus on speed
```

**Step 2: Measure Baseline Using Module 19's Benchmark**
```python
from tinytorch.benchmarking import Benchmark

# Create benchmark harness (from Module 19)
benchmark = Benchmark([baseline_model], [{"name": "baseline"}])

# Run latency benchmark with statistical rigor
baseline_results = benchmark.run_latency_benchmark()
# Returns: BenchmarkResult with mean, std, confidence intervals
```

**Step 3: Apply Optimizations (Modules 14-18)**
```python
from tinytorch.optimization.quantization import quantize_model
from tinytorch.optimization.compression import magnitude_prune

optimized = quantize_model(baseline_model, bits=8)
optimized = magnitude_prune(optimized, sparsity=0.6)
```

**Step 4: Measure Optimized Model**
```python
benchmark_opt = Benchmark([optimized], [{"name": "optimized"}])
optimized_results = benchmark_opt.run_latency_benchmark()
```

**Step 5: Generate Submission**
```python
from tinytorch.competition.submit import generate_submission

submission = generate_submission(
    event=OlympicEvent.LATENCY_SPRINT,
    baseline_results=baseline_results,
    optimized_results=optimized_results
)
# Creates submission.json with all required fields
```

### Key Workflow Principles

**1. Use Module 19's Benchmark Harness**: All measurements use the same statistical rigor
**2. Apply Optimizations Systematically**: Use techniques from Modules 14-18
**3. Generate Standardized Submissions**: Module 20 provides the submission format
**4. Validate Before Submitting**: Ensure your submission meets event requirements

Let's implement the submission generation function that ties everything together.
"""

# %% nbgrader={"grade": false, "grade_id": "submission-generation", "solution": true}
#| export
def generate_submission(baseline_results: Dict[str, Any],
                        optimized_results: Dict[str, Any],
                        event: OlympicEvent = OlympicEvent.ALL_AROUND,
                        athlete_name: str = "YourName",
                        github_repo: str = "",
                        techniques: List[str] = None) -> Dict[str, Any]:
    """
    Generate standardized TinyTorch Olympics competition submission.
    
    This function uses Benchmark results from Module 19 and creates a
    standardized submission JSON following MLPerf-style format.
    
    Args:
        baseline_results: Dict with 'latency', 'memory', 'accuracy' from Benchmark
        optimized_results: Dict with same keys as baseline_results
        event: OlympicEvent enum specifying competition category
        athlete_name: Your name for submission
        github_repo: GitHub repository URL (optional)
        techniques: List of optimization techniques applied
        
    Returns:
        Submission dictionary ready to be saved as JSON
        
    Example:
        >>> baseline = {'latency': 100.0, 'memory': 12.0, 'accuracy': 0.85}
        >>> optimized = {'latency': 40.0, 'memory': 3.0, 'accuracy': 0.83}
        >>> submission = generate_submission(baseline, optimized, OlympicEvent.LATENCY_SPRINT)
        >>> submission['normalized_scores']['speedup']
        2.5
    """
    ### BEGIN SOLUTION
    # Calculate normalized scores
    normalized = calculate_normalized_scores(baseline_results, optimized_results)
    
    # Gather system information
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version.split()[0],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create submission dictionary
    submission = {
        'submission_version': '1.0',
        'event': event.value,
        'athlete_name': athlete_name,
        'github_repo': github_repo,
        'baseline': baseline_results.copy(),
        'optimized': optimized_results.copy(),
        'normalized_scores': {
            'speedup': normalized['speedup'],
            'compression_ratio': normalized['compression_ratio'],
            'accuracy_delta': normalized['accuracy_delta'],
            'efficiency_score': normalized['efficiency_score']
        },
        'techniques_applied': techniques or [],
        'system_info': system_info,
        'timestamp': system_info['timestamp']
    }
    
    return submission
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "submission-validation", "solution": true}
#| export
def validate_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate competition submission meets requirements.
    
    Args:
        submission: Submission dictionary to validate
        
    Returns:
        Dict with 'valid' (bool), 'checks' (list), 'warnings' (list), 'errors' (list)
    """
    ### BEGIN SOLUTION
    checks = []
    warnings = []
    errors = []
    
    # Check required fields
    required_fields = ['event', 'baseline', 'optimized', 'normalized_scores']
    for field in required_fields:
        if field not in submission:
            errors.append(f"Missing required field: {field}")
        else:
            checks.append(f"✅ {field} present")
    
    # Validate event constraints
    event = submission.get('event')
    normalized = submission.get('normalized_scores', {})
    optimized = submission.get('optimized', {})
    
    if event == OlympicEvent.LATENCY_SPRINT.value:
        if optimized.get('accuracy', 0) < 0.85:
            errors.append(f"Latency Sprint requires accuracy >= 85%, got {optimized.get('accuracy', 0)*100:.1f}%")
        else:
            checks.append(f"✅ Accuracy constraint met: {optimized.get('accuracy', 0)*100:.1f}% >= 85%")
    
    elif event == OlympicEvent.MEMORY_CHALLENGE.value:
        if optimized.get('accuracy', 0) < 0.85:
            errors.append(f"Memory Challenge requires accuracy >= 85%, got {optimized.get('accuracy', 0)*100:.1f}%")
        else:
            checks.append(f"✅ Accuracy constraint met: {optimized.get('accuracy', 0)*100:.1f}% >= 85%")
    
    elif event == OlympicEvent.ACCURACY_CONTEST.value:
        if optimized.get('latency', float('inf')) >= 100.0:
            errors.append(f"Accuracy Contest requires latency < 100ms, got {optimized.get('latency', 0):.1f}ms")
        elif optimized.get('memory', float('inf')) >= 10.0:
            errors.append(f"Accuracy Contest requires memory < 10MB, got {optimized.get('memory', 0):.2f}MB")
        else:
            checks.append("✅ Latency and memory constraints met")
    
    elif event == OlympicEvent.EXTREME_PUSH.value:
        if optimized.get('accuracy', 0) < 0.80:
            errors.append(f"Extreme Push requires accuracy >= 80%, got {optimized.get('accuracy', 0)*100:.1f}%")
        else:
            checks.append(f"✅ Accuracy constraint met: {optimized.get('accuracy', 0)*100:.1f}% >= 80%")
    
    # Check for unrealistic improvements
    if normalized.get('speedup', 1.0) > 50:
        errors.append(f"Speedup {normalized['speedup']:.1f}x seems unrealistic (>50x)")
    elif normalized.get('speedup', 1.0) > 20:
        warnings.append(f"⚠️  Very high speedup {normalized['speedup']:.1f}x - please verify")
    
    if normalized.get('compression_ratio', 1.0) > 32:
        errors.append(f"Compression {normalized['compression_ratio']:.1f}x seems unrealistic (>32x)")
    elif normalized.get('compression_ratio', 1.0) > 16:
        warnings.append(f"⚠️  Very high compression {normalized['compression_ratio']:.1f}x - please verify")
    
    return {
        'valid': len(errors) == 0,
        'checks': checks,
        'warnings': warnings,
        'errors': errors
    }
    ### END SOLUTION

def test_unit_submission_generation():
    """🔬 Test submission generation."""
    print("🔬 Unit Test: Submission Generation...")
    
    baseline = {'latency': 100.0, 'memory': 12.0, 'accuracy': 0.85}
    optimized = {'latency': 40.0, 'memory': 3.0, 'accuracy': 0.83}
    
    submission = generate_submission(
        baseline_results=baseline,
        optimized_results=optimized,
        event=OlympicEvent.LATENCY_SPRINT,
        athlete_name="TestUser",
        techniques=["quantization_int8", "pruning_60"]
    )
    
    assert submission['event'] == 'latency_sprint'
    assert submission['normalized_scores']['speedup'] == 2.5
    assert submission['normalized_scores']['compression_ratio'] == 4.0
    assert 'system_info' in submission
    
    # Test validation
    validation = validate_submission(submission)
    assert validation['valid'] == True
    
    print("✅ Submission generation works correctly!")

test_unit_submission_generation()

# %% [markdown]
"""
## 4. Complete Workflow Example

Now let's see a complete example that demonstrates the full competition workflow from start to finish.
"""

# %% nbgrader={"grade": false, "grade_id": "complete-workflow", "solution": true}
def demonstrate_competition_workflow():
    """
    Complete competition workflow demonstration.
    
    This shows how to:
    1. Choose an event
    2. Measure baseline using Module 19's Benchmark
    3. Apply optimizations
    4. Measure optimized model
    5. Generate and validate submission
    """
    ### BEGIN SOLUTION
    print("🏅 TinyTorch Olympics - Complete Workflow Demonstration")
    print("=" * 70)
    
    # Step 1: Choose event
    event = OlympicEvent.LATENCY_SPRINT
    print(f"\n📋 Step 1: Chosen Event: {event.value.replace('_', ' ').title()}")
    
    # Step 2: Create mock baseline model (in real workflow, use your actual model)
    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)  # Simulate computation
            return np.random.rand(10)
    
    baseline_model = MockModel("baseline_cnn")
    
    # Step 3: Measure baseline using Benchmark from Module 19
    print("\n📊 Step 2: Measuring Baseline (using Module 19 Benchmark)...")
    benchmark = Benchmark([baseline_model], [{"name": "baseline"}])
    # In real workflow, this would run actual benchmarks
    baseline_metrics = {'latency': 45.2, 'memory': 12.4, 'accuracy': 0.85}
    print(f"   Baseline Latency: {baseline_metrics['latency']:.1f}ms")
    print(f"   Baseline Memory: {baseline_metrics['memory']:.2f}MB")
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy']:.1%}")
    
    # Step 4: Apply optimizations (Modules 14-18)
    print("\n🔧 Step 3: Applying Optimizations...")
    print("   - Quantization (INT8): 4x memory reduction")
    print("   - Pruning (60%): Additional compression")
    optimized_model = MockModel("optimized_cnn")
    optimized_metrics = {'latency': 22.1, 'memory': 1.24, 'accuracy': 0.835}
    print(f"   Optimized Latency: {optimized_metrics['latency']:.1f}ms")
    print(f"   Optimized Memory: {optimized_metrics['memory']:.2f}MB")
    print(f"   Optimized Accuracy: {optimized_metrics['accuracy']:.1%}")
    
    # Step 5: Measure optimized (using Benchmark again)
    print("\n📊 Step 4: Measuring Optimized Model (using Module 19 Benchmark)...")
    benchmark_opt = Benchmark([optimized_model], [{"name": "optimized"}])
    # Results already calculated above
    
    # Step 6: Generate submission
    print("\n📤 Step 5: Generating Submission...")
    submission = generate_submission(
        baseline_results=baseline_metrics,
        optimized_results=optimized_metrics,
        event=event,
        athlete_name="DemoUser",
        techniques=["quantization_int8", "magnitude_prune_0.6"]
    )
    
    # Step 7: Validate submission
    print("\n🔍 Step 6: Validating Submission...")
    validation = validate_submission(submission)
    
    for check in validation['checks']:
        print(f"   {check}")
    for warning in validation['warnings']:
        print(f"   {warning}")
    for error in validation['errors']:
        print(f"   {error}")
    
    if validation['valid']:
        print("\n✅ Submission is valid!")
        
        # Save submission
        output_file = Path("submission.json")
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        print(f"📄 Submission saved to: {output_file}")
        
        # Display normalized scores
        print("\n📊 Normalized Scores:")
        scores = submission['normalized_scores']
        print(f"   Speedup: {scores['speedup']:.2f}x faster ⚡")
        print(f"   Compression: {scores['compression_ratio']:.2f}x smaller 💾")
        print(f"   Accuracy Δ: {scores['accuracy_delta']:+.2f}pp")
        print(f"   Efficiency Score: {scores['efficiency_score']:.2f}")
    else:
        print("\n❌ Submission has errors - please fix before submitting")
    
    print("\n" + "=" * 70)
    print("🎉 Competition workflow demonstration complete!")
    ### END SOLUTION

demonstrate_competition_workflow()

# %% [markdown]
"""
## 5. Module Integration Test

Final comprehensive test validating the competition workflow works correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire competition module functionality.

    This final test runs before module summary to ensure:
    - OlympicEvent enum works correctly
    - calculate_normalized_scores computes correctly
    - generate_submission creates valid submissions
    - validate_submission checks requirements properly
    - Complete workflow demonstration executes
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 60)

    # Test 1: OlympicEvent enum
    print("🔬 Testing OlympicEvent enum...")
    assert OlympicEvent.LATENCY_SPRINT.value == "latency_sprint"
    assert OlympicEvent.MEMORY_CHALLENGE.value == "memory_challenge"
    assert OlympicEvent.ALL_AROUND.value == "all_around"
    print("  ✅ OlympicEvent enum works")

    # Test 2: Normalized scoring
    print("\n🔬 Testing normalized scoring...")
    baseline = {'latency': 100.0, 'memory': 12.0, 'accuracy': 0.85}
    optimized = {'latency': 40.0, 'memory': 3.0, 'accuracy': 0.83}
    scores = calculate_normalized_scores(baseline, optimized)
    assert abs(scores['speedup'] - 2.5) < 0.01
    assert abs(scores['compression_ratio'] - 4.0) < 0.01
    print("  ✅ Normalized scoring works")

    # Test 3: Submission generation
    print("\n🔬 Testing submission generation...")
    submission = generate_submission(
        baseline_results=baseline,
        optimized_results=optimized,
        event=OlympicEvent.LATENCY_SPRINT,
        athlete_name="TestUser"
    )
    assert submission['event'] == 'latency_sprint'
    assert 'normalized_scores' in submission
    assert 'system_info' in submission
    print("  ✅ Submission generation works")

    # Test 4: Submission validation
    print("\n🔬 Testing submission validation...")
    validation = validate_submission(submission)
    assert validation['valid'] == True
    assert len(validation['checks']) > 0
    print("  ✅ Submission validation works")

    # Test 5: Complete workflow
    print("\n🔬 Testing complete workflow...")
    demonstrate_competition_workflow()
    print("  ✅ Complete workflow works")

    print("\n" + "=" * 60)
    print("🎉 ALL COMPETITION MODULE TESTS PASSED!")
    print("✅ Competition workflow fully functional!")
    print("📊 Ready to generate submissions!")
    print("\nRun: tito module complete 20")

# Call the comprehensive test
test_module()

# %% nbgrader={"grade": false, "grade_id": "main_execution", "solution": false}
if __name__ == "__main__":
    print("🚀 Running TinyTorch Olympics Competition module...")

    # Run the comprehensive test
    test_module()

    print("\n✅ Competition module ready!")
    print("📤 Use generate_submission() to create your competition entry!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Competition Workflow Reflection

This capstone teaches the workflow of professional ML competitions. Let's reflect on the systems thinking behind competition participation.

### Question 1: Statistical Confidence
You use Module 19's Benchmark harness which runs multiple trials and reports confidence intervals.
If baseline latency is 50ms ± 5ms and optimized is 25ms ± 3ms, can you confidently claim improvement?

**Answer:** [Yes/No] _______

**Reasoning:** Consider whether confidence intervals overlap and what that means for statistical significance.

### Question 2: Event Selection Strategy
Different Olympic events have different constraints (Latency Sprint: accuracy ≥ 85%, Extreme Push: accuracy ≥ 80%).
If your optimization reduces accuracy from 87% to 82%, which events can you still compete in?

**Answer:** _______

**Reasoning:** Check which events' accuracy constraints you still meet.

### Question 3: Normalized Scoring
Normalized scores enable fair comparison across hardware. If Baseline A runs on fast GPU (10ms) and Baseline B runs on slow CPU (100ms), both optimized to 5ms:
- Which has better absolute time? _______
- Which has better speedup? _______
- Why does normalized scoring matter? _______

### Question 4: Submission Validation
Your validate_submission() function checks event constraints and flags unrealistic improvements.
If someone claims 100× speedup, what should the validation do?

**Answer:** _______

**Reasoning:** Consider how to balance catching errors vs allowing legitimate breakthroughs.

### Question 5: Workflow Integration
Module 20 uses Benchmark from Module 19 and optimization techniques from Modules 14-18.
What's the key insight about how these modules work together?

a) Each module is independent
b) Module 20 provides workflow that uses tools from other modules
c) You need to rebuild everything in Module 20
d) Competition is separate from benchmarking

**Answer:** _______

**Explanation:** Module 20 teaches workflow and packaging - you use existing tools, not rebuild them.
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: TinyTorch Olympics - Competition & Submission

Congratulations! You've completed the capstone module - learning how to participate in professional ML competitions!

### Key Accomplishments
- **Understood competition events** and how to choose the right event for your optimization goals
- **Used Benchmark harness** from Module 19 to measure performance with statistical rigor
- **Generated standardized submissions** following MLPerf-style format
- **Validated submissions** meet competition requirements
- **Demonstrated complete workflow** from measurement to submission
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **Competition workflow**: How professional ML competitions are structured and participated in
- **Submission packaging**: How to format results for fair comparison and validation
- **Event constraints**: How different events require different optimization strategies
- **Workflow integration**: How to use benchmarking tools (Module 19) + optimization techniques (Modules 14-18)

### The Complete Journey
```
Module 01-18: Build ML Framework
    ↓
Module 19: Learn Benchmarking Methodology
    ↓
Module 20: Learn Competition Workflow
    ↓
Milestone 05: Build TinyGPT (Historical Achievement)
    ↓
Milestone 06: Torch Olympics (Optimization Competition)
```

### Ready for Competition
Your competition workflow demonstrates:
- **Professional submission format** following industry standards (MLPerf-style)
- **Statistical rigor** using Benchmark harness from Module 19
- **Event understanding** knowing which optimizations fit which events
- **Validation mindset** ensuring submissions meet requirements before submitting

**Export with:** `tito module complete 20`

**Achievement Unlocked:** 🏅 **Competition Ready** - You know how to participate in professional ML competitions!

You now understand how ML competitions work - from measurement to submission. The benchmarking tools you built in Module 19 and the optimization techniques from Modules 14-18 come together in Module 20's competition workflow.

**What's Next:**
- Build TinyGPT in Milestone 05 (historical achievement)
- Compete in Torch Olympics (Milestone 06) using this workflow
- Use `tito olympics submit` to generate your competition entry!
"""
