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
# Module 20: Capstone - Benchmarking & Submission

Welcome to the TinyTorch capstone! You've built an entire ML framework from scratch across 19 modules. Now it's time to demonstrate your work by benchmarking a model and generating a submission that showcases your framework's capabilities.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with benchmarking tools (Module 19)
**You'll Build**: Benchmark submission workflow
**You'll Enable**: Shareable results demonstrating framework performance

**Connection Map**:
```
Modules 01-19 → Benchmarking (M19) → Submission (M20)
(Framework)     (Measurement)        (Results)
```

## Learning Objectives
By the end of this capstone, you will:
1. **Use** Module 19's benchmarking tools to measure model performance
2. **Apply** optimization techniques from Modules 14-18
3. **Generate** standardized JSON submissions
4. **Share** your results with the TinyTorch community

**Key Insight**: This module teaches you the complete workflow from model to measurable results - the foundation of ML systems engineering.
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `src/20_capstone/20_capstone.py`  
**Building Side:** Code exports to `tinytorch.capstone`

```python
# How to use this module:
from tinytorch.capstone import generate_submission, BenchmarkReport

# Benchmark your model
report = BenchmarkReport()
report.benchmark_model(my_model, X_test, y_test)

# Generate submission
submission = generate_submission(report)
submission.save("my_submission.json")
```

**Why this matters:**
- **Learning:** Complete workflow from model to shareable results
- **Production:** Professional submission format for benchmarking
- **Community:** Share and compare results with other builders
"""

# %% nbgrader={"grade": false, "grade_id": "exports", "solution": true}
#| default_exp capstone
#| export

# %% [markdown]
"""
## Introduction: From Framework to Results

Over the past 19 modules, you built a complete ML framework. Now we bring it all together by demonstrating that your framework works end-to-end and can produce measurable, shareable results.

This capstone shows you how to:
1. Benchmark a model using your framework
2. Apply optimizations and measure improvements
3. Generate a standardized submission JSON
4. Share your results

Let's get started!
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import platform
import sys

# Import TinyTorch modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import CrossEntropyLoss

print("✅ Capstone modules imported!")
print("📊 Ready to benchmark and submit results")

# %% [markdown]
"""
## Part 1: Building a Simple Benchmark Model

For this capstone, we'll use a simple MLP model. This keeps the focus on the benchmarking workflow rather than model complexity.

Students can later apply this same workflow to more complex models from milestones!
"""

# %% nbgrader={"grade": false, "grade_id": "toy-model", "solution": true}
#| export
class SimpleMLP:
    """
    Simple 2-layer MLP for benchmarking demonstration.
    
    This is a toy model to demonstrate the benchmarking workflow.
    Students can later apply the same workflow to milestone models.
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        """Initialize simple MLP with random weights."""
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

        # Initialize with small random weights
        # Linear layer expects weight shape: (in_features, out_features)
        self.fc1.weight.data = np.random.randn(input_size, hidden_size) * 0.01
        self.fc1.bias.data = np.zeros(hidden_size)
        self.fc2.weight.data = np.random.randn(hidden_size, output_size) * 0.01
        self.fc2.bias.data = np.zeros(output_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x
    
    def parameters(self):
        """Return model parameters for optimization."""
        return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]
    
    def count_parameters(self):
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total

print("✅ SimpleMLP model defined")

# %% [markdown]
"""
## Part 2: Benchmark Report Class

The BenchmarkReport class encapsulates all benchmark results and provides methods for measurement and reporting.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-report", "solution": true}
#| export
class BenchmarkReport:
    """
    Benchmark report for model performance.
    
    Measures and stores:
    - Model characteristics (parameters, size)
    - Performance metrics (accuracy, latency)
    - Optimization info (techniques applied)
    """
    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.metrics = {}
        self.system_info = self._get_system_info()
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    def _get_system_info(self):
        """Collect system information."""
        return {
            'platform': platform.platform(),
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__
        }
    
    def benchmark_model(self, model, X_test, y_test, num_runs=100):
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            X_test: Test inputs
            y_test: Test labels  
            num_runs: Number of inference runs for latency measurement
        """
        # Count parameters
        param_count = model.count_parameters()
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Assuming FP32
        
        # Measure accuracy
        predictions = model.forward(X_test)
        pred_labels = np.argmax(predictions.data, axis=1)
        accuracy = np.mean(pred_labels == y_test)
        
        # Measure latency (average over multiple runs)
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_test[:1])  # Single sample inference
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Store metrics
        self.metrics = {
            'parameter_count': int(param_count),
            'model_size_mb': float(model_size_mb),
            'accuracy': float(accuracy),
            'latency_ms_mean': float(avg_latency),
            'latency_ms_std': float(std_latency),
            'throughput_samples_per_sec': float(1000 / avg_latency)
        }
        
        print(f"\n📊 Benchmark Results for {self.model_name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
        
        return self.metrics

print("✅ BenchmarkReport class defined")

# %% [markdown]
"""
## Part 3: Submission Generation

The core function that generates a standardized JSON submission from benchmark results.
"""

# %% nbgrader={"grade": false, "grade_id": "generate-submission", "solution": true}
#| export
def generate_submission(
    baseline_report: BenchmarkReport,
    optimized_report: Optional[BenchmarkReport] = None,
    student_name: Optional[str] = None,
    techniques_applied: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a standardized benchmark submission.
    
    Args:
        baseline_report: Benchmark results for baseline model
        optimized_report: Optional benchmark results for optimized model
        student_name: Optional student/submitter name
        techniques_applied: List of optimization techniques used
    
    Returns:
        Dictionary containing submission data (ready for JSON export)
    """
    submission = {
        'tinytorch_version': '0.1.0',
        'submission_type': 'capstone_benchmark',
        'timestamp': baseline_report.timestamp,
        'system_info': baseline_report.system_info,
        'baseline': {
            'model_name': baseline_report.model_name,
            'metrics': baseline_report.metrics
        }
    }
    
    # Add student name if provided
    if student_name:
        submission['student_name'] = student_name
    
    # Add optimization results if provided
    if optimized_report:
        submission['optimized'] = {
            'model_name': optimized_report.model_name,
            'metrics': optimized_report.metrics,
            'techniques_applied': techniques_applied or []
        }
        
        # Calculate improvement metrics
        baseline_latency = baseline_report.metrics['latency_ms_mean']
        optimized_latency = optimized_report.metrics['latency_ms_mean']
        baseline_size = baseline_report.metrics['model_size_mb']
        optimized_size = optimized_report.metrics['model_size_mb']
        
        submission['improvements'] = {
            'speedup': float(baseline_latency / optimized_latency),
            'compression_ratio': float(baseline_size / optimized_size),
            'accuracy_delta': float(
                optimized_report.metrics['accuracy'] - baseline_report.metrics['accuracy']
            )
        }
    
    return submission

def save_submission(submission: Dict[str, Any], filepath: str = "submission.json"):
    """Save submission to JSON file."""
    Path(filepath).write_text(json.dumps(submission, indent=2))
    print(f"\n✅ Submission saved to: {filepath}")
    return filepath

print("✅ Submission generation functions defined")

# %% [markdown]
"""
## Part 4: Complete Example Workflow

This section demonstrates the complete workflow from model to submission.
Students can modify this to benchmark their own models!
"""

# %% nbgrader={"grade": false, "grade_id": "example-workflow", "solution": true}
def run_example_benchmark():
    """
    Complete example showing the full benchmarking workflow.
    
    Students can modify this to benchmark their own models!
    """
    print("="*70)
    print("TINYTORCH CAPSTONE: BENCHMARKING WORKFLOW EXAMPLE")
    print("="*70)
    
    # Step 1: Create toy dataset
    print("\n🔧 Step 1: Creating toy dataset...")
    np.random.seed(42)
    X_test = Tensor(np.random.randn(100, 10))
    y_test = np.random.randint(0, 3, 100)
    print(f"  Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features, 3 classes")
    
    # Step 2: Create baseline model
    print("\n🔧 Step 2: Creating baseline model...")
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    print(f"  Model: {baseline_model.count_parameters():,} parameters")
    
    # Step 3: Benchmark baseline
    print("\n📊 Step 3: Benchmarking baseline model...")
    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)
    
    # Step 4: Generate submission
    print("\n📝 Step 4: Generating submission...")
    submission = generate_submission(
        baseline_report=baseline_report,
        student_name="TinyTorch Student"
    )
    
    # Step 5: Save submission
    print("\n💾 Step 5: Saving submission...")
    save_submission(submission, "capstone_submission.json")
    
    print("\n" + "="*70)
    print("🎉 WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try optimizing the model (quantization, pruning, etc.)")
    print("  2. Benchmark the optimized version")
    print("  3. Generate a new submission with both baseline and optimized results")
    print("  4. Share your submission.json with the TinyTorch community!")
    
    return submission

print("✅ Example workflow defined")

# %% [markdown]
"""
## Part 5: Module Testing

The test_module() function validates that the complete workflow works correctly.
"""

# %% nbgrader={"grade": false, "grade_id": "test-module", "solution": true}
def test_module():
    """
    Test the capstone module functionality.
    
    Validates:
    - Model creation
    - Benchmarking
    - Submission generation
    - JSON export
    """
    print("\n" + "="*70)
    print("MODULE 20: CAPSTONE - TEST EXECUTION")
    print("="*70)
    
    # Test 1: Model creation
    print("\n🔬 Test 1: Model creation...")
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    assert model.count_parameters() > 0, "Model should have parameters"
    print("  ✅ Model created successfully")
    
    # Test 2: Benchmarking
    print("\n🔬 Test 2: Benchmarking...")
    np.random.seed(42)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)
    
    report = BenchmarkReport(model_name="test_model")
    metrics = report.benchmark_model(model, X_test, y_test, num_runs=10)
    
    assert 'accuracy' in metrics, "Should have accuracy metric"
    assert 'latency_ms_mean' in metrics, "Should have latency metric"
    assert 'parameter_count' in metrics, "Should have parameter count"
    print("  ✅ Benchmarking completed successfully")
    
    # Test 3: Submission generation
    print("\n🔬 Test 3: Submission generation...")
    submission = generate_submission(baseline_report=report)
    
    assert 'baseline' in submission, "Should have baseline results"
    assert 'timestamp' in submission, "Should have timestamp"
    assert 'system_info' in submission, "Should have system info"
    print("  ✅ Submission generated successfully")
    
    # Test 4: JSON export
    print("\n🔬 Test 4: JSON export...")
    test_file = "/tmp/test_submission.json"
    save_submission(submission, test_file)
    assert Path(test_file).exists(), "Submission file should exist"
    
    # Verify JSON is valid
    loaded = json.loads(Path(test_file).read_text())
    assert loaded['baseline']['metrics']['accuracy'] == metrics['accuracy']
    print("  ✅ JSON export successful")
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 20")
    print("="*70)

print("✅ Test module defined")

# %% [markdown]
"""
## Main Execution

When run as a script, this demonstrates the complete workflow.
"""

# %% nbgrader={"grade": false, "grade_id": "main", "solution": true}
if __name__ == "__main__":
    # Run the test module to validate everything works
    test_module()
