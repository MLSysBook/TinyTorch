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
## Part 4b: Advanced Workflow - Using TinyTorch Optimization APIs

This section demonstrates using the complete optimization pipeline from Modules 14-19:
- Module 14 (Profiling): Measure baseline performance
- Module 15 (Quantization): Reduce precision
- Module 16 (Compression): Prune weights
- Module 19 (Benchmarking): Professional measurement

This is the COMPLETE story: Profile → Optimize → Benchmark → Submit
"""

# %% nbgrader={"grade": false, "grade_id": "optimization-workflow", "solution": true}
def run_optimization_workflow_example():
    """
    Advanced example showing the complete optimization workflow.

    This demonstrates:
    1. Profiling baseline model (Module 14)
    2. Applying optimizations (Modules 15, 16)
    3. Benchmarking with best practices (Module 19)
    4. Generating submission with before/after comparison

    Students learn how to use TinyTorch as a complete framework!
    """
    print("="*70)
    print("TINYTORCH CAPSTONE: OPTIMIZATION WORKFLOW")
    print("="*70)
    print("\nThis workflow demonstrates using Modules 14-19 together:")
    print("  📊 Module 14: Profiling")
    print("  🔢 Module 15: Quantization (optional - API imported for demonstration)")
    print("  ✂️  Module 16: Compression (optional - API imported for demonstration)")
    print("  ⚡ Module 18: Acceleration (optional - API imported for demonstration)")
    print("  📈 Module 19: Benchmarking")
    print("  📝 Module 20: Submission Generation")

    # Demonstrate API imports (students can use these for their own optimizations)
    print("\n🔧 Importing optimization APIs...")
    try:
        from tinytorch.profiling.profiler import Profiler, quick_profile
        print("  ✅ Module 14 (Profiling) imported")
    except ImportError:
        print("  ⚠️  Module 14 (Profiling) not available - using basic profiling")
        Profiler = None

    try:
        from tinytorch.optimization.compression import magnitude_prune, structured_prune
        print("  ✅ Module 16 (Compression) imported")
    except ImportError:
        print("  ⚠️  Module 16 (Compression) not available - skipping pruning demo")
        magnitude_prune = None

    try:
        from tinytorch.benchmarking import Benchmark, BenchmarkResult
        print("  ✅ Module 19 (Benchmarking) imported")
    except ImportError:
        print("  ⚠️  Module 19 (Benchmarking) not available - using basic benchmarking")
        Benchmark = None

    # Step 1: Create dataset
    print("\n" + "="*70)
    print("STEP 1: Create Test Dataset")
    print("="*70)
    np.random.seed(42)
    X_test = Tensor(np.random.randn(100, 10))
    y_test = np.random.randint(0, 3, 100)
    print(f"  Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features, 3 classes")

    # Step 2: Create and profile baseline model
    print("\n" + "="*70)
    print("STEP 2: Baseline Model - Profile & Benchmark")
    print("="*70)
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    print(f"  Model: {baseline_model.count_parameters():,} parameters")

    # Benchmark baseline using BenchmarkReport
    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_metrics = baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

    # Optional: Demonstrate using Module 14's Profiler if available
    if Profiler:
        print("\n  📊 Optional: Using Module 14's Profiler for detailed analysis...")
        profiler = Profiler()
        # Note: Profiler integration would go here
        # This demonstrates the API is available for students to use

    # Step 3: (DEMO ONLY) Show optimization APIs available
    print("\n" + "="*70)
    print("STEP 3: Optimization APIs Available (Demo)")
    print("="*70)
    print("\n  📚 Students can apply these optimizations:")
    print("     - Module 15: quantize_model(model, bits=8)")
    print("     - Module 16: magnitude_prune(model, sparsity=0.5)")
    print("     - Module 17: enable_kv_cache(model)  # For transformers")
    print("     - Module 18: Use accelerated ops (vectorized_matmul, etc.)")
    print("\n  💡 For this demo, we'll simulate an optimized model")
    print("     (Students can replace this with real optimizations!)")

    # Create "optimized" model (students would apply real optimizations here)
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)  # Smaller for demo
    optimized_report = BenchmarkReport(model_name="optimized_mlp")
    optimized_metrics = optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=50)

    # Step 4: Generate submission with before/after comparison
    print("\n" + "="*70)
    print("STEP 4: Generate Submission with Improvements")
    print("="*70)

    submission = generate_submission(
        baseline_report=baseline_report,
        optimized_report=optimized_report,
        student_name="TinyTorch Optimizer",
        techniques_applied=["model_sizing", "architecture_search"]  # Students list real techniques
    )

    # Display improvement summary
    if 'improvements' in submission:
        improvements = submission['improvements']
        print("\n  📈 Optimization Results:")
        print(f"     Speedup: {improvements['speedup']:.2f}x")
        print(f"     Compression: {improvements['compression_ratio']:.2f}x")
        print(f"     Accuracy change: {improvements['accuracy_delta']*100:+.1f}%")

    # Step 5: Save submission
    print("\n" + "="*70)
    print("STEP 5: Save Submission")
    print("="*70)
    filepath = save_submission(submission, "optimization_submission.json")

    print("\n" + "="*70)
    print("🎉 OPTIMIZATION WORKFLOW COMPLETE!")
    print("="*70)
    print("\n📚 What students learned:")
    print("  ✅ How to import and use optimization APIs from Modules 14-19")
    print("  ✅ How to benchmark before and after optimization")
    print("  ✅ How to generate professional submissions with improvement metrics")
    print("  ✅ How TinyTorch modules work together as a complete framework")
    print("\n💡 Next steps:")
    print("  - Apply real optimizations (quantization, pruning, etc.)")
    print("  - Benchmark milestone models (XOR, MNIST, CNN, etc.)")
    print("  - Share your optimized results with the community!")

    return submission

print("✅ Optimization workflow example defined")

# %% [markdown]
"""
## Part 5: Module Testing

Individual unit tests for each component, following TinyTorch testing patterns.
"""

# %% nbgrader={"grade": false, "grade_id": "test-simple-mlp", "solution": true}
def test_unit_simple_mlp():
    """Test SimpleMLP model creation and forward pass."""
    print("🔬 Unit Test: SimpleMLP...")

    # Test model creation with default parameters
    model = SimpleMLP()
    assert model is not None, "Model should be created"

    # Test with custom parameters
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)

    # Test parameter count
    param_count = model.count_parameters()
    expected_params = (10 * 20 + 20) + (20 * 3 + 3)  # fc1 + fc2
    assert param_count == expected_params, f"Expected {expected_params} parameters, got {param_count}"

    # Test forward pass
    np.random.seed(42)
    X = Tensor(np.random.randn(5, 10))  # 5 samples, 10 features
    output = model.forward(X)

    assert output.shape == (5, 3), f"Expected output shape (5, 3), got {output.shape}"
    assert not np.isnan(output.data).any(), "Output should not contain NaN values"

    print("✅ SimpleMLP works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_simple_mlp()

# %% nbgrader={"grade": false, "grade_id": "test-benchmark-report", "solution": true}
def test_unit_benchmark_report():
    """Test BenchmarkReport class functionality."""
    print("🔬 Unit Test: BenchmarkReport...")

    # Create report
    report = BenchmarkReport(model_name="test_model")

    # Check initialization
    assert report.model_name == "test_model", "Model name should be set correctly"
    assert report.timestamp is not None, "Timestamp should be set"
    assert report.system_info is not None, "System info should be collected"
    assert 'platform' in report.system_info, "Should have platform info"
    assert 'python_version' in report.system_info, "Should have Python version"

    # Create test data
    np.random.seed(42)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)

    # Benchmark model
    metrics = report.benchmark_model(model, X_test, y_test, num_runs=10)

    # Check metrics exist
    required_metrics = [
        'parameter_count', 'model_size_mb', 'accuracy',
        'latency_ms_mean', 'latency_ms_std', 'throughput_samples_per_sec'
    ]
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    # Check metric types and ranges
    assert isinstance(metrics['parameter_count'], int), "Parameter count should be int"
    assert metrics['parameter_count'] > 0, "Should have positive parameter count"
    assert metrics['model_size_mb'] > 0, "Model size should be positive"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
    assert metrics['latency_ms_mean'] > 0, "Latency should be positive"
    assert metrics['latency_ms_std'] >= 0, "Standard deviation should be non-negative"
    assert metrics['throughput_samples_per_sec'] > 0, "Throughput should be positive"

    print("✅ BenchmarkReport works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_benchmark_report()

# %% nbgrader={"grade": false, "grade_id": "test-submission-generation", "solution": true}
def test_unit_submission_generation():
    """Test generate_submission() function."""
    print("🔬 Unit Test: Submission Generation...")

    # Create baseline report
    np.random.seed(42)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)

    baseline_report = BenchmarkReport(model_name="baseline_model")
    baseline_report.benchmark_model(model, X_test, y_test, num_runs=10)

    # Generate submission with baseline only
    submission = generate_submission(baseline_report)

    # Check submission structure
    assert isinstance(submission, dict), "Submission should be a dictionary"
    assert 'tinytorch_version' in submission, "Should have version field"
    assert 'submission_type' in submission, "Should have submission type"
    assert 'timestamp' in submission, "Should have timestamp"
    assert 'system_info' in submission, "Should have system info"
    assert 'baseline' in submission, "Should have baseline results"

    # Check baseline structure
    baseline = submission['baseline']
    assert 'model_name' in baseline, "Baseline should have model name"
    assert 'metrics' in baseline, "Baseline should have metrics"
    assert baseline['model_name'] == "baseline_model", "Model name should match"

    # Test with student name
    submission_with_name = generate_submission(baseline_report, student_name="Test Student")
    assert 'student_name' in submission_with_name, "Should include student name when provided"
    assert submission_with_name['student_name'] == "Test Student", "Student name should match"

    print("✅ Submission generation works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_submission_generation()

# %% nbgrader={"grade": false, "grade_id": "test-submission-schema", "solution": true}
def validate_submission_schema(submission: Dict[str, Any]) -> bool:
    """Validate submission JSON conforms to required schema."""
    # Check required top-level fields
    required_fields = ['tinytorch_version', 'submission_type', 'timestamp', 'system_info', 'baseline']
    for field in required_fields:
        if field not in submission:
            raise AssertionError(f"Missing required field: {field}")

    # Check field types
    assert isinstance(submission['tinytorch_version'], str), "Version should be string"
    assert isinstance(submission['submission_type'], str), "Submission type should be string"
    assert isinstance(submission['timestamp'], str), "Timestamp should be string"
    assert isinstance(submission['system_info'], dict), "System info should be dict"
    assert isinstance(submission['baseline'], dict), "Baseline should be dict"

    # Check baseline structure
    baseline = submission['baseline']
    assert 'model_name' in baseline, "Baseline missing model_name"
    assert 'metrics' in baseline, "Baseline missing metrics"

    # Check metrics structure and types
    metrics = baseline['metrics']
    required_metrics = ['parameter_count', 'model_size_mb', 'accuracy', 'latency_ms_mean']
    for metric in required_metrics:
        if metric not in metrics:
            raise AssertionError(f"Missing metric in baseline: {metric}")

    # Check metric value ranges
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy must be in [0, 1]"
    assert metrics['parameter_count'] > 0, "Parameter count must be positive"
    assert metrics['model_size_mb'] > 0, "Model size must be positive"
    assert metrics['latency_ms_mean'] > 0, "Latency must be positive"

    # Check system info
    system_info = submission['system_info']
    assert 'platform' in system_info, "System info missing platform"
    assert 'python_version' in system_info, "System info missing python_version"

    return True

def test_unit_submission_schema():
    """Test submission schema validation."""
    print("🔬 Unit Test: Submission Schema...")

    # Create valid submission
    np.random.seed(42)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)

    report = BenchmarkReport(model_name="test_model")
    report.benchmark_model(model, X_test, y_test, num_runs=10)

    submission = generate_submission(report)

    # Validate schema
    assert validate_submission_schema(submission), "Submission should pass schema validation"

    # Test with optimized results
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)
    optimized_report = BenchmarkReport(model_name="optimized_model")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=10)

    submission_with_opt = generate_submission(
        report,
        optimized_report,
        techniques_applied=["pruning"]
    )

    # Validate optimized submission
    assert validate_submission_schema(submission_with_opt), "Optimized submission should pass validation"
    assert 'optimized' in submission_with_opt, "Should have optimized section"
    assert 'improvements' in submission_with_opt, "Should have improvements section"

    print("✅ Submission schema validation works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_submission_schema()

# %% nbgrader={"grade": false, "grade_id": "test-submission-with-optimization", "solution": true}
def test_unit_submission_with_optimization():
    """Test submission with baseline + optimized comparison."""
    print("🔬 Unit Test: Submission with Optimization...")

    # Create baseline
    np.random.seed(42)
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)

    baseline_report = BenchmarkReport(model_name="baseline")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=10)

    # Create optimized version (smaller model for demo)
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)
    optimized_report = BenchmarkReport(model_name="optimized")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=10)

    # Generate submission with both
    techniques = ["model_sizing", "pruning"]
    submission = generate_submission(
        baseline_report,
        optimized_report,
        student_name="Test Student",
        techniques_applied=techniques
    )

    # Check optimized section exists
    assert 'optimized' in submission, "Should have optimized section"
    optimized = submission['optimized']
    assert 'model_name' in optimized, "Optimized section should have model name"
    assert 'metrics' in optimized, "Optimized section should have metrics"
    assert 'techniques_applied' in optimized, "Should have techniques list"
    assert optimized['techniques_applied'] == techniques, "Techniques should match"

    # Check improvements section
    assert 'improvements' in submission, "Should have improvements section"
    improvements = submission['improvements']
    assert 'speedup' in improvements, "Should have speedup metric"
    assert 'compression_ratio' in improvements, "Should have compression ratio"
    assert 'accuracy_delta' in improvements, "Should have accuracy delta"

    # Check improvement values are reasonable
    assert improvements['speedup'] > 0, "Speedup should be positive"
    assert improvements['compression_ratio'] > 0, "Compression ratio should be positive"
    assert -1 <= improvements['accuracy_delta'] <= 1, "Accuracy delta should be in [-1, 1]"

    print("✅ Submission with optimization works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_submission_with_optimization()

# %% nbgrader={"grade": false, "grade_id": "test-improvements-calculation", "solution": true}
def test_unit_improvements_calculation():
    """Test speedup/compression/accuracy calculations are correct."""
    print("🔬 Unit Test: Improvements Calculation...")

    # Create baseline with known metrics
    baseline_report = BenchmarkReport(model_name="baseline")
    baseline_report.metrics = {
        'parameter_count': 1000,
        'model_size_mb': 4.0,
        'accuracy': 0.80,
        'latency_ms_mean': 10.0,
        'latency_ms_std': 1.0,
        'throughput_samples_per_sec': 100.0
    }
    baseline_report.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    baseline_report.system_info = {'platform': 'test', 'python_version': '3.9', 'numpy_version': '1.20'}

    # Create optimized with 2x speedup, 2x compression, 5% accuracy loss
    optimized_report = BenchmarkReport(model_name="optimized")
    optimized_report.metrics = {
        'parameter_count': 500,
        'model_size_mb': 2.0,
        'accuracy': 0.75,
        'latency_ms_mean': 5.0,
        'latency_ms_std': 0.5,
        'throughput_samples_per_sec': 200.0
    }
    optimized_report.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    optimized_report.system_info = baseline_report.system_info

    # Generate submission
    submission = generate_submission(baseline_report, optimized_report)

    improvements = submission['improvements']

    # Verify calculations
    # Speedup = baseline_latency / optimized_latency = 10.0 / 5.0 = 2.0
    assert abs(improvements['speedup'] - 2.0) < 0.01, f"Expected speedup 2.0, got {improvements['speedup']}"

    # Compression = baseline_size / optimized_size = 4.0 / 2.0 = 2.0
    assert abs(improvements['compression_ratio'] - 2.0) < 0.01, f"Expected compression 2.0, got {improvements['compression_ratio']}"

    # Accuracy delta = 0.75 - 0.80 = -0.05
    assert abs(improvements['accuracy_delta'] - (-0.05)) < 0.001, f"Expected accuracy delta -0.05, got {improvements['accuracy_delta']}"

    print("✅ Improvements calculation is correct!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_improvements_calculation()

# %% nbgrader={"grade": false, "grade_id": "test-json-serialization", "solution": true}
def test_unit_json_serialization():
    """Test save_submission() creates valid JSON files."""
    print("🔬 Unit Test: JSON Serialization...")

    # Create submission
    np.random.seed(42)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(np.random.randn(50, 10))
    y_test = np.random.randint(0, 3, 50)

    report = BenchmarkReport(model_name="test_model")
    report.benchmark_model(model, X_test, y_test, num_runs=10)

    submission = generate_submission(report, student_name="Test Student")

    # Save to file
    test_file = "/tmp/test_submission_unit.json"
    filepath = save_submission(submission, test_file)

    # Check file exists
    assert Path(filepath).exists(), "Submission file should exist"

    # Load and verify JSON is valid
    loaded_json = json.loads(Path(test_file).read_text())

    # Verify structure is preserved
    assert loaded_json['tinytorch_version'] == submission['tinytorch_version'], "Version should match"
    assert loaded_json['student_name'] == submission['student_name'], "Student name should match"
    assert loaded_json['baseline']['model_name'] == submission['baseline']['model_name'], "Model name should match"

    # Verify metrics are preserved
    baseline_metrics = loaded_json['baseline']['metrics']
    original_metrics = submission['baseline']['metrics']
    assert baseline_metrics['accuracy'] == original_metrics['accuracy'], "Accuracy should match"
    assert baseline_metrics['parameter_count'] == original_metrics['parameter_count'], "Parameter count should match"

    # Verify JSON can be dumped again (round-trip test)
    round_trip = json.dumps(loaded_json, indent=2)
    assert len(round_trip) > 0, "JSON should serialize again"

    # Clean up
    Path(test_file).unlink()

    print("✅ JSON serialization works correctly!")

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_json_serialization()

# %% nbgrader={"grade": false, "grade_id": "test-module", "solution": true}
def test_module():
    """
    Test Module 20: Capstone submission infrastructure.

    Runs all unit tests to validate complete functionality.
    """
    print("\n" + "="*70)
    print("MODULE 20: CAPSTONE - UNIT TESTS")
    print("="*70)

    test_unit_simple_mlp()
    test_unit_benchmark_report()
    test_unit_submission_generation()
    test_unit_submission_schema()
    test_unit_submission_with_optimization()
    test_unit_improvements_calculation()
    test_unit_json_serialization()

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\nModule 20 validation complete!")
    print("Run: tito module complete 20")

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
