# 🚀 TinyTorch Capability Showcase

**"Look what you built!" moments for students**

This directory contains showcase files that demonstrate what students have accomplished after completing each module. These are not exercises - they're celebrations of achievement!

## How to Use

After completing a module, run the corresponding showcase file to see your implementation in action:

```bash
# Method 1: Direct execution
python capabilities/01_tensor_operations.py
python capabilities/02_neural_intelligence.py
python capabilities/03_forward_inference.py
# ... and so on

# Method 2: Using tito (if available)
tito demo capability 01
tito demo capability 02
tito demo capability 03
```

Or run all available showcases:
```bash
# Run all showcases you've unlocked
for f in capabilities/*.py; do echo "Running $f"; python "$f"; echo; done
```

## Philosophy

These showcases follow the "Look what you built!" philosophy:
- **No additional coding required** - Just run and watch
- **Uses only your TinyTorch code** - Demonstrates your actual implementations
- **Visually impressive** - Rich terminal output with colors and animations
- **Achievement celebration** - Makes progress tangible and exciting
- **Quick and satisfying** - 30 seconds to 2 minutes of pure awesomeness

## Showcase Files

| File | After Module | What It Shows |
|------|-------------|---------------|
| `01_tensor_operations.py` | 02 (Tensor) | Matrix operations with ASCII visualization |
| `02_neural_intelligence.py` | 03 (Activations) | How activations create intelligence |
| `03_forward_inference.py` | 05 (Dense) | Real digit recognition with your network |
| `04_image_processing.py` | 06 (Spatial) | Convolution edge detection |
| `05_attention_visualization.py` | 07 (Attention) | Attention heatmaps |
| `06_data_pipeline.py` | 09 (DataLoader) | Real CIFAR-10 data loading |
| `07_full_training.py` | 11 (Training) | Live CNN training with progress bars |
| `08_model_compression.py` | 12 (Compression) | Model size optimization |
| `09_performance_profiling.py` | 14 (Benchmarking) | System performance analysis |
| `10_production_systems.py` | 15 (MLOps) | Production deployment simulation |
| `11_tinygpt_mastery.py` | 16 (TinyGPT) | Your GPT generating text! |

## Dependencies

Each showcase file imports only from your TinyTorch implementation:
```python
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU
# etc.
```

Plus Rich for beautiful terminal output:
```python
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
```

## Sample Weights and Data

The `weights/` and `data/` directories contain:
- Pre-trained weights for demo models
- Sample data for quick showcase runs
- All files are small and optimized for fast loading

## Making Your Own Showcases

Want to create more capability showcases? Follow these guidelines:

1. **Import only from tinytorch** - Use what they built
2. **Make it visual** - Use Rich for colors, progress bars, ASCII art
3. **Keep it short** - 30 seconds to 2 minutes max
4. **Celebrate achievement** - End with congratulations
5. **No user input required** - Just run and watch

Example template:
```python
from rich.console import Console
from rich.panel import Panel
from tinytorch.core.tensor import Tensor

console = Console()

def main():
    console.print(Panel.fit("🚀 YOUR CAPABILITY SHOWCASE", style="bold magenta"))
    
    # Show something impressive with their code
    tensor = Tensor([[1, 2], [3, 4]])
    result = tensor @ tensor  # Uses their implementation!
    
    console.print(f"✨ Result: {result}")
    console.print("\n🎉 YOU BUILT THIS! Amazing work!")

if __name__ == "__main__":
    main()
```

---

**Remember**: These showcases exist to make your learning journey tangible and exciting. Each one proves that you're building real, working ML systems from scratch!