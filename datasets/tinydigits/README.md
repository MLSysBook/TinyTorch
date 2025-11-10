# TinyDigits Dataset

A curated subset of the sklearn digits dataset for rapid ML prototyping and educational demonstrations.

## Contents

- **Training**: 150 samples (15 per digit, 0-9)
- **Test**: 47 samples (balanced across digits)
- **Format**: 8×8 grayscale images, float32 normalized [0, 1]
- **Size**: ~51 KB total (vs 67 KB original, 10 MB MNIST)

## Files

```
datasets/tinydigits/
├── train.pkl  # {'images': (150, 8, 8), 'labels': (150,)}
└── test.pkl   # {'images': (47, 8, 8), 'labels': (47,)}
```

## Usage

```python
import pickle

# Load training data
with open('datasets/tinydigits/train.pkl', 'rb') as f:
    data = pickle.load(f)
    train_images = data['images']  # (150, 8, 8)
    train_labels = data['labels']  # (150,)

# Load test data
with open('datasets/tinydigits/test.pkl', 'rb') as f:
    data = pickle.load(f)
    test_images = data['images']   # (47, 8, 8)
    test_labels = data['labels']   # (47,)
```

## Purpose

**Educational Infrastructure**: Designed for teaching ML systems with real data at edge-device scale.

- Fast iteration during development (<5 sec training)
- Instant "it works!" moment for students
- Offline-capable demos (no downloads)
- CI/CD friendly (lightweight tests)
- **Deployable on RasPi0** - tiny footprint for democratizing ML education

## Curation Process

Created from the sklearn digits dataset (8×8 downsampled MNIST):

1. **Balanced Sampling**: 15 training samples per digit class (150 total)
2. **Test Split**: 4-5 samples per digit (47 total) from remaining examples
3. **Random Seeding**: Reproducible selection (seed=42)
4. **Shuffled**: Training and test sets randomly shuffled for fair evaluation

The sklearn digits dataset itself is derived from the UCI ML hand-written digits datasets.

## Why TinyDigits vs Full MNIST?

| Metric | MNIST | TinyDigits | Benefit |
|--------|-------|------------|---------|
| Samples | 60,000 | 150 | 400× fewer samples |
| File size | 10 MB | 51 KB | 200× smaller |
| Train time | 5-10 min | <5 sec | 60-120× faster |
| Download | Network required | Ships with repo | Always available |
| Resolution | 28×28 (784 pixels) | 8×8 (64 pixels) | Faster forward pass |
| Edge deployment | Challenging | Perfect | Works on RasPi0 |

## Educational Progression

TinyDigits serves as the first step in a scaffolded learning path:

1. **TinyDigits (8×8)** ← Start here: Learn MLP/CNN basics with instant feedback
2. **Full MNIST (28×28)** ← Graduate to: Standard benchmark, longer training
3. **CIFAR-10 (32×32 RGB)** ← Advanced: Color images, real-world complexity

## Citation

TinyDigits is curated from the sklearn digits dataset for educational use in TinyTorch.

**Original Source**:
- sklearn.datasets.load_digits()
- Derived from UCI ML hand-written digits datasets
- License: BSD 3-Clause (sklearn)

**TinyTorch Curation**:
```bibtex
@misc{tinydigits2025,
  title={TinyDigits: Curated Educational Dataset for ML Systems Learning},
  author={TinyTorch Project},
  year={2025},
  note={Balanced subset of sklearn digits optimized for edge deployment}
}
```

## Generation

To regenerate this dataset from the original sklearn data:

```bash
python3 datasets/tinydigits/create_tinydigits.py
```

This ensures reproducibility and allows customization for specific educational needs.

## License

See [LICENSE](LICENSE) for details. TinyDigits inherits the BSD 3-Clause license from sklearn.
