# Tiny Datasets for TinyTorch

**Small, curated datasets that ship with TinyTorch** - no downloads required!

These datasets are committed to the repository for instant, offline-friendly learning.

---

## 📊 Available Datasets

### 8×8 Handwritten Digits

**File:** `digits_8x8.npz`  
**Size:** ~67 KB  
**Samples:** 1,797 images  
**Shape:** (8, 8) grayscale  
**Classes:** 10 digits (0-9)  
**Source:** UCI ML Repository via sklearn

**Perfect for:**
- Learning DataLoader mechanics
- Quick CNN testing
- Offline development
- Educational demos

**Usage:**
```python
import numpy as np
from tinytorch import Tensor
from tinytorch.data.loader import TensorDataset, DataLoader

# Load the dataset
data = np.load('datasets/tiny/digits_8x8.npz')
images = Tensor(data['images'])
labels = Tensor(data['labels'])

# Create dataset and loader
dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_images, batch_labels in loader:
    print(f"Batch: {batch_images.shape}, Labels: {batch_labels.shape}")
```

**Visual Sample:**
```
Digit "5":        Digit "3":        Digit "8":
░░████░░         ░█████░          ░█████░░
░█░░░█░          ░░░░░█░          █░░░░░█░
░░░░█░░          ░░███░░          ░█████░░
░░░█░░░          ░░░░░█░          █░░░░░█░
░░█░░░░          ░░████░░          ░█████░░
```

---

## 🎯 Philosophy

**Why ship tiny datasets?**

1. **Zero friction** - Students start learning immediately
2. **Offline-first** - Works in classrooms, planes, anywhere
3. **Fast iteration** - No wait times, instant feedback
4. **Educational focus** - Sized for learning, not production

**Progression:**
- **Tiny datasets** (here) → Learn DataLoader mechanics
- **Downloaded datasets** (../mnist/, ../cifar10/) → Real applications
- **Custom datasets** → Production skills

---

## 📂 File Format

All datasets use NumPy's `.npz` format (compressed):

```python
data = np.load('dataset.npz')
images = data['images']  # Shape: (N, H, W) or (N, H, W, C)
labels = data['labels']  # Shape: (N,)
```

**Benefits:**
- Fast loading
- Compressed storage
- Python-native
- Easy inspection

---

## 🔧 Creating New Tiny Datasets

See `create_digits_8x8.py` for example extraction script.

**Guidelines:**
- Max size: ~100 KB per dataset
- Format: `.npz` with `images` and `labels` keys
- Normalize: Images in [0, 1] range
- License: Verify public domain / open source

---

## 📚 Dataset Information

### Digits 8×8 Credits

**Original Source:** 
- E. Alpaydin, C. Kaynak (1998)
- UCI Machine Learning Repository
- "Optical Recognition of Handwritten Digits"

**Preprocessing:**
- Extracted via `sklearn.datasets.load_digits()`
- Normalized from [0-16] to [0-1]
- Saved as float32 for efficiency

**License:** Public domain

---

## 🚀 Next Steps

After mastering DataLoader with tiny datasets:

1. **Module 08** → Build DataLoader with digits_8x8
2. **Milestone 03** → Train MLP on full MNIST
3. **Milestone 04** → Train CNN on CIFAR-10
4. **Custom datasets** → Apply to your own data

Tiny datasets teach the mechanics.  
Real datasets teach the systems.  
Custom datasets teach the engineering.
