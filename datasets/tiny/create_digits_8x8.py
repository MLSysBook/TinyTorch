#!/usr/bin/env python3
"""
Create 8x8 Digits Dataset
=========================

Extracts the 8×8 handwritten digits dataset from sklearn and saves it
as a compact .npz file for TinyTorch.

Source: UCI Machine Learning Repository
Used by: sklearn.datasets.load_digits()
Size: 1,797 samples, 8×8 grayscale images
License: Public domain
"""

import numpy as np

try:
    from sklearn.datasets import load_digits
except ImportError:
    print("❌ sklearn not installed. Install with: pip install scikit-learn")
    exit(1)

print("📥 Loading 8×8 digits from sklearn...")
digits = load_digits()

print(f"✅ Loaded {len(digits.images)} digit images")
print(f"   Shape: {digits.images.shape}")
print(f"   Classes: {np.unique(digits.target)}")

# Normalize to [0, 1] range (original is 0-16)
images_normalized = digits.images.astype(np.float32) / 16.0
labels = digits.target.astype(np.int64)

# Save as compressed .npz
output_file = 'digits_8x8.npz'
np.savez_compressed(output_file,
                    images=images_normalized,
                    labels=labels)

# Check file size
import os
file_size_kb = os.path.getsize(output_file) / 1024
print(f"\n💾 Saved to {output_file}")
print(f"   File size: {file_size_kb:.1f} KB")
print(f"   Images shape: {images_normalized.shape}")
print(f"   Labels shape: {labels.shape}")
print(f"   Value range: [{images_normalized.min():.2f}, {images_normalized.max():.2f}]")

# Quick verification
print(f"\n✅ Dataset ready for TinyTorch!")
print(f"   Total samples: {len(images_normalized)}")
print(f"   Samples per class: ~{len(images_normalized) // 10}")
print(f"   Perfect for DataLoader demos!")
