#!/usr/bin/env python3
"""
Create TinyDigits Dataset
=========================

Extracts a balanced, curated subset from sklearn's digits dataset (8x8 grayscale).
This creates a TinyTorch-branded educational dataset optimized for fast iteration.

Target sizes:
- Training: 150 samples (15 per digit class 0-9)
- Test: 47 samples (mix of clear and challenging examples)
"""

import numpy as np
import pickle
from pathlib import Path

def create_tinydigits():
    """Create TinyDigits train/test split from full digits dataset."""

    # Load the full sklearn digits dataset (shipped with repo)
    source_path = Path(__file__).parent.parent.parent / "milestones/03_1986_mlp/data/digits_8x8.npz"
    data = np.load(source_path)
    images = data['images']  # (1797, 8, 8)
    labels = data['labels']  # (1797,)

    print(f"📊 Source dataset: {images.shape[0]} samples")
    print(f"   Shape: {images.shape}, dtype: {images.dtype}")
    print(f"   Range: [{images.min():.3f}, {images.max():.3f}]")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create balanced splits
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # For each digit class (0-9)
    for digit in range(10):
        # Get all samples of this digit
        digit_indices = np.where(labels == digit)[0]
        digit_count = len(digit_indices)

        # Shuffle indices
        np.random.shuffle(digit_indices)

        # Split: 15 for training, rest for test pool
        train_count = 15
        test_pool = digit_indices[train_count:]

        # Training: First 15 samples
        train_images.append(images[digit_indices[:train_count]])
        train_labels.extend([digit] * train_count)

        # Test: Select 4-5 samples from remaining (47 total across all digits)
        test_count = 5 if digit < 7 else 4  # 7*5 + 3*4 = 47
        test_indices = np.random.choice(test_pool, size=test_count, replace=False)
        test_images.append(images[test_indices])
        test_labels.extend([digit] * test_count)

        print(f"   Digit {digit}: {train_count} train, {test_count} test (from {digit_count} total)")

    # Stack into arrays
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_images = np.vstack(test_images)
    test_labels = np.array(test_labels, dtype=np.int64)

    # Shuffle both sets
    train_shuffle = np.random.permutation(len(train_images))
    train_images = train_images[train_shuffle]
    train_labels = train_labels[train_shuffle]

    test_shuffle = np.random.permutation(len(test_images))
    test_images = test_images[test_shuffle]
    test_labels = test_labels[test_shuffle]

    print(f"\n✅ Created TinyDigits:")
    print(f"   Training: {train_images.shape} images, {train_labels.shape} labels")
    print(f"   Test: {test_images.shape} images, {test_labels.shape} labels")

    # Save as pickle files
    output_dir = Path(__file__).parent

    train_data = {'images': train_images, 'labels': train_labels}
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\n💾 Saved: train.pkl")

    test_data = {'images': test_images, 'labels': test_labels}
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print(f"💾 Saved: test.pkl")

    # Calculate file sizes
    train_size = (output_dir / 'train.pkl').stat().st_size / 1024
    test_size = (output_dir / 'test.pkl').stat().st_size / 1024
    total_size = train_size + test_size

    print(f"\n📦 File sizes:")
    print(f"   train.pkl: {train_size:.1f} KB")
    print(f"   test.pkl: {test_size:.1f} KB")
    print(f"   Total: {total_size:.1f} KB")

    print(f"\n🎯 TinyDigits created successfully!")
    print(f"   Perfect for TinyTorch on RasPi0 - only {total_size:.1f} KB!")

if __name__ == "__main__":
    create_tinydigits()
