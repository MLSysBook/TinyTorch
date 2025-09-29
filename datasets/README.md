# TinyTorch Datasets

This directory contains datasets for TinyTorch examples and training.

## MNIST Dataset

The `mnist/` directory should contain the MNIST or Fashion-MNIST dataset files:
- `train-images-idx3-ubyte.gz` - Training images (60,000 samples)
- `train-labels-idx1-ubyte.gz` - Training labels
- `t10k-images-idx3-ubyte.gz` - Test images (10,000 samples)
- `t10k-labels-idx1-ubyte.gz` - Test labels

### Downloading the Dataset

Run the provided download script:
```bash
cd datasets
python download_mnist.py
```

This will download Fashion-MNIST (which has the same format as MNIST but is more accessible).

### Dataset Format

Both MNIST and Fashion-MNIST use the same IDX file format:
- Images: 28x28 grayscale pixels
- Labels: Integer values 0-9
- Gzipped for compression

Fashion-MNIST classes:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

The examples will work with either original MNIST digits or Fashion-MNIST items.