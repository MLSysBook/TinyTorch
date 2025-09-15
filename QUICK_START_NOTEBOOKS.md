# TinyTorch: Quick Start for Interactive Notebooks

Get up and running with TinyTorch interactive notebooks in under 5 minutes!

## 🚀 One-Command Setup

```bash
# Clone repository
git clone https://github.com/your-org/TinyTorch.git
cd TinyTorch

# Run automated setup
chmod +x setup-dev.sh
./setup-dev.sh

# Activate environment
source .venv/bin/activate
```

## 📓 Convert Modules to Notebooks

```bash
# Convert all modules to interactive notebooks
python -m tito.main module notebooks

# Or convert specific modules
python -m tito.main module notebooks --module 03_activations
```

## 🎯 Start Learning

```bash
# Open Jupyter Lab
jupyter lab

# Navigate to modules/source/ and open any .ipynb file
# Try: 03_activations/activations_dev.ipynb
```

## ✅ Verify Everything Works

```bash
# Run environment check
python -m tito.main system doctor

# Should show:
# ✅ Python 3.8+
# ✅ Virtual Environment Active
# ✅ Essential Dependencies Installed
```

## 🆘 Need Help?

- **Full Documentation**: [NOTEBOOK_WORKFLOW.md](NOTEBOOK_WORKFLOW.md)
- **Environment Issues**: `python -m tito.main system doctor`
- **Command Help**: `python -m tito.main --help`

## 📋 Available Modules

After conversion, you'll have interactive notebooks for:

- **01_setup**: Environment setup and course introduction
- **02_tensor**: Tensor operations and broadcasting
- **03_activations**: Neural network activation functions
- **04_layers**: Building neural network layers
- **05_dense**: Fully connected networks
- **06_spatial**: Convolutional neural networks
- **07_attention**: Attention mechanisms and transformers
- **08_dataloader**: Data loading and preprocessing
- **09_autograd**: Automatic differentiation
- **10_optimizers**: Gradient descent and optimization
- **11_training**: Training loops and validation
- **12_compression**: Model compression techniques
- **13_kernels**: Custom CUDA kernels
- **14_benchmarking**: Performance measurement
- **15_mlops**: Production deployment

Each module includes:
- 📚 Educational content with explanations
- 💻 Interactive code cells
- 🧪 Comprehensive tests via NBGrader
- 🎯 Hands-on exercises and experiments

Happy learning! 🔥