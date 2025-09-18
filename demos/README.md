# TinyTorch Demo System

This directory contains progressive AI capability demonstrations for TinyTorch. Each demo showcases what becomes possible as you export more modules to the TinyTorch package.

## 🎯 Available Demos

Run any demo using: `tito demo <demo_name>`

### Core Demos

| Demo | Command | Module Requirements | Description |
|------|---------|-------------------|-------------|
| **Mathematical Operations** | `tito demo math` | Module 02 (Tensor) | Linear algebra, matrix operations, geometric transformations |
| **Logical Reasoning** | `tito demo logic` | Module 03 (Activations) | Boolean functions, XOR problem, decision boundaries |
| **Single Neuron Learning** | `tito demo neuron` | Module 04 (Layers) | Watch a neuron learn the AND gate with gradient descent |
| **Multi-Layer Networks** | `tito demo network` | Module 05 (Dense) | Solve the famous XOR problem with 2-layer network |
| **Computer Vision** | `tito demo vision` | Module 06 (Spatial) | Image processing, edge detection, CNN pattern recognition |
| **Attention Mechanisms** | `tito demo attention` | Module 07 (Attention) | Sequence processing, self-attention, transformer foundations |
| **End-to-End Training** | `tito demo training` | Module 11 (Training) | Complete ML pipeline with optimization and evaluation |
| **Language Generation** | `tito demo language` | Module 16 (TinyGPT) | AI text generation and language modeling |

### Demo Commands

```bash
# Show capability matrix
tito demo

# Run specific demo
tito demo math
tito demo vision
tito demo attention

# Run all available demos
tito demo --all

# Show matrix only (no module testing)
tito demo --matrix
```

## 🚀 Demo Progression

The demos unlock progressively as you export modules:

### Foundation (Modules 2-5)
- **Tensor Math**: Matrix operations, linear systems
- **Activations**: Nonlinear functions, sigmoid/ReLU
- **Single Neuron**: Gradient descent learning
- **XOR Network**: Multi-layer breakthrough

### Intelligence (Modules 6-7)
- **Computer Vision**: CNNs, edge detection, pattern recognition
- **Attention**: Sequence understanding, transformer mechanisms

### Complete Systems (Modules 11-16)
- **Training**: End-to-end ML pipelines
- **Language**: Text generation, TinyGPT

## 🎓 Educational Value

Each demo is designed to:

1. **Show Real AI Capabilities**: Not just code, but actual intelligence in action
2. **Explain the "Why"**: Understanding principles behind the implementations
3. **Connect to Production**: How these concepts scale to real ML systems
4. **Build Excitement**: See your framework grow more capable with each module

## 🔧 Technical Details

- **Import Safety**: Each demo gracefully handles missing modules
- **Error Recovery**: Clear messages about which modules need to be exported
- **Rich Output**: Color-coded, formatted demonstrations with explanations
- **Self-Contained**: Each demo can run independently for testing

## 🌟 Demo Highlights

### Mathematical Operations (demo_tensor_math.py)
- Solves real linear algebra problems
- Geometric transformations and rotations
- Preview of neural network computations

### XOR Network (demo_xor_network.py)
- The classic AI milestone problem
- Shows why single layers fail
- Demonstrates hidden layer feature creation

### Computer Vision (demo_vision.py)
- Edge detection with Sobel operators
- Convolutional pattern recognition
- Complete CNN architectures

### Attention Mechanisms (demo_attention.py)
- Self-attention matrix computation
- Multi-head attention concepts
- Connection to modern language models

### Language Generation (demo_language.py)
- Token embeddings and sequence processing
- Autoregressive generation process
- Complete transformer architecture overview

## 📈 Usage Analytics

The demo system tracks:
- Which modules are exported and available
- Demo availability status (✅ Ready, ⚡ Partial, ❌ Not Available)
- Integration with TinyTorch package exports

Students can see their progress through the capability matrix and immediately test new functionality as they complete modules.