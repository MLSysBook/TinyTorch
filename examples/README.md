# TinyTorch Examples

**Complete Applications Built with Your Framework**

These examples demonstrate that the ML framework you built from scratch actually works! Each example is a real application that uses the components you created.

## 📁 Example Structure

Each example folder contains clearly named files:
- `train_*.py` - Training scripts that teach the model
- `test_*.py` - Testing scripts that evaluate performance
- `demo_*.py` - Interactive demonstrations
- `utils.py` - Helper functions specific to that example
- `README.md` - Detailed documentation for students

## 🎯 The Three Capstone Examples

### 1. **xornet/** - Neural Network Fundamentals
**Proves**: Your neural networks can learn non-linear functions

Files:
- `train_xor_network.py` - Trains a network to solve XOR
- `visualize_decision_boundary.py` - Shows what the network learned
- `README.md` - Explains why XOR is important

**What students learn**: XOR can't be solved linearly, but neural networks with hidden layers can solve it perfectly.

### 2. **cifar10/** - Computer Vision 
**Proves**: Your framework can handle real-world image classification

Files:
- `train_image_classifier.py` - Trains CNN on CIFAR-10 images
- `test_random_baseline.py` - Shows random guessing gets ~10%
- `evaluate_model.py` - Tests your trained model
- `visualize_predictions.py` - Shows what the model sees
- `README.md` - Explains computer vision concepts

**What students learn**: How convolutions extract features and how real ML systems train on actual data.

### 3. **tinygpt/** - Language Models
**Proves**: Your framework can build transformers and generate text

Files:
- `train_language_model.py` - Trains GPT on text data
- `generate_text.py` - Interactive text generation
- `test_simple_patterns.py` - Verifies the model can learn
- `tokenizer.py` - Text processing utilities
- `README.md` - Explains language modeling

**What students learn**: How attention mechanisms enable language understanding and generation.

## 🚀 Running the Examples

Each example can be run immediately:

```bash
# XOR - Takes seconds, shows 100% accuracy
cd examples/xornet
python train_xor_network.py

# CIFAR-10 - Takes minutes, achieves 55%+ accuracy  
cd examples/cifar10
python train_image_classifier.py

# TinyGPT - Takes minutes, generates text
cd examples/tinygpt
python train_language_model.py
python generate_text.py
```

## 📊 What Success Looks Like

- **XORNet**: 100% accuracy on XOR problem
- **CIFAR-10**: 55%+ accuracy (5.5x better than random)
- **TinyGPT**: Generates coherent character sequences

## 💡 For Students

These examples are the **proof that you succeeded**. You didn't just learn about neural networks - you built a framework capable of:
- Learning any function (XORNet)
- Classifying real images (CIFAR-10)
- Generating language (TinyGPT)

This is what ML engineers do in production!