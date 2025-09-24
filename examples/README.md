# TinyTorch Examples: A Journey Through AI History

These examples tell the story of neural networks through historical breakthroughs. Each example represents a pivotal moment in AI history, and you'll build the same architectures that changed the field.

## The Historical Journey

### 1957: The Perceptron - Where It All Began
**`perceptron_1957/rosenblatt_perceptron.py`** (Run after Module 4)
- Frank Rosenblatt's first trainable neural network
- Could learn linearly separable patterns
- Sparked dreams of artificial intelligence
- **You'll build:** Single-layer network for linear classification

### 1969: The XOR Problem - The First AI Winter
**`xor_1969/minsky_xor_problem.py`** (Run after Module 6)
- Minsky & Papert proved perceptrons can't solve XOR
- Led to decade-long "AI Winter" (1969-1980s)
- Solution required hidden layers + nonlinearity + backpropagation
- **You'll build:** Multi-layer perceptron that solves XOR

### 1998: LeNet - The Convolution Revolution
**`lenet_1998/train_mlp.py`** (Run after Module 9)
- Yann LeCun's convolutional neural network
- First practical system for reading handwritten digits
- Deployed in banks for check processing
- **You'll build:** Network for MNIST digit recognition

### 2012: AlexNet - The Deep Learning Explosion
**`alexnet_2012/train_cnn.py`** (Run after Module 10)
- Alex Krizhevsky's ImageNet breakthrough
- Proved deep networks could surpass traditional CV
- Triggered the modern deep learning boom
- **You'll build:** Deep CNN for CIFAR-10 classification

### 2018: GPT - The Transformer Era
**`gpt_2018/simple_tinygpt.py`** (Run after Module 14)
- OpenAI's transformer architecture
- Self-attention revolutionized NLP
- Foundation for ChatGPT and modern AI
- **You'll build:** Character-level language model

## Running the Examples

Each example shows which modules are required:

```bash
# After Module 4: Can build architectures
python examples/perceptron_1957/rosenblatt_perceptron.py

# After Module 6: Can train with gradients  
python examples/xor_1969/minsky_xor_problem.py

# After Module 9: Can use convolutions
python examples/lenet_1998/train_mlp.py

# After Module 10: Full training pipeline
python examples/alexnet_2012/train_cnn.py

# After Module 14: Transformers work!
python examples/gpt_2018/simple_tinygpt.py
```

## The Learning Flow

1. **Build modules** → Core engine development
2. **Pass unit tests** → Verify your implementation
3. **Complete module** → `tito module complete XX_modulename`
4. **Pass integration tests** → Automatic validation with other modules
5. **Unlock capability** → New historical example available!
6. **Run example** → See what you've enabled!

📚 **See [CAPABILITIES.md](CAPABILITIES.md) for the complete progression system**

## PyTorch-Style Code

All examples follow modern PyTorch conventions:

```python
class HistoricNetwork:
    def __init__(self):
        # Define layers
        self.fc1 = Dense(input_size, hidden_size)
        self.activation = ReLU()
        self.fc2 = Dense(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
```

## What You're Building

You're not just learning ML - you're rebuilding the breakthroughs that created modern AI:

- **1957**: Linear models that could learn
- **1969**: Multi-layer networks for complex patterns  
- **1998**: Convolutional networks for vision
- **2012**: Deep networks that changed everything
- **2018**: Attention mechanisms powering ChatGPT

Each example runs on YOUR implementation. When GPT works, it's because YOU built every component from scratch!