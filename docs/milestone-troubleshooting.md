# 🔧 TinyTorch Milestone Troubleshooting Guide

## Common Issues and Solutions

This guide helps you overcome the most frequent challenges students encounter while pursuing TinyTorch milestones. Each section provides symptoms, diagnoses, and concrete solutions.

---

## 🎯 Milestone 1: Basic Inference

### Issue: "My neural network outputs don't make sense"

**Symptoms:**
- Network outputs NaN or inf values
- All predictions are the same number
- Accuracy stuck at random chance (10% for MNIST)
- Gradients exploding or vanishing

**Diagnosis & Solutions:**

#### Weight Initialization Problems
```python
# ❌ WRONG: Weights too large
self.weight = Tensor(np.random.randn(input_size, output_size))

# ✅ CORRECT: Xavier initialization
scale = np.sqrt(2.0 / (input_size + output_size))
self.weight = Tensor(np.random.randn(input_size, output_size) * scale)
```

#### Shape Mismatch Issues
```python
# Debug shapes at each step
print(f"Input shape: {x.shape}")
output = self.dense1(x)
print(f"After dense1: {output.shape}")
output = self.activation(output)
print(f"After activation: {output.shape}")
```

#### Learning Rate Problems
```python
# ❌ TOO HIGH: Learning rate 1.0 causes instability
optimizer = SGD(model.parameters(), learning_rate=1.0)

# ✅ GOOD: Start with smaller learning rate
optimizer = SGD(model.parameters(), learning_rate=0.01)
```

### Issue: "MNIST accuracy stuck below 85%"

**Symptoms:**
- Network trains but plateaus at 60-70% accuracy
- Loss decreases but accuracy doesn't improve
- Similar performance on training and test sets

**Diagnosis & Solutions:**

#### Insufficient Network Capacity
```python
# ❌ TOO SIMPLE: Not enough parameters
model = Sequential([
    Dense(784, 10),  # Only 7,850 parameters
    Softmax()
])

# ✅ BETTER: More capacity for complex patterns
model = Sequential([
    Dense(784, 128), ReLU(),  # Hidden layer for feature learning
    Dense(128, 64), ReLU(),   # Additional feature refinement
    Dense(64, 10), Softmax()  # Final classification
])
```

#### Activation Function Issues
```python
# ❌ WRONG: No activation between layers
model = Sequential([
    Dense(784, 128),
    Dense(128, 10),  # Linear combinations of linear functions = linear
    Softmax()
])

# ✅ CORRECT: Nonlinearity enables complex patterns
model = Sequential([
    Dense(784, 128), ReLU(),  # Nonlinearity crucial!
    Dense(128, 10), Softmax()
])
```

---

## 👁️ Milestone 2: Computer Vision

### Issue: "Convolution implementation is too slow"

**Symptoms:**
- Conv2D forward pass takes >10 seconds for small images
- Memory usage explodes during convolution
- System becomes unresponsive during training

**Diagnosis & Solutions:**

#### Inefficient Convolution Loops
```python
# ❌ SLOW: Nested Python loops
for batch in range(batch_size):
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for h in range(output_height):
                for w in range(output_width):
                    # Convolution computation
                    result[batch, out_ch, h, w] += ...

# ✅ FASTER: Vectorized operations using im2col
def im2col_convolution(input_tensor, weight, bias=None):
    # Convert convolution to matrix multiplication
    input_cols = im2col(input_tensor, weight.shape[2:])
    output = input_cols @ weight.reshape(weight.shape[0], -1).T
    return output.reshape(batch_size, out_channels, output_height, output_width)
```

#### Memory Inefficiency
```python
# ❌ MEMORY HOG: Creating intermediate tensors in loops
for i in range(kernel_height):
    for j in range(kernel_width):
        temp_tensor = input[:, :, i:i+output_height, j:j+output_width]
        result += temp_tensor * kernel[:, :, i, j]

# ✅ MEMORY EFFICIENT: In-place operations
output = Tensor(np.zeros((batch_size, out_channels, output_height, output_width)))
for i in range(kernel_height):
    for j in range(kernel_width):
        # Use views instead of copies
        input_slice = input[:, :, i:i+output_height, j:j+output_width]
        output += input_slice * kernel[:, :, i, j]
```

### Issue: "CNN accuracy worse than dense network"

**Symptoms:**
- Dense network achieves 90%+ on MNIST
- CNN with same parameters gets 70-80%
- CNN training loss decreases slower than dense

**Diagnosis & Solutions:**

#### Poor CNN Architecture
```python
# ❌ BAD: CNN worse than dense
model = Sequential([
    Conv2D(1, 32, kernel_size=7),  # Too large kernel
    ReLU(),
    Flatten(),
    Dense(32 * 22 * 22, 10)  # Huge dense layer
])

# ✅ GOOD: Proper CNN design
model = Sequential([
    Conv2D(1, 16, kernel_size=3), ReLU(),  # Small kernels
    MaxPool2D(kernel_size=2),               # Reduce spatial size
    Conv2D(16, 32, kernel_size=3), ReLU(),
    MaxPool2D(kernel_size=2),
    Flatten(),
    Dense(32 * 5 * 5, 128), ReLU(),        # Reasonable dense size
    Dense(128, 10)
])
```

#### Padding and Stride Issues
```python
# ❌ WRONG: Losing too much spatial information
conv = Conv2D(1, 16, kernel_size=5, stride=2, padding=0)  # Aggressive downsampling

# ✅ CORRECT: Preserve spatial information
conv = Conv2D(1, 16, kernel_size=3, stride=1, padding=1)  # Same size output
pool = MaxPool2D(kernel_size=2)  # Controlled downsampling
```

---

## ⚙️ Milestone 3: Full Training

### Issue: "Training loss not decreasing"

**Symptoms:**
- Loss remains constant across epochs
- Gradients are all zeros or very small
- Model predictions don't change during training

**Diagnosis & Solutions:**

#### Learning Rate Too Small
```python
# ❌ TOO SMALL: No visible progress
optimizer = Adam(model.parameters(), learning_rate=1e-6)

# ✅ GOOD RANGE: Start here and adjust
optimizer = Adam(model.parameters(), learning_rate=1e-3)

# Monitor gradient norms to debug
def check_gradients(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm()**2
    return total_norm**0.5

print(f"Gradient norm: {check_gradients(model)}")
```

#### Incorrect Loss Function Implementation
```python
# ❌ WRONG: CrossEntropy without log-softmax
def cross_entropy_loss(predictions, targets):
    return -np.mean(predictions[range(len(targets)), targets])

# ✅ CORRECT: Proper log-softmax + NLL
def cross_entropy_loss(logits, targets):
    log_probs = log_softmax(logits)
    return -np.mean(log_probs[range(len(targets)), targets])

def log_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.log(exp_x / np.sum(exp_x, axis=1, keepdims=True))
```

### Issue: "CIFAR-10 training diverges or gets stuck"

**Symptoms:**
- Loss starts decreasing then shoots up to infinity
- Accuracy drops during training
- NaN values appear in loss or gradients

**Diagnosis & Solutions:**

#### Data Preprocessing Issues
```python
# ❌ WRONG: Using raw pixel values 0-255
train_data = cifar10_data  # Values in [0, 255]

# ✅ CORRECT: Normalize to reasonable range
train_data = cifar10_data.astype(np.float32) / 255.0  # Values in [0, 1]

# Even better: Zero-center and normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
train_data = (train_data - mean) / std
```

#### Batch Size Too Large
```python
# ❌ PROBLEMATIC: Batch size too large for dataset
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# ✅ BETTER: Moderate batch size for stability
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

#### Learning Rate Scheduling
```python
# ❌ BASIC: Fixed learning rate throughout training
optimizer = Adam(model.parameters(), learning_rate=0.001)

# ✅ ADVANCED: Learning rate decay for convergence
def adjust_learning_rate(optimizer, epoch, initial_lr=0.001):
    lr = initial_lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

---

## 🚀 Milestone 4: Advanced Vision

### Issue: "Can't reach 75% CIFAR-10 accuracy"

**Symptoms:**
- Model plateaus at 65-70% accuracy
- Training and validation accuracy gap is large
- Loss continues decreasing but accuracy doesn't improve

**Diagnosis & Solutions:**

#### Insufficient Model Complexity
```python
# ❌ TOO SIMPLE: Not enough capacity for CIFAR-10
model = Sequential([
    Conv2D(3, 16, 3), ReLU(),
    MaxPool2D(2),
    Flatten(),
    Dense(16 * 16 * 16, 10)
])

# ✅ BETTER: Deeper architecture with more features
model = Sequential([
    Conv2D(3, 32, 3), ReLU(),
    Conv2D(32, 32, 3), ReLU(),
    MaxPool2D(2),
    Conv2D(32, 64, 3), ReLU(),
    Conv2D(64, 64, 3), ReLU(),
    MaxPool2D(2),
    Flatten(),
    Dense(64 * 6 * 6, 256), ReLU(),
    Dropout(0.5),
    Dense(256, 10)
])
```

#### Overfitting Problems
```python
# Add regularization techniques
model = Sequential([
    Conv2D(3, 32, 3), BatchNorm2D(32), ReLU(),
    Conv2D(32, 32, 3), BatchNorm2D(32), ReLU(),
    MaxPool2D(2), Dropout(0.2),
    
    Conv2D(32, 64, 3), BatchNorm2D(64), ReLU(),
    Conv2D(64, 64, 3), BatchNorm2D(64), ReLU(),
    MaxPool2D(2), Dropout(0.3),
    
    Flatten(),
    Dense(64 * 6 * 6, 256), BatchNorm1D(256), ReLU(),
    Dropout(0.5),
    Dense(256, 10)
])
```

#### Data Augmentation Missing
```python
# ✅ ADD: Data augmentation for better generalization
def augment_cifar10(image):
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random crop and pad
    pad_width = 4
    padded = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')
    crop_x = np.random.randint(0, 2 * pad_width + 1)
    crop_y = np.random.randint(0, 2 * pad_width + 1)
    image = padded[crop_y:crop_y+32, crop_x:crop_x+32]
    
    return image

class AugmentedCIFAR10Dataset(CIFAR10Dataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.train:
            image = augment_cifar10(image)
        return image, label
```

### Issue: "Model training takes too long"

**Symptoms:**
- Single epoch takes >10 minutes
- GPU utilization low or no GPU being used
- Memory usage constantly growing

**Diagnosis & Solutions:**

#### Inefficient Convolution Implementation
```python
# Profile your convolution
import time

def time_convolution():
    input_tensor = Tensor(np.random.randn(32, 3, 32, 32))
    conv = Conv2D(3, 64, kernel_size=3)
    
    start_time = time.time()
    for _ in range(100):
        output = conv(input_tensor)
    end_time = time.time()
    
    print(f"100 convolutions took {end_time - start_time:.2f} seconds")
    print(f"Average time per convolution: {(end_time - start_time)/100:.4f} seconds")

time_convolution()
```

#### Memory Leaks in Training Loop
```python
# ❌ MEMORY LEAK: Accumulating computation graphs
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # Missing: optimizer.zero_grad()

# ✅ CORRECT: Clear gradients each iteration
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

---

## 🔥 Milestone 5: Language Generation

### Issue: "GPT generates nonsense text"

**Symptoms:**
- Generated text is random characters
- Model outputs same character repeatedly
- Text has no recognizable patterns or structure

**Diagnosis & Solutions:**

#### Tokenization Problems
```python
# ❌ WRONG: Inconsistent character mapping
def tokenize(text):
    chars = list(set(text))  # Order changes each run!
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    return [char_to_idx[ch] for ch in text]

# ✅ CORRECT: Consistent character vocabulary
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))  # Consistent ordering
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
        
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])
```

#### Sequence Length Issues
```python
# ❌ TOO LONG: Sequence length too large for available data
sequence_length = 1000  # Only have 10,000 chars total

# ✅ REASONABLE: Sequence length appropriate for dataset
sequence_length = min(100, len(text) // 100)  # At least 100 sequences
```

#### Position Encoding Missing
```python
# ❌ MISSING: No positional information
class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim)
        
    def forward(self, x):
        x = x + self.attention(x)  # No position info!
        x = x + self.mlp(x)
        return x

# ✅ CORRECT: Add positional encoding
class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
    def forward(self, x):
        x = x + self.pos_encoding(x)  # Add position information
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x
```

### Issue: "Can't reuse components from vision modules"

**Symptoms:**
- Having to reimplement Dense layers, ReLU, etc.
- Components don't work with sequence data
- Different interfaces for vision vs. language components

**Diagnosis & Solutions:**

#### Shape Incompatibility
```python
# ❌ PROBLEM: Dense layer expects 2D input, sequences are 3D
# Sequence shape: (batch_size, sequence_length, embed_dim)
# Dense expects: (batch_size, features)

# ✅ SOLUTION: Reshape for compatibility
class SequenceDense(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.dense = Dense(input_dim, output_dim)  # Reuse vision component!
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape to 2D for dense layer
        x_flat = x.reshape(batch_size * seq_len, input_dim)
        
        # Apply dense transformation
        output_flat = self.dense(x_flat)
        
        # Reshape back to sequence format
        output_dim = output_flat.shape[-1]
        return output_flat.reshape(batch_size, seq_len, output_dim)
```

#### Different Data Types
```python
# ❌ ISSUE: Vision uses float32, language uses int64 indices
# Vision: image_tensor = Tensor(np.float32([...]))
# Language: token_indices = [1, 5, 12, ...]

# ✅ SOLUTION: Embedding layer converts indices to vectors
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        self.embedding = Tensor(np.random.randn(vocab_size, embed_dim) * 0.1)
        
    def forward(self, token_indices):
        # Convert integer indices to float embeddings
        return self.embedding[token_indices]  # Now compatible with Dense layers!
```

---

## 🛠️ General Debugging Strategies

### Debugging Checklist

**Before Every Milestone Attempt:**
1. [ ] Environment activated: `source .venv/bin/activate`
2. [ ] Dependencies updated: `pip install -r requirements.txt`
3. [ ] Previous modules working: `tito test --all-previous`
4. [ ] Clean workspace: `git status` shows clean state

**During Implementation:**
1. [ ] Print shapes at every step
2. [ ] Test with small data first (batch_size=1, small input)
3. [ ] Use debugger breakpoints at critical functions
4. [ ] Save intermediate results for inspection

**Before Milestone Submission:**
1. [ ] Code runs without errors
2. [ ] Performance benchmarks met
3. [ ] All tests pass: `tito milestone test X`
4. [ ] Code exported successfully: `tito export --module X`

### Performance Debugging

**Memory Usage:**
```python
import tracemalloc

def debug_memory_usage():
    tracemalloc.start()
    
    # Your code here
    model = build_model()
    train_one_epoch(model)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()
```

**Training Speed:**
```python
import time

def benchmark_training_speed():
    model = build_model()
    dummy_data = create_dummy_batch()
    
    # Warm up
    for _ in range(5):
        _ = model(dummy_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        output = model(dummy_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
```

### Getting Help

**Documentation Resources:**
- Module READMEs: `modules/source/XX_module/README.md`
- API Reference: `book/appendices/api-reference.md`
- Troubleshooting: This guide!

**Community Support:**
- Discord/Slack: #tinytorch-help channel
- Office Hours: See course calendar
- Study Groups: Form with classmates working on same milestone

**Instructor Support:**
- Email for conceptual questions
- Office hours for debugging sessions
- Milestone review meetings for stuck students

### When to Ask for Help

**Ask for help if:**
- Stuck on same issue for >2 hours
- Performance far below milestone requirements
- Unclear about milestone requirements
- Suspecting bug in provided code

**Before asking, prepare:**
- Minimal code example reproducing the issue
- Error messages and stack traces
- What you've already tried
- Specific question, not just "it doesn't work"

---

## 🎯 Success Strategies

### Milestone Achievement Tips

**Start Early:**
- Begin milestone attempts when you complete prerequisites
- Don't wait until the deadline to discover issues
- Use intermediate checkpoints to track progress

**Incremental Development:**
- Get basic version working first
- Optimize performance second
- Add advanced features last

**Test-Driven Development:**
- Write tests for your functions before implementation
- Use provided test suites as specification
- Add your own tests for edge cases

**Systematic Debugging:**
- Isolate issues to smallest possible code section
- Use print statements and debugger strategically
- Keep a debugging log of what you've tried

### Building Confidence

**Celebrate Small Wins:**
- First successful forward pass
- First decreasing loss curve
- First accuracy improvement

**Learn from Failures:**
- Every bug teaches you something about the system
- Failed milestones often lead to deeper understanding
- Debugging skills are as valuable as implementation skills

**Connect to Bigger Picture:**
- Each milestone represents real-world capability
- Your implementations mirror industry practices
- Skills transfer directly to research and industry roles

**Remember the Goal:**
You're not just completing assignments—you're building genuine ML systems engineering expertise that will serve you throughout your career. Every challenge overcome makes you a stronger engineer.

🚀 **Keep going! Every milestone brings you closer to ML systems mastery.**