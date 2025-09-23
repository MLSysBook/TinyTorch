#!/usr/bin/env python3
"""Ultra-minimal TinyGPT - every line uses code you built!"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.training import CrossEntropyLoss

# TinyGPT - you built every component!
class TinyGPT(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=32, seq_len=8):
        super().__init__()
        # Embedding layers - using Linear as embedding (you built Linear!)
        self.token_embed = nn.Linear(vocab_size, embed_dim)   # Token embedding
        self.pos_embed = nn.Linear(seq_len, embed_dim)        # Positional encoding
        
        # Attention mechanism - simplified using Linear layers you built
        self.query = nn.Linear(embed_dim, embed_dim)          # You built Linear!
        self.key = nn.Linear(embed_dim, embed_dim)            # You built Linear!
        self.value = nn.Linear(embed_dim, embed_dim)          # You built Linear!
        
        # Feedforward network
        self.ff1 = nn.Linear(embed_dim, 64)                   # You built Linear!
        self.ff2 = nn.Linear(64, embed_dim)                   # You built Linear!
        
        # Output projection
        self.output = nn.Linear(embed_dim, vocab_size)        # You built Linear!
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Convert tokens to one-hot and embed
        x_onehot = F.one_hot(x, num_classes=10)              # You built one_hot!
        tok_emb = self.token_embed(x_onehot.float())         # You built Linear!
        
        # Add positional encoding
        pos = F.one_hot(Tensor(np.arange(seq_len)), num_classes=8)
        pos_emb = self.pos_embed(pos.float())
        x = tok_emb + pos_emb.unsqueeze(0)                   # Broadcasting you built!
        
        # Self-attention (simplified)
        Q = self.query(x)                                     # You built Linear!
        K = self.key(x)                                       # You built Linear!
        V = self.value(x)                                     # You built Linear!
        
        # Attention scores
        scores = F.matmul(Q, K.transpose(-2, -1))            # You built matmul!
        scores = scores / (embed_dim ** 0.5)                 # Scaling
        attn = F.softmax(scores, dim=-1)                     # You built softmax!
        x = F.matmul(attn, V)                                # You built matmul!
        
        # Feedforward
        x = F.relu(self.ff1(x))                              # You built ReLU + Linear!
        x = self.ff2(x)                                      # You built Linear!
        
        # Output
        return self.output(x)                                # You built Linear!

# Simple sequence data: predict next number in pattern
def create_simple_sequences(n_samples=500):
    """Create sequences: [0,1,2,3,4...] where next = (current + 1) % 10"""
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.randint(0, 10)
        seq = [(start + i) % 10 for i in range(9)]
        X.append(seq[:-1])  # Input: first 8
        y.append(seq[1:])   # Target: last 8
    return np.array(X), np.array(y)

# Generate training data
X_train, y_train = create_simple_sequences()

# Training setup - you built everything!
model = TinyGPT(vocab_size=10, embed_dim=32, seq_len=8)
optimizer = optim.Adam(model.parameters(), learning_rate=0.01)   # You built Adam!
loss_fn = CrossEntropyLoss()                                     # You built CrossEntropy!

print("Training TinyGPT to predict number sequences...")
# Training loop - you built every operation!
for epoch in range(50):
    total_loss = 0
    batch_size = 32
    
    for i in range(0, len(X_train), batch_size):
        batch_X = Tensor(X_train[i:i+batch_size])
        batch_y = Tensor(y_train[i:i+batch_size])
        
        outputs = model(batch_X)                         # You built forward pass!
        
        # Reshape for loss computation
        outputs = outputs.reshape(-1, 10)                # Flatten predictions
        targets = batch_y.reshape(-1)                    # Flatten targets
        
        loss = loss_fn(outputs, targets)                 # You built CrossEntropy!
        
        loss.backward()                                  # You built backprop!
        optimizer.step()                                 # You built Adam updates!
        optimizer.zero_grad()                            # You built gradient clearing!
        
        total_loss += float(loss.data)
    
    if epoch % 10 == 0:
        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# Test generation
print("\nGenerating sequences:")
test_input = Tensor(np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))  # Start sequence
with_grad = model(test_input)
pred = F.argmax(with_grad, dim=-1)                          # You built argmax!
print(f"Input:  {test_input.data[0]}")
print(f"Output: {pred.data[0]} (should predict 1,2,3,4,5,6,7,8)")

print("\n✅ TinyGPT trained! You built a transformer from scratch!")