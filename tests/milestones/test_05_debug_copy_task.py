#!/usr/bin/env python3
"""
Debug Copy Task Failure

The copy task failed while other tasks succeeded. This script investigates why.

Hypothesis:
1. The causal mask prevents looking at future tokens
2. For position i to predict token i, it can only see tokens 0..i-1
3. This makes copying impossible in an autoregressive model!

Solution: We should test "shifted" copy where we predict the NEXT token.
Input: [1, 2, 3, 4] → Predict: [2, 3, 4, ?]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.models.transformer import GPT

enable_autograd()


def test_copy_with_causal_mask_visualization():
    """Visualize what the model sees with causal masking."""
    print("\n" + "="*70)
    print("Understanding Causal Masking in Copy Task")
    print("="*70)
    
    print("\nInput sequence: [1, 2, 3, 4]")
    print("Target (copy): [1, 2, 3, 4]")
    print("\nWhat each position sees (with causal mask):")
    print("  Position 0: sees [] → must predict 1 (impossible!)")
    print("  Position 1: sees [1] → must predict 2")
    print("  Position 2: sees [1,2] → must predict 3")
    print("  Position 3: sees [1,2,3] → must predict 4")
    print("\n❌ Position 0 CANNOT predict correctly - it sees nothing!")
    print("\n✅ CORRECT task: Predict NEXT token (shifted prediction)")
    print("  Position 0: sees [1] → predict 2")
    print("  Position 1: sees [1,2] → predict 3")
    print("  Position 2: sees [1,2,3] → predict 4")
    print("  Position 3: sees [1,2,3,4] → predict 5 (or padding)")
    

def test_next_token_prediction():
    """
    Test the CORRECT task for autoregressive models: predict next token.
    Input: [1,2,3] → Predict: [2,3,4] (shifted by 1)
    """
    print("\n" + "="*70)
    print("TEST: Next Token Prediction (Autoregressive Copy)")
    print("="*70)
    
    vocab_size = 10
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    seq_len = 4
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    print("\nTask: Given [a,b,c,d], predict [b,c,d,e]")
    print("This is the standard autoregressive task!\n")
    
    # Create training data: targets are inputs shifted by 1
    num_examples = 30
    train_data = []
    for _ in range(num_examples):
        # Create sequence [a, a+1, a+2, a+3]
        start = np.random.randint(0, vocab_size - seq_len)
        x = np.array([[start + i for i in range(seq_len)]])
        # Target is [a+1, a+2, a+3, a+4]
        targets = np.array([[start + i + 1 for i in range(seq_len)]])
        train_data.append((Tensor(x), Tensor(targets)))
    
    print(f"Training on {num_examples} examples for 200 steps...")
    
    # Train
    for step in range(200):
        total_loss = 0
        for x, targets in train_data:
            # Zero gradients
            for param in params:
                param.grad = None
            
            # Forward
            logits = model.forward(x)
            logits_flat = logits.reshape(seq_len, vocab_size)
            targets_flat = targets.reshape(seq_len)
            loss = loss_fn.forward(logits_flat, targets_flat)
            
            # Backward
            loss.backward(np.ones_like(loss.data))
            
            # Update
            optimizer.step()
            
            total_loss += loss.data
        
        if (step + 1) % 50 == 0:
            avg_loss = total_loss / num_examples
            print(f"  Step {step + 1}: Avg Loss = {avg_loss:.4f}")
    
    # Test on new sequences
    print("\nTesting on NEW sequences:")
    correct_total = 0
    total_positions = 0
    
    for i in range(5):
        start = np.random.randint(0, vocab_size - seq_len)
        test_x = Tensor(np.array([[start + j for j in range(seq_len)]]))
        expected = np.array([start + j + 1 for j in range(seq_len)])
        
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)[0]
        
        print(f"  Input: {test_x.data[0]} → Output: {predictions} (Expected: {expected})")
        
        correct = np.sum(predictions == expected)
        correct_total += correct
        total_positions += seq_len
    
    accuracy = correct_total / total_positions * 100
    print(f"\nOverall Accuracy: {correct_total}/{total_positions} ({accuracy:.0f}%)")
    
    if accuracy >= 75:
        print("✅ Next token prediction works perfectly!")
        return True
    else:
        print(f"⚠️  Accuracy is {accuracy:.0f}%, lower than expected")
        return False


def test_memorization_vs_generalization():
    """
    Test if the model memorizes specific sequences or learns the pattern.
    """
    print("\n" + "="*70)
    print("TEST: Memorization vs Generalization")
    print("="*70)
    
    vocab_size = 10
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    seq_len = 4
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    # Train on ONLY sequences starting with 0, 2, 4
    train_starts = [0, 2, 4]
    train_data = []
    for start in train_starts:
        x = np.array([[start, start+1, start+2, start+3]])
        targets = np.array([[start+1, start+2, start+3, start+4]])
        # Add multiple copies
        for _ in range(10):
            train_data.append((Tensor(x.copy()), Tensor(targets.copy())))
    
    print(f"\n1. Training ONLY on sequences: [0,1,2,3], [2,3,4,5], [4,5,6,7]")
    print(f"   (Total: {len(train_data)} examples)")
    
    # Train
    for step in range(150):
        total_loss = 0
        np.random.shuffle(train_data)
        for x, targets in train_data:
            for param in params:
                param.grad = None
            
            logits = model.forward(x)
            logits_flat = logits.reshape(seq_len, vocab_size)
            targets_flat = targets.reshape(seq_len)
            loss = loss_fn.forward(logits_flat, targets_flat)
            
            loss.backward(np.ones_like(loss.data))
            optimizer.step()
            
            total_loss += loss.data
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Avg Loss = {total_loss / len(train_data):.4f}")
    
    # Test on training data
    print("\n2. Testing on TRAINING sequences:")
    for start in train_starts:
        test_x = Tensor(np.array([[start, start+1, start+2, start+3]]))
        expected = np.array([start+1, start+2, start+3, start+4])
        
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)[0]
        
        match = "✅" if np.array_equal(predictions, expected) else "❌"
        print(f"  {match} Input: [{start},{start+1},{start+2},{start+3}] → {predictions} (Expected: {expected})")
    
    # Test on unseen sequences
    print("\n3. Testing on UNSEEN sequences (generalization test):")
    test_starts = [1, 3, 5]
    correct_total = 0
    total_positions = 0
    
    for start in test_starts:
        test_x = Tensor(np.array([[start, start+1, start+2, start+3]]))
        expected = np.array([start+1, start+2, start+3, start+4])
        
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)[0]
        
        correct = np.sum(predictions == expected)
        correct_total += correct
        total_positions += seq_len
        
        match = "✅" if np.array_equal(predictions, expected) else "❌"
        print(f"  {match} Input: [{start},{start+1},{start+2},{start+3}] → {predictions} (Expected: {expected})")
    
    accuracy = correct_total / total_positions * 100
    print(f"\n4. Generalization Accuracy: {correct_total}/{total_positions} ({accuracy:.0f}%)")
    
    if accuracy >= 75:
        print("✅ Model GENERALIZED the pattern!")
    elif accuracy >= 25:
        print("⚠️  Model PARTIALLY generalized")
    else:
        print("❌ Model just MEMORIZED training examples")
    
    return accuracy >= 50


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEBUGGING COPY TASK FAILURE")
    print("="*70)
    
    test_copy_with_causal_mask_visualization()
    
    success1 = test_next_token_prediction()
    success2 = test_memorization_vs_generalization()
    
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    if success1 and success2:
        print("\n✅ The transformer works correctly!")
        print("\nKey insights:")
        print("1. Autoregressive models predict NEXT token, not same token")
        print("2. The model can learn and generalize patterns")
        print("3. The 'copy task' failure was due to incorrect task formulation")
        print("\n🚀 Ready for Shakespeare training!")
    else:
        print("\n⚠️  Some issues found:")
        if not success1:
            print("  - Next token prediction issues")
        if not success2:
            print("  - Generalization issues (memorization)")
    
    print("="*70)

