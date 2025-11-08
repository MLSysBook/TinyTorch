#!/usr/bin/env python3
"""
Transformer Simple Pattern Learning Tests

These tests verify the transformer can learn VERY SIMPLE patterns that are
easy to verify. If the transformer can't learn these, something is wrong.

Pattern Tasks:
1. Copy Task: Input [1,2,3] → Output [1,2,3]
2. Increment Task: Input [1,2,3] → Output [2,3,4]
3. Repeat Pattern: Input [1,2] → Output [1,2,1,2,1,2,...]
4. Constant Sequence: Always predict the same token

These are MUCH simpler than Shakespeare and should achieve near-perfect accuracy.
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


def test_constant_prediction():
    """
    Task: Always predict token 5, regardless of input.
    
    This is the SIMPLEST possible task - the model should achieve 100% accuracy.
    """
    print("\n" + "="*70)
    print("TEST 1: Constant Prediction (Always predict 5)")
    print("="*70)
    
    vocab_size = 10
    embed_dim = 16
    num_layers = 1
    num_heads = 2
    seq_len = 4
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    # Create training data: random inputs, all targets are 5
    num_examples = 10
    train_data = []
    for _ in range(num_examples):
        x = np.random.randint(0, vocab_size, (1, seq_len))
        targets = np.full((1, seq_len), 5)  # Always 5
        train_data.append((Tensor(x), Tensor(targets)))
    
    print(f"Task: Always predict token 5")
    print(f"Training on {num_examples} examples for 100 steps...")
    
    # Train
    for step in range(100):
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
        
        if (step + 1) % 25 == 0:
            avg_loss = total_loss / num_examples
            print(f"  Step {step + 1}: Avg Loss = {avg_loss:.4f}")
    
    # Test: Check predictions
    test_x = Tensor(np.random.randint(0, vocab_size, (1, seq_len)))
    logits = model.forward(test_x)
    predictions = np.argmax(logits.data, axis=-1)
    
    print(f"\nTest Input: {test_x.data[0]}")
    print(f"Predictions: {predictions[0]}")
    print(f"Target: [5, 5, 5, 5]")
    
    correct = np.sum(predictions[0] == 5)
    accuracy = correct / seq_len * 100
    
    print(f"Accuracy: {correct}/{seq_len} ({accuracy:.0f}%)")
    
    assert accuracy >= 75, f"Should achieve at least 75% accuracy, got {accuracy:.0f}%"
    
    print("✅ Constant prediction works!")
    return True


def test_copy_task():
    """
    Task: Copy the input sequence.
    Input: [1, 3, 7, 2] → Output: [1, 3, 7, 2]
    
    This tests if the model can learn identity mapping.
    """
    print("\n" + "="*70)
    print("TEST 2: Copy Task (Input = Output)")
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
    
    # Create training data: targets = inputs
    num_examples = 20
    train_data = []
    for _ in range(num_examples):
        x = np.random.randint(0, vocab_size, (1, seq_len))
        targets = x.copy()  # Copy task!
        train_data.append((Tensor(x), Tensor(targets)))
    
    print(f"Task: Output = Input (copy)")
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
    
    # Test on new examples
    print("\nTesting on 5 new examples:")
    correct_total = 0
    total_positions = 0
    
    for i in range(5):
        test_x = Tensor(np.random.randint(0, vocab_size, (1, seq_len)))
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)
        
        print(f"  Input:  {test_x.data[0]}")
        print(f"  Output: {predictions[0]}")
        
        correct = np.sum(predictions[0] == test_x.data[0])
        correct_total += correct
        total_positions += seq_len
    
    accuracy = correct_total / total_positions * 100
    print(f"\nOverall Accuracy: {correct_total}/{total_positions} ({accuracy:.0f}%)")
    
    assert accuracy >= 60, f"Should achieve at least 60% accuracy, got {accuracy:.0f}%"
    
    print("✅ Copy task works!")
    return True


def test_sequence_completion():
    """
    Task: Learn to complete simple sequences.
    Pattern: [0,1,2] → predict 3, [1,2,3] → predict 4, etc.
    
    This tests if the model can learn arithmetic patterns.
    """
    print("\n" + "="*70)
    print("TEST 3: Sequence Completion (Next Number)")
    print("="*70)
    
    vocab_size = 10
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    seq_len = 3
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    # Create training data: [a,a+1,a+2] → predict [a+1,a+2,a+3]
    train_data = []
    for start in range(7):  # 0-6, so max is 6+2=8 < vocab_size
        x = np.array([[start, start+1, start+2]])
        targets = np.array([[start+1, start+2, start+3]])
        train_data.append((Tensor(x), Tensor(targets)))
        # Add multiple copies for training
        for _ in range(5):
            train_data.append((Tensor(x.copy()), Tensor(targets.copy())))
    
    print(f"Task: Given [a, a+1, a+2], predict [a+1, a+2, a+3]")
    print(f"Training on {len(train_data)} examples for 150 steps...")
    
    # Train
    for step in range(150):
        total_loss = 0
        # Shuffle data
        np.random.shuffle(train_data)
        
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
            avg_loss = total_loss / len(train_data)
            print(f"  Step {step + 1}: Avg Loss = {avg_loss:.4f}")
    
    # Test on training examples
    print("\nTesting on training sequences:")
    correct_total = 0
    total_positions = 0
    
    test_cases = [
        ([0, 1, 2], [1, 2, 3]),
        ([1, 2, 3], [2, 3, 4]),
        ([3, 4, 5], [4, 5, 6]),
    ]
    
    for input_seq, expected_output in test_cases:
        test_x = Tensor(np.array([input_seq]))
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)
        
        print(f"  Input: {input_seq} → Output: {predictions[0].tolist()} (Expected: {expected_output})")
        
        correct = np.sum(predictions[0] == np.array(expected_output))
        correct_total += correct
        total_positions += len(expected_output)
    
    accuracy = correct_total / total_positions * 100
    print(f"\nOverall Accuracy: {correct_total}/{total_positions} ({accuracy:.0f}%)")
    
    assert accuracy >= 50, f"Should achieve at least 50% accuracy, got {accuracy:.0f}%"
    
    print("✅ Sequence completion works!")
    return True


def test_repeat_pattern():
    """
    Task: Learn to repeat a 2-element pattern.
    Input: [1,2,1,2] → Output: [1,2,1,2]
    
    This tests if the model can learn periodic patterns.
    """
    print("\n" + "="*70)
    print("TEST 4: Repeat Pattern (A,B,A,B)")
    print("="*70)
    
    vocab_size = 10
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    seq_len = 8
    
    model = GPT(vocab_size, embed_dim, num_layers, num_heads)
    
    params = model.parameters()
    for param in params:
        param.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    # Create training data: repeating patterns [a,b,a,b,a,b,...]
    train_data = []
    for a in range(0, vocab_size, 2):
        for b in range(1, vocab_size, 2):
            if a != b:
                pattern = [a, b] * (seq_len // 2)
                x = np.array([pattern])
                targets = x.copy()
                train_data.append((Tensor(x), Tensor(targets)))
                # Add multiple copies
                for _ in range(3):
                    train_data.append((Tensor(x.copy()), Tensor(targets.copy())))
    
    print(f"Task: Learn repeating 2-patterns [a,b,a,b,...]")
    print(f"Training on {len(train_data)} examples for 150 steps...")
    
    # Train
    for step in range(150):
        total_loss = 0
        np.random.shuffle(train_data)
        
        for x, targets in train_data[:30]:  # Use subset for speed
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
            avg_loss = total_loss / 30
            print(f"  Step {step + 1}: Avg Loss = {avg_loss:.4f}")
    
    # Test
    print("\nTesting on patterns:")
    correct_total = 0
    total_positions = 0
    
    test_cases = [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [2, 3, 2, 3, 2, 3, 2, 3],
        [4, 5, 4, 5, 4, 5, 4, 5],
    ]
    
    for pattern in test_cases:
        test_x = Tensor(np.array([pattern]))
        logits = model.forward(test_x)
        predictions = np.argmax(logits.data, axis=-1)
        
        print(f"  Input:  {pattern}")
        print(f"  Output: {predictions[0].tolist()}")
        
        correct = np.sum(predictions[0] == np.array(pattern))
        correct_total += correct
        total_positions += len(pattern)
    
    accuracy = correct_total / total_positions * 100
    print(f"\nOverall Accuracy: {correct_total}/{total_positions} ({accuracy:.0f}%)")
    
    assert accuracy >= 40, f"Should achieve at least 40% accuracy, got {accuracy:.0f}%"
    
    print("✅ Pattern repetition works!")
    return True


def run_all_tests():
    """Run all simple pattern learning tests."""
    print("\n" + "="*70)
    print("TRANSFORMER SIMPLE PATTERN LEARNING TESTS")
    print("="*70)
    print("\nThese tests verify the transformer can learn VERY SIMPLE patterns.")
    print("If these fail, something is fundamentally wrong with learning.\n")
    
    tests = [
        ("Constant Prediction", test_constant_prediction),
        ("Copy Task", test_copy_task),
        ("Sequence Completion", test_sequence_completion),
        ("Repeat Pattern", test_repeat_pattern),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"❌ {test_name}: FAIL")
            print(f"Error: {e}")
            print(f"{'='*70}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL SIMPLE PATTERN TESTS PASSED!")
        print("The transformer can learn basic patterns.")
        print("Ready for more complex tasks like Shakespeare!")
    else:
        print(f"\n❌ {failed} test(s) failed")
        print("The transformer has issues with simple pattern learning.")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

