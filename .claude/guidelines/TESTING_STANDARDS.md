# TinyTorch Testing Standards

## 🎯 Core Testing Philosophy

**Test immediately, test simply, test educationally.**

Testing in TinyTorch serves two purposes:
1. **Verification**: Ensure the code works
2. **Education**: Help students understand what they built

## 📋 Testing Patterns

### The Immediate Testing Pattern

**MANDATORY**: Test immediately after each implementation, not at the end.

```python
# ✅ CORRECT: Implementation followed by immediate test
class Tensor:
    def __init__(self, data):
        self.data = data

# Test Tensor creation immediately
def test_tensor_creation():
    t = Tensor([1, 2, 3])
    assert t.data == [1, 2, 3], "Tensor should store data"
    print("✅ Tensor creation works")

test_tensor_creation()

# ❌ WRONG: All tests grouped at the end
# [100 lines of implementations]
# [Then all tests at the bottom]
```

### Simple Assertion Testing

**Use simple assertions, not complex frameworks.**

```python
# ✅ GOOD: Simple and clear
def test_forward_pass():
    model = SimpleMLP()
    x = Tensor(np.random.randn(32, 784))
    output = model.forward(x)
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    print("✅ Forward pass shapes correct")

# ❌ BAD: Over-engineered
class TestMLPForwardPass(unittest.TestCase):
    def setUp(self):
        self.model = SimpleMLP()
    
    def test_forward_pass_shape_validation_with_mock_data(self):
        # ... 50 lines of test setup
```

### Educational Test Messages

**Tests should teach, not just verify.**

```python
# ✅ GOOD: Educational
def test_backpropagation():
    # Create simple network: 2 inputs → 2 hidden → 1 output
    net = TwoLayerNet(2, 2, 1)
    
    # Forward pass with XOR data
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = Tensor([[0], [1], [1], [0]])
    
    output = net.forward(x)
    loss = mse_loss(output, y)
    
    print(f"Initial loss: {loss.data:.4f}")
    print("This high loss shows the network hasn't learned XOR yet")
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert net.w1.grad is not None, "Gradients should be computed"
    print("✅ Backpropagation computed gradients")
    print("The network can now learn from its mistakes!")

# ❌ BAD: Just verification
def test_backprop():
    net = TwoLayerNet(2, 2, 1)
    # ... minimal test
    assert net.w1.grad is not None
    # No educational value
```

## 🧪 Performance Testing

### Baseline Comparisons

**Always test against a clear baseline.**

```python
def test_model_performance():
    # 1. Test random baseline
    random_model = create_random_network()
    random_acc = evaluate(random_model, test_data)
    print(f"Random network accuracy: {random_acc:.1%}")
    
    # 2. Test trained model
    trained_model = load_trained_model()
    trained_acc = evaluate(trained_model, test_data)
    print(f"Trained network accuracy: {trained_acc:.1%}")
    
    # 3. Show improvement
    improvement = trained_acc / random_acc
    print(f"Improvement: {improvement:.1f}× better than random")
    
    assert trained_acc > random_acc * 2, "Should be at least 2× better than random"
```

### Honest Performance Reporting

```python
# ✅ GOOD: Report actual measurements
def test_training_performance():
    start_time = time.time()
    accuracy = train_model(epochs=10)
    train_time = time.time() - start_time
    
    print(f"Achieved accuracy: {accuracy:.1%}")
    print(f"Training time: {train_time:.1f} seconds")
    print(f"Status: {'✅ PASS' if accuracy > 0.5 else '❌ FAIL'}")

# ❌ BAD: Theoretical claims
def test_training():
    # ... training code
    print("Can achieve 60-70% with proper tuning")  # Unverified claim
```

## 🔍 Test Organization

### Test Placement

```python
# Module structure with immediate tests
# module_name.py

# Part 1: Core implementation
class Tensor:
    ...

# Immediate test
test_tensor_creation()

# Part 2: Operations
def add(a, b):
    ...

# Immediate test
test_addition()

# Part 3: Advanced features
def backward():
    ...

# Immediate test
test_backward()

# At the end: Run all tests when executed directly
if __name__ == "__main__":
    print("Running all tests...")
    test_tensor_creation()
    test_addition()
    test_backward()
    print("✅ All tests passed!")
```

## ⚠️ Common Testing Mistakes

1. **Grouping all tests at the end**
   - Loses educational flow
   - Students don't see immediate verification

2. **Over-complicated test frameworks**
   - Obscures what's being tested
   - Adds unnecessary complexity

3. **Testing without teaching**
   - Missing opportunity to reinforce concepts
   - No educational value

4. **Unverified performance claims**
   - Damages credibility
   - Misleads students

## 📝 Test Documentation

```python
def test_attention_mechanism():
    """
    Test that attention correctly weighs different positions.
    
    This test demonstrates the key insight of attention:
    the model learns what to focus on.
    """
    # Create simple sequence
    sequence = Tensor([[1, 0, 0],  # Position 0: important
                       [0, 0, 0],  # Position 1: padding
                       [0, 0, 1]]) # Position 2: important
    
    attention_weights = compute_attention(sequence)
    
    # Check that important positions get more weight
    assert attention_weights[0] > attention_weights[1]
    assert attention_weights[2] > attention_weights[1]
    
    print("✅ Attention focuses on important positions")
    print(f"Weights: {attention_weights}")
    print("Notice how padding (position 1) gets less attention")
```

## 🎯 Remember

> Tests are teaching tools, not just verification tools.

Every test should help a student understand:
- What the code does
- Why it matters
- How to verify it works
- What success looks like