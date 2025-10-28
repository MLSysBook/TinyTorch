# Transformer Training Validation Plan
## Mission: Ensure the Transformer Actually Learns Shakespeare

**Goal:** Progressively verify that our transformer learns to generate Shakespeare-style text, not just compile and run. No reward hacking - we need real learning.

---

## 🎯 Phase 1: Architecture Verification (Sanity Checks)

### Test 1.1: Forward Pass Shape Validation
**What:** Verify all tensor shapes through the forward pass
**Why:** Catch shape mismatches before training

```python
def test_forward_pass_shapes():
    """Verify tensor shapes at each layer."""
    model = TinyGPT(vocab_size=65, embed_dim=128, num_layers=4, num_heads=4)
    
    # Input: (batch=2, seq=64)
    x = Tensor(np.random.randint(0, 65, (2, 64)))
    
    # Forward pass
    output = model.forward(x)
    
    # Output should be (batch=2, seq=64, vocab=65)
    assert output.shape == (2, 64, 65), f"Expected (2, 64, 65), got {output.shape}"
    
    print("✅ Forward pass shapes correct")
```

### Test 1.2: Gradient Flow Verification
**What:** Ensure gradients flow to ALL parameters
**Why:** Broken gradients = no learning

```python
def test_gradient_flow_all_params():
    """Ensure every parameter receives gradients."""
    model = TinyGPT(vocab_size=65, embed_dim=128, num_layers=4, num_heads=4)
    
    # Forward pass
    x = Tensor(np.random.randint(0, 65, (2, 64)), requires_grad=False)
    targets = Tensor(np.random.randint(0, 65, (2, 64)), requires_grad=False)
    
    logits = model.forward(x)
    loss_fn = CrossEntropyLoss()
    
    # Reshape for loss
    logits_flat = logits.reshape(2 * 64, 65)
    targets_flat = targets.reshape(2 * 64)
    
    loss = loss_fn.forward(logits_flat, targets_flat)
    loss.backward(np.ones_like(loss.data))
    
    # Check ALL parameters have gradients
    params = model.parameters()
    params_without_grads = []
    
    for i, param in enumerate(params):
        if param.grad is None:
            params_without_grads.append(i)
    
    assert len(params_without_grads) == 0, \
        f"Parameters without gradients: {params_without_grads}"
    
    print(f"✅ All {len(params)} parameters receive gradients")
```

### Test 1.3: Single Batch Overfitting Test
**What:** Model should memorize a single batch perfectly
**Why:** If it can't overfit, it can't learn at all

```python
def test_single_batch_overfitting():
    """Model should memorize one batch (loss → 0)."""
    print("\n🧪 Test: Single batch overfitting")
    
    model = TinyGPT(vocab_size=65, embed_dim=128, num_layers=4, num_heads=4)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Single batch
    x = Tensor(np.random.randint(0, 65, (2, 64)), requires_grad=False)
    targets = Tensor(np.random.randint(0, 65, (2, 64)), requires_grad=False)
    
    initial_loss = None
    final_loss = None
    
    # Train for 100 steps on same batch
    for step in range(100):
        # Forward
        logits = model.forward(x)
        logits_flat = logits.reshape(2 * 64, 65)
        targets_flat = targets.reshape(2 * 64)
        
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        if step == 0:
            initial_loss = loss.data.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward(np.ones_like(loss.data))
        optimizer.step()
        
        if step % 20 == 0:
            print(f"  Step {step}: Loss = {loss.data:.4f}")
        
        final_loss = loss.data.item()
    
    # Loss should decrease significantly
    improvement = (initial_loss - final_loss) / initial_loss
    
    assert improvement > 0.5, \
        f"Loss only improved by {improvement:.1%}, expected >50%"
    
    print(f"✅ Single batch overfitting: {initial_loss:.4f} → {final_loss:.4f} ({improvement:.1%} improvement)")
```

---

## 🎯 Phase 2: Data Pipeline Verification

### Test 2.1: Character Encoding/Decoding
**What:** Verify text ↔ indices conversion is lossless
**Why:** Data corruption = garbage learning

```python
def test_character_encoding():
    """Ensure encode/decode is lossless."""
    text = "To be or not to be, that is the question."
    
    # Create vocab
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode
    indices = [char_to_idx[ch] for ch in text]
    
    # Decode
    decoded = ''.join([idx_to_char[i] for i in indices])
    
    assert decoded == text, "Encoding/decoding is lossy!"
    
    print(f"✅ Character encoding lossless: '{text[:20]}...'")
```

### Test 2.2: Sequence Alignment
**What:** Targets should be inputs shifted by 1
**Why:** Misalignment = model learns wrong task

```python
def test_sequence_alignment():
    """Verify input[i+1] == target[i]."""
    text = "Hello world"
    sequences = []
    
    # Create sequences
    seq_length = 5
    for i in range(len(text) - seq_length):
        input_seq = text[i:i+seq_length]
        target_seq = text[i+1:i+seq_length+1]
        sequences.append((input_seq, target_seq))
    
    # Check alignment
    for input_seq, target_seq in sequences:
        for j in range(len(input_seq)):
            # input[j+1] should == target[j]
            if j < len(input_seq) - 1:
                assert input_seq[j+1] == target_seq[j], \
                    f"Misalignment: input[{j+1}]='{input_seq[j+1]}' != target[{j}]='{target_seq[j]}'"
    
    print(f"✅ Sequence alignment correct: {len(sequences)} sequences")
```

### Test 2.3: Batch Diversity
**What:** Batches should contain different sequences
**Why:** Same sequences = no generalization

```python
def test_batch_diversity():
    """Ensure batches have diverse sequences."""
    dataloader = # ... your dataloader
    
    batch = next(iter(dataloader))
    inputs, targets = batch
    
    # Check rows are different
    unique_rows = len(set(tuple(row) for row in inputs.data))
    
    assert unique_rows > 1, "All sequences in batch are identical!"
    
    print(f"✅ Batch diversity: {unique_rows}/{len(inputs.data)} unique sequences")
```

---

## 🎯 Phase 3: Training Dynamics Verification

### Test 3.1: Loss Decreases Monotonically (Small Dataset)
**What:** On small dataset, loss should decrease consistently
**Why:** If loss doesn't decrease, something is wrong

```python
def test_loss_decreases_small_dataset():
    """Train on 1000 characters, loss should decrease."""
    print("\n🧪 Test: Loss decrease on small dataset")
    
    # Small Shakespeare sample (1000 chars)
    text = """To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles..."""[:1000]
    
    # Train for 50 epochs
    model, losses = train_on_text(text, epochs=50, lr=0.001)
    
    # Loss should decrease by at least 30%
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss
    
    assert improvement > 0.3, \
        f"Loss only improved by {improvement:.1%}, expected >30%"
    
    # No increasing trend in last 10 epochs
    last_10 = losses[-10:]
    increasing_steps = sum(1 for i in range(1, len(last_10)) if last_10[i] > last_10[i-1])
    
    assert increasing_steps < 5, \
        f"Loss increased in {increasing_steps}/10 last epochs"
    
    print(f"✅ Loss decreases: {initial_loss:.4f} → {final_loss:.4f} ({improvement:.1%})")
```

### Test 3.2: Perplexity Improvement
**What:** Perplexity = exp(loss) should decrease
**Why:** Lower perplexity = better predictions

```python
def test_perplexity_improvement():
    """Perplexity should decrease during training."""
    # Train model and collect losses
    model, losses = train_on_text(text, epochs=50)
    
    initial_perplexity = np.exp(losses[0])
    final_perplexity = np.exp(losses[-1])
    
    improvement = (initial_perplexity - final_perplexity) / initial_perplexity
    
    print(f"Perplexity: {initial_perplexity:.2f} → {final_perplexity:.2f}")
    print(f"Improvement: {improvement:.1%}")
    
    assert improvement > 0.2, "Perplexity didn't improve enough"
```

### Test 3.3: Learning Rate Sensitivity
**What:** Different learning rates should affect learning
**Why:** If LR doesn't matter, gradients aren't flowing

```python
def test_learning_rate_sensitivity():
    """Different LRs should give different results."""
    print("\n🧪 Test: Learning rate sensitivity")
    
    text = shakespeare_sample[:1000]
    
    # Try 3 learning rates
    lrs = [0.0001, 0.001, 0.01]
    final_losses = []
    
    for lr in lrs:
        _, losses = train_on_text(text, epochs=20, lr=lr)
        final_losses.append(losses[-1])
        print(f"  LR={lr}: Final loss = {losses[-1]:.4f}")
    
    # Should see variation
    loss_range = max(final_losses) - min(final_losses)
    
    assert loss_range > 0.1, \
        f"Learning rate has no effect (range={loss_range:.4f})"
    
    print(f"✅ LR matters: loss range = {loss_range:.4f}")
```

---

## 🎯 Phase 4: Generation Quality Verification

### Test 4.1: Character Distribution Match
**What:** Generated text character distribution should match training data
**Why:** If distribution is way off, model isn't learning patterns

```python
def test_character_distribution():
    """Generated text should have similar character distribution."""
    import collections
    
    # Training data distribution
    training_text = load_shakespeare()
    train_dist = collections.Counter(training_text)
    train_dist = {k: v/len(training_text) for k, v in train_dist.items()}
    
    # Generate 10K characters
    model = trained_model
    generated = generate_text(model, prompt="To be", length=10000)
    gen_dist = collections.Counter(generated)
    gen_dist = {k: v/len(generated) for k, v in gen_dist.items()}
    
    # Compare distributions (KL divergence or simple diff)
    common_chars = set(train_dist.keys()) & set(gen_dist.keys())
    
    diffs = []
    for char in common_chars:
        diff = abs(train_dist[char] - gen_dist[char])
        diffs.append(diff)
    
    avg_diff = np.mean(diffs)
    
    print(f"Avg character distribution diff: {avg_diff:.4f}")
    assert avg_diff < 0.05, "Character distribution too different from training"
    
    print(f"✅ Character distribution matches (diff={avg_diff:.4f})")
```

### Test 4.2: N-gram Overlap
**What:** Generated text should contain bigrams/trigrams from training
**Why:** Model should learn common patterns

```python
def test_ngram_overlap():
    """Generated text should contain training n-grams."""
    import collections
    
    training_text = load_shakespeare()
    
    # Extract training bigrams
    train_bigrams = set()
    for i in range(len(training_text) - 1):
        train_bigrams.add(training_text[i:i+2])
    
    # Generate text
    generated = generate_text(model, prompt="To be", length=1000)
    
    # Extract generated bigrams
    gen_bigrams = set()
    for i in range(len(generated) - 1):
        gen_bigrams.add(generated[i:i+2])
    
    # Calculate overlap
    overlap = len(gen_bigrams & train_bigrams) / len(gen_bigrams)
    
    print(f"Bigram overlap: {overlap:.1%}")
    assert overlap > 0.8, f"Only {overlap:.1%} of bigrams appear in training"
    
    print(f"✅ N-gram overlap good: {overlap:.1%}")
```

### Test 4.3: Progressive Generation Quality
**What:** Generation quality should improve with training
**Why:** Later epochs should produce better text

```python
def test_progressive_quality():
    """Generation quality improves over epochs."""
    print("\n🧪 Test: Progressive generation improvement")
    
    text = load_shakespeare()[:5000]
    prompt = "To be or not"
    
    # Save model at epochs 10, 20, 30, 40, 50
    checkpoints = [10, 20, 30, 40, 50]
    perplexities = []
    
    model = TinyGPT(...)
    
    for epoch in range(1, 51):
        train_one_epoch(model, text)
        
        if epoch in checkpoints:
            # Measure perplexity on held-out text
            test_text = load_shakespeare()[5000:6000]
            perplexity = calculate_perplexity(model, test_text)
            perplexities.append(perplexity)
            
            # Generate sample
            sample = generate_text(model, prompt, length=100)
            print(f"\nEpoch {epoch} (perplexity={perplexity:.2f}):")
            print(f"  '{sample}'")
    
    # Perplexity should generally decrease
    assert perplexities[-1] < perplexities[0], \
        "Perplexity didn't improve from epoch 10 to 50"
    
    print(f"✅ Quality improves: {perplexities[0]:.2f} → {perplexities[-1]:.2f}")
```

---

## 🎯 Phase 5: Shakespeare-Specific Learning

### Test 5.1: Memorization Test (Small Excerpt)
**What:** Model should memorize exact passages after many epochs
**Why:** Proves model can learn specific sequences

```python
def test_memorize_passage():
    """Model should memorize a specific passage."""
    print("\n🧪 Test: Memorize Shakespeare passage")
    
    # Famous passage
    passage = "To be, or not to be, that is the question"
    prompt = "To be"
    
    # Train ONLY on this passage (repeated)
    training_text = (passage + " ") * 100  # Repeat 100 times
    
    model = TinyGPT(...)
    train_on_text(model, training_text, epochs=100, lr=0.001)
    
    # Generate with prompt
    generated = generate_text(model, prompt, length=len(passage), temperature=0.1)
    
    # Should closely match
    similarity = calculate_similarity(passage, generated)
    
    print(f"Target:    '{passage}'")
    print(f"Generated: '{generated}'")
    print(f"Similarity: {similarity:.1%}")
    
    assert similarity > 0.7, f"Only {similarity:.1%} match"
    
    print(f"✅ Memorizes passage: {similarity:.1%} match")
```

### Test 5.2: Style Consistency
**What:** Generated text should have Shakespeare-like features
**Why:** Model should learn the style, not just characters

```python
def test_shakespeare_style():
    """Generated text should have Shakespeare features."""
    print("\n🧪 Test: Shakespeare style features")
    
    model = train_on_full_shakespeare(epochs=50)
    
    # Generate long sample
    generated = generate_text(model, prompt="ROMEO:", length=500)
    
    # Check features:
    checks = {
        "Has colons": ":" in generated,
        "Has capital letters": any(c.isupper() for c in generated),
        "Has punctuation": any(c in ".,;!?" for c in generated),
        "Has spaces": " " in generated,
        "Has newlines": "\n" in generated,
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    for feature, present in checks.items():
        status = "✓" if present else "✗"
        print(f"  {status} {feature}")
    
    assert passed >= 4, f"Only {passed}/{total} style features present"
    
    print(f"✅ Shakespeare style: {passed}/{total} features present")
```

### Test 5.3: Context Coherence
**What:** Model should continue text coherently
**Why:** Real understanding requires context use

```python
def test_context_coherence():
    """Model should use context to generate coherent text."""
    print("\n🧪 Test: Context coherence")
    
    model = train_on_full_shakespeare(epochs=50)
    
    test_prompts = [
        ("ROMEO:", "Should reference love/Juliet themes"),
        ("To be", "Should continue philosophically"),
        ("What light", "Should reference 'through yonder window breaks'"),
    ]
    
    coherent_count = 0
    
    for prompt, expectation in test_prompts:
        generated = generate_text(model, prompt, length=50, temperature=0.7)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print(f"Expectation: {expectation}")
        
        # Manual inspection needed, but check basic coherence
        # At minimum: should continue with Shakespeare-like text
        is_coherent = (
            len(generated) > len(prompt) and
            generated.startswith(prompt) and
            not all(c == generated[0] for c in generated)  # Not all same char
        )
        
        if is_coherent:
            coherent_count += 1
    
    print(f"\n✅ Context coherence: {coherent_count}/{len(test_prompts)} prompts")
```

---

## 🎯 Phase 6: Comprehensive Training Run

### Final Validation: Full Shakespeare Training
**What:** Train on full Shakespeare corpus with proper monitoring
**Why:** This is the real test

```python
def full_shakespeare_training():
    """
    Comprehensive training run with all monitoring.
    """
    print("\n" + "="*70)
    print("FULL SHAKESPEARE TRANSFORMER TRAINING")
    print("="*70)
    
    # Load full dataset
    text = load_shakespeare_corpus()  # ~1.1M characters
    print(f"Dataset: {len(text):,} characters, {len(set(text))} unique")
    
    # Split train/val
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]
    
    # Model
    vocab_size = len(set(text))
    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=256,      # Increased
        num_layers=6,       # Increased
        num_heads=8,        # Increased
        seq_length=128,     # Increased context
    )
    
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Training config
    config = {
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'warmup_steps': 1000,
        'checkpoint_every': 10,
    }
    
    # Metrics to track
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': [],
        'generation_samples': [],
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_one_epoch(model, train_text, config)
        metrics['train_loss'].append(train_loss)
        metrics['train_perplexity'].append(np.exp(train_loss))
        
        # Validate
        val_loss = evaluate(model, val_text)
        metrics['val_loss'].append(val_loss)
        metrics['val_perplexity'].append(np.exp(val_loss))
        
        print(f"Train Loss: {train_loss:.4f} (PPL: {np.exp(train_loss):.2f})")
        print(f"Val Loss:   {val_loss:.4f} (PPL: {np.exp(val_loss):.2f})")
        
        # Generate samples
        if epoch % 10 == 0:
            print("\nGeneration Samples:")
            for prompt in ["To be", "ROMEO:", "What light"]:
                sample = generate_text(model, prompt, length=100, temperature=0.8)
                print(f"\n'{prompt}' → '{sample}'")
                metrics['generation_samples'].append((epoch, prompt, sample))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, f'best_model_epoch{epoch}.npz')
            print(f"✅ New best model saved (val_loss={val_loss:.4f})")
        
        # Early stopping check
        if epoch > 20:
            recent_losses = metrics['val_loss'][-10:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                print("\n⚠️ Validation loss not improving, consider stopping")
    
    # Final report
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"  Val Loss: {metrics['val_loss'][-1]:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Final Train PPL: {metrics['train_perplexity'][-1]:.2f}")
    print(f"  Final Val PPL: {metrics['val_perplexity'][-1]:.2f}")
    
    # Plot learning curves
    plot_metrics(metrics)
    
    return model, metrics
```

---

## 📋 Success Criteria

### Minimum Acceptable Performance
- ✅ Loss decreases from ~4.5 to <3.0 on validation
- ✅ Validation perplexity < 20
- ✅ Generated text has >80% bigram overlap with training
- ✅ Model memorizes short passages when trained on them
- ✅ No gradient explosions or NaN losses
- ✅ Generation continues coherently for 200+ characters

### Good Performance
- ✅ Loss < 2.5 on validation
- ✅ Validation perplexity < 15
- ✅ Generated text is mostly readable English
- ✅ Some Shakespeare-like phrases appear
- ✅ Context awareness (ROMEO: talks about love)

### Excellent Performance
- ✅ Loss < 2.0 on validation
- ✅ Validation perplexity < 10
- ✅ Generated text is coherent and Shakespeare-like
- ✅ Can complete famous quotes with reasonable accuracy
- ✅ Generates grammatically correct sentences

---

## 🚫 Anti-Reward-Hacking Checks

### What NOT to do:
1. ❌ Don't train on validation set
2. ❌ Don't cherry-pick best generation samples
3. ❌ Don't use extremely low temperature (>0.1) to hide problems
4. ❌ Don't claim success with only loss decrease (need quality too)
5. ❌ Don't ignore validation metrics
6. ❌ Don't skip the hard tests (memorization, coherence)

### Honest Reporting:
- Show BOTH best and worst generation samples
- Report validation metrics, not just training
- Include failure cases
- Show learning curves (are we still improving?)
- Test on held-out Shakespeare text

---

## 📁 Implementation Files Needed

1. `test_architecture.py` - Phase 1 tests
2. `test_data_pipeline.py` - Phase 2 tests
3. `test_training_dynamics.py` - Phase 3 tests
4. `test_generation_quality.py` - Phase 4 tests
5. `test_shakespeare_learning.py` - Phase 5 tests
6. `train_shakespeare_full.py` - Phase 6 implementation
7. `evaluate_model.py` - Comprehensive evaluation script
8. `visualize_training.py` - Plot learning curves, attention weights

---

## 🎯 Execution Plan

1. **Day 1**: Implement Phase 1-2 tests, verify they all pass
2. **Day 2**: Implement Phase 3 tests, fix any training issues
3. **Day 3**: Implement Phase 4-5 tests, tune hyperparameters
4. **Day 4-7**: Run full Shakespeare training (Phase 6)
5. **Day 8**: Comprehensive evaluation and documentation

**No skipping steps. No reward hacking. Real learning or nothing.**

