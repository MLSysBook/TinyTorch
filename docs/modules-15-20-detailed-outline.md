# Detailed Module Outlines: 15-20
## Complete Implementation Plans for Optimization Journey

---

## Module 15: Acceleration - From Manual Loops to Optimized Code

### **Core Principle**
Students have been using manual loops since Module 2. Now we show WHY they're slow and HOW to fix them.

### **Module Structure**
1. **Part 1: The Problem - Your Loops Are Slow**
   ```python
   # From Module 2/4 - what students have been using
   def matmul_manual(a, b):
       result = np.zeros((a.shape[0], b.shape[1]))
       for i in range(a.shape[0]):
           for j in range(b.shape[1]):
               for k in range(a.shape[1]):
                   result[i,j] += a[i,k] * b[k,j]
       return result
   ```
   - Profile this: ~1000ms for 512×512 matrices
   - Explain cache misses, no vectorization

2. **Part 2: Optimization 1 - Cache-Friendly Blocking**
   ```python
   def matmul_blocked(a, b, block_size=32):
       # Tile the computation for cache efficiency
       for i_block in range(0, n, block_size):
           for j_block in range(0, n, block_size):
               # Process block - better cache locality
   ```
   - Profile: ~200ms (5x speedup!)
   - Explain L1/L2 cache utilization

3. **Part 3: Optimization 2 - NumPy (The Real Solution)**
   ```python
   def matmul_optimized(a, b):
       return np.matmul(a, b)  # Uses BLAS, SIMD, etc.
   ```
   - Profile: ~10ms (100x speedup!)
   - Explain BLAS, vectorization, SIMD

4. **Part 4: Transparent Backend System**
   ```python
   class OptimizedBackend:
       def __init__(self, mode='auto'):
           self.mode = mode
       
       def matmul(self, a, b):
           if self.mode == 'educational':
               return matmul_manual(a, b)
           elif self.mode == 'optimized':
               return matmul_optimized(a, b)
   ```

### **Student Deliverables**
- Implement blocked matrix multiplication
- Profile all three versions
- Build backend dispatch system
- Update their Tensor class to use optimized backend

---

## Module 16: Memory - KV Caching for Transformers

### **Core Principle**
Transformers recompute attention for ALL tokens every generation step. Fix this with caching.

### **Integration with Module 14 (Transformers)**
```python
# Current transformer (Module 14) - what needs fixing
class TransformerBlock:
    def forward(self, x, position):
        # Currently recomputes K,V for all previous positions
        keys = self.key_projection(x)     # Recomputed every time!
        values = self.value_projection(x)  # Wasteful!
        attention = compute_attention(q, keys, values)
```

### **Module Structure**
1. **Part 1: Profile the Problem**
   ```python
   # Generate 100 tokens with existing transformer
   for i in range(100):
       output = transformer(tokens[:i+1])  # O(n²) complexity!
   # Time: 30 seconds for 100 tokens
   ```

2. **Part 2: Build KV Cache**
   ```python
   class KVCache:
       def __init__(self, max_len, n_heads, head_dim):
           self.k_cache = np.zeros((max_len, n_heads, head_dim))
           self.v_cache = np.zeros((max_len, n_heads, head_dim))
           self.position = 0
       
       def update(self, k, v):
           self.k_cache[self.position] = k
           self.v_cache[self.position] = v
           self.position += 1
       
       def get_keys_values(self):
           return self.k_cache[:self.position], self.v_cache[:self.position]
   ```

3. **Part 3: Modify Transformer for Incremental Computation**
   ```python
   class CachedTransformerBlock(TransformerBlock):
       def forward_incremental(self, x, cache):
           # Only compute K,V for new token
           k_new = self.key_projection(x[-1:])    # Just new token!
           v_new = self.value_projection(x[-1:])  # Much faster!
           cache.update(k_new, v_new)
           
           k_all, v_all = cache.get_keys_values()
           return compute_attention(q_new, k_all, v_all)
   ```

4. **Part 4: Measure Impact**
   - Without cache: 30 seconds for 100 tokens
   - With cache: 0.6 seconds for 100 tokens (50x speedup!)

### **Student Deliverables**
- Implement KVCache class
- Modify their Module 14 transformer to use caching
- Profile memory usage vs speed tradeoff
- Generate text 50x faster!

---

## Module 17: Quantization - Numerical Optimization

### **Core Principle**
FP32 → INT8 reduces model size 4x and speeds inference 2-4x with minimal accuracy loss.

### **Module Structure**
1. **Part 1: Understanding Numerics**
   ```python
   # Visualize FP32 vs INT8 range and precision
   fp32_range = [-3.4e38, 3.4e38]  # Huge range
   int8_range = [-128, 127]         # Limited range
   
   # Show precision differences
   fp32_precision = 7 decimal places
   int8_precision = integer only
   ```

2. **Part 2: Basic Quantization**
   ```python
   def quantize_naive(weights, dtype=np.int8):
       scale = np.max(np.abs(weights)) / 127
       quantized = np.round(weights / scale).astype(dtype)
       return quantized, scale
   
   def dequantize(quantized, scale):
       return quantized.astype(np.float32) * scale
   ```

3. **Part 3: Calibration for Better Accuracy**
   ```python
   def calibrate_quantization(model, calibration_data):
       # Run calibration data through model
       # Track activation ranges
       # Use percentile (99.9%) not min/max
       scales = {}
       for layer in model.layers:
           activations = layer(calibration_data)
           scale = np.percentile(np.abs(activations), 99.9) / 127
           scales[layer.name] = scale
       return scales
   ```

4. **Part 4: Quantized Operations**
   ```python
   def quantized_matmul(a_q, b_q, scale_a, scale_b):
       # Integer computation (fast!)
       result_int = np.matmul(a_q.astype(np.int32), 
                             b_q.astype(np.int32))
       # Rescale to float
       return result_int.astype(np.float32) * scale_a * scale_b
   ```

### **Student Deliverables**
- Quantize their CNN from Module 9
- Implement calibration on CIFAR-10
- Measure: 4x size reduction, <1% accuracy loss
- Build quantized inference pipeline

---

## Module 18: Compression - Removing Unnecessary Weights

### **Core Principle**
Many weights contribute little to accuracy. Remove them for smaller, faster models.

### **Module Structure**
1. **Part 1: Magnitude-Based Pruning**
   ```python
   def prune_magnitude(weights, sparsity=0.9):
       threshold = np.percentile(np.abs(weights), sparsity * 100)
       mask = np.abs(weights) > threshold
       return weights * mask, mask
   ```

2. **Part 2: Structured Pruning (Channels/Filters)**
   ```python
   def prune_channels(conv_layer, keep_fraction=0.5):
       # Remove entire filters (hardware-friendly)
       importance = np.sum(np.abs(conv_layer.weight), axis=(1,2,3))
       n_keep = int(len(importance) * keep_fraction)
       keep_indices = np.argsort(importance)[-n_keep:]
       return conv_layer.weight[keep_indices]
   ```

3. **Part 3: Fine-tuning After Pruning**
   ```python
   def prune_and_finetune(model, data, sparsity):
       # Prune
       for layer in model.layers:
           layer.weight, mask = prune_magnitude(layer.weight, sparsity)
       
       # Fine-tune with mask frozen
       for epoch in range(5):
           train_with_mask(model, data, mask)
   ```

4. **Part 4: Measure Impact**
   - Original model: 10MB, 95% accuracy
   - 90% pruned: 1MB, 93% accuracy
   - Inference speedup: 3x with sparse kernels

### **Student Deliverables**
- Implement magnitude and structured pruning
- Prune their models to 90% sparsity
- Fine-tune to recover accuracy
- Visualize sparsity patterns

---

## Module 19: AutoTuning - Which Optimization When?

### **Core Principle**
Given constraints, automatically choose and apply the right optimizations.

### **Simple Optimization Strategy (Tractable for Students)**
```python
class AutoTuner:
    def __init__(self):
        self.optimization_space = {
            'quantization_bits': [32, 16, 8],
            'pruning_sparsity': [0, 0.5, 0.9],
            'use_kv_cache': [False, True],
            'backend': ['manual', 'optimized']
        }
    
    def optimize(self, model, constraints):
        # Simple Bayesian Optimization with Gaussian Process
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        # Try configurations, model performance
        gp = GaussianProcessRegressor()
        
        for iteration in range(20):  # Limited iterations
            # Choose next config based on acquisition function
            config = self.suggest_config(gp)
            
            # Apply optimizations
            optimized_model = self.apply_config(model, config)
            
            # Measure against constraints
            score = self.evaluate(optimized_model, constraints)
            
            # Update GP model
            gp.fit(config, score)
        
        return best_model
```

### **Module Structure**
1. **Part 1: Define Optimization Space**
   - Which knobs can we turn?
   - What are valid combinations?

2. **Part 2: Simple Search Strategy**
   - Start with grid search
   - Add early stopping
   - Basic Bayesian optimization

3. **Part 3: Constraint Satisfaction**
   ```python
   constraints = {
       'max_memory': 100_000_000,  # 100MB
       'max_latency': 50,           # 50ms
       'min_accuracy': 0.90         # 90%
   }
   ```

4. **Part 4: Hardware-Aware Optimization**
   ```python
   if hardware == 'mobile':
       prioritize(['quantization', 'pruning'])
   elif hardware == 'server':
       prioritize(['kv_cache', 'acceleration'])
   ```

### **Student Deliverables**
- Build optimization search space
- Implement simple Bayesian optimization
- Create hardware-specific strategies
- Auto-optimize their models from previous modules

---

## Module 20: AI Olympics - Competition Infrastructure

### **New Name: "AI Olympics"** ✅

### **Core Infrastructure**
```python
class OlympicsSubmission:
    def __init__(self, team_name, model, optimizer):
        self.team = team_name
        self.model = model
        self.auto_tuner = optimizer
        
    def prepare_submission(self):
        # Standardized profiling
        profile = StandardProfiler()
        
        metrics = {
            'latency': profile.measure_latency(self.model),
            'memory': profile.measure_memory(self.model),
            'accuracy': profile.measure_accuracy(self.model),
            'model_size': profile.measure_size(self.model),
            'innovations': self.describe_innovations()
        }
        
        # Package for submission
        submission = {
            'team': self.team,
            'model': serialize(self.model),
            'metrics': metrics,
            'optimizations_used': self.auto_tuner.get_config()
        }
        
        # Upload to GitHub (for now)
        self.upload_to_github(submission)
        return submission
```

### **Standardized Profiling System**
```python
class StandardProfiler:
    """Ensures fair comparison across all submissions"""
    
    def measure_latency(self, model):
        # Warm up
        for _ in range(10):
            model(self.standard_input)
        
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            model(self.standard_input)
            times.append(time.perf_counter() - start)
        return np.median(times)
    
    def measure_memory(self, model):
        # Peak memory during inference
        # Standardized measurement
```

### **Competition Categories**
1. **Speed Challenge**: Fastest inference time
2. **Size Challenge**: Smallest model with >90% accuracy
3. **Efficiency Challenge**: Best accuracy/resource ratio
4. **Innovation Challenge**: Most creative optimization approach

### **Student Deliverables**
- Complete optimized model
- Standardized profiling results
- Documentation of techniques used
- GitHub submission (temporary solution)
- Innovation report

---

## Next Steps

1. **Get PyTorch expert validation** on:
   - KV cache integration with Module 14 transformers
   - Bayesian optimization simplicity for AutoTuning
   - Standardized profiling fairness

2. **Test integration points**:
   - Module 16 must plug into Module 14 cleanly
   - AutoTuner must work with all optimization techniques

3. **Build competition infrastructure**:
   - Standardized test datasets
   - Fair profiling system
   - Leaderboard visualization (future)