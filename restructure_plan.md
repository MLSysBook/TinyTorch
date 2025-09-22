# TinyTorch Three-Part Restructure Plan

## Current → Target Structure

### Part I: Foundations (Modules 1-5)
- 01_setup      → 01_setup (no change)
- 02_tensor     → 02_tensor (no change)  
- 03_activations → 03_activations (no change)
- 04_layers     → 04_layers (no change)
- 05_dense      → 05_networks (rename for clarity)

### Part II: Computer Vision (Modules 6-11)
- 06_spatial    → 06_spatial (no change)
- 07_attention  → 13_attention (move to Part III)
- 08_dataloader → 07_dataloader (renumber)
- NEW           → 08_normalization (create from scratch or extract)
- 09_autograd   → 09_autograd (no change)
- 10_optimizers → 10_optimizers (no change)
- 11_training   → 11_training (no change)

### Part III: Language Models (Modules 12-17)
- NEW           → 12_embeddings (create new)
- 07_attention  → 13_attention (moved from Part II, expanded)
- NEW           → 14_transformers (create new)
- NEW           → 15_generation (create new)
- 12_compression → 16_regularization (rename and expand)
- 13_kernels + 14_benchmarking + 15_mlops → 17_systems (combine)

## Migration Steps

1. Fix duplicate 09_dataloader issue
2. Rename modules that need renaming
3. Move modules to new positions
4. Create new modules for language model components
5. Update imports and dependencies
6. Update checkpoint system
7. Update documentation

