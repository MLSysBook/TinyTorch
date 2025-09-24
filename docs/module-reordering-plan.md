# TinyTorch Module Reordering Plan

## Current vs New Beautiful Order

### **Current Order (Phase 2 Issues):**
```
01_setup
02_tensor  
03_activations
04_layers
05_losses
06_autograd          ← Problem: Autograd before optimizers
07_dataloader        ← Problem: DataLoader before training
08_optimizers        ← Problem: Optimizers after autograd
09_spatial           ← Problem: Spatial before training
10_training          ← Problem: Training comes last
11_tokenization
12_embeddings  
13_attention
14_transformers
15_acceleration
16_caching
17_precision
18_compression
19_benchmarking
20_capstone
```

### **New Beautiful Order:**
```
01_setup
02_tensor
03_activations  
04_layers
05_losses
06_optimizers        ← Fixed: Optimizers after losses (systematic weight updates)
07_autograd          ← Fixed: Autograd after optimizers (automatic gradients)
08_training          ← Fixed: Training as bridge (systematic procedures)
09_spatial           ← Fixed: Spatial after training (architectural improvements)
10_dataloader        ← Fixed: DataLoader last (efficiency solution)
11_tokenization
12_embeddings
13_attention
14_transformers
15_acceleration
16_caching
17_precision
18_compression
19_benchmarking
20_capstone
```

## Specific Changes Needed:

### **Module Renumbering:**
- `06_autograd` → `07_autograd`
- `07_dataloader` → `10_dataloader` 
- `08_optimizers` → `06_optimizers`
- `09_spatial` → `09_spatial` (stays)
- `10_training` → `08_training`

### **Dependencies to Update:**
- **Training module (new 08)**: Remove DataLoader imports, use single-sample iteration
- **Spatial module (new 09)**: Can now use Training procedures from module 08
- **DataLoader module (new 10)**: Show speedup vs Training module's single-sample approach

### **Step-by-Step Reordering Process:**
1. Create temporary backup
2. Rename modules to new numbers  
3. Update internal imports and references
4. Update module.yaml files with new numbers
5. Update all documentation and examples
6. Update master roadmap and tutorial plans
7. Test integration and exports

## Files That Need Updates:

### **Module Files:**
- Module directories need renaming
- `module.yaml` files need number updates
- README files need prerequisite updates
- Python files need import path updates

### **Documentation Files:**
- `COMPLETE_MODULE_ROADMAP.md`
- `tutorial-design-rationale.md` 
- All example files referencing modules
- Checkpoint system mappings

### **Integration Files:**
- Test files with module dependencies
- Export/import configurations
- CLI command mappings

This reordering will create the beautiful "inevitable discovery" progression we designed!