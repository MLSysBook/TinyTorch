# Marimo Integration for TinyTorch

## What is Marimo?

[Marimo](https://marimo.io/) is a modern, reactive Python notebook platform that:
- **Stores notebooks as pure Python** (`.py` files) - Git-friendly!
- **Reactive execution** - Cells update automatically when dependencies change
- **Interactive elements** - Built-in widgets, sliders, dataframes
- **AI-native** - Built-in AI assistance and copilots
- **Share as apps** - Export to HTML or serve as web apps
- **Reproducible** - Deterministic execution, no hidden state

## Why Marimo for TinyTorch?

**Perfect Fit:**
1. ✅ **Git-friendly** - Notebooks stored as `.py` files (matches TinyTorch's Python-first approach!)
2. ✅ **Reactive** - Great for teaching (students see changes propagate automatically)
3. ✅ **Educational** - Used by Stanford, UC Berkeley, Princeton, etc.
4. ✅ **Modern** - Better than Jupyter for many use cases
5. ✅ **Open source** - Free and community-driven

## Marimo vs Current Options

| Feature | MyBinder | Colab | Marimo |
|---------|----------|-------|--------|
| Git-friendly | ❌ (.ipynb) | ❌ (.ipynb) | ✅ (.py files) |
| Reactive | ❌ | ❌ | ✅ |
| AI assistance | ❌ | ✅ | ✅ |
| Free | ✅ | ✅ | ✅ |
| Zero-setup | ✅ | ⚠️ (needs account) | ✅ |
| GPU access | ❌ | ✅ | ⚠️ (limited) |

## Integration Options

### Option 1: Marimo Molab Badges

Marimo provides "molab" badges that can open notebooks directly from GitHub:

```
https://marimo.app/molab?repo=mlsysbook/TinyTorch&path=path/to/notebook.py
```

**How it works:**
- Notebooks stored as `.py` files in repo
- Badge links to marimo's cloud service
- Opens notebook in marimo's online editor
- No local installation needed

### Option 2: Add to Launch Buttons

Jupyter Book doesn't natively support marimo launch buttons, but we can:
1. Add custom HTML/JavaScript to create marimo badges
2. Use marimo's badge generator
3. Add manual links in notebook pages

### Option 3: Convert Notebooks to Marimo Format

Since marimo uses `.py` files, we could:
1. Keep current `.ipynb` files for Jupyter/Colab/Binder
2. Generate `.py` versions for marimo
3. Add marimo badges alongside existing launch buttons

## Recommendation

**Add Marimo as an Option:**

1. **Keep current setup** (MyBinder + Colab) - they work well
2. **Add marimo badges** to notebook pages for students who want reactive notebooks
3. **Generate `.py` versions** of notebooks for marimo compatibility

**Benefits:**
- Students get choice of notebook platforms
- Marimo's reactive execution helps with learning
- Git-friendly format aligns with TinyTorch's Python-first approach
- Modern, educational tool used by top universities

## Implementation Steps

### Step 1: Generate Marimo-Compatible Notebooks

Since TinyTorch already uses Python-first development (`*_dev.py` files), we could:
- Convert assignment notebooks to marimo format
- Or create marimo-specific versions

### Step 2: Add Marimo Badges

Add to notebook pages:
```html
<a href="https://marimo.app/molab?repo=mlsysbook/TinyTorch&path=site/chapters/modules/01_tensor.py">
  <img src="https://marimo.app/badge.svg" alt="Open in Marimo">
</a>
```

### Step 3: Document Marimo Usage

Add to student documentation:
- How to use marimo with TinyTorch
- Benefits of reactive notebooks
- Comparison with Jupyter/Colab

## Current Status

**Not yet integrated** - but marimo would be a great addition!

**Next steps if you want to add it:**
1. Test marimo with TinyTorch notebooks
2. Generate marimo-compatible `.py` files
3. Add badges to site pages
4. Update documentation

## Resources

- [Marimo Website](https://marimo.io/)
- [Marimo Docs](https://docs.marimo.io/)
- [Marimo Gallery](https://marimo.io/gallery)
- [Marimo Badge Generator](https://marimo.io/badge)

