# VS Code Jupytext Cell Visual Cues

Simple approach using built-in visual indicators to distinguish markdown from code cells.

## Visual Cues:
- **Markdown cells**: `# %% [markdown]` + triple quotes (`"""`) 
- **Code cells**: `# %%` + regular Python syntax

## What you see:
- **Cell markers** clearly indicate cell type
- **Triple quotes** naturally stand out with string highlighting
- **Syntax highlighting** makes content type obvious

## Files:
- `.vscode/settings.json` - Basic Python project settings
- `.vscode/extensions.json` - Recommends Python extension

Clean, simple, and works reliably across all VS Code themes! 