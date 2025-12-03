# TinyTorch Course Book

This directory contains the TinyTorch course content built with [Jupyter Book](https://jupyterbook.org/).

## 🌐 View Online

**Live website:** https://mlsysbook.github.io/TinyTorch/

## 📚 Build Options

### Option 1: HTML (Default Website)

```bash
cd site
jupyter-book build .
```

Output: `_build/html/index.html`

### Option 2: PDF (Simple Method - Recommended)

No LaTeX installation required!

```bash
cd site
make install-pdf    # Install dependencies
make pdf-simple     # Build PDF
```

Output: `_build/tinytorch-course.pdf`

### Option 3: PDF (LaTeX Method - Professional Quality)

Requires LaTeX installation (texlive, mactex, etc.)

```bash
cd site
make pdf
```

Output: `_build/latex/tinytorch-course.pdf`

## 🚀 Quick Commands

Using the Makefile (recommended):

```bash
make html        # Build website
make pdf-simple  # Build PDF (no LaTeX needed)
make pdf         # Build PDF via LaTeX
make clean       # Remove build artifacts
make install     # Install dependencies
make install-pdf # Install PDF dependencies
```

Using scripts directly:

```bash
./build_pdf_simple.sh  # PDF without LaTeX
./build_pdf.sh         # PDF with LaTeX
```

## 📖 Detailed Documentation

See **[PDF_BUILD_GUIDE.md](PDF_BUILD_GUIDE.md)** for:
- Complete setup instructions
- Troubleshooting guide
- Configuration options
- Build performance details

## 🏗️ Structure

```
site/
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents
├── chapters/                # Course chapters (01-20)
├── _static/                 # Images, CSS, JavaScript
├── intro.md                 # Book introduction
├── quickstart-guide.md      # Quick start for students
├── tito-essentials.md       # CLI reference
└── ...                      # Additional course pages
```

## 🎯 Content Overview

### 📚 20 Technical Chapters

**Foundation Tier (01-07):**
- Tensor operations, activations, layers, losses, autograd, optimizers, training

**Architecture Tier (08-13):**
- DataLoader, convolutional networks (CNNs), tokenization, embeddings, attention, transformers

**Optimization Tier (14-19):**
- Profiling, memoization (KV caching), quantization, compression, acceleration, benchmarking

**Capstone (20):**
- Torch Olympics Competition project

## 🔧 Development

### Local Development Server

```bash
jupyter-book build . --path-output ./_build-dev
python -m http.server 8000 -d _build-dev/html
```

Visit: http://localhost:8000

### Auto-rebuild on Changes

```bash
pip install sphinx-autobuild
sphinx-autobuild site site/_build/html
```

## 🤝 Contributing

To contribute to the course content:

1. Edit chapter files in `chapters/`
2. Test your changes: `jupyter-book build .`
3. Preview in browser: Open `_build/html/index.html`
4. Submit PR with your improvements

## 📦 Dependencies

Core dependencies are in `requirements.txt`:
- jupyter-book
- numpy, matplotlib
- sphinxcontrib-mermaid
- rich (for CLI output)

PDF dependencies (optional):
- `pyppeteer` (HTML-to-PDF, no LaTeX)
- LaTeX distribution (for pdflatex method)

## 🎓 For Instructors

**Using this book for teaching:**

1. **Host locally:** Build and serve on your institution's server
2. **Customize content:** Modify chapters for your course
3. **Generate PDFs:** Distribute offline reading material
4. **Track progress:** Use the checkpoint system for assessment

See [instructor guide](instructor-guide.md) for more details.

## 📝 License

MIT License - see LICENSE file in repository root

## 🐛 Issues

Report issues: https://github.com/mlsysbook/TinyTorch/issues

---

**Build ML systems from scratch. Understand how things work.**

