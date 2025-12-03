# TinyTorch Carousel Demo Recordings

This directory contains Terminalizer configurations and generated GIF demos for the TinyTorch workflow carousel.

## Quick Start

### Regenerate all GIFs
```bash
./render-all.sh
```

This will render all 4 carousel GIFs using the pre-configured Terminalizer YAML files.

## Requirements

- **Node.js v16** (managed via nvm) - Required for Terminalizer
- **Terminalizer** - Terminal session recorder/renderer
- **macOS with GUI session** - Electron requires desktop environment

### Installation

```bash
# 1. Install nvm (Node Version Manager) if not already installed
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# 2. Reload shell or run:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 3. Install Node v16
nvm install 16
nvm use 16

# 4. Install Terminalizer globally
npm install -g terminalizer

# 5. Verify installation
terminalizer --version
```

### Configuration

All demo configs inherit settings from `base-config.yml`:
- **Dimensions:** 100 columns × 24 rows (optimized for carousel display)
- **Theme:** Vibrant color scheme with good contrast
- **Font:** Monaco/Lucida Console monospace at 14px
- **Quality:** 100% for crisp, clean output

**Note:** Terminalizer doesn't support config inheritance. `base-config.yml` serves as the reference/source of truth. To change styling across all demos, update `base-config.yml` then manually sync to individual `*.yml` files.

## Files

```
site/_static/demos/
├── base-config.yml             # Shared config (source of truth for styling)
├── 01-clone-setup.yml          # Demo config: Clone & Setup
├── 01-clone-setup.gif          # Generated GIF
├── 02-build-jupyter.yml        # Demo config: Build in Jupyter
├── 02-build-jupyter.gif        # Generated GIF
├── 03-export-tito.yml          # Demo config: Export with TITO
├── 03-export-tito.gif          # Generated GIF
├── 04-validate-history.yml     # Demo config: Validate with History
├── 04-validate-history.gif     # Generated GIF
├── render-all.sh               # Script to regenerate all GIFs
└── README.md                   # This file
```

## Usage

### Render individual GIFs
```bash
nvm use 16
cd site/_static/demos
terminalizer render 01-clone-setup -o 01-clone-setup.gif
```

### Render all GIFs at once
```bash
./render-all.sh
```

### Edit configurations
Edit the `*.yml` files to modify the terminal sessions. Each YAML file contains:
- Terminal appearance settings (theme, font, size)
- Typed commands and delays
- Simulated output text

## How It Works

Terminalizer uses Electron to render pre-scripted terminal sessions as animated GIFs:

1. YAML configs define what to type and display
2. Terminalizer renders frames using Electron
3. Frames are merged into animated GIF
4. GIFs are displayed in the website carousel with emoji fallbacks

## Troubleshooting

### "Cannot read property 'dock' of undefined"
- Terminalizer requires a GUI session (Electron/app.dock API)
- Make sure you're running in a full macOS desktop environment
- Won't work over SSH or in headless mode

### "node-pty build failed"
- You're using Node v17+
- Switch to Node v16: `nvm use 16`

### Want to update the carousel?
- Edit the YAML files to change the terminal sessions
- Run `./render-all.sh` to regenerate GIFs
- Rebuild the site: `jupyter-book build site`
- GIFs will automatically display (with emoji fallbacks if missing)

## Architecture

The carousel in `site/intro.md` references these GIFs with fallback emojis:
- If GIF exists: displays animated terminal recording
- If GIF missing: displays emoji icon (💻 📓 🛠️ 🏆)

This ensures the carousel always works, even without generating GIFs.
