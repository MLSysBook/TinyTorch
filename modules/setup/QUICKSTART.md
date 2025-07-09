# TinyTorch Quick Start Guide

Get your TinyTorch environment set up in 5 minutes!

## 🚀 Option 1: Automated Setup (Recommended)

**Step 1**: Clone and navigate
```bash
git clone <repository-url>
cd TinyTorch
```

**Step 2**: Run the automated setup
```bash
python3 projects/setup/create_env.py
```

**Step 3**: Activate environment and verify
```bash
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

python3 projects/setup/check_setup.py
```

**Done!** If you see all ✅ checkmarks, you're ready to code.

---

## 🔧 Option 2: Manual Setup

**Step 1**: Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 2**: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 3**: Verify setup
```bash
python3 projects/setup/check_setup.py
```

---

## 🎯 Every Day Workflow

**Start working** (run this every time):
```bash
cd TinyTorch
source .venv/bin/activate  # Always activate first!
```

**Check status**:
```bash
python3 bin/tito.py info
```

**Run tests**:
```bash
python3 -m pytest projects/setup/test_setup.py -v
```

---

## ❗ Common Issues

**"ModuleNotFoundError"**: You forgot to activate your virtual environment
```bash
source .venv/bin/activate
```

**"Command not found"**: Make sure you're in the TinyTorch directory
```bash
cd TinyTorch
ls  # Should see bin/, projects/, requirements.txt
```

**Dependencies missing**: Reinstall in the virtual environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📝 What's in Your Environment

After setup, you'll have:
- ✅ Python 3.8+ with exact dependency versions
- ✅ pytest for running tests  
- ✅ All TinyTorch course materials
- ✅ Isolated environment (no conflicts with other projects)

**Next**: Read `projects/setup/README.md` for detailed instructions! 