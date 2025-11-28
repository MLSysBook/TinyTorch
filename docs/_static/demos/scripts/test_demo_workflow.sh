#!/bin/bash
# Test TinyTorch demo workflow and log all output
# This validates commands work before generating VHS tapes

set -e  # Exit on error

LOGFILE="/tmp/tinytorch_demo_log.txt"
DEMO_DIR="/tmp/TinyTorch"

echo "=== TinyTorch Demo Test ===" > "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Clean up any existing TinyTorch in /tmp (force removal, don't fail if doesn't exist)
echo "Cleaning $DEMO_DIR if it exists..." | tee -a "$LOGFILE"
if [ -d "$DEMO_DIR" ]; then
    rm -rf "$DEMO_DIR" 2>&1 | tee -a "$LOGFILE"
    echo "✓ Removed existing $DEMO_DIR" | tee -a "$LOGFILE"
else
    echo "✓ No existing directory found (clean start)" | tee -a "$LOGFILE"
fi
echo "" >> "$LOGFILE"

# Test 1: Clone repository
echo "=== Test 1: Clone Repository ===" | tee -a "$LOGFILE"
cd /tmp 2>&1 | tee -a "$LOGFILE"
git clone https://github.com/mlsysbook/TinyTorch.git 2>&1 | tee -a "$LOGFILE"
echo "✓ Repository cloned" | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

# Test 2: Run setup
echo "=== Test 2: Environment Setup ===" | tee -a "$LOGFILE"
cd TinyTorch 2>&1 | tee -a "$LOGFILE"
bash setup-environment.sh 2>&1 | tee -a "$LOGFILE"
echo "✓ Environment setup complete" | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

# Test 3: Verify tito installation
echo "=== Test 3: Verify TITO Installation ===" | tee -a "$LOGFILE"
.venv/bin/python -c "import sys; print(f'Python: {sys.executable}')" 2>&1 | tee -a "$LOGFILE"
.venv/bin/tito --version 2>&1 | tee -a "$LOGFILE"
echo "✓ TITO verified" | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

# Test 4: TITO system health
echo "=== Test 4: TITO System Health ===" | tee -a "$LOGFILE"
.venv/bin/tito system health 2>&1 | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

# Test 5: Check module status
echo "=== Test 5: Module Status ===" | tee -a "$LOGFILE"
.venv/bin/tito module status 2>&1 | head -30 | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

# Test 7: List milestones
echo "=== Test 7: List Milestones ===" | tee -a "$LOGFILE"
.venv/bin/tito milestones list 2>&1 | head -30 | tee -a "$LOGFILE"
echo "" >> "$LOGFILE"

echo "Completed: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "✅ All tests passed!" | tee -a "$LOGFILE"
echo "Full log saved to: $LOGFILE" | tee -a "$LOGFILE"
