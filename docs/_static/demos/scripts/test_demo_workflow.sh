#!/bin/bash
# Test TinyTorch demo workflow and log all output with timing measurements
# This validates commands work before generating VHS tapes

set -e  # Exit on error

LOGFILE="/tmp/tinytorch_demo_log.txt"
TIMING_FILE="/tmp/tinytorch_timing.txt"
DEMO_DIR="/tmp/TinyTorch"

echo "=== TinyTorch Demo Test ===" > "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Initialize timing log
echo "=== VHS Timing Recommendations ===" > "$TIMING_FILE"
echo "Measured execution times for VHS Sleep commands:" >> "$TIMING_FILE"
echo "" >> "$TIMING_FILE"

# Clean up any existing TinyTorch in /tmp
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
START=$(python3 -c "import time; print(time.time())")
git clone https://github.com/mlsysbook/TinyTorch.git 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
CLONE_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "✓ Repository cloned (${CLONE_TIME}s)" | tee -a "$LOGFILE"
echo "git clone: Sleep ${CLONE_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 2: Run setup
echo "=== Test 2: Environment Setup ===" | tee -a "$LOGFILE"
cd TinyTorch 2>&1 | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
bash setup-environment.sh 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
SETUP_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "✓ Environment setup complete (${SETUP_TIME}s)" | tee -a "$LOGFILE"
echo "./setup-environment.sh: Sleep ${SETUP_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 3: Verify tito is available
echo "=== Test 3: Verify TITO Installation ===" | tee -a "$LOGFILE"
.venv/bin/python -c "import sys; print(f'Python: {sys.executable}')" 2>&1 | tee -a "$LOGFILE"
.venv/bin/tito --version 2>&1 | tee -a "$LOGFILE"
echo "✓ TITO verified" | tee -a "$LOGFILE"
echo "'source activate.sh': Sleep 1s (activation is instant)" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 4: TITO system health
echo "=== Test 4: TITO System Health ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito system health 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
HEALTH_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "tito system health: Sleep ${HEALTH_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 5: Check module status
echo "=== Test 5: Module Status ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito module status 2>&1 | head -30 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
STATUS_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "tito module status: Sleep ${STATUS_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 6: TITO logo
echo "=== Test 6: TITO Logo ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito logo 2>&1 | head -20 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
LOGO_TIME=$(python3 -c "import time; print(f'{$END - $START:.1f}')")
echo "tito logo: Sleep ${LOGO_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 7: List milestones
echo "=== Test 7: List Milestones ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito milestones list 2>&1 | head -30 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
MILESTONES_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "tito milestones list: Sleep ${MILESTONES_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

echo "Completed: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "✅ All tests passed!" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Show timing summary
cat "$TIMING_FILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Full log: $LOGFILE"
echo "Timing file: $TIMING_FILE"
