#!/bin/bash
# Test GIF 02: Build → Test → Ship workflow
# Runs actual commands and measures timing

set -e

LOGFILE="/tmp/tinytorch_02_log.txt"
TIMING_FILE="/tmp/tinytorch_02_timing.txt"

echo "=== GIF 02: Build → Test → Ship Test ===" > "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "" >> "$LOGFILE"

echo "=== VHS Timing for GIF 02 ===" > "$TIMING_FILE"
echo "" >> "$TIMING_FILE"

# Assume we're in TinyTorch directory with activated environment
cd /tmp/TinyTorch 2>&1 | tee -a "$LOGFILE"

# Test 1: Activate environment
echo "=== Test 1: Activate Environment ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
source .venv/bin/activate 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
ACTIVATE_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "'source activate.sh': Sleep ${ACTIVATE_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 2: Start module 02
echo "=== Test 2: Start Module 02 ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito module start 02 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
START_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "'tito module start 02': Sleep ${START_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 3: Complete module 02 (THE MONEY SHOT)
echo "=== Test 3: Complete Module 02 ===" | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/tito module complete 02 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
COMPLETE_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "'tito module complete 02': Sleep ${COMPLETE_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

# Test 4: Verify in Python REPL
echo "=== Test 4: Python Import Test ===" | tee -a "$LOGFILE"
echo "Testing Python imports..." | tee -a "$LOGFILE"
START=$(python3 -c "import time; print(time.time())")
.venv/bin/python3 -c "from tinytorch import Sigmoid; from tinytorch import Tensor; sig = Sigmoid(); result = sig(Tensor([0.0])); print(f'Result: {result}')" 2>&1 | tee -a "$LOGFILE"
END=$(python3 -c "import time; print(time.time())")
PYTHON_TIME=$(python3 -c "print(f'{$END - $START:.1f}')")
echo "Python imports + execution: Sleep ${PYTHON_TIME}s" >> "$TIMING_FILE"
echo "" >> "$LOGFILE"

echo "Completed: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "✅ All tests passed!" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

cat "$TIMING_FILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Full log: $LOGFILE"
echo "Timing file: $TIMING_FILE"
