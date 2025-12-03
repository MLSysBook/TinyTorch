#!/bin/bash
#
# TinyTorch Pre-flight Check
# ===========================
# Run this before releases to verify everything works.
#
# Usage:
#   ./scripts/preflight.sh          # Quick check (~1 min)
#   ./scripts/preflight.sh --full   # Full validation (~10 min)
#
# What professionals call this:
#   - "Smoke tests" (quick)
#   - "Release validation" (full)
#   - "Preflight checks" (before deployment)
#

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Print functions
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}▶ $1${NC}"
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((FAILED++))
}

check_skip() {
    echo -e "  ${YELLOW}○${NC} $1 (skipped)"
    ((SKIPPED++))
}

# Determine test level
FULL_TEST=false
if [[ "$1" == "--full" ]]; then
    FULL_TEST=true
fi

# Start
print_header "🔥 TinyTorch Pre-flight Check"
echo ""
if $FULL_TEST; then
    echo -e "${YELLOW}Mode: FULL VALIDATION (~10 minutes)${NC}"
else
    echo -e "${YELLOW}Mode: QUICK CHECK (~1 minute)${NC}"
    echo -e "${YELLOW}      Run with --full for complete validation${NC}"
fi

START_TIME=$(date +%s)

# ============================================================================
# PHASE 1: Environment Checks
# ============================================================================
print_section "Phase 1: Environment"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    check_pass "Python installed: $PYTHON_VERSION"
else
    check_fail "Python not found"
fi

# Check we're in the right directory
if [[ -f "pyproject.toml" && -d "tito" ]]; then
    check_pass "In TinyTorch project root"
else
    check_fail "Not in TinyTorch project root"
    echo -e "${RED}  Run from the TinyTorch directory${NC}"
    exit 1
fi

# Check virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    check_pass "Virtual environment active: $(basename $VIRTUAL_ENV)"
else
    check_skip "No virtual environment active"
fi

# ============================================================================
# PHASE 2: Package Structure
# ============================================================================
print_section "Phase 2: Package Structure"

# Check critical directories
for dir in "src" "modules" "milestones" "tito" "tinytorch" "tests"; do
    if [[ -d "$dir" ]]; then
        check_pass "Directory exists: $dir/"
    else
        check_fail "Directory missing: $dir/"
    fi
done

# Check critical files
for file in "pyproject.toml" "requirements.txt" "README.md"; do
    if [[ -f "$file" ]]; then
        check_pass "File exists: $file"
    else
        check_fail "File missing: $file"
    fi
done

# ============================================================================
# PHASE 3: TinyTorch Package Import
# ============================================================================
print_section "Phase 3: Package Import"

# Test tinytorch import
if python3 -c "import tinytorch" 2>/dev/null; then
    check_pass "import tinytorch"
else
    check_fail "import tinytorch"
fi

# Test core imports (these may fail if modules not exported)
for module in "Tensor" "ReLU" "Linear"; do
    if python3 -c "from tinytorch import $module" 2>/dev/null; then
        check_pass "from tinytorch import $module"
    else
        check_skip "from tinytorch import $module (not exported yet)"
    fi
done

# ============================================================================
# PHASE 4: CLI Commands (tito)
# ============================================================================
print_section "Phase 4: CLI Commands"

# Test basic tito commands
if python3 -m tito.main --version >/dev/null 2>&1; then
    check_pass "tito --version"
else
    check_fail "tito --version"
fi

if python3 -m tito.main --help >/dev/null 2>&1; then
    check_pass "tito --help"
else
    check_fail "tito --help"
fi

# Test key subcommands
for cmd in "module status" "milestones list --simple" "system info"; do
    if python3 -m tito.main $cmd >/dev/null 2>&1; then
        check_pass "tito $cmd"
    else
        check_fail "tito $cmd"
    fi
done

# ============================================================================
# PHASE 5: E2E Quick Tests (pytest)
# ============================================================================
print_section "Phase 5: E2E Quick Tests"

if [[ -f "tests/e2e/test_user_journey.py" ]]; then
    if python3 -m pytest tests/e2e/test_user_journey.py -k quick -q --tb=no 2>/dev/null; then
        check_pass "E2E quick tests passed"
    else
        check_fail "E2E quick tests failed"
    fi
else
    check_skip "E2E tests not found"
fi

# ============================================================================
# PHASE 6: Full Validation (if --full)
# ============================================================================
if $FULL_TEST; then
    print_section "Phase 6: Module Tests"
    
    # Run module 01 tests
    if python3 -m pytest tests/01_tensor/ -q --tb=no 2>/dev/null; then
        check_pass "Module 01 (Tensor) tests"
    else
        check_fail "Module 01 (Tensor) tests"
    fi
    
    # Run CLI tests
    if python3 -m pytest tests/cli/ -q --tb=no 2>/dev/null; then
        check_pass "CLI tests"
    else
        check_fail "CLI tests"
    fi
    
    print_section "Phase 7: Module Flow Tests"
    
    if python3 -m pytest tests/e2e/test_user_journey.py -k module_flow -q --tb=short 2>/dev/null; then
        check_pass "Module flow E2E tests"
    else
        check_fail "Module flow E2E tests"
    fi
    
    print_section "Phase 8: Milestone Verification"
    
    # Check if milestone tests exist and can run
    if [[ -f "tests/milestones/test_learning_verification.py" ]]; then
        echo -e "  ${YELLOW}⏳${NC} Running milestone learning tests (this takes ~90s)..."
        if python3 -m pytest tests/milestones/test_learning_verification.py -q --tb=no 2>/dev/null; then
            check_pass "Milestone learning verification"
        else
            check_fail "Milestone learning verification"
        fi
    else
        check_skip "Milestone tests not found"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_header "📊 Pre-flight Summary"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo ""
echo -e "  ${CYAN}Duration:${NC} ${DURATION}s"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✅ PRE-FLIGHT CHECK PASSED${NC}"
    echo ""
    if ! $FULL_TEST; then
        echo -e "${YELLOW}💡 For complete validation, run: ./scripts/preflight.sh --full${NC}"
    fi
    exit 0
else
    echo -e "${RED}${BOLD}❌ PRE-FLIGHT CHECK FAILED${NC}"
    echo ""
    echo -e "${YELLOW}Fix the issues above before release.${NC}"
    exit 1
fi

