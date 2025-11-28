#!/bin/bash
# Validate all carousel GIFs are present and meet quality standards
#
# Usage:
#   ./scripts/validate-gifs.sh

set -e

DEMO_DIR="site/_static/demos"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "✅ TinyTorch GIF Validator"
echo "=========================="
echo ""

# Expected GIFs
EXPECTED_GIFS=(
    "01-zero-to-ready.gif"
    "02-build-test-ship.gif"
    "03-milestone-unlocked.gif"
    "04-share-journey.gif"
)

# Quality thresholds
MAX_SIZE_MB=6
MIN_SIZE_KB=100

# Track validation status
all_valid=true

echo "Checking GIF presence and quality..."
echo ""

for gif_name in "${EXPECTED_GIFS[@]}"; do
    gif_path="$DEMO_DIR/$gif_name"
    
    echo -n "📦 $gif_name ... "
    
    # Check if file exists
    if [ ! -f "$gif_path" ]; then
        echo -e "${RED}❌ MISSING${NC}"
        all_valid=false
        continue
    fi
    
    # Check file size
    size=$(wc -c < "$gif_path" | awk '{print $1}')
    size_mb=$(echo "scale=2; $size / 1048576" | bc)
    size_kb=$(echo "scale=0; $size / 1024" | bc)
    
    # Validate size range
    if (( $(echo "$size < $MIN_SIZE_KB * 1024" | bc -l) )); then
        echo -e "${RED}❌ TOO SMALL${NC} (${size_kb} KB < ${MIN_SIZE_KB} KB minimum)"
        all_valid=false
        continue
    fi
    
    if (( $(echo "$size > $MAX_SIZE_MB * 1048576" | bc -l) )); then
        echo -e "${YELLOW}⚠️  LARGE${NC} (${size_mb} MB > ${MAX_SIZE_MB} MB recommended)"
        echo "   Consider running: ./scripts/optimize-gifs.sh $gif_name"
    else
        echo -e "${GREEN}✅ OK${NC} (${size_mb} MB)"
    fi
    
    # Check if it's actually a GIF
    if ! file "$gif_path" | grep -q "GIF image data"; then
        echo -e "   ${RED}❌ NOT A VALID GIF${NC}"
        all_valid=false
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$all_valid" = true ]; then
    echo -e "${GREEN}🎉 All GIFs validated successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Preview: open $DEMO_DIR/*.gif"
    echo "  2. Deploy: cd site && ./build.sh"
    exit 0
else
    echo -e "${RED}❌ Validation failed. Fix the issues above.${NC}"
    echo ""
    echo "To regenerate GIFs:"
    echo "  ./scripts/generate-demo-gifs.sh"
    exit 1
fi

