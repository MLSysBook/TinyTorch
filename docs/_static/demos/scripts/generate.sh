#!/bin/bash
# Generate all 4 carousel GIFs using VHS
# 
# Prerequisites:
#   - VHS installed: brew install vhs
#   - TinyTorch environment active: source activate.sh
#
# Usage:
#   ./scripts/generate-demo-gifs.sh              # Generate all GIFs
#   ./scripts/generate-demo-gifs.sh --single 01  # Generate only GIF 1

set -e

DEMO_DIR="docs/_static/demos"
TAPE_DIR="$DEMO_DIR/tapes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "🎬 TinyTorch GIF Producer"
echo "========================="
echo ""

# Check VHS installation
if ! command -v vhs &> /dev/null; then
    echo -e "${RED}❌ VHS not found${NC}"
    echo ""
    echo "Install VHS with:"
    echo "  macOS:  brew install vhs"
    echo "  Linux:  go install github.com/charmbracelet/vhs@latest"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ VHS found:${NC} $(vhs --version)"

# Check TITO availability
if ! command -v tito &> /dev/null; then
    echo -e "${YELLOW}⚠️  TITO not found${NC}"
    echo "Activate environment: source activate.sh"
    echo ""
    echo "Note: VHS will still generate GIFs, but they'll use simulated output"
    echo ""
fi

# Create directories
mkdir -p "$DEMO_DIR"
mkdir -p "$TAPE_DIR"

# Clean /tmp/TinyTorch to ensure fresh demo environment
echo "🧹 Cleaning /tmp/TinyTorch for fresh demo..."
rm -rf /tmp/TinyTorch 2>/dev/null || true
echo "✅ Demo environment ready"
echo ""

# Function to generate a single GIF
generate_gif() {
    local num=$1
    local name=$2
    local tape_file="$TAPE_DIR/${num}-${name}.tape"
    local output_file="$DEMO_DIR/${num}-${name}.gif"
    
    echo -e "${BLUE}🎬 Generating GIF ${num}: ${name}...${NC}"
    
    if [ ! -f "$tape_file" ]; then
        echo -e "${RED}❌ Tape file not found: ${tape_file}${NC}"
        return 1
    fi
    
    # Generate GIF with VHS
    if vhs "$tape_file"; then
        echo -e "${GREEN}✅ GIF ${num} complete${NC}"
        
        # Check file size
        if [ -f "$output_file" ]; then
            size=$(wc -c < "$output_file" | awk '{print $1}')
            size_mb=$(echo "scale=2; $size / 1048576" | bc)
            echo -e "   📁 Size: ${size_mb} MB"
            
            # Warn if too large
            if (( $(echo "$size > 6000000" | bc -l) )); then
                echo -e "   ${YELLOW}⚠️  Large file size. Consider optimization.${NC}"
            fi
        fi
        echo ""
        return 0
    else
        echo -e "${RED}❌ Failed to generate GIF ${num}${NC}"
        echo ""
        return 1
    fi
}

# Parse command line arguments
SINGLE_GIF=""
if [ "$1" = "--single" ] && [ -n "$2" ]; then
    SINGLE_GIF="$2"
fi

# Generate GIFs
if [ -n "$SINGLE_GIF" ]; then
    case $SINGLE_GIF in
        01|1)
            generate_gif "01" "zero-to-ready"
            ;;
        02|2)
            generate_gif "02" "build-test-ship"
            ;;
        03|3)
            generate_gif "03" "milestone-unlocked"
            ;;
        04|4)
            generate_gif "04" "share-journey"
            ;;
        *)
            echo -e "${RED}❌ Invalid GIF number: $SINGLE_GIF${NC}"
            echo "Valid options: 01, 02, 03, 04"
            exit 1
            ;;
    esac
else
    # Generate all GIFs
    generate_gif "01" "zero-to-ready"
    generate_gif "02" "build-test-ship"
    generate_gif "03" "milestone-unlocked"
    generate_gif "04" "share-journey"
fi

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 GIF generation complete!${NC}"
echo ""
echo "📁 Output directory: $DEMO_DIR"
echo ""
echo "Generated GIFs:"
for gif in "$DEMO_DIR"/*.gif; do
    if [ -f "$gif" ]; then
        basename "$gif"
    fi
done
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Preview GIFs: open $DEMO_DIR/*.gif"
echo "  2. Optimize if needed: ./scripts/optimize-gifs.sh"
echo "  3. Update website carousel: cd docs && ./build.sh"
echo ""

