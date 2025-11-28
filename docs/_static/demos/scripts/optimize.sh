#!/bin/bash
# Optimize GIF file sizes using gifsicle
#
# Prerequisites:
#   - gifsicle installed: brew install gifsicle
#
# Usage:
#   ./scripts/optimize-gifs.sh              # Optimize all GIFs
#   ./scripts/optimize-gifs.sh 01-*.gif     # Optimize specific GIF

set -e

DEMO_DIR="site/_static/demos"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "🗜️  TinyTorch GIF Optimizer"
echo "==========================="
echo ""

# Check gifsicle installation
if ! command -v gifsicle &> /dev/null; then
    echo -e "${RED}❌ gifsicle not found${NC}"
    echo ""
    echo "Install gifsicle with:"
    echo "  macOS:  brew install gifsicle"
    echo "  Linux:  apt-get install gifsicle"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ gifsicle found:${NC} $(gifsicle --version | head -n1)"
echo ""

# Function to optimize a single GIF
optimize_gif() {
    local input_file=$1
    local temp_file="${input_file}.tmp"
    
    if [ ! -f "$input_file" ]; then
        echo -e "${RED}❌ File not found: $input_file${NC}"
        return 1
    fi
    
    echo -e "📦 Optimizing: $(basename "$input_file")"
    
    # Get original size
    original_size=$(wc -c < "$input_file" | awk '{print $1}')
    original_mb=$(echo "scale=2; $original_size / 1048576" | bc)
    
    # Optimize with gifsicle
    # -O3: Maximum optimization
    # --lossy=80: Lossy compression (reduces size significantly)
    # --colors 256: Reduce to 256 colors if needed
    if gifsicle -O3 --lossy=80 --colors 256 "$input_file" -o "$temp_file" 2>/dev/null; then
        # Get new size
        new_size=$(wc -c < "$temp_file" | awk '{print $1}')
        new_mb=$(echo "scale=2; $new_size / 1048576" | bc)
        
        # Calculate savings
        saved=$(( original_size - new_size ))
        saved_mb=$(echo "scale=2; $saved / 1048576" | bc)
        percent=$(echo "scale=1; ($saved * 100) / $original_size" | bc)
        
        # Replace original with optimized version
        mv "$temp_file" "$input_file"
        
        echo -e "   ${GREEN}✅ ${original_mb} MB → ${new_mb} MB${NC} (saved ${saved_mb} MB, ${percent}%)"
    else
        echo -e "   ${RED}❌ Optimization failed${NC}"
        rm -f "$temp_file"
        return 1
    fi
    
    echo ""
}

# Determine which GIFs to optimize
if [ $# -gt 0 ]; then
    # Optimize specific GIFs provided as arguments
    for pattern in "$@"; do
        for gif in "$DEMO_DIR"/$pattern; do
            if [ -f "$gif" ]; then
                optimize_gif "$gif"
            fi
        done
    done
else
    # Optimize all GIFs
    for gif in "$DEMO_DIR"/*.gif; do
        if [ -f "$gif" ]; then
            optimize_gif "$gif"
        fi
    done
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 Optimization complete!${NC}"
echo ""
echo "Final sizes:"
for gif in "$DEMO_DIR"/*.gif; do
    if [ -f "$gif" ]; then
        size=$(wc -c < "$gif" | awk '{print $1}')
        size_mb=$(echo "scale=2; $size / 1048576" | bc)
        echo "  $(basename "$gif"): ${size_mb} MB"
    fi
done
echo ""

