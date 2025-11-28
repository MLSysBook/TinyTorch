# VHS Tape: Quick Test
# Purpose: Test that VHS setup works with torch prompt
# Duration: 5 seconds

Output "gifs/00-test.gif"

# Window bar for realistic terminal look (must be at top)
Set WindowBar Colorful

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme "Catppuccin Mocha"
Set Padding 60
Set Framerate 30
Set TypingSpeed 100ms
Set LoopOffset 0%

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

# Simple test
Type "echo 'Testing TinyTorch prompt...'"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 1s

Type "echo 'Torch emoji: 🔥'"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 1s

Type "echo 'Setup works!'"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 2s
