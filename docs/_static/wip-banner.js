/**
 * TinyTorch Top Bar
 * Smart sticky bar: hides on scroll down, shows on scroll up
 * Matches MLSysBook navbar style for consistency
 */

document.addEventListener('DOMContentLoaded', function() {
    // Only inject if not already present
    if (document.getElementById('tinytorch-bar')) return;
    
    const barHTML = `
        <div class="tinytorch-bar" id="tinytorch-bar">
            <div class="tinytorch-bar-content">
                <div class="tinytorch-bar-left">
                    <a href="intro.html" class="tinytorch-bar-brand">
                        <span class="icon">🔥</span>
                        <span>Tiny🔥Torch</span>
                    </a>
                    <span class="tinytorch-bar-wip">🚧 Under Construction</span>
                </div>
                <div class="tinytorch-bar-links">
                    <a href="https://mlsysbook.ai" target="_blank" class="link-book">
                        <span class="link-icon">📖</span>
                        <span class="link-text">MLSysBook</span>
                    </a>
                    <a href="https://buttondown.email/mlsysbook" target="_blank" class="link-subscribe">
                        <span class="link-icon">✉️</span>
                        <span class="link-text">Subscribe</span>
                    </a>
                    <a href="https://github.com/harvard-edge/TinyTorch" target="_blank" class="link-star">
                        <span class="link-icon">⭐</span>
                        <span class="link-text">Star</span>
                    </a>
                    <a href="https://tinytorch.ai/join" target="_blank" class="link-community">
                        <span class="link-icon">🌍</span>
                        <span class="link-text">Community</span>
                    </a>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('afterbegin', barHTML);
    
    // Smart sticky: hide on scroll down, show on scroll up
    const bar = document.getElementById('tinytorch-bar');
    let lastScrollY = window.scrollY;
    let ticking = false;
    
    function updateBar() {
        const currentScrollY = window.scrollY;
        
        if (currentScrollY < 50) {
            // Always show at top of page
            bar.classList.remove('hidden');
        } else if (currentScrollY > lastScrollY) {
            // Scrolling down - hide
            bar.classList.add('hidden');
        } else {
            // Scrolling up - show
            bar.classList.remove('hidden');
        }
        
        lastScrollY = currentScrollY;
        ticking = false;
    }
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(updateBar);
            ticking = true;
        }
    }, { passive: true });
});