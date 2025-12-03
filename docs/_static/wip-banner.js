/**
 * TinyTorch Top Bar
 * Simple branding bar with navigation links on every page
 * Matches MLSysBook navbar style for consistency
 */

document.addEventListener('DOMContentLoaded', function() {
    // Only inject if not already present
    if (document.getElementById('tinytorch-bar')) return;
    
    const barHTML = `
        <div class="tinytorch-bar" id="tinytorch-bar">
            <div class="tinytorch-bar-content">
                <a href="intro.html" class="tinytorch-bar-brand">
                    <span class="icon">🔥</span>
                    <span>Tiny🔥Torch</span>
                </a>
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
});