/**
 * TinyTorch Announcement Banner
 * Displays branding and quick links on every page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Banner injection: Create banner dynamically if not present in HTML
    let banner = document.getElementById('wip-banner');
    if (!banner) {
        const bannerHTML = `
            <div class="wip-banner" id="wip-banner">
                <div class="wip-banner-content">
                    <div class="wip-banner-title">
                        <span class="icon">🔥</span>
                        <span>Tiny🔥Torch</span>
                    </div>
                    <div class="wip-banner-description">
                        Hands-on labs for <a href="https://mlsysbook.ai" target="_blank">MLSysBook</a>
                        <span class="separator">•</span>
                        <a href="https://tinytorch.ai/join" target="_blank">Join the Community</a>
                        <span class="separator">•</span>
                        <a href="https://github.com/harvard-edge/TinyTorch" target="_blank">GitHub ⭐</a>
                    </div>
                </div>
                <button class="wip-banner-toggle" id="wip-banner-toggle" title="Collapse banner" aria-label="Toggle banner">▲</button>
            </div>
        `;
        document.body.insertAdjacentHTML('afterbegin', bannerHTML);
        banner = document.getElementById('wip-banner');
    }

    const toggleBtn = document.getElementById('wip-banner-toggle');

    if (!banner) return;

    // Check if banner was previously collapsed
    const collapsed = localStorage.getItem('wip-banner-collapsed');
    if (collapsed === 'true') {
        banner.classList.add('collapsed');
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
            toggleBtn.title = 'Expand banner';
        }
    }

    // Toggle collapse/expand
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            const isCollapsed = banner.classList.contains('collapsed');

            if (isCollapsed) {
                banner.classList.remove('collapsed');
                toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
                toggleBtn.title = 'Collapse banner';
                localStorage.setItem('wip-banner-collapsed', 'false');
            } else {
                banner.classList.add('collapsed');
                toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
                toggleBtn.title = 'Expand banner';
                localStorage.setItem('wip-banner-collapsed', 'true');
            }
        });
    }

    // Add smooth transitions
    banner.style.transition = 'all 0.3s ease';
});