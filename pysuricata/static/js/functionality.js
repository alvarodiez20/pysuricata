function toggleDarkModeScoped() {
  const root = document.getElementById('pysuricata-report');
  if (!root) return;
  root.classList.toggle('light');

  const body = document.body;
  if (body && body.classList.contains('suricata-standalone')) {
    body.classList.toggle('light');
  }

  const icon = document.getElementById('toggle-icon');
  if (icon) icon.textContent = root.classList.contains('light') ? '🌙' : '☀️';
}

function downloadReport() {
  try {
    // Grab only our report
    var root = document.getElementById('pysuricata-report');
    if (!root) throw new Error('Report root not found');

    // Keep current theme
    var isLight = root.classList.contains('light');
    var title = (document.title && document.title.trim()) || 'PySuricata Report';

    // Favicon (embedded/base64 if present)
    var fav = document.querySelector('link[rel="icon"][href^="data:image"]');
    var favHTML = fav ? fav.outerHTML : '';

    // Pull only our inline styles (look for our selectors)
    var styles = Array.from(document.querySelectorAll('style'))
      .filter(s => /#pysuricata-report|suricata-standalone/.test(s.textContent || ''))
      .map(s => s.textContent)
      .join('\n');

    // Include dark-mode toggle script if present
    var toggleScriptEl = Array.from(document.querySelectorAll('script'))
      .find(s => /toggleDarkMode/.test(s.textContent || ''));
    var toggleScript = toggleScriptEl ? toggleScriptEl.textContent : '';

    // Build a clean, standalone document
    var standalone = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title>
${favHTML}
<style>${styles}</style>
<script>${toggleScript}<\/script>
</head>
<body class="suricata-standalone${isLight ? ' light' : ''}">
${root.outerHTML}
</body>
</html>`;

    // Download
    var blob = new Blob([standalone], { type: 'text/html;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    var ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    a.download = 'pysuricata-report-' + ts + '.html';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.error('Download failed', e);
  }
  return false; // prevent default navigation
}

// --- Scroll to top functionality for logo/report icon ---
(function () {
  const ROOT_ID = 'pysuricata-report';

  // Add click handler for logo to scroll to top
  document.addEventListener('click', function (e) {
    const logo = e.target.closest('#logo-container, .logo');
    if (!logo) return;

    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(logo)) return;

    // Smooth scroll to top
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });

    e.preventDefault();
    return false;
  });
})();

// --- Header pin/unpin toggle (scoped to #pysuricata-report) ---
(function () {
  const ROOT_ID = 'pysuricata-report';
  const PIN_BTN_ID = 'pin-button';
  const STORAGE_KEY = 'headerPinned';

  function setPinned(pinned) {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const btn = document.getElementById(PIN_BTN_ID);
    const iconOn = document.getElementById('pinIconOn');
    const iconOff = document.getElementById('pinIconOff');
    if (pinned) {
      root.classList.remove('unpinned');
      try { localStorage.setItem(STORAGE_KEY, 'true'); } catch (e) { }
      if (btn) btn.setAttribute('aria-label', 'Unpin header');
      if (iconOn) iconOn.style.display = '';
      if (iconOff) iconOff.style.display = 'none';
    } else {
      root.classList.add('unpinned');
      try { localStorage.setItem(STORAGE_KEY, 'false'); } catch (e) { }
      if (btn) btn.setAttribute('aria-label', 'Pin header');
      if (iconOn) iconOn.style.display = 'none';
      if (iconOff) iconOff.style.display = '';
    }
  }

  // Public toggle for inline onclick hooks
  window.toggleHeaderPinScoped = function () {
    const current = (function () { try { return localStorage.getItem(STORAGE_KEY) !== 'false'; } catch (e) { return true; } })();
    setPinned(!current);
    return false;
  };

  // Insert a pin link into the quick nav if one isn't present
  function ensurePinButton() {
    if (document.getElementById(PIN_BTN_ID)) return;
    const quickNav = document.querySelector('#pysuricata-report .quick');
    if (!quickNav) return;
    const a = document.createElement('a');
    a.href = '#';
    a.id = PIN_BTN_ID;
    a.title = 'Unpin header';
    a.setAttribute('aria-label', 'Unpin header');
    a.setAttribute('onclick', 'return toggleHeaderPinScoped()');
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('width', '16');
    svg.setAttribute('height', '16');
    const pathOn = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathOn.setAttribute('id', 'pinIconOn');
    pathOn.setAttribute('fill', 'currentColor');
    pathOn.setAttribute('d', 'M6.75 1.5h2.5c.414 0 .75.336.75.75V5h1.25c.414 0 .75.336.75.75s-.336.75-.75.75H10v2.1l1.97 1.97a.75.75 0 1 1-1.06 1.06L8.94 9.81 8.75 10v4.25a.75.75 0 0 1-1.5 0V10l-.19-.19-1.97 1.97a.75.75 0 1 1-1.06-1.06L6 8.6V6.5H4.75a.75.75 0 0 1 0-1.5H6V2.25c0-.414.336-.75.75-.75Z');
    const pathOff = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathOff.setAttribute('id', 'pinIconOff');
    pathOff.setAttribute('fill', 'currentColor');
    pathOff.setAttribute('d', 'M3.22 2.22a.75.75 0 0 1 1.06 0l8.5 8.5a.75.75 0 1 1-1.06 1.06L9.5 9.56l-.56.56V14a.75.75 0 0 1-1.5 0V10.12l-.56-.56-2.72 2.72a.75.75 0 1 1-1.06-1.06L5.38 8.5V6.5H4.25a.75.75 0 0 1 0-1.5H6V2.25c0-.414.336-.75.75-.75h2.5c.209 0 .398.085.535.222l-1.06 1.06H7.5V5.94L3.22 2.22Z');
    pathOff.style.display = 'none';
    svg.appendChild(pathOn); svg.appendChild(pathOff);
    a.appendChild(svg);
    quickNav.appendChild(a);
  }

  document.addEventListener('DOMContentLoaded', function () {
    ensurePinButton();
    const pinned = (function () { try { return localStorage.getItem(STORAGE_KEY) !== 'false'; } catch (e) { return true; } })();
    setPinned(pinned);
  });
})();

// --- Histogram controls: bins + scale (per-card, scoped to #pysuricata-report) ---
(function () {
  const ROOT_ID = 'pysuricata-report';
  document.addEventListener('click', function (e) {
    const btn = e.target.closest('.hist-controls button');
    if (!btn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(btn)) return;
    // Ignore Details toggle buttons; handled by dedicated listener below
    if (btn.classList && btn.classList.contains('details-toggle')) return;

    const controls = btn.closest('.hist-controls');
    const card = btn.closest('.var-card');
    if (!controls || !card) return;

    // Update state from the clicked button
    if (btn.hasAttribute('data-bin')) {
      const b = btn.getAttribute('data-bin');
      controls.dataset.bin = b;
      // Activate only within the bin-group
      const binGroup = controls.querySelector('.bin-group');
      if (binGroup) {
        binGroup.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === btn));
      }
    }
    if (btn.hasAttribute('data-scale')) {
      const s = btn.getAttribute('data-scale');
      controls.dataset.scale = s;
      // Activate only within the scale-group
      const scaleGroup = controls.querySelector('.scale-group');
      if (scaleGroup) {
        scaleGroup.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === btn));
      }
    }

    const scale = controls.dataset.scale || 'lin';
    const bin = controls.dataset.bin || '25';
    let targetId = `${card.id}-${scale}-bins-${bin}`;

    // Toggle active variant via class (CSS controls display)
    card.querySelectorAll('.hist.variant').forEach(v => v.classList.remove('active'));
    let target = document.getElementById(targetId);
    if (!target) {
      targetId = `${card.id}-${scale}-bins-25`;
      target = document.getElementById(targetId);
    }
    if (target) target.classList.add('active');
  }, { passive: true });
})();


/* --- Histogram/Datetime hover tooltip --- */
(function () {
  const ROOT_ID = 'pysuricata-report';
  function ensureTip() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return null;
    let tip = root.querySelector('.hist-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'hist-tooltip';
      root.appendChild(tip);
    }
    return tip;
  }
  function showTip(e, html) {
    const root = document.getElementById(ROOT_ID);
    const tip = ensureTip();
    if (!root || !tip) return;
    tip.innerHTML = html;
    tip.style.display = 'block';
    positionTip(e, tip, root);
  }
  function hideTip() {
    const root = document.getElementById(ROOT_ID);
    const tip = root && root.querySelector('.hist-tooltip');
    if (tip) tip.style.display = 'none';
    // Also hide donut tooltip
    const donutTip = root && root.querySelector('.donut-tooltip');
    if (donutTip) donutTip.style.display = 'none';
  }
  function positionTip(e, tip, root) {
    const r = root.getBoundingClientRect();
    let x = e.clientX - r.left + 12;
    let y = e.clientY - r.top + 12;
    const maxX = r.width - tip.offsetWidth - 8;
    const maxY = r.height - tip.offsetHeight - 8;
    if (x > maxX) x = Math.max(8, maxX);
    if (y > maxY) y = Math.max(8, maxY);
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function formatCount(count) {
    if (count >= 1_000_000) {
      return `${(count / 1_000_000).toFixed(1)}M (${count.toLocaleString()})`;
    } else if (count >= 10_000) {
      return `${(count / 1_000).toFixed(1)}K (${count.toLocaleString()})`;
    } else {
      return count.toLocaleString();
    }
  }

  // Timeline histogram hover effects with mathematical notation
  document.addEventListener('mousemove', function (e) {
    const timelineHot = e.target.closest('.dt-svg .hot');
    if (timelineHot) {
      const count = timelineHot.getAttribute('data-count') || '0';
      const pct = timelineHot.getAttribute('data-pct') || '0.0';
      const label = timelineHot.getAttribute('data-label') || '';
      const html = `<div class="line"><strong>${count}</strong> rows <span class="muted">(${pct}%)</span></div>` +
        `<div class="line"><span class="muted">Range:</span> [${label}]</div>`;
      showTip(e, html);
      return;
    }

    // Temporal distribution chart tooltips (hour/dow/month/year)
    const temporalBar = e.target.closest('.temporal-chart .temporal-bar');
    if (temporalBar) {
      const count = parseInt(temporalBar.getAttribute('data-count') || '0');
      const pct = temporalBar.getAttribute('data-pct') || '0.0';
      const label = temporalBar.getAttribute('data-label') || '';

      // Smart number formatting
      const formattedCount = formatCount(count);

      const html = `<div class="line"><strong>${label}</strong></div>` +
        `<div class="line">${formattedCount} records <span class="muted">(${pct}%)</span></div>`;
      showTip(e, html);
      return;
    }

    // Donut chart segment tooltips (Column types pie chart)
    const donutSegment = e.target.closest('.segment-path');
    if (donutSegment) {
      const type = donutSegment.getAttribute('data-type') || 'Unknown';
      const count = donutSegment.getAttribute('data-count') || '0';
      const pct = donutSegment.getAttribute('data-percentage') || '0.0';

      const html = `<div class="line"><strong>${type}</strong></div>` +
        `<div class="line">${count} ${count === '1' ? 'column' : 'columns'} <span class="muted">(${pct}%)</span></div>`;

      // Use donut-tooltip class instead of hist-tooltip
      const root = document.getElementById(ROOT_ID);
      if (!root) return;
      let tip = root.querySelector('.donut-tooltip');
      if (!tip) {
        tip = document.createElement('div');
        tip.className = 'donut-tooltip';
        root.appendChild(tip);
      }
      tip.innerHTML = html;
      tip.style.display = 'block';
      positionTip(e, tip, root);
      return;
    }

    // Missing values distribution tooltips (chunk and spectrum segments)
    const missingSegment = e.target.closest('.chunk-segment, .spectrum-segment');
    if (missingSegment) {
      const startRow = parseInt(missingSegment.getAttribute('data-start') || '0');
      const endRow = parseInt(missingSegment.getAttribute('data-end') || '0');
      const missingCount = parseInt(missingSegment.getAttribute('data-missing') || '0');
      const pct = missingSegment.getAttribute('data-pct') || '0.0';

      // Smart number formatting for row ranges and counts
      const formattedStart = formatCount(startRow);
      const formattedEnd = formatCount(endRow);
      const formattedMissing = formatCount(missingCount);

      const html = `<div class="line"><strong>Rows ${formattedStart}–${formattedEnd}</strong></div>` +
        `<div class="line">${formattedMissing} missing <span class="muted">(${pct}%)</span></div>`;
      showTip(e, html);
      return;
    }

    const bar = e.target.closest('.hist-svg .bar, .dt-svg .bar');
    if (!bar) { hideTip(); return; }
    const isDt = !!bar.closest('.dt-svg');
    const count = bar.getAttribute('data-count') || '0';
    const pct = bar.getAttribute('data-pct') || '0.0';
    if (isDt) {
      const label = bar.getAttribute('data-label') || '';
      const html = `<div class="line"><strong>${count}</strong> rows <span class="muted">(${pct}%)</span></div>` +
        `<div class="line"><span class="muted">Value:</span> ${label}</div>`;
      showTip(e, html);
    } else {
      const x0 = bar.getAttribute('data-x0') || '';
      const x1 = bar.getAttribute('data-x1') || '';
      const html = `<div class="line"><strong>${count}</strong> rows <span class="muted">(${pct}%)</span></div>` +
        `<div class="line"><span class="muted">Range:</span> [${x0}, ${x1})</div>`;
      showTip(e, html);
    }
  }, { passive: true });

  // Hide when leaving a histogram entirely
  document.addEventListener('mouseleave', function (e) {
    if (e.target && e.target.closest &&
        (e.target.closest('.hist-svg') || e.target.closest('.dt-svg') ||
         e.target.closest('.temporal-chart') || e.target.closest('.chunk-distribution') ||
         e.target.closest('.chunk-spectrum') || e.target.closest('.missing-spectrum-bar') ||
         e.target.closest('.dataprep-spectrum'))) {
      hideTip();
    }
  }, true);
})();

/* --- Details section + tabs (full-width) --- */
(function () {
  const ROOT_ID = 'pysuricata-report';

  // Toggle full-width details section controlled via aria-controls
  document.addEventListener('click', function (e) {
    const btn = e.target.closest('.details-toggle');
    if (!btn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(btn)) return;

    const id = btn.getAttribute('aria-controls');
    const panel = id && document.getElementById(id);

    if (panel) {
      const isOpen = !panel.hasAttribute('hidden');

      if (isOpen) {
        panel.setAttribute('hidden', '');
        btn.setAttribute('aria-expanded', 'false');
      } else {
        panel.removeAttribute('hidden');
        btn.setAttribute('aria-expanded', 'true');

        // Ask dt mini‑charts to render with actual widths now that panel is visible
        try {
          const ev = new CustomEvent('suricata:dt:render', { detail: { container: panel } });
          // Single delayed render after layout settles
          setTimeout(() => document.dispatchEvent(ev), 100);
        } catch (e) {
          console.error('Failed to trigger chart render:', e);
        }
      }
      // Prevent any other listeners (e.g., legacy inline) from double-toggling
      e.stopImmediatePropagation();
      e.preventDefault();
      return;
    }

    // Legacy fallback: inline dropdown panel inside .details
    const details = btn.closest('.details');
    const legacy = details && details.querySelector('.details-panel');

    if (!legacy) {
      return;
    }

    const open = !legacy.hasAttribute('hidden');

    if (open) {
      legacy.setAttribute('hidden', '');
      btn.setAttribute('aria-expanded', 'false');
    } else {
      legacy.removeAttribute('hidden');
      btn.setAttribute('aria-expanded', 'true');
    }
  }, { passive: false });

  // Tab switching inside the details section (or legacy panel)
  document.addEventListener('click', function (e) {
    const tabBtn = e.target.closest('.tabs [role="tab"]');
    if (!tabBtn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(tabBtn)) return;

    const container = tabBtn.closest('.details-section') || tabBtn.closest('.details-panel');
    if (!container) return;

    const name = tabBtn.getAttribute('data-tab');
    if (!name) return;

    container.querySelectorAll('.tabs [role="tab"]').forEach(b => b.classList.toggle('active', b === tabBtn));
    container.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.getAttribute('data-tab') === name));
    if (name === 'breakdown') {  // Updated tab name
      try {
        const ev = new CustomEvent('suricata:dt:render', { detail: { container } });
        // Single delayed render after layout settles
        setTimeout(() => document.dispatchEvent(ev), 100);
      } catch (e) { }
    }
  }, { passive: true });
})();

// --- Categorical controls: Top-N + scale (per-card, scoped to #pysuricata-report) ---
(function () {
  const ROOT_ID = 'pysuricata-report';
  document.addEventListener('click', function (e) {
    const btn = e.target.closest('.hist-controls button');
    if (!btn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(btn)) return;

    const controls = btn.closest('.hist-controls');
    const card = btn.closest('.var-card');
    if (!controls || !card) return;

    // Only handle cards that have categorical variants
    const hasCat = card.querySelector('.cat.variant');
    if (!hasCat) return; // let the numeric handler manage others

    // Read current state; set sensible defaults
    let topn = controls.dataset.topn || '10';
    let scale = controls.dataset.scale || 'count';

    // Update state & active styles
    if (btn.hasAttribute('data-topn')) {
      topn = btn.getAttribute('data-topn') || topn;
      controls.dataset.topn = topn;
      const binGroup = controls.querySelector('.bin-group');
      if (binGroup) {
        binGroup.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === btn));
      }
    }
    if (btn.hasAttribute('data-scale')) {
      scale = btn.getAttribute('data-scale') || scale;
      controls.dataset.scale = scale;
      const scaleGroup = controls.querySelector('.scale-group');
      if (scaleGroup) {
        scaleGroup.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === btn));
      }
    }

    // Prefer a scale-specific variant id if present, else fallback to simple top-N id
    let targetId = `${card.id}-cat-${scale}-top-${topn}`;
    let target = document.getElementById(targetId);
    if (!target) {
      targetId = `${card.id}-cat-top-${topn}`;
      target = document.getElementById(targetId);
    }

    // Toggle via active class to align with CSS
    card.querySelectorAll('.cat.variant').forEach(v => v.classList.remove('active'));
    if (target) target.classList.add('active');
  }, { passive: true });
})();

/* --- Variables section and controls --- */
(function () {
  // Variables section controls
  let searchTerm = '';
  let currentFilter = 'all';
  const SEARCH_DEBOUNCE = 300;

  function setupVariablesControls() {
    const controls = document.querySelector('.vars-controls');
    const cardsGrid = document.getElementById('cards-grid');
    const pagination = document.getElementById('pagination');

    if (!controls || !cardsGrid) return;

    // Hide controls if there are 8 or fewer variables
    const totalCards = cardsGrid.children.length;
    if (totalCards <= 8) {
      if (controls) controls.style.display = 'none';
      if (pagination) pagination.style.display = 'none';
      return;
    }

    setupSearch();
    setupFilters();
    setupPagination();
    applyFilters();
  }

  function setupSearch() {
    const searchInput = document.getElementById('search-input');

    if (!searchInput) return;

    let timeout;
    searchInput.addEventListener('input', (e) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        searchTerm = e.target.value.toLowerCase();
        applyFilters();
      }, SEARCH_DEBOUNCE);
    });
  }

  function setupFilters() {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
        currentFilter = e.target.dataset.filter;
        applyFilters();
      });
    });
  }

  function setupPagination() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const paginationInfo = document.getElementById('pagination-info');

    if (!prevBtn || !nextBtn || !paginationInfo) return;

    let currentPage = 1;
    const cardsPerPage = 8;

    function updatePagination() {
      const cards = Array.from(document.querySelectorAll('.var-card:not([style*="display: none"])'));
      const totalPages = Math.ceil(cards.length / cardsPerPage);

      prevBtn.disabled = currentPage === 1;
      nextBtn.disabled = currentPage === totalPages || totalPages === 0;

      const start = (currentPage - 1) * cardsPerPage + 1;
      const end = Math.min(currentPage * cardsPerPage, cards.length);

      if (cards.length === 0) {
        paginationInfo.textContent = 'No variables found';
      } else {
        paginationInfo.textContent = `Showing ${start}-${end} of ${cards.length}`;
      }

      // Show/hide cards based on current page
      cards.forEach((card, index) => {
        const shouldShow = index >= (currentPage - 1) * cardsPerPage &&
          index < currentPage * cardsPerPage;
        card.style.display = shouldShow ? 'block' : 'none';
      });
    }

    prevBtn.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        updatePagination();
      }
    });

    nextBtn.addEventListener('click', () => {
      const cards = Array.from(document.querySelectorAll('.var-card:not([style*="display: none"])'));
      const totalPages = Math.ceil(cards.length / cardsPerPage);
      if (currentPage < totalPages) {
        currentPage++;
        updatePagination();
      }
    });

    // Store update function for use in applyFilters
    window.updatePagination = updatePagination;
  }

  function applyFilters() {
    const cards = document.querySelectorAll('.var-card');

    cards.forEach(card => {
      const cardName = card.querySelector('.var-card__header h3')?.textContent?.toLowerCase() || '';
      const cardType = card.dataset.type || '';

      const matchesSearch = !searchTerm || cardName.includes(searchTerm);
      const matchesFilter = currentFilter === 'all' || cardType === currentFilter;

      if (matchesSearch && matchesFilter) {
        card.style.display = 'block';
      } else {
        card.style.display = 'none';
      }
    });

    // Update pagination if it exists
    if (window.updatePagination) {
      window.updatePagination();
    }
  }

  // Run now if DOM is already loaded (e.g., inside notebooks), else on DOMContentLoaded
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    try {
      setupVariablesControls();
    } catch (e) { }
  } else {
    document.addEventListener('DOMContentLoaded', () => {
      try {
        setupVariablesControls();
      } catch (e) { }
    });
  }
})();

/**
 * Smoothly scroll to the top of the page when the logo is clicked
 */
function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}

function toggleSampleText(detailsElement) {
  const textElement = document.getElementById('sample-toggle-text');
  if (!textElement) return;

  if (detailsElement.open) {
    textElement.textContent = 'Hide sample';
  } else {
    textElement.textContent = 'Show sample';
  }
}

/* Missing Values Section Tab Switching */
(function () {
  function initMissingValuesTabs() {
    const tabButtons = document.querySelectorAll('.missing-section-tabs button');

    if (tabButtons.length === 0) {
      // Not loaded yet, try again
      return;
    }

    tabButtons.forEach(button => {
      // Remove existing listeners to prevent duplicates
      button.replaceWith(button.cloneNode(true));
    });

    // Re-query after cloning
    const freshTabButtons = document.querySelectorAll('.missing-section-tabs button');

    freshTabButtons.forEach(button => {
      button.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();

        const targetTab = this.getAttribute('data-tab');
        console.log('Tab clicked:', targetTab);

        // Update button states
        freshTabButtons.forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');

        // Update content visibility
        const tabContents = document.querySelectorAll('.missing-tab-content');
        tabContents.forEach(content => {
          const contentTab = content.getAttribute('data-tab');
          if (contentTab === targetTab) {
            content.classList.add('active');
            console.log('Activating tab:', contentTab);
          } else {
            content.classList.remove('active');
          }
        });
      });
    });
  }

  // Try to initialize immediately
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMissingValuesTabs);
  } else {
    initMissingValuesTabs();
  }

  // Also try after a short delay to catch dynamically rendered content
  setTimeout(initMissingValuesTabs, 100);
})();

/* Toggle description text */
function toggleDescriptionText(detailsEl) {
  const toggleText = document.getElementById('description-toggle-text');
  if (toggleText) {
    toggleText.textContent = detailsEl.open ? 'Description' : 'Description';
  }
}
