/* --- Core action handlers (no inline onclick) --- */
(function () {
  'use strict';

  var ROOT_ID = 'pysuricata-report';

  function toggleDarkMode() {
    var root = document.getElementById(ROOT_ID);
    if (!root) return;

    var body = document.body;

    // Suppress transitions across the flip, or the theme arrives in waves: the
    // stylesheet's hover transitions animate colour too, at every duration from
    // .12s to .3s, and each section catches up in its own time. The reflows are
    // load-bearing — they flush the suppression before the colours move, and the
    // new colours before it is lifted. Without them the browser coalesces all
    // three class changes into one style pass and animates anyway.
    root.classList.add('theme-switching');
    if (body) body.classList.add('theme-switching');
    void root.offsetHeight;

    root.classList.toggle('light');
    if (body && body.classList.contains('suricata-standalone')) {
      body.classList.toggle('light');
    }

    void root.offsetHeight;
    root.classList.remove('theme-switching');
    if (body) body.classList.remove('theme-switching');

    var icon = document.getElementById('toggle-icon');
    if (icon) {
      var moonSvg = '<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
      var sunSvg = '<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
      icon.innerHTML = root.classList.contains('light') ? moonSvg : sunSvg;
    }
  }

  function downloadReport() {
    try {
      var root = document.getElementById(ROOT_ID);
      if (!root) throw new Error('Report root not found');

      var isLight = root.classList.contains('light');
      var title = (document.title && document.title.trim()) || 'PySuricata Report';

      var fav = document.querySelector('link[rel="icon"][href^="data:image"]');
      var favHTML = fav ? fav.outerHTML : '';

      var styles = Array.from(document.querySelectorAll('style'))
        .filter(function (s) { return /#pysuricata-report|suricata-standalone/.test(s.textContent || ''); })
        .map(function (s) { return s.textContent; })
        .join('\n');

      var toggleScriptEl = Array.from(document.querySelectorAll('script'))
        .find(function (s) { return /toggleDarkMode/.test(s.textContent || ''); });
      var toggleScript = toggleScriptEl ? toggleScriptEl.textContent : '';

      var standalone = '<!DOCTYPE html>\n<html lang="en">\n<head>\n' +
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n' +
        '<title>' + title + '</title>\n' + favHTML + '\n' +
        '<style>' + styles + '</style>\n' +
        '<script>' + toggleScript + '<\/script>\n' +
        '</head>\n<body class="suricata-standalone' + (isLight ? ' light' : '') + '">\n' +
        root.outerHTML + '\n</body>\n</html>';

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
  }

  // An explicit `behavior` beats the CSS `scroll-behavior` property, so the
  // reduced-motion rule in _01-base.css cannot reach these on its own.
  function motionBehavior() {
    return window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
      ? 'auto'
      : 'smooth';
  }

  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: motionBehavior() });
  }

  // Delegated click handler for data-action attributes
  document.addEventListener('click', function (e) {
    var actionEl = e.target.closest('[data-action]');
    if (!actionEl) return;

    var root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(actionEl)) return;

    var action = actionEl.getAttribute('data-action');
    switch (action) {
      case 'scroll-to-top':
        scrollToTop();
        e.preventDefault();
        break;
      case 'download-report':
        downloadReport();
        e.preventDefault();
        break;
      case 'toggle-dark-mode':
        toggleDarkMode();
        break;
      case 'toggle-pin':
        if (window._pysuricataTogglePin) window._pysuricataTogglePin();
        break;
      case 'edit-description':
        if (typeof startDescriptionEdit === 'function') startDescriptionEdit();
        break;
    }
  });

  // Sample details toggle text
  var sampleDetails = document.getElementById('sample-details');
  if (sampleDetails) {
    sampleDetails.addEventListener('toggle', function () {
      var textEl = document.getElementById('sample-toggle-text');
      if (textEl) {
        textEl.textContent = sampleDetails.open ? 'Hide sample' : 'Show sample';
      }
    });
  } else {
    // Retry on DOMContentLoaded if not yet available
    document.addEventListener('DOMContentLoaded', function () {
      var el = document.getElementById('sample-details');
      if (el) {
        el.addEventListener('toggle', function () {
          var textEl = document.getElementById('sample-toggle-text');
          if (textEl) {
            textEl.textContent = el.open ? 'Hide sample' : 'Show sample';
          }
        });
      }
    });
  }

  // Expose for backward compatibility (e.g. downloaded reports)
  window.toggleDarkModeScoped = toggleDarkMode;
  window.downloadReport = downloadReport;
  window.scrollToTop = scrollToTop;
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
      // title as well as aria-label: updating only the latter left the
      // tooltip saying "Unpin header" on a header that was already unpinned,
      // so the accessible name and the visible one disagreed.
      if (btn) { btn.setAttribute('aria-label', 'Unpin header'); btn.title = 'Unpin header'; }
      if (iconOn) iconOn.style.display = '';
      if (iconOff) iconOff.style.display = 'none';
    } else {
      root.classList.add('unpinned');
      try { localStorage.setItem(STORAGE_KEY, 'false'); } catch (e) { }
      if (btn) { btn.setAttribute('aria-label', 'Pin header'); btn.title = 'Pin header'; }
      if (iconOn) iconOn.style.display = 'none';
      if (iconOff) iconOff.style.display = '';
    }
  }

  // Public toggle — called via data-action delegation
  function togglePin() {
    const current = (function () { try { return localStorage.getItem(STORAGE_KEY) !== 'false'; } catch (e) { return true; } })();
    setPinned(!current);
    return false;
  }
  window._pysuricataTogglePin = togglePin;
  // Legacy alias for backward compatibility (downloaded reports)
  window.toggleHeaderPinScoped = togglePin;

  // Insert a pin link into the quick nav if one isn't present
  function ensurePinButton() {
    if (document.getElementById(PIN_BTN_ID)) return;
    // The icon group, not the nav. A pin dropped into .quick lands among the
    // text section links, and on mobile into the rail that has to fit five
    // labels at 390px without scrolling.
    const quickNav = document.querySelector('#pysuricata-report .bar-actions')
      || document.querySelector('#pysuricata-report .quick');
    if (!quickNav) return;
    const a = document.createElement('a');
    a.href = '#';
    a.id = PIN_BTN_ID;
    a.title = 'Unpin header';
    a.setAttribute('aria-label', 'Unpin header');
    a.setAttribute('data-action', 'toggle-pin');
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
    // `.hot` rather than `.dt-svg .hot`: #219 rebuilt the timeline as a
    // `figure.hist` so it could reuse the histogram's classes, and the SVG
    // became `.hist-svg`. Nothing has carried `.dt-svg` since, so this
    // selector matched nothing and the timeline's tooltip was dead while its
    // 60 hotspots carried count, percentage and bucket label (#233).
    const timelineHot = e.target.closest('.hot');
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

    // The donut's segment tooltip is gone with the donut. It existed because
    // an arc cannot be read to a value, so the number had to be revealed on
    // hover; the stacked bar prints each count inside its own segment and
    // lists every count in the legend, including the zeros. A tooltip that
    // repeats what is already on screen is a hover target that does nothing.

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

    const bar = e.target.closest('.hist-svg .bar');
    if (!bar) { hideTip(); return; }
    // Decided by the data the branch needs rather than by an ancestor's class.
    // A datetime bar labels a bucket; a numeric one spans a range. Keying on
    // `.dt-svg` meant a container rename silently switched every datetime bar
    // to the numeric format, printing `Range: [, )` from attributes it does
    // not have (#233).
    const isDt = bar.hasAttribute('data-label');
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
        (e.target.closest('.hist-svg') ||
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

    // A `.details-panel` fallback for a pre-refactor layout used to run here.
    // No renderer has emitted that class since the details section landed, so
    // it was a second code path nobody exercised, reachable only if the first
    // branch above failed to find its target -- which is precisely when a
    // silent `return` is the wrong behaviour.
  }, { passive: false });

  // Tab switching inside the details section (or legacy panel)
  document.addEventListener('click', function (e) {
    const tabBtn = e.target.closest('.tabs [role="tab"]');
    if (!tabBtn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(tabBtn)) return;

    const container = tabBtn.closest('.details-section');
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

/* The missing-values tab switcher lived here. #120 replaced the tabs with a
   route on chunk count, and test_missing_section_views.py asserts the markup
   stays gone -- so this listened for clicks on elements no report emits. */
