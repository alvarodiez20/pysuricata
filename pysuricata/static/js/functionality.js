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
    var ts = new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
    a.download = 'pysuricata-report-' + ts + '.html';
    document.body.appendChild(a);
    a.click();
    setTimeout(function(){ URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.error('Download failed', e);
  }
  return false; // prevent default navigation
}

// --- Header pin/unpin toggle (scoped to #pysuricata-report) ---
(function(){
  const ROOT_ID = 'pysuricata-report';
  const PIN_BTN_ID = 'pin-button';
  const STORAGE_KEY = 'headerPinned';

  function setPinned(pinned){
    const root = document.getElementById(ROOT_ID);
    if(!root) return;
    const btn = document.getElementById(PIN_BTN_ID);
    const iconOn = document.getElementById('pinIconOn');
    const iconOff = document.getElementById('pinIconOff');
    if(pinned){
      root.classList.remove('unpinned');
      try { localStorage.setItem(STORAGE_KEY, 'true'); } catch(e){}
      if(btn) btn.setAttribute('aria-label','Unpin header');
      if(iconOn) iconOn.style.display = '';
      if(iconOff) iconOff.style.display = 'none';
    } else {
      root.classList.add('unpinned');
      try { localStorage.setItem(STORAGE_KEY, 'false'); } catch(e){}
      if(btn) btn.setAttribute('aria-label','Pin header');
      if(iconOn) iconOn.style.display = 'none';
      if(iconOff) iconOff.style.display = '';
    }
  }

  // Public toggle for inline onclick hooks
  window.toggleHeaderPinScoped = function(){
    const current = (function(){ try { return localStorage.getItem(STORAGE_KEY) !== 'false'; } catch(e){ return true; } })();
    setPinned(!current);
    return false;
  };

  // Insert a pin link into the quick nav if one isn't present
  function ensurePinButton(){
    if (document.getElementById(PIN_BTN_ID)) return;
    const quickNav = document.querySelector('#pysuricata-report .quick');
    if(!quickNav) return;
    const a = document.createElement('a');
    a.href = '#';
    a.id = PIN_BTN_ID;
    a.title = 'Unpin header';
    a.setAttribute('aria-label','Unpin header');
    a.setAttribute('onclick','return toggleHeaderPinScoped()');
    const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('aria-hidden','true');
    svg.setAttribute('viewBox','0 0 16 16');
    svg.setAttribute('width','16');
    svg.setAttribute('height','16');
    const pathOn = document.createElementNS('http://www.w3.org/2000/svg','path');
    pathOn.setAttribute('id','pinIconOn');
    pathOn.setAttribute('fill','currentColor');
    pathOn.setAttribute('d','M6.75 1.5h2.5c.414 0 .75.336.75.75V5h1.25c.414 0 .75.336.75.75s-.336.75-.75.75H10v2.1l1.97 1.97a.75.75 0 1 1-1.06 1.06L8.94 9.81 8.75 10v4.25a.75.75 0 0 1-1.5 0V10l-.19-.19-1.97 1.97a.75.75 0 1 1-1.06-1.06L6 8.6V6.5H4.75a.75.75 0 0 1 0-1.5H6V2.25c0-.414.336-.75.75-.75Z');
    const pathOff = document.createElementNS('http://www.w3.org/2000/svg','path');
    pathOff.setAttribute('id','pinIconOff');
    pathOff.setAttribute('fill','currentColor');
    pathOff.setAttribute('d','M3.22 2.22a.75.75 0 0 1 1.06 0l8.5 8.5a.75.75 0 1 1-1.06 1.06L9.5 9.56l-.56.56V14a.75.75 0 0 1-1.5 0V10.12l-.56-.56-2.72 2.72a.75.75 0 1 1-1.06-1.06L5.38 8.5V6.5H4.25a.75.75 0 0 1 0-1.5H6V2.25c0-.414.336-.75.75-.75h2.5c.209 0 .398.085.535.222l-1.06 1.06H7.5V5.94L3.22 2.22Z');
    pathOff.style.display = 'none';
    svg.appendChild(pathOn); svg.appendChild(pathOff);
    a.appendChild(svg);
    quickNav.appendChild(a);
  }

  document.addEventListener('DOMContentLoaded', function(){
    ensurePinButton();
    const pinned = (function(){ try { return localStorage.getItem(STORAGE_KEY) !== 'false'; } catch(e){ return true; } })();
    setPinned(pinned);
  });
})();

// --- Histogram controls: bins + scale (per-card, scoped to #pysuricata-report) ---
(function(){
  const ROOT_ID = 'pysuricata-report';
  document.addEventListener('click', function(e){
    const btn = e.target.closest('.hist-controls button');
    if (!btn) return;
    const root = document.getElementById(ROOT_ID);
    if (!root || !root.contains(btn)) return;

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
    const targetId = `${card.id}-${scale}-bins-${bin}`;

    // Hide all variants within this card, then show selected
    card.querySelectorAll('.hist.variant').forEach(v => { v.style.display = 'none'; });
    const target = document.getElementById(targetId);
    if (target) target.style.display = '';
  }, {passive:true});
})();


/* --- Histogram/Datetime hover tooltip --- */
(function(){
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
    const tip  = ensureTip();
    if (!root || !tip) return;
    tip.innerHTML = html;
    tip.style.display = 'block';
    positionTip(e, tip, root);
  }
  function hideTip() {
    const root = document.getElementById(ROOT_ID);
    const tip  = root && root.querySelector('.hist-tooltip');
    if (tip) tip.style.display = 'none';
  }
  function positionTip(e, tip, root) {
    const r = root.getBoundingClientRect();
    let x = e.clientX - r.left + 12;
    let y = e.clientY - r.top  + 12;
    const maxX = r.width  - tip.offsetWidth  - 8;
    const maxY = r.height - tip.offsetHeight - 8;
    if (x > maxX) x = Math.max(8, maxX);
    if (y > maxY) y = Math.max(8, maxY);
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }

  // Delegated mouse tracking over bars (numeric hist + datetime mini charts)
  document.addEventListener('mousemove', function(e){
    const bar = e.target.closest('.hist-svg .bar, .dt-svg .bar');
    if (!bar) { hideTip(); return; }
    const isDt = !!bar.closest('.dt-svg');
    const count = bar.getAttribute('data-count') || '0';
    const pct   = bar.getAttribute('data-pct')   || '0.0';
    if (isDt) {
      const label = bar.getAttribute('data-label') || '';
      const html = `<div class="line"><strong>${count}</strong> rows <span class="muted">(${pct}%)</span></div>` +
                   `<div class="line"><span class="muted">Value:</span> ${label}</div>`;
      showTip(e, html);
    } else {
      const x0    = bar.getAttribute('data-x0')    || '';
      const x1    = bar.getAttribute('data-x1')    || '';
      const html = `<div class="line"><strong>${count}</strong> rows <span class="muted">(${pct}%)</span></div>` +
                   `<div class="line"><span class="muted">Range:</span> ${x0} – ${x1}</div>`;
      showTip(e, html);
    }
  }, {passive:true});

  // Hide when leaving a histogram entirely
  document.addEventListener('mouseleave', function(e){
    if (e.target && e.target.closest && (e.target.closest('.hist-svg') || e.target.closest('.dt-svg'))) {
      hideTip();
    }
  }, true);
})();

/* --- Details section + tabs (full-width) --- */
(function(){
  const ROOT_ID = 'pysuricata-report';

  // Toggle full-width details section controlled via aria-controls
  document.addEventListener('click', function(e){
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
          document.dispatchEvent(ev);
          // Run again after layout settles
          setTimeout(() => document.dispatchEvent(ev), 50);
        } catch(e) {}
      }
      // Prevent any other listeners (e.g., legacy inline) from double-toggling
      e.stopImmediatePropagation();
      e.preventDefault();
      return;
    }

    // Legacy fallback: inline dropdown panel inside .details
    const details = btn.closest('.details');
    const legacy = details && details.querySelector('.details-panel');
    if (!legacy) return;
    const open = !legacy.hasAttribute('hidden');
    if (open) {
      legacy.setAttribute('hidden','');
      btn.setAttribute('aria-expanded','false');
    } else {
      legacy.removeAttribute('hidden');
      btn.setAttribute('aria-expanded','true');
    }
  }, {passive: false});

  // Tab switching inside the details section (or legacy panel)
  document.addEventListener('click', function(e){
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
    if (name === 'dist') {
      try {
        const ev = new CustomEvent('suricata:dt:render', { detail: { container } });
        document.dispatchEvent(ev);
        setTimeout(() => document.dispatchEvent(ev), 50);
      } catch(e) {}
    }
  }, {passive: true});
})();

// --- Categorical controls: Top-N + scale (per-card, scoped to #pysuricata-report) ---
(function(){
  const ROOT_ID = 'pysuricata-report';
  document.addEventListener('click', function(e){
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

    // Hide/show only categorical variants
    card.querySelectorAll('.cat.variant').forEach(v => { v.style.display = 'none'; });
    if (target) target.style.display = '';
  }, {passive:true});
})();

/* --- Datetime mini-charts renderer (hour/DOW/month/YEAR) --- */
(function(){
  const ROOT_ID = 'pysuricata-report';
  const DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];

  function renderAll(){
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const metas = root.querySelectorAll('script[type="application/json"][id$="-dt-meta"]');
    metas.forEach(m => {
      let data; try { data = JSON.parse(m.textContent || '{}'); } catch(e){ data = null; }
      if (!data) return;
      const colId = m.id.replace(/-dt-meta$/, '');
      // Hour / DOW / Month
      drawBar(colId + '-dt-hour',  data.counts && data.counts.hour,  Array.from({length:24}, (_,i)=>String(i).padStart(2,'0')+':00'));
      drawBar(colId + '-dt-dow',   data.counts && data.counts.dow,   DOW);
      drawBar(colId + '-dt-month', data.counts && data.counts.month, MONTHS.map(m => m.slice(0,3)));
      // YEAR (dynamic labels)
      const yr = data.counts && data.counts.year;
      if (yr && Array.isArray(yr.values) && Array.isArray(yr.labels)) {
        drawBar(colId + '-dt-year', yr.values, yr.labels.map(String));
      }
    });
  }

  // Render all charts within a provided container (DOM node)
  function renderIn(container){
    const metas = container.querySelectorAll && container.querySelectorAll('script[type="application/json"][id$="-dt-meta"]');
    if (!metas || !metas.length) return;
    metas.forEach(m => {
      let data; try { data = JSON.parse(m.textContent || '{}'); } catch(e){ data = null; }
      if (!data) return;
      const colId = m.id.replace(/-dt-meta$/, '');
      drawBar(colId + '-dt-hour',  data.counts && data.counts.hour,  Array.from({length:24}, (_,i)=>String(i).padStart(2,'0')+':00'));
      drawBar(colId + '-dt-dow',   data.counts && data.counts.dow,   DOW);
      drawBar(colId + '-dt-month', data.counts && data.counts.month, MONTHS);
      const yr = data.counts && data.counts.year;
      if (yr && Array.isArray(yr.values) && Array.isArray(yr.labels)) {
        drawBar(colId + '-dt-year', yr.values, yr.labels.map(String));
      }
    });
  }

  function drawBar(containerId, values, labels){
    const el = document.getElementById(containerId);
    if (!el || !values || !labels || values.length === 0) return;
    const n = values.reduce((a,b)=>a+(+b||0), 0) || 1;
    const W = el.clientWidth || 420;
    const H = el.clientHeight || 90;
    const ML = 36, MR = 8, MT = 8, MB = 20;
    const iw = W - ML - MR;
    const ih = H - MT - MB;
    const max = Math.max(1, Math.max.apply(null, values));

    function sx(i){ return ML + (i + 0.5) / values.length * iw; }
    function sy(v){ return MT + (1 - v / max) * ih; }

    // Label density
    const dense = labels.length > 12;
    const ticks = [];
    for (let i = 0; i < labels.length; i++) {
      if (!dense || i % Math.ceil(labels.length/6) === 0 || i === labels.length - 1) {
        ticks.push({i, label: String(labels[i])});
      }
    }

    const parts = [];
    parts.push(`<svg class="dt-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`);
    // Axis
    const xAxisY = MT + ih;
    parts.push(`<line class="axis" x1="${ML}" y1="${xAxisY}" x2="${ML+iw}" y2="${xAxisY}"></line>`);

    // Bars
    const bw = Math.max(1, iw / values.length * 0.9);
    for (let i = 0; i < values.length; i++) {
      const v = +values[i] || 0;
      const x = sx(i) - bw/2;
      const y = sy(v);
      const h = (MT + ih) - y;
      const pct = ((v / n) * 100).toFixed(1);
      const label = labels[i];
      parts.push(
        `<rect class="bar" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${bw.toFixed(2)}" height="${Math.max(0,h).toFixed(2)}" data-count="${v}" data-pct="${pct}" data-label="${label}"><title>${v} rows\n${label}</title></rect>`
      );
    }

    // X tick labels
    ticks.forEach(t => {
      const tx = sx(t.i);
      parts.push(`<text class="tick-label" x="${tx.toFixed(2)}" y="${xAxisY + 12}" text-anchor="middle">${t.label}</text>`);
    });

    parts.push('</svg>');
    el.innerHTML = parts.join('');
  }

  // Debounce helper
  function debounce(fn, ms){ let t; return function(){ clearTimeout(t); t = setTimeout(() => fn.apply(this, arguments), ms); }; }
  const onResize = debounce(() => { try { renderAll(); } catch(e) {} }, 120);
  window.addEventListener('resize', onResize);

  document.addEventListener('suricata:dt:render', function(e){
    const c = e && e.detail && e.detail.container;
    if (c) { try { renderIn(c); } catch(e2) {} }
  });

  // Run now if DOM is already loaded (e.g., inside notebooks), else on DOMContentLoaded
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    try { renderAll(); } catch (e) {}
  } else {
    document.addEventListener('DOMContentLoaded', renderAll);
  }
})();