/* Six interactive figures for the PySuricata docs.
 *
 * Authored as a Claude artifact; the drawing code below is unmodified. Only the
 * runtime around it changed: dc-lite.js replaces React + ReactDOM + Babel from
 * a CDN with a small vanilla shim, so the figures load offline and add no
 * third-party requests to the documentation site.
 *
 * FIG 1 reservoir sampling   -> algorithms/sampling.md
 * FIG 2 Misra-Gries eviction -> algorithms/sketches.md
 * FIG 3 memory curve         -> why-pysuricata.md   (PLACEHOLDER NUMBERS, see #68)
 * FIG 4 chunk lifecycle      -> architecture.md
 * FIG 5 Welford -> Pebay     -> algorithms/streaming.md
 * FIG 6 annotated report card-> examples.md
 */
const NS = "http://www.w3.org/2000/svg";
const FONT = "system-ui,-apple-system,sans-serif";
const MONO = "ui-monospace,SFMono-Regular,Menlo,monospace";

function mk(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function txt(parent, x, y, s, attrs) {
  const n = mk("text", Object.assign({ x: x, y: y, fill: "var(--muted)", style: "font:11px " + FONT }, attrs || {}), parent);
  n.textContent = s;
  return n;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const fmt = (n) => n.toLocaleString("en-US");
const LIGHT = { "--fg": "#0b0b0b", "--muted": "#52514e", "--hair": "#b6b5ae", "--bg": "#ffffff", "--panel": "rgba(0,0,0,0.018)", "--chipbr": "rgba(0,0,0,0.10)", "--chipbg": "rgba(0,0,0,0.05)" };
const DARK = { "--fg": "#e8e7e3", "--muted": "#a5a49e", "--hair": "#4b4a46", "--bg": "#17171a", "--panel": "rgba(255,255,255,0.035)", "--chipbr": "rgba(255,255,255,0.13)", "--chipbg": "rgba(255,255,255,0.08)" };

class Component extends DCLogic {
  constructor(props) {
    super(props);
    this.state = { theme: props.startTheme === "dark" ? "dark" : "light", resPlaying: true, mgPlaying: true };
    const r = () => React.createRef();
    this.R = {
      root: r(), mem: r(), res: r(), mg: r(), chunk: r(), peb: r(), card: r(), cardWrap: r(),
      legend: r(), mermaid: r(), mgKeys: r(), resSpeed: r(), resK: r(), resKLabel: r(),
      pebNa: r(), pebNb: r(), pebNaLabel: r(), pebNbLabel: r(),
      ram8: r(), ram16: r(), ram32: r(), ram64: r()
    };
    this.ram = 16;
    this.resSpeedVal = 8;
    this.resKVal = 8;
    this.peb = { ma: -1.4, mb: 1.1, na: 400, nb: 900, M2a: 1200, M2b: 900 };
  }

  componentDidMount() {
    this.applyTheme();
    if (this.R.resSpeed.current) this.R.resSpeed.current.value = this.resSpeedVal;
    if (this.R.resK.current) this.R.resK.current.value = this.resKVal;
    if (this.R.pebNa.current) this.R.pebNa.current.value = this.peb.na;
    if (this.R.pebNb.current) this.R.pebNb.current.value = this.peb.nb;
    this.buildMem();
    this.buildRes();
    this.buildMG();
    this.buildChunk();
    this.buildPeb();
    this.buildCard();
    if (this.R.mermaid.current) this.R.mermaid.current.textContent = this.mermaidSrc();
    this.last = performance.now();
    this.loop = (t) => {
      const dt = Math.min(0.05, (t - this.last) / 1000);
      this.last = t;
      try { this.tickRes(dt); } catch (e) { console.error("tickRes", e); }
      try { this.tickMG(dt); } catch (e) { console.error("tickMG", e); }
      try { this.tickChunk(dt); } catch (e) { console.error("tickChunk", e); }
      this.raf = requestAnimationFrame(this.loop);
    };
    this.raf = requestAnimationFrame(this.loop);
  }
  componentDidUpdate() { this.applyTheme(); }
  componentWillUnmount() { cancelAnimationFrame(this.raf); }

  applyTheme() {
    const el = this.R.root.current;
    if (!el) return;
    const v = this.state.theme === "dark" ? DARK : LIGHT;
    for (const k in v) el.style.setProperty(k, v[k]);
    el.style.setProperty("--accent", this.props.accent || (this.state.theme === "dark" ? "#6aa6ec" : "#2a78d6"));
    el.style.setProperty("--accent2", this.props.accent2 || (this.state.theme === "dark" ? "#f08a5d" : "#eb6834"));
    this.markRam();
  }

  /* ---------------- 3 · memory curve ---------------- */
  buildMem() {
    const s = this.R.mem.current;
    if (!s) return;
    s.addEventListener("pointermove", (e) => {
      const b = s.getBoundingClientRect();
      this.memHover = ((e.clientX - b.left) * 820) / b.width;
      this.drawMem();
    });
    s.addEventListener("pointerleave", () => { this.memHover = null; this.drawMem(); });
    this.drawMem();
  }
  memGeom() {
    return { X0: 66, X1: 654, Y0: 40, Y1: 286, lo: 4, hi: 8 };
  }
  drawMem() {
    const s = this.R.mem.current;
    if (!s) return;
    s.innerHTML = "";
    const g = this.memGeom();
    const cap = this.ram * 1024;
    const ymax = cap * 1.12;
    const X = (lr) => g.X0 + ((lr - g.lo) / (g.hi - g.lo)) * (g.X1 - g.X0);
    const Y = (mb) => g.Y1 - (Math.min(mb, ymax) / ymax) * (g.Y1 - g.Y0);
    const ps = (rows) => 168 + 9 * (Math.log10(rows) - 4);
    const yd = (rows) => rows * 0.0014;
    const failRows = cap / 0.0014;
    const failLr = Math.log10(failRows);

    for (let i = 1; i <= 3; i++) {
      const mb = (cap / 3) * i;
      mk("line", { x1: g.X0, x2: g.X1, y1: Y(mb), y2: Y(mb), stroke: "var(--hair)", "stroke-width": 1, opacity: i === 3 ? 0.9 : 0.45, "stroke-dasharray": i === 3 ? "4 3" : "" }, s);
      txt(s, g.X0 - 10, Y(mb) + 4, i === 3 ? fmt(Math.round(mb / 1024)) + " GB" : fmt(Math.round(mb / 1024)) + " GB", { "text-anchor": "end", opacity: i === 3 ? 1 : 0.75 });
    }
    txt(s, g.X0 - 10, Y(0) + 4, "0", { "text-anchor": "end", opacity: 0.75 });
    txt(s, 20, 22, "peak memory", { style: "font:11px " + FONT, "text-anchor": "start" });
    mk("line", { x1: g.X0, x2: g.X1, y1: g.Y1, y2: g.Y1, stroke: "var(--hair)", "stroke-width": 1.5 }, s);

    const ticks = [4, 5, 6, 7, 8];
    const tl = ["10k", "100k", "1M", "10M", "100M"];
    ticks.forEach((t, i) => {
      mk("line", { x1: X(t), x2: X(t), y1: g.Y1, y2: g.Y1 + 6, stroke: "var(--hair)" }, s);
      txt(s, X(t), g.Y1 + 20, tl[i], { "text-anchor": "middle" });
    });
    txt(s, (g.X0 + g.X1) / 2, g.Y1 + 40, "rows · log scale", { "text-anchor": "middle" });

    let dy = "";
    const endLr = Math.min(failLr, g.hi);
    for (let lr = g.lo; lr <= endLr + 0.001; lr += 0.05) {
      const l = Math.min(lr, endLr);
      dy += (dy ? "L" : "M") + X(l).toFixed(1) + " " + Y(yd(Math.pow(10, l))).toFixed(1);
    }
    mk("path", { d: dy, fill: "none", stroke: "var(--hair)", "stroke-width": 2 }, s);

    let dp = "";
    for (let lr = g.lo; lr <= g.hi + 0.001; lr += 0.1) {
      dp += (dp ? "L" : "M") + X(lr).toFixed(1) + " " + Y(ps(Math.pow(10, lr))).toFixed(1);
    }
    mk("path", { d: dp, fill: "none", stroke: "var(--accent)", "stroke-width": 3, "stroke-linecap": "round" }, s);

    if (failLr <= g.hi) {
      const fx = X(failLr), fy = Y(cap);
      mk("path", { d: "M" + (fx - 6) + " " + (fy - 6) + "L" + (fx + 6) + " " + (fy + 6) + "M" + (fx + 6) + " " + (fy - 6) + "L" + (fx - 6) + " " + (fy + 6), stroke: "var(--accent2)", "stroke-width": 2.4, "stroke-linecap": "round" }, s);
      txt(s, fx + 12, fy - 10, "MemoryError", { fill: "var(--accent2)", style: "font:600 13px " + FONT });
      txt(s, fx + 12, fy + 6, "at ~" + this.rowsLabel(Math.pow(10, failLr)) + " rows", { fill: "var(--accent2)", opacity: 0.85 });
    }

    txt(s, g.X1 + 12, Y(ps(1e8)) - 8, "PySuricata", { fill: "var(--accent)", style: "font:600 13px " + FONT });
    txt(s, g.X1 + 12, Y(ps(1e8)) + 8, Math.round(ps(1e8)) + " MB at 100M rows", { fill: "var(--accent)", opacity: 0.8 });
    const ylab = Math.min(failLr, g.hi) - 0.55;
    txt(s, X(ylab), Y(yd(Math.pow(10, ylab))) - 12, "ydata-profiling", { "text-anchor": "middle", style: "font:600 12px " + FONT });

    const mr = 1e6;
    mk("circle", { cx: X(6), cy: Y(ps(mr)), r: 3.5, fill: "var(--accent)" }, s);
    txt(s, X(6) + 8, Y(ps(mr)) - 8, "[" + Math.round(ps(mr)) + " MB @ 1M]", { fill: "var(--accent)", style: "font:11px " + MONO, opacity: 0.9 });

    if (this.memHover != null && this.memHover >= g.X0 && this.memHover <= g.X1) {
      const lr = g.lo + ((this.memHover - g.X0) / (g.X1 - g.X0)) * (g.hi - g.lo);
      const rows = Math.pow(10, lr);
      mk("line", { x1: X(lr), x2: X(lr), y1: g.Y0 - 8, y2: g.Y1, stroke: "var(--fg)", "stroke-width": 1, opacity: 0.35 }, s);
      mk("circle", { cx: X(lr), cy: Y(ps(rows)), r: 4.5, fill: "var(--accent)" }, s);
      const failed = lr > failLr;
      if (!failed) mk("circle", { cx: X(lr), cy: Y(yd(rows)), r: 4.5, fill: "var(--fg)", opacity: 0.55 }, s);
      const bx = Math.min(X(lr) + 14, 560);
      const t1 = txt(s, bx, g.Y0 + 4, this.rowsLabel(rows) + " rows", { fill: "var(--fg)", style: "font:600 13px " + FONT });
      t1.setAttribute("x", bx);
      txt(s, bx, g.Y0 + 22, "PySuricata " + Math.round(ps(rows)) + " MB", { fill: "var(--accent)", style: "font:12px " + MONO });
      txt(s, bx, g.Y0 + 38, failed ? "ydata-profiling — failed" : "ydata-profiling " + fmt(Math.round(yd(rows))) + " MB", { fill: failed ? "var(--accent2)" : "var(--muted)", style: "font:12px " + MONO });
    } else {
      txt(s, 812, 22, "one streaming pass · fixed-size sketches", { style: "font:12px " + FONT, "text-anchor": "end" });
    }

    txt(s, g.X0, 338, "[machine] · pysuricata [version] · ydata-profiling [version] · [date]", { style: "font:10px " + MONO, opacity: 0.8 });
    txt(s, g.X0, 352, "placeholder — fill from benchmarks/end_to_end.py", { style: "font:10px " + MONO, opacity: 0.8 });
  }
  rowsLabel(r) {
    if (r >= 1e6) return (r / 1e6).toFixed(r >= 1e7 ? 0 : 1) + "M";
    if (r >= 1e3) return (r / 1e3).toFixed(r >= 1e4 ? 0 : 1) + "k";
    return Math.round(r);
  }
  setRam(v) { this.ram = v; this.markRam(); this.drawMem(); }
  markRam() {
    [[8, this.R.ram8], [16, this.R.ram16], [32, this.R.ram32], [64, this.R.ram64]].forEach(([v, ref]) => {
      const b = ref.current;
      if (!b) return;
      const on = this.ram === v;
      b.style.background = on ? "var(--accent)" : "transparent";
      b.style.color = on ? "#fff" : "var(--fg)";
      b.style.borderColor = on ? "var(--accent)" : "var(--chipbr)";
    });
  }

  /* ---------------- 1 · reservoir ---------------- */
  buildRes() {
    this.resInit();
    this.drawResStatic();
  }
  resInit() {
    const k = this.resKVal;
    this.rs = {
      k: k, n: 0, frac: 0,
      rngR: mulberry32(20250815), rngL: mulberry32(991),
      R: { slots: new Array(k).fill(null), examined: 0, flash: new Array(k).fill(0) },
      L: { slots: new Array(k).fill(null), examined: 0, flash: new Array(k).fill(0), accepts: [], W: 1, next: k + 1 }
    };
  }
  resAdvance() {
    const S = this.rs, n = ++S.n, k = S.k;
    S.R.examined++;
    if (n <= k) { S.R.slots[n - 1] = n; S.R.flash[n - 1] = 1; }
    else if (S.rngR() < k / n) { const i = Math.floor(S.rngR() * k); S.R.slots[i] = n; S.R.flash[i] = 1; }
    if (n <= k) {
      S.L.slots[n - 1] = n; S.L.flash[n - 1] = 1; S.L.examined++; S.L.accepts.push(n);
      if (n === k) {
        S.L.W = Math.exp(Math.log(S.rngL()) / k);
        S.L.next = k + Math.floor(Math.log(S.rngL()) / Math.log(1 - S.L.W)) + 1;
      }
    } else if (n === S.L.next) {
      const i = Math.floor(S.rngL() * k);
      S.L.slots[i] = n; S.L.flash[i] = 1; S.L.examined++; S.L.accepts.push(n);
      S.L.W *= Math.exp(Math.log(S.rngL()) / k);
      S.L.next = n + Math.floor(Math.log(S.rngL()) / Math.log(1 - S.L.W)) + 1;
    }
    if (S.L.accepts.length > 40) S.L.accepts.splice(0, S.L.accepts.length - 40);
  }
  drawResStatic() {
    const s = this.R.res.current;
    if (!s) return;
    s.innerHTML = "";
    const S = this.rs;
    this.resEls = {};
    const lanes = [
      { key: "R", y: 30, title: "Algorithm R — tests every element", track: 74, slots: 96 },
      { key: "L", y: 186, title: "Algorithm L — skips to the next acceptance", track: 258, slots: 284 }
    ];
    mk("line", { x1: 40, x2: 780, y1: 158, y2: 158, stroke: "var(--hair)", opacity: 0.5 }, s);
    this.resEls.lanes = {};
    lanes.forEach((L) => {
      const G = mk("g", {}, s);
      txt(G, 40, L.y, L.title, { fill: "var(--fg)", style: "font:600 13px " + FONT });
      const c1 = txt(G, 780, L.y, "0", { "text-anchor": "end", fill: L.key === "R" ? "var(--fg)" : "var(--accent)", style: "font:600 13px " + MONO });
      txt(G, 780, L.y + 15, "elements examined", { "text-anchor": "end", opacity: 0.85 });
      const stream = mk("g", {}, G);
      const slotG = mk("g", {}, G);
      const slots = [];
      const w = Math.min(56, 700 / S.k - 8);
      for (let i = 0; i < S.k; i++) {
        const x = 40 + i * (w + 8);
        const rect = mk("rect", { x: x, y: L.slots, width: w, height: 30, rx: 7, fill: "none", stroke: "var(--hair)", "stroke-width": 1.2 }, slotG);
        const t = txt(slotG, x + w / 2, L.slots + 20, "—", { "text-anchor": "middle", fill: "var(--fg)", style: "font:12px " + MONO });
        slots.push({ rect: rect, t: t });
      }
      txt(G, 40 + S.k * (w + 8) + 6, L.slots + 20, "k = " + S.k + " slots", { opacity: 0.85 });
      this.resEls.lanes[L.key] = { counter: c1, stream: stream, slots: slots, geom: L };
    });
    txt(s, 40, 334, "Both reservoirs hold a uniform sample: every element seen is present with probability k/n. The run restarts at 260 elements.", { style: "font:11px " + FONT });
  }
  tickRes(dt) {
    const S = this.rs;
    if (!S || !this.resEls) return;
    if (this.state.resPlaying) {
      S.frac += dt * this.resSpeedVal;
      while (S.frac >= 1) { S.frac -= 1; this.resAdvance(); }
      if (S.n >= 260) { this.resInit(); return; }
    }
    const nf = S.n + (this.state.resPlaying ? S.frac : 0);
    const SP = 15, X0 = 40, X1 = 700;
    const xOf = (i) => X1 - (nf - i) * SP;
    const first = Math.max(1, Math.ceil(nf - (X1 - X0) / SP));
    ["R", "L"].forEach((key) => {
      const lane = this.resEls.lanes[key], gy = lane.geom.track;
      const f = document.createDocumentFragment();
      if (key === "R") {
        for (let i = first; i <= Math.floor(nf); i++) {
          const x = xOf(i);
          mk("line", { x1: x, x2: x, y1: gy - 12, y2: gy + 2, stroke: "var(--hair)", "stroke-width": 1.4 }, f);
          mk("circle", { cx: x, cy: gy + 10, r: 1.9, fill: "var(--accent)", opacity: 0.85 }, f);
        }
        mk("line", { x1: X1, x2: X1, y1: gy - 22, y2: gy + 16, stroke: "var(--accent)", "stroke-width": 2 }, f);
        const tt = mk("text", { x: X1 + 8, y: gy - 14, fill: "var(--accent)", style: "font:600 11px " + FONT }, f);
        tt.textContent = "tests each";
      } else {
        const acc = new Set(S.L.accepts);
        for (let i = first; i <= Math.floor(nf); i++) {
          const x = xOf(i), hit = acc.has(i);
          mk("line", { x1: x, x2: x, y1: gy - 12, y2: gy + 2, stroke: hit ? "var(--accent)" : "var(--hair)", "stroke-width": hit ? 2 : 1.4, opacity: hit ? 1 : 0.45 }, f);
        }
        for (let j = 1; j < S.L.accepts.length; j++) {
          const a = S.L.accepts[j - 1], b = S.L.accepts[j];
          if (b < first) continue;
          const xa = xOf(a), xb = xOf(b);
          const h = Math.min(26, 8 + (xb - xa) * 0.22);
          mk("path", { d: "M" + xa.toFixed(1) + " " + (gy - 14) + " Q " + ((xa + xb) / 2).toFixed(1) + " " + (gy - 14 - h * 2).toFixed(1) + " " + xb.toFixed(1) + " " + (gy - 14), fill: "none", stroke: "var(--accent)", "stroke-width": 1.6, opacity: 0.8 }, f);
        }
        const lastA = S.L.accepts[S.L.accepts.length - 1] || 1;
        mk("circle", { cx: xOf(lastA), cy: gy - 14, r: 4.5, fill: "var(--accent)" }, f);
        const gap = Math.max(0, S.L.next - Math.floor(nf));
        const nx = Math.min(xOf(S.L.next), 716);
        mk("line", { x1: nx, x2: nx, y1: gy - 26, y2: gy + 14, stroke: "var(--accent2)", "stroke-width": 2, "stroke-dasharray": "3 3" }, f);
        const t2 = mk("text", { x: nx, y: gy - 34, fill: "var(--accent2)", "text-anchor": "middle", style: "font:600 11px " + FONT }, f);
        t2.textContent = "next acceptance";
        const t3 = mk("text", { x: nx, y: gy + 30, fill: "var(--accent2)", "text-anchor": "middle", style: "font:600 11px " + MONO }, f);
        t3.textContent = "+" + gap;
      }
      lane.stream.replaceChildren(f);
      const src = key === "R" ? S.R : S.L;
      lane.counter.textContent = fmt(src.examined);
      for (let i = 0; i < lane.slots.length; i++) {
        const v = src.slots[i];
        lane.slots[i].t.textContent = v == null ? "—" : fmt(v);
        if (src.flash[i] > 0) {
          src.flash[i] = Math.max(0, src.flash[i] - dt * 2.2);
          lane.slots[i].rect.setAttribute("fill", "var(--accent)");
          lane.slots[i].rect.setAttribute("fill-opacity", (src.flash[i] * 0.35).toFixed(3));
          lane.slots[i].rect.setAttribute("stroke", "var(--accent)");
        } else {
          lane.slots[i].rect.setAttribute("fill", "none");
          lane.slots[i].rect.setAttribute("stroke", v == null ? "var(--hair)" : "var(--hair)");
        }
      }
    });
  }

  /* ---------------- 2 · misra-gries ---------------- */
  buildMG() {
    this.mgInit();
    const box = this.R.mgKeys.current;
    if (box) {
      box.replaceChildren();
      ["US", "DE", "FR", "new key"].forEach((kk) => {
        const b = document.createElement("button");
        b.textContent = kk;
        b.style.cssText = "padding:4px 10px;border-radius:9999px;border:1px solid var(--chipbr);background:var(--chipbg);color:var(--fg);font:500 11px " + MONO + ";cursor:pointer";
        b.onclick = () => { this.setState({ mgPlaying: false }); this.mgFeed(kk === "new key" ? null : kk); };
        box.appendChild(b);
      });
    }
    this.drawMGStatic();
  }
  mgInit() {
    this.mgS = {
      k: 6, slots: [], rng: mulberry32(4242), t: 0, hold: 0.7, phase: "idle",
      incoming: null, label: "", labelT: 0, seen: 0, distinct: new Set(), evictions: 0, novel: 0
    };
    for (let i = 0; i < 6; i++) this.mgS.slots.push({ key: null, c: 0, disp: 0 });
    ["US", "DE", "FR"].forEach((k, i) => { this.mgS.slots[i] = { key: k, c: [7, 5, 3][i], disp: [7, 5, 3][i] }; });
    this.mgS.slots[3] = { key: "JP", c: 1, disp: 1 };
    this.mgS.seen = 16;
  }
  mgNextKey() {
    const r = this.mgS.rng();
    if (r < 0.5) return ["US", "DE", "FR"][Math.floor(this.mgS.rng() * 3)];
    if (r < 0.62) return "JP";
    return null;
  }
  mgFeed(key) {
    const S = this.mgS;
    if (key == null) { S.novel++; key = "x" + String(1000 + S.novel).slice(1); }
    S.seen++;
    const hit = S.slots.find((s) => s.key === key);
    if (hit) { hit.c += 1; S.incoming = { key: key, target: S.slots.indexOf(hit), t: 0 }; S.label = "known key · increment"; S.labelT = 1; S.hold = 0.75; return; }
    const free = S.slots.findIndex((s) => s.key == null);
    if (free >= 0) { S.slots[free] = { key: key, c: 1, disp: 0 }; S.incoming = { key: key, target: free, t: 0 }; S.label = "new key · free slot"; S.labelT = 1; S.hold = 0.85; return; }
    S.slots.forEach((s) => { s.c -= 1; });
    S.evictions++;
    let freed = -1;
    S.slots.forEach((s, i) => { if (s.c <= 0) { s.key = null; s.c = 0; if (freed < 0) freed = i; } });
    if (freed >= 0) S.slots[freed] = { key: key, c: 1, disp: 0 };
    S.incoming = { key: key, target: freed, t: 0 };
    S.label = "all counters decrement";
    S.labelT = 1;
    S.hold = 2.0;
  }
  drawMGStatic() {
    const s = this.R.mg.current;
    if (!s) return;
    s.innerHTML = "";
    const S = this.mgS;
    const W = 104, GAP = 18, X0 = 46, BASE = 250, HMAX = 130;
    this.mgEls = { bars: [], keys: [], counts: [] };
    for (let i = 0; i < S.k; i++) {
      const x = X0 + i * (W + GAP);
      mk("rect", { x: x, y: BASE - HMAX, width: W, height: HMAX, rx: 8, fill: "none", stroke: "var(--hair)", "stroke-width": 1, "stroke-dasharray": "3 4", opacity: 0.6 }, s);
      const bar = mk("rect", { x: x + 6, y: BASE - 10, width: W - 12, height: 10, rx: 5, fill: "var(--accent)", opacity: 0.85 }, s);
      const key = txt(s, x + W / 2, BASE + 22, "—", { "text-anchor": "middle", fill: "var(--fg)", style: "font:600 12px " + MONO });
      const cnt = txt(s, x + W / 2, BASE - 20, "", { "text-anchor": "middle", fill: "var(--fg)", style: "font:600 13px " + MONO });
      this.mgEls.bars.push(bar); this.mgEls.keys.push(key); this.mgEls.counts.push(cnt);
    }
    mk("line", { x1: X0 - 8, x2: X0 + S.k * (W + GAP) - GAP + 8, y1: BASE, y2: BASE, stroke: "var(--hair)", "stroke-width": 1.5 }, s);
    txt(s, X0, BASE + 48, "k = 6 counters · memory does not grow with the number of distinct values", { style: "font:11px " + FONT });
    this.mgEls.label = txt(s, X0, 34, "", { fill: "var(--accent2)", style: "font:600 14px " + FONT });
    this.mgEls.stats = txt(s, 780, 34, "", { "text-anchor": "end", style: "font:12px " + MONO });
    this.mgEls.incoming = mk("g", { opacity: 0 }, s);
    this.mgEls.incRect = mk("rect", { x: 0, y: 52, width: 62, height: 26, rx: 13, fill: "var(--accent2)", "fill-opacity": 0.14, stroke: "var(--accent2)", "stroke-width": 1.2 }, this.mgEls.incoming);
    this.mgEls.incText = txt(this.mgEls.incoming, 0, 69, "", { "text-anchor": "middle", fill: "var(--accent2)", style: "font:600 12px " + MONO });
    this.mgGeom = { W: W, GAP: GAP, X0: X0, BASE: BASE, HMAX: HMAX };
  }
  tickMG(dt) {
    const S = this.mgS;
    if (!S || !this.mgEls) return;
    if (this.state.mgPlaying) {
      S.t += dt;
      if (S.t >= S.hold) { S.t = 0; this.mgFeed(this.mgNextKey()); }
    }
    const g = this.mgGeom;
    const maxC = Math.max(6, ...S.slots.map((s) => s.c));
    S.slots.forEach((sl, i) => {
      sl.disp += (sl.c - sl.disp) * Math.min(1, dt * 7);
      const h = Math.max(sl.key ? 10 : 0, (Math.max(0, sl.disp) / maxC) * g.HMAX);
      const bar = this.mgEls.bars[i];
      bar.setAttribute("y", (g.BASE - h).toFixed(1));
      bar.setAttribute("height", h.toFixed(1));
      bar.setAttribute("fill", sl.key == null ? "var(--hair)" : (S.incoming && S.incoming.target === i && S.labelT > 0.25 ? "var(--accent2)" : "var(--accent)"));
      bar.setAttribute("opacity", sl.key == null ? 0.25 : 0.9);
      this.mgEls.keys[i].textContent = sl.key || "empty";
      this.mgEls.keys[i].setAttribute("opacity", sl.key ? 1 : 0.45);
      this.mgEls.counts[i].textContent = sl.key ? Math.round(Math.max(0, sl.disp)) : "";
      this.mgEls.counts[i].setAttribute("y", (g.BASE - h - 10).toFixed(1));
    });
    if (S.labelT > 0) S.labelT = Math.max(0, S.labelT - dt / Math.max(0.6, S.hold));
    this.mgEls.label.textContent = S.label;
    this.mgEls.label.setAttribute("opacity", Math.min(1, S.labelT * 2.2).toFixed(2));
    this.mgEls.label.setAttribute("fill", S.label.indexOf("decrement") >= 0 ? "var(--accent2)" : "var(--muted)");
    this.mgEls.stats.textContent = fmt(S.seen) + " keys seen · " + S.evictions + " decrement rounds";
    if (S.incoming) {
      S.incoming.t = Math.min(1, S.incoming.t + dt / 0.5);
      const tx = g.X0 + Math.max(0, S.incoming.target) * (g.W + g.GAP) + g.W / 2;
      const x = 760 + (tx - 760) * (1 - Math.pow(1 - S.incoming.t, 3));
      this.mgEls.incoming.setAttribute("opacity", (1 - Math.max(0, S.incoming.t - 0.75) * 4).toFixed(2));
      this.mgEls.incRect.setAttribute("x", (x - 31).toFixed(1));
      this.mgEls.incText.setAttribute("x", x.toFixed(1));
      this.mgEls.incText.textContent = S.incoming.key;
    }
  }

  /* ---------------- 4 · chunk lifecycle ---------------- */
  buildChunk() {
    const s = this.R.chunk.current;
    if (!s) return;
    s.innerHTML = "";
    const box = (x, y, w, h, label, sub, dashed) => {
      const g = mk("g", {}, s);
      mk("rect", { x: x, y: y, width: w, height: h, rx: 8, fill: "var(--bg)", stroke: dashed ? "var(--accent)" : "var(--hair)", "stroke-width": dashed ? 1.6 : 1.2, "stroke-dasharray": dashed ? "5 4" : "" }, g);
      txt(g, x + 14, y + (sub ? 24 : h / 2 + 4), label, { fill: "var(--fg)", style: "font:600 13px " + FONT });
      if (sub) txt(g, x + 14, y + 42, sub, { style: "font:11px " + MONO });
      return g;
    };
    const arrow = (x1, y1, x2, y2) => {
      mk("path", { d: "M" + x1 + " " + y1 + "L" + x2 + " " + y2, stroke: "var(--hair)", "stroke-width": 1.4, "marker-end": "url(#psArrow)" }, s);
    };
    const defs = mk("defs", {}, s);
    const m = mk("marker", { id: "psArrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
    mk("path", { d: "M0 0 L10 5 L0 10 z", fill: "var(--hair)" }, m);

    mk("rect", { x: 30, y: 118, width: 430, height: 212, rx: 10, fill: "var(--accent)", "fill-opacity": 0.04, stroke: "var(--hair)", "stroke-width": 1, "stroke-dasharray": "4 4" }, s);
    txt(s, 42, 138, "per chunk · repeats", { style: "font:11px " + MONO });

    this.chunkNodes = [];
    this.chunkNodes.push(box(60, 14, 370, 54, "source", "DataFrame | LazyFrame | iterator of chunks"));
    this.chunkNodes.push(box(60, 76, 370, 30, "adapter selection · peek one chunk, splice back"));
    this.chunkNodes.push(box(60, 150, 370, 52, "chunk", "O(chunk_size) · released after each iteration"));
    this.chunkNodes.push(box(60, 216, 370, 30, "per column: convert to array"));
    this.chunkNodes.push(box(60, 260, 370, 52, "accumulator.update(array, row_offset)"));
    arrow(245, 68, 245, 74);
    arrow(245, 106, 245, 148);
    arrow(245, 202, 245, 214);
    arrow(245, 246, 245, 258);

    mk("rect", { x: 500, y: 118, width: 290, height: 212, rx: 10, fill: "none", stroke: "var(--accent)", "stroke-width": 1.6, "stroke-dasharray": "5 4" }, s);
    txt(s, 514, 140, "O(1) in rows — fixed-size sketches", { fill: "var(--accent)", style: "font:600 12px " + FONT });
    ["numeric", "categorical", "datetime", "boolean"].forEach((n, i) => {
      const y = 156 + i * 42;
      mk("rect", { x: 514, y: y, width: 262, height: 32, rx: 7, fill: "var(--bg)", stroke: "var(--hair)", "stroke-width": 1.1 }, s);
      txt(s, 526, y + 21, n, { fill: "var(--fg)", style: "font:600 12px " + FONT });
      const sub = ["Welford/Pébay · reservoir", "Misra-Gries · KMV", "min/max · histogram", "counts"][i];
      txt(s, 764, y + 21, sub, { "text-anchor": "end", style: "font:10px " + MONO, opacity: 0.8 });
    });
    arrow(432, 286, 512, 286);
    mk("path", { d: "M58 286 Q 44 286 44 231 Q 44 176 56 176", fill: "none", stroke: "var(--hair)", "stroke-width": 1.4, "stroke-dasharray": "4 3", "marker-end": "url(#psArrow)" }, s);
    txt(s, 68, 325, "next chunk", { style: "font:10px " + MONO, opacity: 0.85 });

    this.chunkNodes.push(box(60, 348, 370, 30, "after the last chunk: finalize() per column"));
    this.chunkNodes.push(box(60, 392, 370, 30, "render: HTML report · summarize(): JSON"));
    arrow(245, 330, 245, 346);
    arrow(245, 378, 245, 390);

    mk("rect", { x: 500, y: 344, width: 290, height: 86, rx: 8, fill: "none", stroke: "var(--hair)", "stroke-width": 1 }, s);
    txt(s, 514, 364, "live chunk memory", { style: "font:10px " + MONO });
    mk("rect", { x: 514, y: 370, width: 262, height: 8, rx: 4, fill: "var(--hair)", opacity: 0.35 }, s);
    this.chunkBar1 = mk("rect", { x: 514, y: 370, width: 0, height: 8, rx: 4, fill: "var(--accent2)" }, s);
    txt(s, 514, 396, "accumulator state", { style: "font:10px " + MONO });
    mk("rect", { x: 514, y: 402, width: 262, height: 8, rx: 4, fill: "var(--hair)", opacity: 0.35 }, s);
    this.chunkBar2 = mk("rect", { x: 514, y: 402, width: 30, height: 8, rx: 4, fill: "var(--accent)" }, s);
    this.chunkStat = txt(s, 514, 424, "", { style: "font:11px " + MONO, fill: "var(--fg)" });

    mk("line", { x1: 446, x2: 446, y1: 42, y2: 286, stroke: "var(--hair)", "stroke-width": 1, "stroke-dasharray": "2 5", opacity: 0.5 }, s);
    this.chunkTok = mk("circle", { cx: 446, cy: 42, r: 6, fill: "var(--accent2)", opacity: 0 }, s);
    this.ch = { mode: "idle", stage: -1, t: 0, queue: 0, chunks: 0 };
    this.chunkTick0();
  }
  chunkTick0() {
    if (this.chunkStat) this.chunkStat.textContent = "0 chunks processed · 1.2 MB state";
  }
  tickChunk(dt) {
    const C = this.ch;
    if (!C) return;
    const LOOP = [[446, 42], [446, 91], [446, 176], [446, 231], [446, 286], [660, 286]];
    const FIN = [[660, 286], [446, 330], [446, 363], [446, 407], [446, 444]];
    if (C.mode === "idle") {
      if (C.queue > 0) { C.queue--; C.mode = "loop"; C.stage = 0; C.t = 0; }
      else {
        this.chunkTok.setAttribute("opacity", 0);
        this.chunkBar1.setAttribute("width", 0);
        this.chunkNodes.forEach((n) => { const r = n.querySelector("rect"); r.setAttribute("stroke", "var(--hair)"); r.setAttribute("stroke-width", 1.2); });
        return;
      }
    }
    const path = C.mode === "loop" ? LOOP : FIN;
    C.t += dt / (C.mode === "loop" ? 0.45 : 0.6);
    const i = Math.min(path.length - 2, C.stage);
    const a = path[i], b = path[i + 1];
    const u = Math.min(1, C.t);
    this.chunkTok.setAttribute("cx", (a[0] + (b[0] - a[0]) * u).toFixed(1));
    this.chunkTok.setAttribute("cy", (a[1] + (b[1] - a[1]) * u).toFixed(1));
    this.chunkTok.setAttribute("opacity", C.mode === "finish" && C.stage >= 3 ? (1 - u).toFixed(2) : 1);
    this.chunkTok.setAttribute("fill", C.mode === "finish" ? "var(--accent)" : "var(--accent2)");
    const hot = C.mode === "loop" ? Math.min(4, C.stage + 1) : (C.stage === 1 ? 5 : C.stage >= 2 ? 6 : -1);
    this.chunkNodes.forEach((n, idx) => {
      const r = n.querySelector("rect");
      const on = idx === hot;
      r.setAttribute("stroke", on ? (C.mode === "finish" ? "var(--accent)" : "var(--accent2)") : "var(--hair)");
      r.setAttribute("stroke-width", on ? 1.8 : 1.2);
    });
    const mem = C.mode === "finish" ? 0 : C.stage >= 2 ? 34 : C.stage >= 1 ? 20 : 6;
    this.chunkBar1.setAttribute("width", (mem * 6.5 * (C.mode === "loop" && C.stage >= 4 ? 1 - u : 1)).toFixed(1));
    this.chunkBar2.setAttribute("width", (46 + Math.min(6, C.chunks * 0.2)).toFixed(1));
    if (u >= 1) {
      C.stage++;
      C.t = 0;
      if (C.mode === "loop" && C.stage >= LOOP.length - 1) {
        C.chunks++;
        this.chunkStat.textContent = fmt(C.chunks) + " chunk" + (C.chunks === 1 ? "" : "s") + " processed · 1.2 MB state";
        if (C.queue > 0) { C.queue--; C.stage = 2; }
        else { C.mode = "finish"; C.stage = 0; }
      } else if (C.mode === "finish") {
        const nc = fmt(C.chunks) + " chunk" + (C.chunks === 1 ? "" : "s");
        if (C.stage === 1) this.chunkStat.textContent = "finalize() · " + nc + " · 1.2 MB state";
        if (C.stage === 3) this.chunkStat.textContent = "report rendered · " + nc + " · 1.2 MB state";
        if (C.stage >= FIN.length - 1) { C.mode = "idle"; C.stage = -1; }
      }
    }
  }
  mermaidSrc() {
    return [
      "flowchart TD",
      "  SRC[\"source: DataFrame, LazyFrame,<br/>iterator of chunks\"] --> AD[\"adapter selection<br/>peek one chunk, splice back\"]",
      "  AD --> LOOP",
      "  subgraph LOOP [\"per chunk — repeats\"]",
      "    CH[\"chunk<br/>O(chunk_size), released each iteration\"] --> ARR[\"per column: convert to array\"]",
      "    ARR --> UPD[\"accumulator.update(array, row_offset)\"]",
      "  end",
      "  UPD --> STATE",
      "  subgraph STATE [\"O(1) in rows — fixed-size sketches\"]",
      "    NUM[\"numeric accumulator\"]",
      "    CAT[\"categorical accumulator\"]",
      "    DTM[\"datetime accumulator\"]",
      "    BOO[\"boolean accumulator\"]",
      "  end",
      "  UPD -.->|next chunk| CH",
      "  STATE --> FIN[\"after last chunk: finalize()\"]",
      "  FIN --> OUT[\"render: HTML report<br/>summarize(): JSON\"]"
    ].join("\n");
  }

  /* ---------------- 5 · pébay ---------------- */
  buildPeb() {
    const s = this.R.peb.current;
    if (!s) return;
    s.addEventListener("pointerdown", (e) => this.pebGrab(e));
    s.addEventListener("pointermove", (e) => this.pebMove(e));
    window.addEventListener("pointerup", () => { this.pebDrag = null; });
    this.drawPeb();
  }
  pebX(v) { return 410 + v * 150; }
  pebV(x) { return Math.max(-2.2, Math.min(2.2, (x - 410) / 150)); }
  pebGrab(e) {
    const s = this.R.peb.current, b = s.getBoundingClientRect();
    const x = ((e.clientX - b.left) * 820) / b.width;
    const da = Math.abs(x - this.pebX(this.peb.ma)), db = Math.abs(x - this.pebX(this.peb.mb));
    if (Math.min(da, db) < 40) this.pebDrag = da < db ? "ma" : "mb";
  }
  pebMove(e) {
    if (!this.pebDrag) return;
    const s = this.R.peb.current, b = s.getBoundingClientRect();
    this.peb[this.pebDrag] = this.pebV(((e.clientX - b.left) * 820) / b.width);
    this.drawPeb();
  }
  drawPeb() {
    const s = this.R.peb.current;
    if (!s) return;
    s.innerHTML = "";
    const P = this.peb;
    const n = P.na + P.nb;
    const d = P.mb - P.ma;
    const corr = (d * d * P.na * P.nb) / n;
    const M2 = P.M2a + P.M2b + corr;
    const mean = (P.na * P.ma + P.nb * P.mb) / n;

    const panel = (x, label, nn, mu, m2, col) => {
      mk("rect", { x: x, y: 24, width: 200, height: 92, rx: 9, fill: "none", stroke: "var(--hair)", "stroke-width": 1.2 }, s);
      txt(s, x + 14, 46, label, { fill: col, style: "font:600 13px " + FONT });
      txt(s, x + 14, 68, "n = " + fmt(nn), { style: "font:12px " + MONO, fill: "var(--fg)" });
      txt(s, x + 14, 86, "mean = " + mu.toFixed(3), { style: "font:12px " + MONO, fill: "var(--fg)" });
      txt(s, x + 14, 104, "M2 = " + fmt(Math.round(m2)), { style: "font:12px " + MONO, fill: "var(--fg)" });
    };
    panel(40, "partition A", P.na, P.ma, P.M2a, "var(--fg)");
    panel(270, "partition B", P.nb, P.mb, P.M2b, "var(--fg)");
    mk("rect", { x: 540, y: 24, width: 240, height: 92, rx: 9, fill: "var(--accent)", "fill-opacity": 0.06, stroke: "var(--accent)", "stroke-width": 1.4 }, s);
    txt(s, 554, 46, "union A ∪ B — exact", { fill: "var(--accent)", style: "font:600 13px " + FONT });
    txt(s, 554, 68, "n = " + fmt(n), { style: "font:12px " + MONO, fill: "var(--fg)" });
    txt(s, 554, 86, "mean = " + mean.toFixed(3), { style: "font:12px " + MONO, fill: "var(--fg)" });
    txt(s, 554, 104, "M2 = " + fmt(Math.round(M2)), { style: "font:12px " + MONO, fill: "var(--fg)" });
    mk("path", { d: "M240 70 L268 70 M470 70 L538 70", stroke: "var(--hair)", "stroke-width": 1.4 }, s);

    const Y = 190;
    mk("line", { x1: 60, x2: 780, y1: Y, y2: Y, stroke: "var(--hair)", "stroke-width": 1.5 }, s);
    txt(s, 60, Y + 34, "value", { style: "font:11px " + FONT });
    [-2, -1, 0, 1, 2].forEach((v) => {
      mk("line", { x1: this.pebX(v), x2: this.pebX(v), y1: Y, y2: Y + 5, stroke: "var(--hair)" }, s);
      txt(s, this.pebX(v), Y + 20, String(v), { "text-anchor": "middle", opacity: 0.7 });
    });
    const xa = this.pebX(P.ma), xb = this.pebX(P.mb);
    [[xa, "mean A"], [xb, "mean B"]].forEach(([x, l]) => {
      mk("line", { x1: x, x2: x, y1: Y - 40, y2: Y, stroke: "var(--fg)", "stroke-width": 2 }, s);
      mk("circle", { cx: x, cy: Y - 40, r: 7, fill: "var(--bg)", stroke: "var(--fg)", "stroke-width": 2, style: "cursor:ew-resize" }, s);
      txt(s, x, Y - 52, l, { "text-anchor": "middle", fill: "var(--fg)", style: "font:600 11px " + FONT });
    });
    const lo = Math.min(xa, xb), hi = Math.max(xa, xb);
    mk("path", { d: "M" + lo + " " + (Y - 14) + " L" + lo + " " + (Y - 22) + " L" + hi + " " + (Y - 22) + " L" + hi + " " + (Y - 14), fill: "none", stroke: "var(--accent2)", "stroke-width": 1.6 }, s);
    txt(s, (lo + hi) / 2, Y - 28, "δ = " + d.toFixed(3), { "text-anchor": "middle", fill: "var(--accent2)", style: "font:600 12px " + MONO });

    const BY = 268, BX = 60, BW = 720;
    const total = Math.max(1, M2);
    const wa = (P.M2a / total) * BW, wb = (P.M2b / total) * BW, wc = (corr / total) * BW;
    mk("rect", { x: BX, y: BY, width: wa, height: 26, fill: "var(--accent)", opacity: 0.35 }, s);
    mk("rect", { x: BX + wa, y: BY, width: wb, height: 26, fill: "var(--accent)", opacity: 0.6 }, s);
    mk("rect", { x: BX + wa + wb, y: BY, width: Math.max(0, wc), height: 26, fill: "var(--accent2)" }, s);
    mk("rect", { x: BX, y: BY, width: BW, height: 26, fill: "none", stroke: "var(--hair)" }, s);
    txt(s, BX + 6, BY + 18, "M2a", { fill: "var(--fg)", style: "font:11px " + MONO });
    txt(s, BX + wa + 6, BY + 18, "M2b", { fill: "var(--fg)", style: "font:11px " + MONO });
    if (wc > 70) txt(s, BX + wa + wb + 8, BY + 18, "correction " + fmt(Math.round(corr)), { fill: "#fff", style: "font:600 11px " + MONO });
    else txt(s, BX + wa + wb + Math.max(0, wc) + 8, BY + 18, "correction " + fmt(Math.round(corr)), { fill: "var(--accent2)", style: "font:600 11px " + MONO });

    txt(s, BX, 330, "M2 = M2a + M2b + δ² · na · nb / n", { fill: "var(--fg)", style: "font:600 15px " + MONO });
    txt(s, BX, 352, fmt(Math.round(M2)) + "  =  " + fmt(P.M2a) + " + " + fmt(P.M2b) + " + " + d.toFixed(3) + "² · " + fmt(P.na) + "·" + fmt(P.nb) + " / " + fmt(n), { style: "font:12px " + MONO });
    txt(s, BX, 378, "M3 and M4 follow the same shape, with larger correction terms.", { style: "font:11px " + FONT });

    if (this.R.pebNaLabel.current) this.R.pebNaLabel.current.textContent = fmt(P.na);
    if (this.R.pebNbLabel.current) this.R.pebNbLabel.current.textContent = fmt(P.nb);
  }

  /* ---------------- 6 · annotated card ---------------- */
  buildCard() {
    const host = this.R.card.current;
    if (!host) return;
    const chipbr = "var(--chipbr)";
    const wrap = document.createElement("div");
    wrap.style.cssText = "font-family:" + FONT + ";color:var(--fg)";
    const badge = (n, top, left) => "<span data-badge=\"" + n + "\" style=\"position:absolute;top:" + (top == null ? -9 : top) + "px;left:" + (left == null ? -9 : left) + "px;width:18px;height:18px;border-radius:9999px;background:var(--accent);color:#fff;font:600 10px " + FONT + ";display:flex;align-items:center;justify-content:center;z-index:3\">" + n + "</span>";
    const bars = [3, 9, 22, 41, 63, 88, 74, 52, 37, 28, 19, 14, 9, 6, 4, 3, 2, 2, 1, 1];
    const hist = bars.map((h, i) => "<rect x=\"" + (6 + i * 26) + "\" y=\"" + (104 - h) + "\" width=\"22\" height=\"" + h + "\" rx=\"2\" fill=\"#60a5fa\"></rect>").join("");
    wrap.innerHTML =
      "<div data-region=\"1\" style=\"position:relative;padding:12px 14px;border-bottom:1px solid " + chipbr + ";display:flex;align-items:center;gap:8px;flex-wrap:wrap\">" +
        badge(1) +
        "<span style=\"font-weight:700;font-size:15px\">fare</span>" +
        "<span style=\"font-size:11px;padding:2px 6px;border-radius:6px;background:rgba(96,165,250,.16);color:#3b82f6;border:1px solid rgba(96,165,250,.35)\">Numeric</span>" +
        "<span style=\"font-size:12px;padding:2px 6px;border-radius:6px;background:var(--chipbg);border:1px solid " + chipbr + "\">float64</span>" +
        "<span data-region=\"2\" style=\"position:relative;display:inline-flex;gap:6px;margin-left:18px\">" + badge(2, -13, -20) +
          "<span style=\"padding:.18rem .45rem;border-radius:9999px;font-size:12px;border:1px solid #f1c54e;background:rgba(241,197,78,.12)\">Skewed Right</span>" +
          "<span style=\"padding:.18rem .45rem;border-radius:9999px;font-size:12px;border:1px solid #f15e4e;background:rgba(241,94,78,.12)\">Heavy‑tailed</span>" +
          "<span style=\"padding:.18rem .45rem;border-radius:9999px;font-size:12px;border:1px solid #f15e4e;background:rgba(241,94,78,.12)\">Many outliers</span>" +
        "</span>" +
      "</div>" +
      "<div style=\"padding:12px 14px;display:grid;grid-template-columns:200px 200px minmax(0,1fr);gap:16px;align-items:start\">" +
        "<div data-region=\"3\" style=\"position:relative\">" + badge(3) +
          "<table style=\"border-collapse:collapse;width:100%;font-size:12.5px\">" +
          [["Count", "891"], ["Unique (≈)", "248"], ["Missing", "0 (0.0%)"], ["Outliers", "116 (13.0%)"], ["Zeros", "15 (1.7%)"], ["Infinites", "0 (0.0%)"], ["Negatives", "0 (0.0%)"]]
            .map(([k, v], i) => "<tr><td style=\"padding:3px 0;opacity:.85\">" + k + "</td><td style=\"padding:3px 0;text-align:right;white-space:nowrap;" + (i === 3 ? "color:#dc2626;font-weight:700;background:rgba(241,94,78,0.18)" : "") + "\">" + v + "</td></tr>").join("") +
        "</table></div>" +
        "<div data-region=\"4\" style=\"position:relative\">" + badge(4) +
          "<table style=\"border-collapse:collapse;width:100%;font-size:12.5px\">" +
          [["Min", "0.00"], ["Q1 (P25)", "7.91"], ["Median", "14.45"], ["Mean", "32.20"], ["Q3 (P75)", "31.00"], ["Max", "512.33"], ["Processed bytes (≈)", "7.0 KB"]]
            .map(([k, v]) => "<tr><td style=\"padding:3px 0;opacity:.85\">" + k + "</td><td style=\"padding:3px 0;text-align:right;white-space:nowrap\">" + v + "</td></tr>").join("") +
        "</table></div>" +
        "<div style=\"display:flex;flex-direction:column;gap:8px\">" +
          "<div data-region=\"5\" style=\"position:relative\">" + badge(5) +
            "<svg viewBox=\"0 0 528 112\" style=\"width:100%;height:auto;display:block;border:1px solid " + chipbr + ";border-radius:6px;background:var(--bg)\">" + hist +
            "<line x1=\"0\" y1=\"104\" x2=\"528\" y2=\"104\" stroke=\"var(--hair)\"></line></svg>" +
          "</div>" +
          "<div data-region=\"6\" style=\"position:relative;display:flex;flex-wrap:wrap;gap:6px;padding-left:14px\">" + badge(6, 5, -6) +
            ["Histogram", "ECDF", "lin", "log", "bins 10", "bins 25", "bins 50"].map((p, i) =>
              "<span style=\"display:inline-flex;padding:4px 8px;border-radius:9999px;font-size:12px;border:1px solid " + chipbr + ";background:" + (i === 0 || i === 2 || i === 5 ? "rgba(96,165,250,.16)" : "var(--chipbg)") + "\">" + p + "</span>").join("") +
          "</div>" +
          "<div data-region=\"7\" style=\"position:relative;border-top:1px solid " + chipbr + ";padding-top:8px;font-size:12.5px;opacity:.85\">" + badge(7) + "▸ Details" +
          "</div>" +
        "</div>" +
      "</div>";
    host.replaceChildren(wrap);

    const legend = [
      ["1", "Column name, type badge and dtype badge"],
      ["2", "Quality chips — the thresholds that fired, with the measured value on hover"],
      ["3", "Left table: count, unique, missing, outliers, zeros, infinites, negatives"],
      ["4", "Right table: quartiles, min, max, mean, processed bytes"],
      ["5", "Histogram drawn from the reservoir sample"],
      ["6", "Scale and bin controls — CSS-only tabstrip, no JS"],
      ["7", "Details disclosure: quantile table, extremes, correlations"]
    ];
    const lg = this.R.legend.current;
    lg.replaceChildren();
    this.cardRegions = Array.from(wrap.querySelectorAll("[data-region]"));
    legend.forEach(([n, t]) => {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:8px;align-items:flex-start;font:12px/1.45 " + FONT + ";color:var(--muted);cursor:pointer;padding:2px 0";
      row.innerHTML = "<span style=\"flex:none;width:18px;height:18px;border-radius:9999px;background:var(--chipbg);border:1px solid var(--chipbr);color:var(--fg);font:600 10px " + FONT + ";display:flex;align-items:center;justify-content:center\">" + n + "</span><span>" + t + "</span>";
      row.onmouseenter = () => this.highlight(n);
      row.onmouseleave = () => this.highlight(null);
      lg.appendChild(row);
    });
    this.cardRegions.forEach((el) => {
      el.onmouseenter = () => this.highlight(el.getAttribute("data-region"));
      el.onmouseleave = () => this.highlight(null);
    });
  }
  highlight(n) {
    if (!this.cardRegions) return;
    this.cardRegions.forEach((el) => {
      const on = n && el.getAttribute("data-region") === n;
      el.style.outline = on ? "2px solid var(--accent2)" : "none";
      el.style.outlineOffset = "4px";
      el.style.borderRadius = "4px";
      el.style.opacity = n && !on ? "0.35" : "1";
      el.style.transition = "opacity .18s ease";
      const b = el.querySelector("[data-badge]");
      if (b) b.style.background = on ? "var(--accent2)" : "var(--accent)";
    });
  }

  /* ---------------- render values ---------------- */
  renderVals() {
    return {
      rootRef: this.R.root, memRef: this.R.mem, resRef: this.R.res, mgRef: this.R.mg,
      chunkRef: this.R.chunk, pebRef: this.R.peb, cardRef: this.R.card, cardWrapRef: this.R.cardWrap,
      legendRef: this.R.legend, mermaidRef: this.R.mermaid, mgKeysRef: this.R.mgKeys,
      resSpeedRef: this.R.resSpeed, resKRef: this.R.resK, resKLabel: this.R.resKLabel,
      pebNaRef: this.R.pebNa, pebNbRef: this.R.pebNb, pebNaLabel: this.R.pebNaLabel, pebNbLabel: this.R.pebNbLabel,
      ram8Ref: this.R.ram8, ram16Ref: this.R.ram16, ram32Ref: this.R.ram32, ram64Ref: this.R.ram64,
      themeLabel: this.state.theme === "dark" ? "Light theme" : "Dark theme",
      resPlayLabel: this.state.resPlaying ? "Pause" : "Play",
      mgPlayLabel: this.state.mgPlaying ? "Pause" : "Play",
      toggleTheme: () => this.setState((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),
      ram8: () => this.setRam(8), ram16: () => this.setRam(16), ram32: () => this.setRam(32), ram64: () => this.setRam(64),
      resToggle: () => this.setState((s) => ({ resPlaying: !s.resPlaying })),
      resStep: () => { this.setState({ resPlaying: false }); this.resAdvance(); },
      resReset: () => { this.resInit(); this.drawResStatic(); },
      resSpeed: (e) => { this.resSpeedVal = Number(e.target.value); },
      resKChange: (e) => {
        this.resKVal = Number(e.target.value);
        if (this.R.resKLabel.current) this.R.resKLabel.current.textContent = this.resKVal;
        this.resInit(); this.drawResStatic();
      },
      mgToggle: () => this.setState((s) => ({ mgPlaying: !s.mgPlaying })),
      mgReset: () => { this.mgInit(); this.drawMGStatic(); },
      chunkRun: () => { this.ch.queue += 1; },
      chunkRunAll: () => { this.ch.queue += 12; },
      chunkReset: () => { this.ch = { mode: "idle", stage: -1, t: 0, queue: 0, chunks: 0 }; this.chunkTick0(); },
      pebNa: (e) => { this.peb.na = Number(e.target.value); this.drawPeb(); },
      pebNb: (e) => { this.peb.nb = Number(e.target.value); this.drawPeb(); }
    };
  }
}
