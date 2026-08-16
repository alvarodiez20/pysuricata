/* PySuricata browser demo — Pyodide runtime.
 *
 * Runs off the main thread so a large file never freezes the tab. The file is
 * mounted through WORKERFS rather than copied into the WASM heap, so pandas
 * reads it lazily off the Blob and pysuricata sees a chunk generator. That is
 * what keeps a 500 MB CSV inside a browser tab's memory budget.
 */

let pyodide = null;
let cfg = null;

/* pysuricata imports psutil in no code path — it is a test-only dependency that
 * is declared as a runtime one. psutil has no WASM wheel, so micropip's resolver
 * would fail on it. A mock distribution satisfies the resolver and gives any
 * lazy third-party import something that degrades instead of raising.
 * Remove once psutil moves to an optional extra upstream. */
const PSUTIL_SHIM = `__version__ = "7.1.0"

class _VMem:
    total = 2 * 1024 ** 3
    available = 1 * 1024 ** 3
    percent = 50.0
    used = 1 * 1024 ** 3
    free = 1 * 1024 ** 3

def virtual_memory():
    return _VMem()

def cpu_count(logical=True):
    return 1

class _MemInfo:
    rss = 0
    vms = 0

class Process:
    def __init__(self, pid=None):
        self.pid = pid or 0
    def memory_info(self):
        return _MemInfo()
    def memory_percent(self):
        return 0.0
    def cpu_percent(self, interval=None):
        return 0.0
`;

/* The run this worker is currently serving. Echoed back on every message so the
 * page can drop anything belonging to a run it has already moved past. */
let runId;

const post = (type, payload = {}) =>
  self.postMessage(runId === undefined ? { type, ...payload } : { type, run: runId, ...payload });
const status = (text, detail = null) => post("status", { text, detail });

/** WASM heap size — a fair proxy for what the runtime is actually holding. */
function heapMB() {
  try {
    return Math.round(pyodide._module.HEAPU8.length / 1048576);
  } catch {
    return null;
  }
}

/* Drop the previous run's report from the Python heap. A failed run used to
 * leave it pinned, so the next run started from an inflated baseline and the
 * heap figure the page prints stopped being true. Best-effort by design: on an
 * early failure the names may never have been bound. */
function releasePython() {
  try {
    pyodide.runPython(`
import gc
for _n in ("_report", "_stats", "_ds"):
    globals().pop(_n, None)
gc.collect()
`);
  } catch {
    /* nothing to release */
  }
}

async function boot(config) {
  cfg = config;
  const t0 = performance.now();

  status("Downloading the Python runtime…", "~13 MB, cached by your browser after this");
  self.importScripts(cfg.pyodideBase + "pyodide.js");
  pyodide = await loadPyodide({
    indexURL: cfg.pyodideBase,
    stdout: (m) => post("log", { text: m }),
    stderr: (m) => post("log", { text: m }),
  });

  status("Loading pandas and numpy…", "~8 MB");
  await pyodide.loadPackage(["micropip", "pandas", "numpy"]);

  status("Installing pysuricata from PyPI…", "~0.6 MB");
  pyodide.globals.set("PSUTIL_SHIM", PSUTIL_SHIM);
  await pyodide.runPythonAsync(`
import micropip
micropip.add_mock_package("psutil", "7.1.0", modules={"psutil": PSUTIL_SHIM})
`);
  const indexArg = cfg.indexUrls ? `, index_urls=${JSON.stringify(cfg.indexUrls)}` : "";
  await pyodide.runPythonAsync(`await micropip.install("pysuricata"${indexArg})`);

  const versions = pyodide.runPython(`
import json, sys, pandas, numpy, pysuricata
json.dumps({
    "pysuricata": pysuricata.__version__,
    "pandas": pandas.__version__,
    "numpy": numpy.__version__,
    "python": sys.version.split()[0],
})
`);

  post("ready", {
    versions: JSON.parse(versions),
    bootSecs: (performance.now() - t0) / 1000,
    heapMB: heapMB(),
  });
}

/* Every mount gets its own directory. Unmounting WORKERFS leaves the mount
 * point's child nodes behind, so re-mounting onto the same path fails on the
 * second file with an opaque ErrnoError — which is what used to break a second
 * profile in the same session. A fresh path per run sidesteps it, and the old
 * mount is released so the previous Blob is not pinned. */
let mounted = null;

function unmountLast() {
  if (!mounted) return;
  try {
    pyodide.FS.unmount(mounted);
  } catch {
    /* already gone */
  }
  mounted = null;
}

/** Mount a File into the WASM filesystem without copying its bytes. */
function mountFile(file) {
  unmountLast();
  const dir = `/data/run-${runId ?? 0}`;
  pyodide.FS.mkdirTree(dir);
  pyodide.FS.mount(pyodide.FS.filesystems.WORKERFS, { files: [file] }, dir);
  mounted = dir;
  return `${dir}/${file.name}`;
}

async function profileFile({ file, name, isParquet, chunkSize, maxColumns }) {
  const t0 = performance.now();
  status("Mounting the file…", "Nothing is uploaded — this stays in your browser");
  const path = mountFile(file);

  if (isParquet) {
    status("Loading the Parquet reader…", "fastparquet, ~2 MB, first time only");
    await pyodide.loadPackage("fastparquet");
  }

  pyodide.globals.set("_PATH", path);
  pyodide.globals.set("_NAME", name);
  pyodide.globals.set("_IS_PARQUET", isParquet);
  pyodide.globals.set("_CHUNK", chunkSize);
  pyodide.globals.set("_MAX_COLS", maxColumns);

  // Cheap pre-flight. Wide frames are the one shape that breaks the memory
  // guarantee (~1.5 MB and ~66 KB of HTML per column), so refuse early with a
  // real number rather than dying on an out-of-memory white screen.
  status("Inspecting the schema…");
  const preflight = JSON.parse(
    await pyodide.runPythonAsync(`
import csv, json, os
import pandas as pd

size = os.path.getsize(_PATH)
if _IS_PARQUET:
    from fastparquet import ParquetFile
    pf = ParquetFile(_PATH)
    cols = [c for c in pf.columns]
    rows = pf.count()
else:
    with open(_PATH, "r", newline="", errors="replace") as fh:
        cols = next(csv.reader(fh))
    rows = None
json.dumps({"n_cols": len(cols), "n_rows": rows, "bytes": size,
            "columns": [str(c) for c in cols[:12]]})
`),
  );

  if (preflight.n_cols > maxColumns) {
    post("refused", {
      reason: `This file has ${preflight.n_cols} columns. The demo caps at ${maxColumns}.`,
      detail:
        "PySuricata's bounded memory applies to rows, not columns — each column " +
        "carries its own sketches and its own card in the report (~1.5 MB of RAM " +
        "and ~66 KB of HTML per column). A browser tab runs out first. Run wide " +
        "frames locally, or pass columns=[...] to profile a subset.",
    });
    unmountLast();  // a refusal still holds the Blob until the mount goes
    return;
  }

  post("preflight", preflight);
  status(
    isParquet ? "Profiling…" : `Profiling in ${chunkSize.toLocaleString()}-row chunks…`,
    "Streaming — the whole file is never held in memory at once",
  );

  const tRun = performance.now();
  const meta = JSON.parse(
    await pyodide.runPythonAsync(`
import json
import pandas as pd
import pysuricata

if _IS_PARQUET:
    from fastparquet import ParquetFile
    pf = ParquetFile(_PATH)
    source = pf.iter_row_groups()
else:
    # A TextFileReader is a generator of frames. Handing it straight to
    # profile() is the out-of-core path: pandas never materialises the file.
    source = pd.read_csv(_PATH, chunksize=_CHUNK, low_memory=False)

_report = pysuricata.profile(source, title=_NAME)
_stats = dict(_report.stats or {})
_ds = dict(_stats.get("dataset", {}))
json.dumps({
    "rows": _ds.get("n_rows") or _ds.get("rows_est") or _ds.get("rows"),
    "cols": _ds.get("cols") or len(_stats.get("columns", {})),
    "missing_pct": _ds.get("missing_cells_pct"),
    "duplicate_pct": _ds.get("duplicate_rows_pct_est"),
    "html_bytes": len(_report.html),
})
`),
  );

  status("Rendering…");
  const html = pyodide.runPython("_report.html");
  const stats = pyodide.runPython("import json; json.dumps(_stats, default=str)");
  releasePython();

  post("done", {
    html,
    stats,
    meta,
    profileSecs: (performance.now() - tRun) / 1000,
    totalSecs: (performance.now() - t0) / 1000,
    heapMB: heapMB(),
  });
}

/** Stream a synthetic dataset straight into profile() as a chunk generator.
 *  Nothing is written to disk and no chunk outlives the loop, so peak memory is
 *  one chunk regardless of the row count. This is the bounded-memory claim in
 *  its purest form — and it costs the visitor no download at all. */
async function generate({ rows, chunkSize }) {
  const t0 = performance.now();
  const heapBefore = heapMB();
  status(
    `Streaming ${rows.toLocaleString()} synthetic rows through the profiler…`,
    "Chunks are built on demand and dropped — nothing is ever held whole",
  );
  pyodide.globals.set("_ROWS", rows);
  pyodide.globals.set("_CHUNK", chunkSize);

  const tRun = performance.now();
  const meta = JSON.parse(
    await pyodide.runPythonAsync(`
import json
import numpy as np, pandas as pd, pysuricata

_SENSORS = [f"S-{i:03d}" for i in range(250)]

def _sensor_chunks(total, batch):
    """Yield frames on demand. The caller never sees two at once."""
    rng = np.random.default_rng(7)
    written = 0
    while written < total:
        n = min(batch, total - written)
        idx = np.arange(written, written + n)
        df = pd.DataFrame({
            "reading_id": idx,
            "sensor": rng.choice(_SENSORS, n),
            "temperature_c": rng.normal(21.5, 4.2, n).round(2),
            "humidity_pct": rng.uniform(20, 95, n).round(1),
            "pressure_hpa": rng.lognormal(6.9, 0.02, n).round(1),
            "battery_pct": rng.integers(0, 101, n),
            "status": rng.choice(["ok", "ok", "ok", "degraded", "fault"], n),
            "firmware": rng.choice(["4.1.2", "4.2.0", "5.0.1"], n),
            "online": rng.random(n) > 0.03,
            "recorded_at": pd.Timestamp("2026-01-01") + pd.to_timedelta(idx, unit="s"),
        })
        # Uneven nullity, so the missing-value panel has something to say.
        df.loc[rng.random(n) < 0.12, "humidity_pct"] = np.nan
        df.loc[rng.random(n) < 0.04, "temperature_c"] = np.nan
        yield df
        written += n

_report = pysuricata.profile(
    _sensor_chunks(_ROWS, _CHUNK),
    title=f"Streamed sensor readings ({_ROWS:,} rows)",
)
_stats = dict(_report.stats or {})
_ds = dict(_stats.get("dataset", {}))
json.dumps({
    "rows": _ds.get("n_rows") or _ds.get("rows_est") or _ds.get("rows"),
    "cols": _ds.get("cols") or len(_stats.get("columns", {})),
    "missing_pct": _ds.get("missing_cells_pct"),
    "duplicate_pct": _ds.get("duplicate_rows_pct_est"),
    "bytes_seen": _ds.get("memory_bytes"),
    "html_bytes": len(_report.html),
})
`),
  );

  status("Rendering…");
  const html = pyodide.runPython("_report.html");
  const stats = pyodide.runPython("import json; json.dumps(_stats, default=str)");
  releasePython();

  post("done", {
    html,
    stats,
    meta,
    streamed: true,
    heapBefore,
    profileSecs: (performance.now() - tRun) / 1000,
    totalSecs: (performance.now() - t0) / 1000,
    heapMB: heapMB(),
  });
}

self.onmessage = async (e) => {
  const { type, run } = e.data;
  runId = run;
  try {
    if (type === "boot") await boot(e.data.config);
    else if (type === "profile") await profileFile(e.data);
    else if (type === "generate") await generate(e.data);
  } catch (err) {
    // Report the failure and then clean up, so the next run starts from the
    // same state a successful run would have left behind.
    post("error", {
      message: err && err.message ? err.message : String(err),
      stack: err && err.stack ? String(err.stack).slice(0, 4000) : null,
    });
    releasePython();
    unmountLast();
  }
};
