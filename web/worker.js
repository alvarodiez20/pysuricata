/* PySuricata browser demo — Pyodide runtime.
 *
 * Runs off the main thread so a large file never freezes the tab. The file is
 * mounted through WORKERFS rather than copied into the WASM heap, so pandas
 * reads it lazily off the Blob and pysuricata sees a chunk generator. That is
 * what keeps a 500 MB CSV inside a browser tab's memory budget.
 *
 * Three input kinds, and only two of them stream. CSV goes through
 * `read_csv(chunksize=…)` and Parquet through `iter_row_groups()`; Excel has no
 * streaming reader in pandas at all, so a workbook is decompressed and
 * materialised whole. The page caps Excel far lower for exactly that reason and
 * says so on screen — the bounded-memory claim is about the streaming formats,
 * and pretending otherwise here would be the one dishonest thing this demo could
 * do.
 */

let pyodide = null;
let cfg = null;

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
 * early failure the names may never have been bound.
 *
 * `source` and `wb` matter as much as the report for a workbook: read_excel
 * hands back a whole materialised frame, and leaving it bound would carry a
 * run's peak into the next one's baseline. */
function releasePython() {
  try {
    pyodide.runPython(`
import gc
for _n in ("_report", "_stats", "_ds", "source", "wb", "sh", "pf", "sheets", "picked"):
    globals().pop(_n, None)
gc.collect()
`);
  } catch {
    /* nothing to release */
  }
}

/** The one line of an exception worth showing in the log.
 *
 * For a plain JS error that is the first line. For a **Python** traceback --
 * which is what every failure from `runPythonAsync` is -- the first line is
 * always the literal `Traceback (most recent call last):`, and taking it threw
 * away the only informative part. `pysuricata==0.2.0 would not install here
 * (Traceback (most recent call last):)` is what a visitor saw on the day 0.2.0
 * was published, describing a failure that was easy to explain.
 *
 * The exception is at the bottom, below the frames: the last unindented
 * `SomeError: message` line. Scanned from the end because a message may itself
 * span lines, and matched on the type name rather than on "last line" because
 * micropip appends a bare `See: <url>` hint after some of its errors.
 */
const EXCEPTION_LINE =
  /^(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\s*:|^[A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit)\s*:/;

const errLine = (err) => {
  const text = String((err && err.message) || err).trim();
  const lines = text.split("\n").filter((line) => line.trim());
  if (!lines.length) return text;
  if (!/^Traceback \(most recent call last\)/.test(lines[0])) return lines[0];
  for (let i = lines.length - 1; i >= 0; i--) {
    if (EXCEPTION_LINE.test(lines[i])) return lines[i].trim();
  }
  // A traceback whose tail does not look like a typed exception at all. The
  // last line is still a better guess than the header we know says nothing.
  return lines[lines.length - 1].trim();
};

/* What PyPI is serving right now.
 *
 * `micropip.install("pysuricata")` already means "newest", but only as far as
 * the resolver can see: when the newest release is unsatisfiable inside this
 * Pyodide, micropip settles on an older one and says nothing, and a demo three
 * releases behind looks exactly like a current one. Asking PyPI directly gives
 * the install something to be checked against.
 *
 * `no-store` covers the half of the staleness this page controls. The JSON API
 * is served `max-age=900`, so a visitor who booted the demo minutes before a
 * release would otherwise be handed their own cached copy of the old answer.
 *
 * It does **not** make the two halves agree. This request bypasses the browser
 * cache; micropip's own query to the same API does not, so for a window after
 * an upload this function can report a version micropip cannot yet see. That
 * is not hypothetical -- it is what happened when 0.2.0 was published: the log
 * read `newest pysuricata is 0.2.0` and then `0.2.0 would not install here`,
 * and the fallback below served 0.1.5 for about a quarter of an hour. Nothing
 * here can fix that, and nothing here should try: the fallback keeps the demo
 * working, and the version line keeps it honest about which one it is running.
 */
/** A final release: digits and dots, optionally a `.postN`. Not `1.2.0rc1`. */
const STABLE_VERSION = /^\d+(\.\d+)*(\.post\d+)?$/;

async function latestOnPyPI() {
  const res = await fetch("https://pypi.org/pypi/pysuricata/json", { cache: "no-store" });
  if (!res.ok) throw new Error(`PyPI answered ${res.status}`);
  const version = (await res.json())?.info?.version;
  if (!version) throw new Error("no version in the PyPI response");
  return version;
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
  await pyodide.runPythonAsync("import micropip");
  const indexArg = cfg.indexUrls ? `, index_urls=${JSON.stringify(cfg.indexUrls)}` : "";

  /* A mirror (?local=1) serves whatever it was populated with, so asking
   * pypi.org what is newest would only manufacture a mismatch nobody running
   * offline can act on. */
  let latest = null;
  if (!cfg.indexUrls) {
    try {
      const newest = await latestOnPyPI();
      /* An alpha or a release candidate is newest without being what a visitor
       * should be handed. Unpinned micropip ignores pre-releases already, so
       * leaving `latest` null hands the choice back to the resolver. */
      if (STABLE_VERSION.test(newest)) {
        latest = newest;
        post("log", { text: `pypi.org: newest pysuricata is ${latest}` });
      } else {
        post("log", { text: `pypi.org: newest pysuricata is ${newest}, a pre-release — installing the newest stable instead` });
      }
    } catch (err) {
      post("log", { text: `could not read the PyPI index (${errLine(err)}) — installing unpinned` });
    }
  }

  /* Pinned first, unpinned as the fallback. Pinning is what makes "latest"
   * checkable; falling back is what keeps the demo alive on the day a release
   * cannot be satisfied inside this Pyodide, which pinning alone would turn
   * from a stale demo into no demo. Either way the version that actually
   * loaded is compared against PyPI below. */
  let pinFailed = null;
  if (latest) {
    try {
      await pyodide.runPythonAsync(`await micropip.install("pysuricata==${latest}"${indexArg})`);
    } catch (err) {
      pinFailed = errLine(err);
      post("log", { text: `pysuricata==${latest} would not install here (${pinFailed})` });
    }
  }
  if (!latest || pinFailed) {
    await pyodide.runPythonAsync(`await micropip.install("pysuricata"${indexArg})`);
  }

  const versions = pyodide.runPython(`
import json, sys, pandas, numpy, pysuricata
json.dumps({
    "pysuricata": pysuricata.__version__,
    "pandas": pandas.__version__,
    "numpy": numpy.__version__,
    "python": sys.version.split()[0],
})
`);

  const parsed = JSON.parse(versions);

  /* Only ever a warning. A demo running an older release still profiles files
   * correctly — what would be wrong is letting it pass for the current one. */
  const stale = latest && parsed.pysuricata !== latest
    ? { installed: parsed.pysuricata, latest, reason: pinFailed }
    : null;

  post("ready", {
    versions: parsed,
    latest,
    stale,
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

async function profileFile({ file, name, kind, sheet, title, chunkSize, maxColumns }) {
  const t0 = performance.now();
  const isParquet = kind === "parquet";
  const isExcel = kind === "excel";
  status("Mounting the file…", "Nothing is uploaded — this stays in your browser");
  const path = mountFile(file);

  if (isParquet) {
    status("Loading the Parquet reader…", "fastparquet, ~2 MB, first time only");
    await pyodide.loadPackage("fastparquet");
  }
  if (isExcel) {
    // calamine rather than openpyxl: it is the one spreadsheet reader in the
    // Pyodide distribution (openpyxl is not there at all), it covers xlsx, xlsm,
    // xlsb, xls and ods through a single engine, and pandas has spoken to it
    // natively since 2.2. Loaded on demand so a CSV visitor never pays for it.
    status("Loading the spreadsheet reader…", "python-calamine, ~1 MB, first time only");
    await pyodide.loadPackage("python-calamine");
  }

  pyodide.globals.set("_PATH", path);
  pyodide.globals.set("_NAME", title || name);
  pyodide.globals.set("_IS_PARQUET", isParquet);
  pyodide.globals.set("_IS_EXCEL", isExcel);
  // An empty string, not null: a JS null crosses into Python as JsNull, which is
  // not None and passes every `is not None` guard on the way to a TypeError deep
  // inside read_excel. "" is unambiguous on both sides of the bridge.
  pyodide.globals.set("_SHEET", sheet || "");
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
sheets = chosen = None
names = []

if _IS_EXCEL:
    from python_calamine import CalamineWorkbook

    # The workbook is opened for its table of contents only. A sheet's cells are
    # not read until one is chosen, so listing even a large book is cheap.
    wb = CalamineWorkbook.from_path(_PATH)
    sheets = []
    for sheet_name in wb.sheet_names:
        sh = wb.get_sheet_by_name(sheet_name)
        # height counts the header row; the report's row count will not.
        sheets.append({"name": sheet_name, "rows": max(sh.height - 1, 0), "cols": sh.width})
    if _SHEET:
        chosen = _SHEET
    elif len(sheets) == 1:
        chosen = sheets[0]["name"]
    picked = next((s for s in sheets if s["name"] == chosen), None)
    n_cols = picked["cols"] if picked else 0
    rows = picked["rows"] if picked else None
elif _IS_PARQUET:
    from fastparquet import ParquetFile
    pf = ParquetFile(_PATH)
    names = [str(c) for c in pf.columns]
    n_cols = len(names)
    rows = pf.count()
else:
    with open(_PATH, "r", newline="", errors="replace") as fh:
        names = [str(c) for c in next(csv.reader(fh))]
    n_cols = len(names)
    rows = None

json.dumps({"n_cols": n_cols, "n_rows": rows, "bytes": size, "sheets": sheets,
            "sheet": chosen, "columns": names[:12]})
`),
  );

  // A workbook with more than one sheet is a question, not a dataset. Picking
  // the first one silently is how a visitor ends up reading a confident report
  // about the wrong table.
  if (isExcel && sheet == null && preflight.sheets && preflight.sheets.length > 1) {
    post("sheets", { book: name, sheets: preflight.sheets });
    releasePython();
    unmountLast();  // release the Blob until a sheet is chosen
    return;
  }

  if (preflight.n_cols > maxColumns) {
    post("refused", {
      reason: `This file has ${preflight.n_cols} columns. The demo caps at ${maxColumns}.`,
      detail:
        "PySuricata's bounded memory applies to rows, not columns — each column " +
        "carries its own sketches and its own card in the report (~1.5 MB of RAM " +
        "and ~66 KB of HTML per column). A browser tab runs out first. Run wide " +
        "frames locally, or pass columns=[...] to profile a subset.",
    });
    releasePython();
    unmountLast();  // a refusal still holds the Blob until the mount goes
    return;
  }

  // A workbook can also arrive with nothing in it — every sheet blank, or no
  // sheet at all. Nothing above catches that, and read_excel would go on to
  // raise something unreadable, so say it plainly here.
  if (isExcel && (!preflight.sheet || !preflight.n_cols)) {
    post("refused", {
      reason: preflight.sheet
        ? `The sheet “${preflight.sheet}” has no data in it.`
        : "This workbook has no sheet with any data in it.",
      detail:
        "A profile needs at least one column of values. Check that the data is on a " +
        "sheet of its own and starts at the top-left cell, then try again.",
    });
    releasePython();
    unmountLast();
    return;
  }

  post("preflight", preflight);
  // The chosen sheet is resolved in the preflight (a single-sheet workbook needs
  // no question asked), so hand the answer back before the read.
  if (isExcel) pyodide.globals.set("_SHEET", preflight.sheet);
  status(
    isExcel
      ? `Reading “${preflight.sheet}”…`
      : isParquet
        ? "Profiling…"
        : `Profiling in ${chunkSize.toLocaleString()}-row chunks…`,
    isExcel
      ? "Excel has no streaming reader — the sheet is read whole"
      : "Streaming — the whole file is never held in memory at once",
  );

  const tRun = performance.now();
  const meta = JSON.parse(
    await pyodide.runPythonAsync(`
import json, re
import pandas as pd
import pysuricata

unnamed = 0
if _IS_EXCEL:
    # No chunksize on read_excel, in any engine: the sheet is decompressed and
    # built whole. This is the same call a visitor would make locally, which is
    # the point — the report they get here is the report they would get there.
    source = pd.read_excel(_PATH, sheet_name=_SHEET, engine="calamine")
    unnamed = sum(bool(re.fullmatch(r"Unnamed: \\d+", str(c))) for c in source.columns)
elif _IS_PARQUET:
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
    "unnamed_cols": unnamed,
    "sheet": _SHEET if _IS_EXCEL else None,
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
