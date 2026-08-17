# PySuricata browser demo

A single static page that runs PySuricata itself — not a screenshot of it — inside
the visitor's browser. Drop a CSV or Parquet file, get the real report back. No
server, no upload, no compute cost.

## Why it exists

It demonstrates the one claim that is hard to show any other way: bounded memory,
in the single environment with a hard memory ceiling. It also turns "no server"
from a hosting detail into a privacy guarantee, which is what reaches people who
cannot put their data into a SaaS profiler.

## How it works

| Piece | What it does |
|---|---|
| `index.html` | The page. All CSS and UI logic inline; no framework, no build step. |
| `worker.js` | Pyodide runtime in a Web Worker, so a large file never freezes the tab. |
| `assets/`, `sample/` | Logo, favicon, and the sample dataset. |
| `_headers` | Cloudflare Pages response headers (CSP, caching). |

The runtime comes from the jsDelivr CDN and `pysuricata` is installed from PyPI at
page load with `micropip`. Nothing here is vendored, so **the demo picks up every
new PyPI release with no redeploy.**

Two details worth knowing:

- **WORKERFS, not a copy.** The dropped `File` is mounted into the WASM filesystem
  rather than read into memory, so `pandas.read_csv(..., chunksize=…)` streams off
  the Blob and `profile()` receives a chunk generator. That is what makes a large
  file survive a browser tab.
- **A mocked `psutil`.** `psutil` was declared as a runtime dependency but is
  imported in no code path in the library (only in tests and docs). It has no WASM
  wheel, so `micropip` would fail to resolve it. `worker.js` registers a mock
  distribution to satisfy the resolver. It has since moved to the
  `pysuricata[system]` extra, but this worker installs from PyPI and the published
  0.1.0 metadata still requires it. **Delete that block once 0.1.1 is published**,
  and confirm `micropip.install("pysuricata")` resolves in a bare Pyodide session
  with no mock package — see the comment in `worker.js`.

## Guardrails

Memory is bounded in rows, not in columns: roughly 1.5 MB of RAM and 66 KB of HTML
per column. A wide frame exhausts a tab long before a tall one does, so the page
refuses files above `MAX_COLUMNS` (250) with an explanation, and caps file size at
600 MB.

## Local development

```bash
cd web
python3 -m http.server 8000
open http://localhost:8000
```

That hits the real CDN and the real PyPI. Append `?local=1` only if you have set up
an offline mirror at `/pyodide/` and `/pypi/`.

## Deployment

Cloudflare Pages, git-connected to this repository:

- **Build command:** *(none)*
- **Build output directory:** `web`
- **Root directory:** *(repository root)*

Every push to `main` redeploys. Pull requests get preview URLs.

## Pinning

`index.html` pins the Pyodide version in `CONFIG.pyodideBase`. Bump it deliberately
and re-test — a new Pyodide ships a new pandas, and `pysuricata`'s pandas
requirement has to keep being satisfiable inside it.
