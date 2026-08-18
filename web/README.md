# PySuricata browser demo

A single static page that runs PySuricata itself — not a screenshot of it — inside
the visitor's browser. Drop a CSV, a Parquet file or an Excel workbook, get the
real report back. No server, no upload, no compute cost.

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

One detail worth knowing:

- **WORKERFS, not a copy.** The dropped `File` is mounted into the WASM filesystem
  rather than read into memory, so `pandas.read_csv(..., chunksize=…)` streams off
  the Blob and `profile()` receives a chunk generator. That is what makes a large
  file survive a browser tab.

## Input formats

| Kind | Reader | Streams? |
|---|---|---|
| `.csv`, `.tsv`, `.txt` | `pd.read_csv(chunksize=200_000)` | yes |
| `.parquet`, `.pq` | `fastparquet.iter_row_groups()` | yes |
| `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods` | `pd.read_excel(engine="calamine")` | **no** |

The spreadsheet reader is `python-calamine`, chosen because **openpyxl is not in
the Pyodide distribution** while calamine is, and because one engine covers all
five spreadsheet formats. pandas has spoken to it natively since 2.2. Both it and
fastparquet load on demand, so a CSV visitor downloads neither.

Two things follow from Excel being the one format that cannot stream — pandas has
no `chunksize` on `read_excel` in any engine, so the sheet is built whole before
the profiler sees a row:

- The page **says so on screen**. The bounded-memory claim belongs to CSV and
  Parquet, and quietly letting it cover a workbook would be the one dishonest
  thing this demo could do.
- Workbooks are capped at `MAX_EXCEL_BYTES` (40 MB) rather than 600 MB, because
  a compressed workbook expands several times over on the way into the heap.

A multi-sheet workbook **pauses and asks which sheet**, with each sheet's row and
column counts; empty sheets are not offered. Silently taking the first one is how
a visitor ends up reading a confident report about the wrong table. Two Excel
failure modes are named rather than rendered: a sheet whose first row is a title
or a blank spacer comes back as `Unnamed: n` columns and raises a warning, and a
workbook with no data in any sheet is refused with a reason.

## Guardrails

Memory is bounded in rows, not in columns: roughly 1.5 MB of RAM and 66 KB of HTML
per column. A wide frame exhausts a tab long before a tall one does, so the page
refuses files above `MAX_COLUMNS` (250) with an explanation, and caps file size at
600 MB (40 MB for workbooks, above).

## Local development

```bash
cd web
python3 -m http.server 8000
open http://localhost:8000
```

That hits the real CDN and the real PyPI. Append `?local=1` only if you have set up
an offline mirror at `/pyodide/` and `/pypi/`.

## Deployment

Live at **<https://pysuricata.pages.dev>**.

Cloudflare Pages, git-connected to this repository:

- **Build command:** *(none)*
- **Build output directory:** `web`
- **Root directory:** *(repository root)*

Every push to `main` redeploys. Pull requests get preview URLs.

## Pinning

`index.html` pins the Pyodide version in `CONFIG.pyodideBase`. Bump it deliberately
and re-test — a new Pyodide ships a new pandas, and `pysuricata`'s pandas
requirement has to keep being satisfiable inside it.
