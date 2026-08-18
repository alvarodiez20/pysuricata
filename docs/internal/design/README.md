# Report redesign — the design package, and what is left of it

This directory is the **fourth and current** design handoff for the report redesign,
copied into the repo on 2026-08-18 so that the issues can point at paths that exist
here rather than at a folder on one laptop.

Not published: `mkdocs.yml` carries `exclude_docs: internal/`, so nothing under
`docs/internal/` reaches the site.

## Where to start

| File | What it is |
| --- | --- |
| `../integration.md` | **the plan.** Fifteen phases mapped to files in this repo, each with acceptance criteria, then three code findings, a commit sequence, and nine open questions |
| `HANDOFF.md` | the designer's orientation note — what each file decides, and the five rules a contrast test cannot check |
| `*.dc.html` | the designs. Open directly in a browser; `support.js` sits beside them and is the only thing they load. Newest turn at the top of each file |
| `tokens.css` | the token layer, with the measurements and the five rules in comments |
| `contrast_test.reference.py` | the reference form of what is now `tests/test_contrast.py` |
| `flag_reference.reference.py` | the reference form of what is now `pysuricata/render/flag_reference.py`. **The shipped copy diverges deliberately** — five of its slugs did not exist in `triage.py`, and eight raised slugs had no entry. Read the shipped file, not this one |

`source/` from the original package — 4 MB of generated reports every figure was read
off — is **not** copied. Regenerate with `scripts/regenerate_example_report.py`.

## Read the status here, not the one in `integration.md`

`integration.md` opens with a status table written against the source **as it was at
handoff**. Five of its rows are now stale: the numeric restack reached all four card
kinds, the emoji are gone, the correlations empty state and the `--data-4` violation
are fixed, and the two dead `missing_section.py` methods were deleted.

The table below was verified against `HEAD` on 2026-08-18 by grepping the renderers.

### Landed

Phases 1–4b in full, 5, 5b, 5c, 5d, 6, 7, and the three code findings.

That includes: the token layer and `test_contrast.py`; the header, summary and sample;
the numeric restack on all four card kinds; every details pane for numeric, categorical
and datetime, and the removal of the boolean one; histogram geometry, including the
split between SVG bars and HTML labels; correlations routing, list and matrix; missing
values routed on chunk count; and the whole of 4b — cards collapse instead of
`display: none`, chips carry their threshold, the flag reference renders from the flags
a report actually raised, and the toolbar says what it is showing.

Phase 9's mobile and accessibility pass is done except for the parts that need a human
looking at a screen — see #300.

### Open

| Item | Issue | Note |
| --- | --- | --- |
| 5e.2 · flat baselines on the two datetime ratios | #291 | the baseline sits in a code comment at `datetime_card.py:109` |
| 5e.3 · thirteen datetime statistics become eight | #292 | blocked behind #291, which takes two of them off the grid |
| 5e.4 · bars or a line for the timeline | #293 | **decision.** The plan calls it the one genuine trade in the phase |
| 5e.5 · the per-card missing strip | #294 | also touches `numeric_card.py`; 5b.7 asked for the same removal and it did not happen |
| 5f.1 · suppress statistics that cannot be true | #295 | the largest single win left — categorical is eight of Titanic's twelve columns |
| 5f.2 · the even-split rule on every level bar | #296 | `even_split_pct()` exists at `flag_reference.py:254` and has no callers |
| 5f.3 · concentration figures for many-level columns | #297 | extends 5.4's high-cardinality branch |
| 8 · an HTML view for `compare()` | #121 | colour settled: `--data-1` and `--data-3`. Deltas are not verdicts |
| — · the Summary heading | #298 | the only section not using `.section-title` |
| — · degenerate frames | #299 | never designed. One column, zero rows, one row, all one type |
| — · dark mode in situ | #300 | values derived, never reviewed on a screen |
| — · should pysuricata recommend actions? | #301 | **decision.** Option 15a. The only thing in the plan that makes a claim beyond the data |

5f.4 — routing categorical to three faces by level count — is **held**, not open. 5f.1
plus 5f.2 get the two real wins without three code paths, and `integration.md` says to
revisit only if suppression still produces cards that feel wrong.

## Before touching any of it

Three things in `CLAUDE.md` were learned on exactly this work and will cost an
afternoon each if rediscovered:

- **The report inlines its own CSS and JS.** Searching the whole document for a class
  name finds it in the source that references it. `"dt-svg" in html` was `True` for a
  class no element carried.
- **`functionality.js` and the renderers never import each other.** A class renamed on
  one side produces no error and no console warning, just a control that goes quiet.
  `tests/test_js_selectors_match_markup.py` is what catches it.
- **A fixture that misses a branch reports "absent", not "unknown"** — and absent reads
  as broken. A frame of five distinct values profiles as *categorical*, so every
  numeric-card selector looks dead. Confirm in a browser before calling anything dead.

And three ratchets fail in **both** directions — growth is a regression, and shrinking
asks you to lower the baseline: report bytes and elements per card
(`tests/test_report_layout.py`), untokenised colours (`tests/test_colour_tokens.py`),
and `Processed bytes` staying out of the stat row
(`tests/test_processed_bytes_placement.py`).
