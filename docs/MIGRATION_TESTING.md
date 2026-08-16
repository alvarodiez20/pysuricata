# Testing the report UI migration

The integration plan is seventeen commits across fourteen stylesheets and fifteen
render modules. It lists tests to update. This is the other half: what the
migration can actually break, and the harness that catches it.

The organising idea is one sentence.

> **Presentation changes on every commit. The facts change on exactly two.**

Every test below exists to make that sentence checkable. Without it, seventeen
commits of 100%-churn diffs are unreviewable, and a real regression rides in
behind the noise that everyone has stopped reading.

---

## 1. What can actually break

| | Risk | Can it be silent? | Caught by |
|---|---|---|---|
| **A** | A number changes | **Yes, and catastrophically** | §2.1 golden payload, §2.2 fact coverage |
| **B** | A statistic silently disappears from the report | **Yes** | §2.2 fact coverage |
| **C** | The report stops being self-contained | Yes, until someone opens it offline | §2.3 |
| **D** | The compatibility shim hides breakage until commit 15 | **Yes, by design** | §2.4 |
| **E** | A tap target, rail or frozen column regresses | No — but only if measured | §2.5 |
| **F** | Contrast regresses | Yes | §2.6, extended |
| **G** | Dark mode was never verified | Yes — it is unverified *now* | §2.6, from commit 1 |
| **H** | The report gets bigger or slower | Yes | §2.7 |

Only **two commits in the whole plan can change a number**: the KMV clamp
(commit 12) and the correlations change, where `_collect_correlations` must start
returning below-threshold pairs. Those are the only places a *rendering* project
reaches into *computation*. Everything else is presentation, and must prove it.

---

## 2. The harness

### 2.1 Golden `summarize()` payload — the compute firewall

`summarize()` is already a versioned contract (`schema_version: 1`). Snapshot it
on three fixtures and assert byte-equality. Zero churn expected across sixteen of
the seventeen commits; the one exception is a one-line diff you can read.

```python
# tests/test_summarize_golden.py
@pytest.mark.parametrize("fixture", ["titanic", "daily", "wide_mixed"])
def test_payload_is_unchanged(fixture):
    got = summarize(load_fixture(fixture))
    expected = json.loads(GOLDEN[fixture].read_text())
    assert got == expected
```

This is the cheapest test in the set and the one that matters most. It runs in
milliseconds and it is the only thing standing between "a CSS refactor" and "a
CSS refactor that quietly changed the median".

### 2.2 Fact coverage — does the report still *show* what it computes

The real risk of phase 5.1 is not a wrong number, it is a **missing** one. The
numeric card today has two `.kv` tables holding fourteen statistics; the new
layout is a four-cell stat row plus detail tabs. Things move. Something will be
dropped, and no snapshot diff will make that visible in a document where every
line changed.

Measured against 0.0.42: **117 of the 125 numeric statistics** in `summarize()`
appear somewhere in the rendered HTML. Pin that set and require it not to shrink.

```python
# tests/test_report_renders_the_facts.py
BASELINE = load("tests/fixtures/rendered_facts.txt")   # (column, statistic) pairs

def test_no_statistic_disappears():
    doc = normalise(profile(FIXTURE).html)      # tags stripped, separators removed
    missing = [(c, k) for (c, k) in BASELINE if not appears(summary[c][k], doc)]
    assert not missing, f"statistics no longer rendered: {missing}"
```

`appears()` accepts any of the formats the report uses (`1234`, `1,234`,
`1.2e+03`, one to four decimal places) so that the deliberate reformatting in
phase 1 does not register as a loss. The eight statistics that legitimately do
not render — raw epoch nanoseconds, `mem_bytes` shown as `45 KB`,
`avg_interval_seconds` shown as `4.3 hours` — go in an explicit allow-list with
a reason each, so the exemptions are decisions rather than accidents.

### 2.3 Self-containment

The single-file promise is a constraint in the plan and nothing enforces it. It
is also the constraint most likely to be broken *by accident*, because the design
comps themselves break it — `Report Screen.dc.html` loads React from
`unpkg.com` and renders an empty page offline.

```python
def test_report_has_no_external_dependency():
    doc = profile(FIXTURE).html
    for pattern in (r'src="https?:', r'href="https?:[^"]*\.css',
                    r'@import', r'url\(\s*https?:', r'<link[^>]+stylesheet'):
        assert not re.search(pattern, doc), pattern
```

Add `assert "data:image/png" not in doc` once the logo becomes SVG (commit 3),
and the 578 KB regression can never come back.

### 2.4 The compatibility shim — make it shrink

The shim maps fifteen legacy variable names onto the new scale so nothing breaks
mid-migration. That is exactly the mechanism by which something *can* break
silently: `--chip-bg` now resolves to `transparent`, so a chip that relied on a
background for legibility becomes invisible with no error and no failing test.

Two changes to how the plan handles it:

1. **Delete shim entries per phase, not all at commit 15.** When phase 3 rewrites
   `_03-summary.css`, the legacy names that file used come out of the shim in the
   same commit. Commit 15 then deletes an empty block instead of detonating
   fifteen deferred breakages at once, four commits before release.
2. **Assert it only ever shrinks**, so the ratchet is mechanical:

```python
def test_shim_does_not_grow():
    shim = extract_shim_names(TOKENS_CSS)
    used = {n for n in shim if references(n, STATIC_CSS_DIR)}
    assert used <= PREVIOUS_SHIM_USE, f"legacy tokens came back: {used - PREVIOUS_SHIM_USE}"
```

### 2.5 The acceptance criteria are already tests — write them as tests

Every phase in the integration plan ends with numeric acceptance criteria. They
are phrased as assertions and should be executed as assertions, not read as a
checklist. Playwright is already available.

```python
# tests/test_report_layout.py — one file, parametrised over breakpoints and themes
BREAKPOINTS = [390, 768, 1240]
THEMES = ["light", "dark"]

def test_header_height(page, width):          assert bar_height(page) <= (48 if width < 640 else 52)
def test_every_target_is_44px(page):          assert not [t for t in targets(page) if t.w < 44 or t.h < 44]
def test_nav_rail_is_not_clipped(page):       assert rail.clientHeight >= 44 and rail.scrollWidth == rail.clientWidth
def test_frozen_index_aligns(page):           assert max(abs(a - b) for a, b in row_tops(page)) == 0
def test_no_horizontal_scroll(page):          assert doc.scrollWidth <= doc.clientWidth   # except the two known panes
def test_summary_height_at_390(page):         assert summary_height <= 560
def test_matrix_is_lower_triangle(page):      assert cells == n * (n - 1) // 2
def test_month_chart_has_12_slots(page):      assert len(month_slots) == 12
def test_no_chart_for_high_cardinality(page): assert chart_for("Name") is None
```

Nine of these come straight out of the plan's own acceptance lines. Writing them
once, at commit 3, means the remaining fourteen commits cannot regress them —
which is worth far more than checking each one by hand at the end.

### 2.6 Contrast — extend the provided test

`assets/test_contrast.py` works and should go in at commit 2 as planned. **It
currently fails on the tokens it ships with** — see §3.1 — and it does not cover
every pair the design uses. Add:

| Pair | Where the design uses it | Light | Dark |
|---|---|---:|---:|
| `data-4` on `paper` | below-threshold correlation bars (6.1), third series | **1.83** | **1.87** |
| `data-4` on `track` | palest step inside its own track | **1.55** | **1.60** |
| `data-3` on `paper` | second series | **2.63** | 3.21 |
| `data-1` / `data-3` / `data-4` adjacent | stacked composition bar (3.2) | **1.44–2.33** | **1.44–1.95** |
| `q-warn-text` on `track` | warning figure over a bar track | **4.23** | 7.46 |
| `muted` on `track` | captions over a bar track | 6.25 | 5.82 |

Bold fails its minimum (3:1 non-text, 4.5:1 text). Also run **dark from commit 1**,
not commit 16 — the test already parametrises over both themes, so a fifteen-commit
window of unverified dark mode is a choice, not a constraint.

### 2.7 Budgets

```python
def test_report_size():     assert len(profile(TITANIC).html) < 400_000   # from 1,176,000
def test_render_time():     assert render_seconds(TITANIC) < baseline * 1.15
def test_svg_element_count(): assert elements_per_card < 400   # full-width 50-bin histograms, 40-chunk strips
```

Size has a floor to beat and should ratchet down; render time and element count
have ceilings, because a full-width histogram at 50 bins and a per-chunk strip at
forty chunks both multiply SVG nodes, and rendering is on the hot path of the
library's own pitch.

### 2.8 Visual diffing — a review aid, not a gate

Screenshot every section at three breakpoints in both themes, per commit, and
publish the contact sheet as a CI artifact. Do **not** gate on pixel equality:
seventeen commits are *supposed* to change every pixel, so a pixel gate would be
disabled by commit 2 and stay disabled. The structural tests above are the gate;
the images are how a human reviews a phase in thirty seconds instead of thirty
minutes.

---

## 3. Two things to settle before commit 1

### 3.1 The contrast test fails on the tokens it ships with

```
[light] container edge and axis line: --rule-strong (#CFC7B8) on --paper (#FBF9F5) is 1.60:1, needs 3.0:1
[dark]  container edge and axis line: --rule-strong (#46413A) on --paper (#1C1A17) is 1.72:1, needs 3.0:1
```

36 pass, 2 fail. The dark failure is expected — the file says dark is a proposal.
The light one is not.

This is worth resolving *properly* rather than by relaxing the assertion, because
the test's entire value is that it is not negotiable. The token is doing two jobs
with different requirements: a **decorative row divider**, which WCAG 1.4.11 does
not cover, and a **chart axis line**, which it does, as "part of a graphic
required to understand the content".

**Split the token.** Keep `--rule-strong` as the hairline it is, and add
`--axis` at ≥3:1 against the paper for chart axes and gridlines. One extra token,
both uses honest, the test stays strict.

The stacked-bar adjacencies (1.44:1 between neighbouring segments) have a fix the
designer has already written down elsewhere: the correlation matrix spec calls for
"a 2px `--paper` gutter, so the grid reads as tiles rather than a table".
**Apply the same gutter between segments of the composition bar** and the
adjacency requirement disappears without touching the palette.

`--data-4` at 1.83:1 on the paper is the one that needs a palette decision: as the
fill for below-threshold correlation bars it is a graphic conveying information,
and at that ratio it is not reliably visible. Either darken it or give those bars
an outline in `--rule-strong`.

### 3.2 The data fixes are at commit 12 and belong at commit 0

Commit 12 clamps the KMV estimate. That fix is currently **eleven commits deep
into a UI migration**, which has three consequences:

- The fingerprint and golden-payload baselines get taken from a report that still
  says `892 distinct in 891 rows`, so the migration's reference point is a known-wrong
  number.
- It is the one bug that gates publishing anything, and it is a one-line clamp
  that has nothing to do with the redesign.
- It touches `accumulators/`, so it needs the accuracy oracle, not a snapshot — a
  different review from every other commit in the sequence, buried in the middle.

**Move the KMV clamp and the flag-threshold display to a commit 0, before phase 1.**
Then every baseline in this document is taken from correct output.

While it is open: **the quasi-constant flag is not in the integration plan and it
should be.** `age` — 67 distinct integers over 18–85 in 20,000 rows — is flagged
`Quasi-Constant`, severity `bad`, from a `data-value="0.3%"` unique *ratio*. Phase
5.7 changes how a flag is displayed without changing which flags fire, so the
redesign will render this false alarm more legibly and more prominently than
today. Fix the rule in commit 0 too, or the new report opens with
*"1 of 2 columns need a look"* pointing at the healthy column.

---

## 4. Per-commit test additions

| # | Commit | Test work |
|---|---|---|
| **0** | **KMV clamp, quasi-constant rule, flag thresholds** | **accuracy oracle; then take every baseline in this document** |
| 1 | tokens, typography, drop shadows and gradients | legacy-hex assertion in `test_css_integrity`; §2.4 shim ratchet |
| 2 | contrast test | as provided, plus §2.6 pairs, both themes |
| 3 | header | §2.5 skeleton — header height, tap targets, nav rail; §2.3 self-containment |
| 4–5 | summary, description | donut test rewritten as stacked-bar widths summing to 100; empty-description-is-one-row |
| 6–7 | sample | frozen-index alignment; `nan` never literal; no cell borders |
| 8–9 | numeric card, histogram axes | **§2.2 fact coverage matters most here** — the restack is where a statistic gets lost |
| 10–11 | categorical, high-cardinality, boolean, datetime | no chart element for high-cardinality; month chart has 12 slots |
| 12 | *(now empty — moved to 0)* | — |
| 13 | correlations | empty-state text names pair count and strongest value; triangle emits `n(n-1)/2`; no emoji; **`_collect_correlations` change re-runs the golden payload** |
| 14 | missing values | view routed by chunk count; no tab markup |
| 15 | remove the shim | shim ratchet reaches zero |
| 16 | dark mode | already covered since commit 2 — this becomes a confirmation, not a discovery |
| 17 | accessibility pass | greyscale render; full §2.5 matrix at all three breakpoints |

## 5. CI

Everything above runs on `ubuntu-latest`, which is free and uncapped for public
repositories. Add it to `ci.yml` — and note that `ci.yml` currently has **no
`push` trigger**, so none of it will ever run on `main` until that is fixed.

Three jobs:

- **fast** (every push and PR): golden payload, fact coverage, contrast, self-containment, budgets. Seconds.
- **layout** (every PR): Playwright matrix, 3 breakpoints × 2 themes. A minute or two.
- **contact sheet** (every PR): screenshots uploaded as an artifact. Never fails the build.
