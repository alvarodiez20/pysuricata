---
title: Versioning
description: What a pysuricata version number promises, and what it does not
---

# Versioning

A version number means nothing until the surface it describes is written down.
This page is that surface.

## The contract, at 0.x

pysuricata follows [Semantic Versioning](https://semver.org/), with the
convention Cargo uses for pre-1.0 crates:

!!! note "At 0.x, a **minor** bump is what a major bump becomes at 1.0"

    `0.1.0 → 0.2.0` is the release allowed to break you.
    `0.1.0 → 0.1.1` never is.

That makes `pysuricata~=0.1.0` a real guarantee: you will get fixes, and you
will not get a breaking change without changing that line yourself.

Semver's own text says `0.y.z` means "anything may change" and that `1.0.0` is
what defines a public API — so "a stable release at 0.1.0" is, read strictly, a
contradiction. It is resolved by writing the contract down rather than by
picking a bigger number. **A 0.x with a written contract beats a 1.0 you have to
walk back.**

## What is covered

A breaking change to anything in this list requires a minor bump at 0.x, and a
major bump at 1.0 and after.

| Surface | Covered |
|---|---|
| **The public API** | The names exported from `pysuricata` — `profile`, `summarize`, `compare`, `check`, `ProfileConfig`, `ComputeOptions`, `RenderOptions`, and the result types they return |
| **The CLI** | The three subcommands, their documented flags, and the exit codes `0` (pass), `1` (findings), `2` (usage or input error) |
| **The `summarize()` payload** | Its shape, gated by `schema_version`. A field is never removed or repurposed without a bump |
| **The baseline file** | The format `check` reads and writes |
| **Documented defaults** | Changing a default that alters results — `max_uniques`, `numeric_sample_size`, `chunk_size` — is a behavioural break |

### What `schema_version` gates, and what it does not

`schema_version` describes the payload's **shape**, not the values in it.

- **Adding a key does not bump it.** Nothing that read the payload before can
  break by a key appearing beside the ones it already reads.
- **Removing or repurposing a key bumps it.** A consumer reading that key is
  broken by definition.
- **Correcting a wrong value does not bump it.** A statistic that was wrong and
  is now right is a bug fix; pinning it under the schema would mean the contract
  guaranteed the bug.

That last rule was decided on `duplicate_rows_est` (#202), which is the case
worth recording because it looks like a break and is not. The figure was
published raw while the HTML report suppressed it below the sketch's resolution,
so `summarize()` reported over a thousand duplicate rows in frames that had
exactly zero. It now carries the same suppression the report always applied, and
`duplicate_rows_uncertainty` was added beside it.

A consumer gating on `duplicate_rows_est > 0` sees that gate stop firing on
frames where it should never have fired. That is the fix, not a break — and it
is why the correction belongs where the figure is produced rather than where it
is drawn. Two call sites producing one statistic, with the threshold on only one
of them, is how the two surfaces drifted apart in the first place.

## What is not covered

Depending on any of these is depending on an implementation detail. They change
in patch releases without notice.

- **Anything `_private`**, and anything reachable only through a module path
  rather than the top-level package.
- **The HTML structure of the report** — class names, element order, the markup
  of any card. The redesign rewrote all of it across fourteen releases, and
  will again.
- **The exact value of any approximate figure.** Distinct counts, quantiles,
  duplicate counts and the bounds printed beside them are sketch estimates.
  Improving one is a fix, not a break: `0.0.73` changed how quantiles are
  *presented* and what error bound the duplicate count carries, and neither
  should have forced a minor bump.
- **Log output, progress reporting and timing figures.**
- **The wheel's dependency floors**, except where raising one drops a Python
  version — that is covered.

Python version support is covered: dropping a Python is a minor bump at 0.x.

## How a release happens

Publishing is triggered by **pushing a tag**, not by merging.

```bash
# after the version bump and changelog section have merged to main
git tag v0.1.0
git push origin v0.1.0
```

That is the whole reform, and everything else follows from it. Until 0.0.72,
`cd.yml` triggered on `push: branches: [main]` while `version-check` required a
bump on every pull request — so **one merged pull request was exactly one PyPI
release, unconditionally.** A rewritten kernel and a fixed typo were the same
size of event, and the version could not describe a change because it was
incremented by the act of merging rather than by a judgement about what merged.

The pipeline runs `guard → build → smoke → publish → release`, in that order:

1. **guard** — the tag matches `pyproject.toml`, and the changelog has notes.
2. **build** — one set of artifacts, reused by everything downstream.
3. **smoke** — the **wheel** is installed into an empty virtualenv on 3.10 and
   3.14, asked for its version, then made to profile a frame and check
   `schema_version`. CI tests the repository; this tests the artifact, which is
   not the same object.
4. **publish** — Trusted Publishing over OIDC, so no long-lived credential
   lives in the repository.

    !!! warning "Pushing the tag publishes. There is no confirmation step."

        The `pypi` environment has no protection rules, so nothing pauses
        between the tag and PyPI. A published version cannot be replaced, only
        yanked — and a yanked version keeps its number forever.

        The three jobs above are what stand in for a reviewer: the tag must
        match `pyproject.toml`, the changelog must carry notes, and the built
        wheel must install and profile a frame on 3.10 and 3.14. Check the
        version you are about to tag before you push it.

        A required reviewer can be added back under **Settings → Environments
        → pypi**.
5. **release** — GitHub release created **after** PyPI has the package, with
   notes lifted from `CHANGELOG.md`.
6. **demo-check** — the live demo, not a local checkout, profiles the sample
   dataset against the version `publish` just shipped and asserts the report
   frame is visibly painted (`web/e2e.py`, #1). `worker.js` installs
   `pysuricata==<latest>` from PyPI at page load, so every publish above edits
   the demo's launch asset in production, and this is what tests that edit
   instead of the first visitor doing it. Runs after `publish`, not before —
   the demo cannot see a version that is not on PyPI yet — and does not gate
   `release`: PyPI already has the package by then and nothing can take that
   back, so a demo failure fails the workflow on its own rather than
   delaying release notes that are already accurate.

    !!! warning "Freeze releases during a launch window"

        A release cannot be un-shipped, and the demo re-installs whatever is
        newest on every page load with no redeploy — so a release pushed
        during a traffic spike (a front-page post, a scheduled announcement)
        can break the demo in front of the traffic it was timed for, and
        `demo-check` only reports that after the fact. Do not push a version
        tag inside a planned launch window. If a release must go out during
        one anyway, watch the `demo-check` job to completion before treating
        the window as safe, and be ready to yank.

A pull request does **not** have to bump the version. If it does,
`scripts/check_version.py` asserts the step is legal — one component raised, the
ones below it reset, nothing skipped, no downgrade — and that a matching
changelog section exists.

## The gates for 1.0.0

Deliberately about evidence rather than dates.

1. **Two consecutive minor releases with no breaking change, unforced.** A
   settled shape, not a freeze.
2. **The deprecation queue is empty.** `ReportConfig` is aliased to
   `ProfileConfig` and now warns on use, naming **0.3.0** as the release that
   removes it (#210). The queue is not empty until that removal happens.
3. **Every approximate value carries its error bound.** The quantiles (#146) and
   the duplicate count (#161) are done; the distinct count already was.
4. **No known correctness bug in a covered path.**
5. **The covered surface above has not changed for a full minor cycle** — if the
   list is still moving, the contract is not ready to be permanent.
