---
title: Versioning
description: What a pysuricata version number promises, and what it does not
---

# Versioning

A version number means nothing until the surface it describes is written down.
This page is that surface.

## The contract

pysuricata follows [Semantic Versioning](https://semver.org/), and means the
part everyone skips:

!!! note "Only a **major** bump may break you"

    `0.1.0 → 1.0.0` is the release allowed to break you.
    `0.1.0 → 0.2.0` adds; it does not break.
    `0.1.0 → 0.1.1` fixes.

That makes `pysuricata>=0.1,<1` a real guarantee, not just `~=0.1.0`: every
release before 1.0.0 keeps the surface below working.

Semver's own text says `0.y.z` means "anything may change" and that `1.0.0` is
what defines a public API — so read strictly, a 0.x promises nothing. Some
projects resolve that by adopting Cargo's convention, where a 0.x minor bump
does the job of a major one. **This project does not.** A minor number that
sometimes breaks and sometimes does not is a number a consumer cannot act on
without reading the changelog, which is the thing the version number exists to
save them from.

The cost is deliberate and worth stating: a breaking change costs 1.0.0. If the
surface below has to change before then, the change waits, ships behind a new
name beside the old one, or the project goes to 1.0. **A 0.x with a written
contract beats a 1.0 you have to walk back**, and a contract that only holds
when convenient is not one.

## What is covered

A breaking change to anything in this list requires a **major** bump.

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

`outliers_iqr_est` is the same call made again (#327): it was a count inside the
reservoir published beside a population `count`, 49x low at a million rows, and
correcting it is a fix. It moved `schema_version` to 2 anyway, for a reason the
rule above does not cover: a baseline stored before the fix holds counts on the
old scale, and `check` refusing across schema versions is the only thing that
stops it being compared against the new ones as though it were drift.

The rename that travelled with it, `outliers_mod_zscore` to
`outliers_mod_zscore_est`, is a break, and a break costs a major bump. So it did
not ship as one: **both names are published**, and the old one goes at 1.0.0.
That is the shape every rename takes here.

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
  *presented* and what error bound the duplicate count carries, and neither is
  a breaking change.
- **Log output, progress reporting and timing figures.**
- **The wheel's dependency floors**, except where raising one drops a Python
  version — that is covered.

Python version support is covered: dropping a Python is a break, so it waits
for a major bump.

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
