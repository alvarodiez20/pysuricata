#!/usr/bin/env python3
"""The one command behind a published comparison table (#2).

`end_to_end.py` is a general-purpose harness: tools, suites, scale and round
count are all flags. That is right for exploring the space and wrong for a
number someone is about to publish -- the flags a headline ratio was actually
measured with are easy to lose between the terminal and the blog post, and "a
table of ratios against named competitors, trust me" is exactly the shape
that gets taken apart in a thread. This file is the fixed point: one
scenario, one tool set, one round count, nothing to reconstruct.

    python -m benchmarks.field
    python -m benchmarks.field --markdown field-results.md

The scenario is `datasets.mixed()` at `SCALE` -- the suite already built to
read as "the column mix of a real analytics table" (see its docstring),
rather than one of `numeric_wide` / `categorical_heavy`'s isolation shapes,
which exist to pin down a single kernel and would flatter or punish one tool
by the shape of the data alone. `ROUNDS` is `end_to_end.MIN_QUOTABLE_ROUNDS`:
the fewest rounds this project is willing to call a ratio quotable at.

Tools measured: `pysuricata`, `ydata` (imports `fg-data-profiling`'s
`data_profiling` first -- `ydata-profiling` renamed itself in 4.18.4 and
receives no further updates under the old name; see
`end_to_end.TOOLS["ydata"]` for the fallback and the note it attaches when
the fallback is what actually ran), `sweetviz`, `skimpy`. Whatever is not
installed is reported `skipped`, not silently dropped -- a comparison table
must never imply a tool lost that was never run.

Same load guard, same round-robin schedule, same environment block as
`end_to_end.py` -- this file pins a scenario on top of that machinery, it
does not duplicate it. If the published scenario needs to change, change the
constants below in their own pull request, so the change is visible in the
diff rather than buried in a flag someone typed once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmarks import end_to_end as e2e  # noqa: E402
from benchmarks.end_to_end import (  # noqa: E402
    MIN_QUOTABLE_ROUNDS,
    _report_load,
    environment,
    load_average,
    load_guard,
    round_robin,
    to_markdown,
)

#: The scenario a published comparison table is measured on. Constants, not
#: defaults -- a default can be overridden by a flag typed once and never
#: recorded; a constant can only change by editing this file, which leaves a
#: diff.
SUITE = "mixed"
SCALE = 0.2
ROUNDS = MIN_QUOTABLE_ROUNDS
TIMEOUT = 900
TOOLS = ["pysuricata", "ydata", "sweetviz", "skimpy"]

assert set(TOOLS) <= set(e2e.TOOLS), (
    f"field.py names a tool end_to_end.TOOLS does not: {set(TOOLS) - set(e2e.TOOLS)}"
)


def run(rounds: int = ROUNDS, force: bool = False) -> dict:
    """Run the fixed scenario and return the same payload shape `end_to_end.py`
    writes to `--json` -- `to_markdown()` and any downstream tooling built
    against that shape works on this unchanged."""
    load_start, refusal = load_guard(force)
    if refusal:
        raise SystemExit(f"refusing to measure: {refusal}")

    from benchmarks import datasets

    payload = {
        "environment": environment(),
        "rounds": rounds,
        "quotable": rounds >= MIN_QUOTABLE_ROUNDS,
        # Both ends: a job that starts *during* the run does not show up in
        # the reading taken before it.
        "load_start": load_start,
        "load_end": None,
        "forced": bool(force),
        "suites": {},
    }
    if load_start is not None:
        print(f"load average at start: {load_start:.2f}")
    print(json.dumps(payload["environment"], indent=2), "\n")

    print(f"=== {SUITE} (scale={SCALE}, rounds={rounds}) ===")
    df = datasets.build(SUITE, scale=SCALE)
    payload["suites"][SUITE] = {"_shape": datasets.describe(df)}
    del df
    payload["suites"][SUITE].update(
        round_robin(TOOLS, SUITE, SCALE, REPO, TIMEOUT, rounds)
    )

    payload["load_end"] = load_average()
    _report_load(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help=f"Interleaved rounds (default {ROUNDS}, the quotable minimum). "
        "Raise it for a tighter spread; do not lower it for a published number.",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Measure even when the machine is busy. The load is still "
        "recorded with the results, so the caveat travels with them.",
    )
    args = ap.parse_args(argv)

    if args.rounds < 1:
        ap.error("--rounds must be at least 1")
    if args.rounds < MIN_QUOTABLE_ROUNDS:
        print(
            f"WARNING: --rounds {args.rounds} is below the "
            f"{MIN_QUOTABLE_ROUNDS} this scenario is normally run at. "
            "Fine for a quick check, not for what you publish.\n"
        )

    payload = run(rounds=args.rounds, force=args.force)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(to_markdown(json.loads(json.dumps(payload))))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
