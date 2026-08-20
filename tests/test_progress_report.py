"""#211 — one option for *show me something before it finishes*.

Checkpointing was five options in `ComputeOptions`, and the intent is not what
any of them is named after. Nobody thinks "I would like a checkpoint prefix".
They also encoded an implementation -- pickle files on disk, rotated -- as the
interface, which is why `checkpoint_write_html` existed as a fifth boolean
rather than being the whole point.

`progress_report=N` replaces them as the surface. The pickle rotation still runs
underneath, because resuming a run needs it; it just stops being the thing you
have to understand to get a report while you wait.

The removal half of the issue is **not** here. Deleting the five names is a
break, `docs/versioning.md` says a break costs a major bump, so they are
deprecated against **1.0.0** and still work. The issue body names 0.2.0, which
was correct under the Cargo-style reading the project has since dropped.

On the invariant this rests on: the issue was blocked on #205, `finalize()`
consuming reservoir randomness -- a progressive report calls `finalize()`
repeatedly by construction, so if that were true the median would move while
the user watched it. #205 closed as not reproducible: the bit generator's state
is byte-identical across `finalize()`, and `TestFinalizeIsIdempotent` in
`benchmarks/accuracy.py` pins it. `test_the_numbers_do_not_move` below is the
same claim from the public API.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from pysuricata import ComputeOptions, ProfileConfig, profile, summarize

#: Six chunks at this size, so `progress_report=2` writes three reports and the
#: count is a number rather than "more than zero".
ROWS = 12_000
CHUNK = 2_000


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "a": rng.normal(size=ROWS),
            "b": rng.choice(list("abcd"), ROWS),
            "c": rng.random(ROWS) > 0.5,
        }
    )


def _config(**compute) -> ProfileConfig:
    return ProfileConfig(
        compute=ComputeOptions(chunk_size=CHUNK, random_seed=0, **compute)
    )


#: The things a report stamps from the clock. None is a statistic, and none
#: can be equal across two runs of anything.
#:
#: Matched by their place in the template rather than by the shape of a
#: duration. `human_time` renders `0.02 s`, `1.50 s`, `1 min 5 s` or
#: `1 h 1 min 1 s` depending on magnitude, so a numeric pattern is a bet that
#: both runs land in the same format -- and the first version of this file made
#: that bet and lost it on one CI runner out of six, where the slower of the
#: two runs crossed a unit boundary. `[^<]*` inside the exact element cannot
#: care.
_CLOCK_STAMPED = (
    (re.compile(r'data-report-id="[^"]*"'), 'data-report-id="STAMP"'),
    # `Generated` is `datetime.now()` at second resolution, so two runs agree
    # only when they land in the same second. That made this comparison fail
    # about one full-suite run in three -- intermittently, which is the worst
    # way to learn that a normalisation is incomplete.
    (
        re.compile(r'(Generated</span>\s*<span class="v">)[^<]*(</span>)'),
        r"\1GENERATED\2",
    ),
    (
        re.compile(r'(Profiled in</span>\s*<span class="v">)[^<]*(</span>)'),
        r"\1ELAPSED\2",
    ),
    (
        re.compile(r'(stat--elapsed".*?<div class="stat__val">)[^<]*(</div>)', re.S),
        r"\1ELAPSED\2",
    ),
)


def _without_the_clock(html: str) -> str:
    for pattern, replacement in _CLOCK_STAMPED:
        html, n = pattern.subn(replacement, html)
        assert n, (
            f"nothing matched {pattern.pattern!r}; the template moved and this "
            f"normalisation is now silently doing nothing, which would make "
            f"the comparison below pass or fail for an unrelated reason"
        )
    return html


class TestTheComparisonItselfIsSound:
    """The helper above is the part that broke, so it gets its own test.

    A normalisation that quietly stops matching turns the comparison into
    either a tautology or an unrelated failure, and neither announces itself.
    """

    #: The two ends of `human_time`'s range, which is what the first version
    #: could not span: a fast local run and a slow CI one do not render the
    #: elapsed time in the same units.
    @pytest.mark.parametrize(
        "fast,slow",
        [("0.02 s", "1 min 5 s"), ("340 ms", "0.90 s"), ("1.50 s", "1 h 1 min 1 s")],
    )
    def test_two_durations_in_different_units_normalise_alike(self, fast, slow):
        def page(duration: str, report_id: str) -> str:
            return (
                f'<div data-report-id="{report_id}">'
                f'<span class="k">Generated</span> '
                f'<span class="v">2026-08-20 {report_id}0:00:00</span>'
                f'<span class="k">Profiled in</span> '
                f'<span class="v">{duration}</span>'
                f'<div class="stat stat--elapsed"><div class="stat__cap">Elapsed</div>'
                f'<div class="stat__val">{duration}</div></div>'
                f'<div class="stat__val">0.02 s</div>'
                "</div>"
            )

        assert _without_the_clock(page(fast, "A")) == _without_the_clock(
            page(slow, "B")
        )

    def test_it_leaves_other_values_alone(self):
        """The trailing `stat__val` in the fixture above is deliberately the
        same text as a plausible duration. Normalising by *value* rather than
        by position would erase it too, and then a real difference in a real
        statistic could hide behind the clock."""
        page = (
            '<div data-report-id="A">'
            '<span class="k">Generated</span> <span class="v">2026-08-20 10:00:00</span>'
            '<span class="k">Profiled in</span> <span class="v">0.02 s</span>'
            '<div class="stat stat--elapsed"><div class="stat__val">0.02 s</div></div>'
            '<div class="stat stat--rows"><div class="stat__val">0.02 s</div></div>'
            "</div>"
        )
        cleaned = _without_the_clock(page)

        assert cleaned.count("ELAPSED") == 2, "both clock sites normalised"
        assert (
            '<div class="stat stat--rows"><div class="stat__val">0.02 s</div>'
            in cleaned
        ), (
            "a statistic that happens to read like a duration was erased; "
            "normalising by value rather than by position would hide a real "
            "difference behind the clock"
        )

    def test_it_refuses_to_pass_silently_if_the_template_moves(self):
        with pytest.raises(AssertionError, match="silently doing nothing"):
            _without_the_clock("<html>nothing the report would emit</html>")


class TestItRendersWhileYouWait:
    def test_a_partial_report_lands_every_n_chunks(self, frame, tmp_path):
        profile(
            frame,
            config=_config(progress_report=2, checkpoint_dir=str(tmp_path)),
        )
        partials = sorted(p.name for p in tmp_path.glob("*.html"))

        assert len(partials) == 3, (
            f"{ROWS} rows at {CHUNK} is six chunks, so progress_report=2 owes "
            f"three partial reports; got {partials}"
        )

    def test_it_is_off_by_default(self, frame, tmp_path):
        profile(frame, config=_config(checkpoint_dir=str(tmp_path)))

        assert not list(tmp_path.iterdir()), (
            "a default run wrote files; progress_report defaults to 0 and "
            "nothing else may turn it on"
        )


class TestTurningItOnChangesNothing:
    """The acceptance box that matters. A progressive report that perturbs the
    result is worse than no progressive report."""

    def test_the_numbers_do_not_move(self, frame, tmp_path):
        """`summarize()` has no timestamps in it, so this is an exact equality
        over the statistics themselves rather than over their rendering."""
        on = summarize(frame, chunk_size=CHUNK, seed=0, progress_report=2)
        off = summarize(frame, chunk_size=CHUNK, seed=0)

        # `dataset` carries the wall-clock duration, which is not a statistic.
        assert on["columns"] == off["columns"]

    def test_the_final_report_is_the_same_document(self, frame, tmp_path):
        """Identical apart from the two fields stamped from the clock -- the
        elapsed time, which writing three extra reports necessarily changes,
        and the report id, which is generated per run."""
        on = profile(
            frame, config=_config(progress_report=2, checkpoint_dir=str(tmp_path))
        ).html
        off = profile(frame, config=_config()).html

        assert _without_the_clock(on) == _without_the_clock(off)


class TestTheKeywordReachesIt:
    @pytest.mark.parametrize("entry", [profile, summarize])
    def test_it_is_in_the_passthrough(self, frame, entry, tmp_path):
        """#211 asks for it in the keyword passthrough, not only on the
        dataclass -- the passthrough is the surface most callers use."""
        entry(frame.head(100), progress_report=0)

    def test_a_negative_interval_is_refused(self):
        with pytest.raises(ValueError, match="progress_report must be non-negative"):
            ComputeOptions(progress_report=-1)


class TestTheOldNamesStillWorkAndSaySo:
    @pytest.mark.parametrize(
        "setting",
        [
            {"checkpoint_every_n_chunks": 2},
            {"checkpoint_write_html": True},
        ],
    )
    def test_each_replaced_name_warns(self, setting):
        with pytest.warns(DeprecationWarning) as record:
            ComputeOptions(**setting)

        message = str(record[0].message)
        assert next(iter(setting)) in message
        assert "1.0.0" in message, (
            "the warning must name the release that removes it, and it must be "
            "one the versioning contract can carry -- a break needs a major"
        )
        assert "progress_report" in message, "and it must name the replacement"

    @pytest.mark.parametrize(
        "setting",
        [
            {"checkpoint_dir": "/tmp/x"},
            {"checkpoint_prefix": "other"},
            {"checkpoint_max_to_keep": 5},
        ],
    )
    def test_the_three_without_a_replacement_do_not_warn(self, setting):
        """`progress_report` is an interval that turns HTML on. It cannot say
        where the files go, what they are called or how many are kept, and
        there is nowhere else to say it either.

        `__init__.py` records the rule for the release a warning names -- it
        must be one that can carry the removal, or the deadline cannot happen.
        The replacement it names is subject to the same rule, and pointing
        these three at `progress_report` would be telling people to migrate to
        something that does not do their job. They stay until #211's placement
        half is designed.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ComputeOptions(**setting)

    def test_the_replacement_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ComputeOptions(progress_report=4)

    def test_a_default_passed_explicitly_does_not_warn(self):
        """Deprecating by value cannot see whether a field was passed. Someone
        who passes the default gets no warning, which costs them nothing --
        the setting is doing nothing either way."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ComputeOptions(checkpoint_every_n_chunks=0, checkpoint_max_to_keep=3)

    def test_the_lens_warns_too(self):
        """`opts.checkpoint.dir = ...` is the spelling `CheckpointView` exists
        to encourage, so it is the one that must not go quiet."""
        options = ComputeOptions()
        with pytest.warns(DeprecationWarning, match="1.0.0"):
            options.checkpoint.every_n_chunks = 4

    def test_they_still_do_what_they_did(self, frame, tmp_path):
        """Deprecated is not removed. Someone who ignores the warning until
        1.0.0 must keep getting the behaviour they have today."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile(
                frame,
                config=_config(
                    checkpoint_every_n_chunks=3,
                    checkpoint_write_html=True,
                    checkpoint_dir=str(tmp_path),
                ),
            )

        assert len(list(tmp_path.glob("*.html"))) == 2

    def test_placement_still_needs_the_old_name(self, frame, tmp_path):
        """Recorded because it is the gap, not because it is the behaviour we
        want. `progress_report` alone writes to the working directory, and
        `checkpoint_dir` is the only way to say otherwise -- which is why that
        name is not deprecated yet. When #211's placement half lands, this test
        should start failing and be replaced by one that names the new option.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            profile(
                frame,
                config=_config(progress_report=2, checkpoint_dir=str(tmp_path)),
            )

        assert len(list(tmp_path.glob("*.html"))) == 3
