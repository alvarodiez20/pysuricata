"""Every estimate must track truth, and say so when it cannot (#331).

Three defects shipped in one release -- #327, #328 and #329 -- and they were
found by an outside benchmark rather than by this suite. The reason is
structural, and worth stating plainly: **the suite tested that estimators run,
not that they are right.** `outliers_iqr_est` was 49x low at a million rows and
every test passed, because no test compared it to a number computed another way.

The per-defect regressions live next door (`test_outlier_population_estimate.py`,
`test_topk_error_bound.py`) and pin the two bugs that were found. This module
pins the *class* they belong to, so the next estimator cannot ship the same way.

Two ideas do the work.

**An oracle per estimate.** `_ORACLES` pairs every published estimate with an
independent exact computation of the same quantity. A test that walks the table
cannot be satisfied by an estimator agreeing with itself.

**Adding an estimate forces a decision.** `test_every_estimate_declares_an_oracle`
fails when a key matching the estimate-shaped naming convention appears in the
payload without an entry here -- the same forcing move `SUMMARY_FIELDS_WITHHELD`
makes for publication, applied to accuracy.

The sizes straddle every internal budget deliberately. `numeric_sample_size` is
20,000 and `chunk_size` 50,000, so a suite whose largest frame is 4,000 rows --
which was the case -- exercises the sampled path in none of its columns and the
chunked path not at all. #327 lived entirely above 20,000 and was invisible
below it.
"""

from __future__ import annotations

import dataclasses as dc
from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import pysuricata

#: The internal budgets an estimate changes behaviour at. Named rather than
#: inlined so a test that straddles one says which one it is straddling.
NUMERIC_SAMPLE_SIZE = 20_000
CHUNK_SIZE = 50_000
TOP_K = 50
UNIQUES_SKETCH = 2_048

#: Row counts either side of the reservoir and the chunk boundary. The pairs
#: matter more than the absolute values: 19,999/20,001 isolates the sampling
#: threshold, and 49,999/50,001 the chunking one.
SIZES = (5_000, NUMERIC_SAMPLE_SIZE - 1, NUMERIC_SAMPLE_SIZE + 1, CHUNK_SIZE + 1)

#: Sampling error, not a fudge factor. The reservoir is a uniform sample, so a
#: scaled count carries roughly `1/sqrt(k)` relative error; measured spread over
#: these cases is under 4%. Set at 15% so a genuine regression -- the 60-98%
#: error the unscaled count had -- cannot hide inside it.
TOLERANCE = 0.15


def _lognormal(n: int) -> np.ndarray:
    """A column with a real tail, so the IQR fence has something to count."""
    return np.random.default_rng(0).lognormal(0.0, 1.0, n)


def _cased_words(n: int) -> np.ndarray:
    """Strings whose distinct count changes under folding and trimming.

    Built so the three distinct counts genuinely differ -- raw > case-folded >
    nothing, and raw > trimmed -- because an oracle over a column where they
    coincide would pass against an estimator that computed any of the three.
    """
    rng = np.random.default_rng(3)
    bases = [f"word{i}" for i in range(max(8, n // 10))]
    picked = rng.choice(bases, size=n)
    out = []
    for i, word in enumerate(picked):
        if i % 3 == 0:
            out.append(word.upper())
        elif i % 5 == 0:
            out.append(f"  {word} ")
        else:
            out.append(word)
    return np.array(out, dtype=object)


def _exact_iqr_outliers(values: np.ndarray) -> int:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def _exact_distinct(values: np.ndarray) -> int:
    return int(len(np.unique(values)))


def _exact_distinct_folded(values: np.ndarray) -> int:
    return int(len({str(v).lower() for v in values}))


def _exact_distinct_trimmed(values: np.ndarray) -> int:
    return int(len({str(v).strip() for v in values}))


@dc.dataclass(frozen=True)
class _Estimate:
    """A published estimate, and how to check it against truth.

    `make` builds a column of a given size so the scale tests can grow it, and
    `exact` computes the same quantity another way. `kind` says which column of
    the payload carries the key.
    """

    kind: str
    make: Callable[[int], np.ndarray]
    exact: Callable[[np.ndarray], float]


#: Published estimate -> an independent exact computation of the same quantity.
#:
#: Every entry is a claim that can be wrong, which is the point. `unique_est`
#: comes from KMV and `outliers_iqr_est` from the scaled reservoir; no oracle
#: here shares a line of code with the thing it checks.
_ORACLES: dict[str, _Estimate] = {
    "outliers_iqr_est": _Estimate("numeric", _lognormal, _exact_iqr_outliers),
    "unique_est": _Estimate("numeric", _lognormal, _exact_distinct),
    "case_variants_est": _Estimate("categorical", _cased_words, _exact_distinct_folded),
    "trim_variants_est": _Estimate(
        "categorical", _cased_words, _exact_distinct_trimmed
    ),
}

#: Keys that look like estimates and are deliberately not oracled here, with the
#: reason. Keeping this beside `_ORACLES` is what makes the forcing test honest:
#: a new estimate lands in one list or the other, never in neither.
_NO_ORACLE = {
    "outliers_mod_zscore_est": (
        "the modified z-score fence is defined against the sample's own median "
        "and MAD, so an 'exact' oracle over the full column is answering a "
        "different question. Covered by scale invariance in "
        "test_outlier_population_estimate.py instead"
    ),
    "unique_ratio_approx": "a ratio of unique_est, which is oracled above",
    "rows_est": "the row count is exact; the name is historical",
    "duplicate_rows_est": (
        "deliberately suppressed below the sketch's resolution, so on a frame "
        "with few duplicates it is 0 by design rather than wrong, and an "
        "oracle counting real duplicates would read that as an error. The "
        "readable form is the interval #329 added, and the contract below "
        "asserts that interval contains the truth"
    ),
    "duplicate_rows_pct_est": "the same figure as a percentage",
}


def _column_of_kind(payload: dict, kind: str) -> dict:
    wanted = ("numeric", "identifier") if kind == "numeric" else (kind,)
    for stats in payload["columns"].values():
        if stats["type"] in wanted:
            return stats
    raise AssertionError(f"no {kind} column in the fixture")


def _numeric_column(payload: dict) -> dict:
    return _column_of_kind(payload, "numeric")


def _reported(key: str, values: np.ndarray) -> float:
    """Profile a one-column frame and read the estimate back off the payload."""
    spec = _ORACLES[key]
    payload = pysuricata.summarize(pd.DataFrame({"x": values}))
    return _column_of_kind(payload, spec.kind)[key]


class TestEstimatesTrackTruth:
    """Contract 1: a `*_est` must not drift with `n`.

    A ratio to truth that tracks `numeric_sample_size / n` is the signature of
    a count made in the sample and published against the population. That is
    #327 exactly, and it is invisible to any test that never crosses 20,000
    rows.
    """

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("key", sorted(_ORACLES))
    def test_the_estimate_is_within_tolerance_of_the_exact_value(
        self, n: int, key: str
    ) -> None:
        spec = _ORACLES[key]
        values = spec.make(n)
        reported = _reported(key, values)
        exact = spec.exact(values)

        assert exact > 0, "fixture must exercise the statistic, or this proves nothing"
        ratio = reported / exact
        assert abs(ratio - 1.0) < TOLERANCE, (
            f"{key} at n={n:,}: reported {reported:,} against a true {exact:,} "
            f"(ratio {ratio:.3f}). A ratio that tracks a sample-size fraction "
            "means the estimate describes the sample, not the column."
        )

    @pytest.mark.parametrize("key", sorted(_ORACLES))
    def test_the_error_does_not_grow_with_the_frame(self, key: str) -> None:
        """The sharper form: drift is a trend, and a single size cannot show one.

        #327 passed every fixed-size check ever written for it. What convicts it
        is that the ratio fell monotonically -- 1.000, 0.396, 0.101, 0.021 -- as
        the frame grew.
        """
        spec = _ORACLES[key]
        ratios = []
        for n in SIZES:
            values = spec.make(n)
            ratios.append(_reported(key, values) / spec.exact(values))

        assert max(ratios) - min(ratios) < TOLERANCE, (
            f"{key} ratios across {SIZES}: "
            f"{[f'{r:.3f}' for r in ratios]}. A spread this wide is a scale "
            "error, not sampling noise."
        )


class TestTheApproxContract:
    """Contract 2: `approx` must mean what it says, in both directions.

    #328 was the False direction failing: eviction *deletes* Misra-Gries
    counters, so the published list shrinks below the budget precisely when the
    sketch is under most pressure, and a length comparison read that as exact.
    """

    def test_approx_false_means_the_counts_are_exact(self) -> None:
        """The direction that actually hurts. `approx is False` is a promise."""
        values = np.array([f"v{i % 20}" for i in range(4_000)], dtype=object)
        payload = pysuricata.summarize(pd.DataFrame({"c": values}))
        stats = payload["columns"]["c"]

        assert stats["approx"] is False, "20 distinct values fit in 50 counters"

        truth = Counter(values)
        for value, count in stats["top_items"]:
            assert count == truth[value], (
                f"{value!r} published as {count:,} against a true "
                f"{truth[value]:,}, with approx=False claiming exactness"
            )

    def test_approx_true_carries_a_bound_that_brackets_the_truth(self) -> None:
        """The other direction: an estimate must publish its own error."""
        rng = np.random.default_rng(7)
        values = rng.choice([f"v{i}" for i in range(TOP_K * 20)], size=200_000)
        payload = pysuricata.summarize(pd.DataFrame({"c": values}))
        stats = payload["columns"]["c"]

        assert stats["approx"] is True, "1,000 distinct values cannot fit in 50"
        bound = stats["top_items_uncertainty"]
        assert bound > 0, "an approximate column must publish a non-zero bound"

        truth = Counter(values)
        for value, count in stats["top_items"]:
            assert count <= truth[value] <= count + bound, (
                f"{value!r}: truth {truth[value]:,} outside the published "
                f"[{count:,}, {count + bound:,}]. Misra-Gries guarantees this "
                "interval, so a miss means the decrement mass is undercounted."
            )

    @pytest.mark.parametrize("duplicate_fraction", [0.0, 0.5])
    def test_the_duplicate_interval_contains_the_truth(
        self, duplicate_fraction: float
    ) -> None:
        """#329's interval, held to the same rule as every other bound.

        Both ends of the range matter, which is why this runs at 0% and 50%.
        A near-clean frame has duplicates below the sketch's resolution, where
        `duplicate_rows_est` is 0 *by design* rather than wrong -- and the
        interval is the only thing that distinguishes "none" from "none I can
        resolve". A frame that is half duplicates is well above resolution,
        where an interval that still contained the truth only by being
        enormous would be no use.
        """
        rng = np.random.default_rng(0)
        n = 50_000
        values = rng.integers(0, 1_000_000, n)
        if duplicate_fraction:
            repeated = int(n * duplicate_fraction)
            values[:repeated] = values[0]
        frame = pd.DataFrame({"a": values})

        exact = int(frame.duplicated().sum())
        dataset = pysuricata.summarize(frame)["dataset"]

        assert {"duplicate_rows_lo", "duplicate_rows_hi"} <= set(dataset)
        assert dataset["duplicate_rows_lo"] <= exact <= dataset["duplicate_rows_hi"], (
            f"{exact:,} real duplicates outside the published "
            f"[{dataset['duplicate_rows_lo']:,}, {dataset['duplicate_rows_hi']:,}]"
        )


class TestTheThresholdsAreCrossed:
    """Contract 3: test either side of every budget, not comfortably inside it.

    #328 lived entirely in the gap between 50 and 51 distinct values. A suite
    whose categorical fixtures have three levels cannot see it at any row count.
    """

    @pytest.mark.parametrize(
        "distinct", [TOP_K - 1, TOP_K, TOP_K + 1, TOP_K * 10, UNIQUES_SKETCH + 1]
    )
    def test_the_top_k_budget(self, distinct: int) -> None:
        """Exactness must follow whether the counters can hold the column."""
        rng = np.random.default_rng(7)
        values = rng.choice([f"v{i}" for i in range(distinct)], size=20_000)
        stats = pysuricata.summarize(pd.DataFrame({"c": values}))["columns"]["c"]

        truth = Counter(values)
        if stats["approx"] is False:
            for value, count in stats["top_items"]:
                assert count == truth[value]
        else:
            bound = stats["top_items_uncertainty"]
            for value, count in stats["top_items"]:
                assert count <= truth[value] <= count + bound

    @pytest.mark.parametrize("n", [NUMERIC_SAMPLE_SIZE - 1, NUMERIC_SAMPLE_SIZE + 1])
    def test_the_reservoir_budget(self, n: int) -> None:
        """Below it the count is exact; above it, an estimate within tolerance.

        Both halves matter. Scaling applied where it does not belong would make
        a small column's exact count approximate, which is this bug pointing
        the other way.
        """
        values = _lognormal(n)
        stats = _numeric_column(pysuricata.summarize(pd.DataFrame({"x": values})))
        exact = _exact_iqr_outliers(values)

        if n < NUMERIC_SAMPLE_SIZE:
            assert stats["outliers_iqr_est"] == exact
        else:
            assert abs(stats["outliers_iqr_est"] / exact - 1.0) < TOLERANCE

    @pytest.mark.parametrize("n", [CHUNK_SIZE - 1, CHUNK_SIZE + 1])
    def test_the_chunk_budget(self, n: int) -> None:
        """Chunked and unchunked must agree -- the invariant `CLAUDE.md` names
        as the one most likely to break, checked here on the estimates."""
        values = _lognormal(n)
        frame = pd.DataFrame({"x": values})
        whole = _numeric_column(pysuricata.summarize(frame))
        chunked = _numeric_column(
            pysuricata.summarize(
                [frame.iloc[i : i + 10_000] for i in range(0, n, 10_000)]
            )
        )
        assert whole["count"] == chunked["count"]


class TestAddingAnEstimateForcesADecision:
    """The forcing move, and the reason this module is more than three tests.

    Without it, the next estimator is published with no oracle and no bound, and
    nothing fails -- which is exactly how #327 shipped.
    """

    def test_every_estimate_declares_an_oracle_or_a_reason(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "amount": rng.lognormal(3, 1.2, 4_000),
                "region": rng.choice(["north", "south", "east"], 4_000).astype(object),
            }
        )
        payload = pysuricata.summarize(frame)

        candidates = set()
        for stats in payload["columns"].values():
            candidates.update(k for k in stats if k.endswith(("_est", "_approx")))
        candidates.update(k for k in payload["dataset"] if k.endswith("_est"))

        undeclared = sorted(candidates - set(_ORACLES) - set(_NO_ORACLE))
        assert not undeclared, (
            "published estimates with neither an oracle nor a stated reason: "
            + ", ".join(undeclared)
            + ". Add an entry to _ORACLES so it is checked against truth, or to "
            "_NO_ORACLE saying why it cannot be."
        )

    def test_every_excused_estimate_names_a_reason(self) -> None:
        assert all(reason.strip() for reason in _NO_ORACLE.values())

    def test_the_string_fixture_can_tell_the_three_counts_apart(self) -> None:
        """An oracle is only as good as the fixture it runs on.

        If raw, case-folded and trimmed distinct counts all coincided, the
        case and trim oracles would pass against an estimator that computed
        any one of the three -- including the wrong one. The fixture has to
        separate them before the assertions above mean anything. This is the
        `[1.0, 2, 3, 4, 5] * 40` lesson: a fixture that misses the branch
        reports "absent", and absent reads as passing.
        """
        values = _cased_words(5_000)
        raw = _exact_distinct(values)
        folded = _exact_distinct_folded(values)
        trimmed = _exact_distinct_trimmed(values)

        assert folded < raw, "no case variants in the fixture"
        assert trimmed < raw, "no whitespace variants in the fixture"
        assert folded != trimmed, "folding and trimming collapse the same values"

    def test_the_two_lists_do_not_overlap(self) -> None:
        """An entry in both would be checked and excused at once, and the excuse
        would win the next time the oracle was inconvenient."""
        assert not (set(_ORACLES) & set(_NO_ORACLE))
