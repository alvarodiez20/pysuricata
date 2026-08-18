"""Categorical card rendering functionality."""

import math
from collections.abc import Sequence

from .card_base import CardRenderer
from .card_config import DEFAULT_CAT_CONFIG, DEFAULT_CHART_DIMS
from .card_types import BarData, CategoricalStats, QualityFlags
from .flag_reference import even_split_pct
from .format_utils import ordinal_number
from .triage import annotate_flags

# Top-5 coverage below this means a bar chart of the top values would be a row
# of near-identical slivers -- ten bars of one row each on Titanic's `Name`.
_LOW_COVERAGE = 0.02

# Or the column is distinct enough that the same is true by construction. 0.5
# is the ceiling the report already uses for "high-cardinality categorical" in
# the summary, so the card and the summary agree about which columns those are.
# Titanic's `Cabin` is 147 distinct in 204 rows -- 0.72 -- and its top five
# cover 8.8%, which clears the coverage arm but is still a chart of five values
# out of a hundred and forty-seven.
_HIGH_CARDINALITY = 0.5

# A stronger claim than "high cardinality", and only this one licenses saying
# every value is different.
_NEAR_UNIQUE = 0.90


def describe_high_cardinality(stats: CategoricalStats) -> dict | None:
    """Whether to replace the chart with a sentence, and what it should say.

    Returns None for an ordinary column. For a high-cardinality one, returns
    the facts the sentence needs.

    Two things make this rule harder than a threshold on `unique_est`:

    **The inputs are approximate.** `unique_est` carries about 2.2% of KMV
    error, so a column sitting on the boundary can flip between runs of the
    same data -- and a card that changes shape on re-profiling is worse than
    either shape. Coverage is computed from Misra-Gries counts, which are
    *lower bounds*: the test can only under-state coverage, so it errs towards
    the sentence, which is the safe direction. The distinct-count arm is set at
    0.90 rather than near 1.0 so the sketch error cannot reach it.

    **`top_items` may be empty rather than full of singletons.** Misra-Gries is
    gated off entirely on high-cardinality columns (#62), so the branch has to
    handle *no top values at all*, which is the case that looks like a bug if
    it falls through to the chart.
    """
    count = int(getattr(stats, "count", 0) or 0)
    if count <= 0:
        return None

    items = list(getattr(stats, "top_items", None) or [])
    unique = int(getattr(stats, "unique_est", 0) or 0)
    distinct_ratio = unique / count if count else 0.0

    # No counters at all: the sketch was switched off because the column is
    # high-cardinality, so the absence *is* the signal.
    if not items:
        # No counters at all. That only means high cardinality if the distinct
        # count says so -- an all-missing column also has no top values.
        if distinct_ratio < _HIGH_CARDINALITY or unique <= 1:
            return None
        # Unknown, and *not* zero. Misra-Gries is gated off entirely on
        # high-cardinality columns (#62), so there are no counts to sum --
        # which is a different thing from having summed them and got nothing.
        # `Cabin` shipped "the five most common cover 0.0%", a fabricated
        # figure for a column that simply was not measured (#155, 5c.3).
        coverage = None
    else:
        coverage = sum(c for _, c in items[:5]) / count
        if coverage > _LOW_COVERAGE and distinct_ratio < _HIGH_CARDINALITY:
            return None

    return {
        "unique": unique,
        "count": count,
        #: `None` means *not measured*, never zero. See above.
        "coverage": coverage,
        "distinct_ratio": distinct_ratio,
        "identifier_like": distinct_ratio >= _NEAR_UNIQUE,
    }


def high_cardinality_sentence(facts: dict) -> str:
    """What to say in place of the chart.

    Two sentences, because two different things are true. A column where every
    value is distinct can say so outright. One that is merely high-cardinality
    -- `Cabin` is 147 values in 204 rows -- cannot: claiming every value is
    different there would be false, and the reason the chart is useless is that
    the top few cover almost nothing, which is worth stating as a number.
    """
    unique = facts["unique"]
    count = facts["count"]
    if facts["identifier_like"]:
        return (
            "Every value is different. A top-values chart would be "
            "bars of one row each, so there is nothing to plot."
        )
    if facts["coverage"] is None:
        # The counters were never kept, so the coverage is unmeasured. Saying
        # so is shorter than the alternative and does not invent a number.
        return (
            f"{unique:,} distinct values in {count:,} rows. The top-values "
            "counters are not kept for a column this varied, so there is no "
            "ranking to plot."
        )
    return (
        f"{unique:,} distinct values in {count:,} rows, and the five most "
        f"common cover {facts['coverage'] * 100:.1f}% of them. A top-values "
        "chart would be a row of slivers, so there is nothing worth plotting."
    )


# `Entropy`, `Rare levels` and `Top 5 coverage` all describe how a distribution
# spreads across its levels, and one card face renders all three for every
# categorical column. Categorical is the most common column type -- eight of
# Titanic's twelve -- and the three were written for exactly one of the four
# kinds it covers: a handful of levels with meaningfully different shares.
# `Sex` gets entropy 0.936, rare levels 0 and top-5 coverage 100%, three
# statistics describing the spread of a distribution that has two members and
# no spread (#295, 5f.1).
#
# The rule is **per statistic**, not per column kind. That is the whole reason
# suppression was taken over routing to three card faces (5f.4, held): there is
# no level boundary here to defend, and no argument about where a 12-level
# column belongs. Each of the three drops out exactly where its own arithmetic
# stops carrying information, so a column can lose one and keep the others --
# `Embarked` loses top-5 coverage and keeps entropy and rare levels.

#: Keys returned by `suppressed_statistics`, matching the rows they silence.
ENTROPY = "entropy"
RARE_LEVELS = "rare_levels"
TOP5_COVERAGE = "top5_coverage"


def _levels_are_complete(items: Sequence[tuple[str, int]], total: int) -> bool:
    """Whether `items` is every level of the column, with exact counts.

    Misra-Gries counters only ever *decrease* below the true count, and they
    decrease only when an eviction round runs -- which happens only once more
    distinct values arrive than the sketch has counters for. So the counters
    summing to the number of non-missing rows is not evidence that the sketch
    is complete, it is proof: no eviction can have happened, every level is
    present and every count is exact.

    That matters because the alternative is `unique_est`, and reading a level
    count off a KMV estimate would make suppression flip between runs of the
    same data for a column sitting on the boundary. This test cannot flip. When
    it cannot prove completeness it returns False and the statistic renders,
    which is the safe direction: an unnecessary statistic is a smaller error
    than a suppressed one the reader needed.
    """
    return bool(items) and total > 0 and sum(c for _, c in items) == total


def suppressed_statistics(cat_stats: dict) -> frozenset[str]:
    """Which of the three spread statistics say nothing about this column.

    Not one rule -- three, because the three fail for three different reasons
    and a single threshold would be a coincidence rather than an argument:

    **Top 5 coverage** is suppressed at five levels or fewer. There, the top
    five *are* all of them, so the figure is 100% by construction. It is not a
    measurement of the column, it is a restatement of `Unique`, and it reads as
    the former.

    **Rare levels** is suppressed at two levels. `Rare` names a tail, and it
    exists to summarise the levels the chart is not showing; with two levels
    the chart shows both, with their exact shares, immediately above. This one
    is suppressed even when it would be non-zero -- a 99.9/0.1 split does have
    a level under the 1% threshold, and the bar already says so.

    **Entropy** is suppressed at two levels, where it is a monotone restatement
    of the mode share already on the card and is read against the wrong scale
    (its maximum is 1 bit, not the `log2(levels)` a reader assumes). It is also
    suppressed when every tracked level occurs exactly once, where it collapses
    to `log2(n)` -- a function of the level count, computed over values that
    never repeat, presented as a measure of how they repeat.

    An untracked column is not handled here. Its three cells already render as
    `_unknown_cell`, and that is a different statement: *unknown* is a sketch
    that could not answer, *absent* is a question that does not apply.
    """
    if not cat_stats.get("tracked", True):
        return frozenset()

    suppressed = set()

    if cat_stats["all_singletons"]:
        suppressed.add(ENTROPY)

    if cat_stats["levels_complete"]:
        levels = cat_stats["n_levels"]
        if levels <= 5:
            suppressed.add(TOP5_COVERAGE)
        if levels <= 2:
            suppressed.update((ENTROPY, RARE_LEVELS))

    return frozenset(suppressed)


class CategoricalCardRenderer(CardRenderer):
    """Renders categorical data cards."""

    def __init__(self):
        super().__init__()
        self.cat_config = DEFAULT_CAT_CONFIG
        self.chart_dims = DEFAULT_CHART_DIMS

    def render_card(self, stats: CategoricalStats) -> str:
        """Render a complete categorical card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and quality flags
        total = int(getattr(stats, "count", 0) + getattr(stats, "missing", 0))
        miss_pct = (stats.missing / max(1, total)) * 100.0
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 0 else "")

        quality_flags = self.quality_assessor.assess_categorical_quality(stats)
        quality_flags_html = self._build_quality_flags_html(
            quality_flags, miss_pct, stats
        )

        # Compute derived stats
        cat_stats = self._compute_categorical_stats(stats)

        # Build components
        approx_badge = self._build_approx_badge(stats.approx)
        stat_row = self._build_stat_row(
            self._left_stats(stats, miss_cls, miss_pct, cat_stats)
            + self._right_stats(stats, cat_stats)
        )

        # Chart and details
        items = stats.top_items or []
        topn_list, default_topn = self._get_topn_candidates(items)

        # A high-cardinality column gets a sentence instead of a chart, and no
        # chart box: an empty box the height of a chart reads as a failed
        # render rather than as "there is nothing to draw here".
        high_card = describe_high_cardinality(stats)
        if high_card is not None:
            chart_html = self._build_high_cardinality_note(stats, high_card)
            # There is no chart, so there is nothing for a Top-N control to
            # control. `Name` and `Ticket` rendered five buttons above a
            # sentence (#155, 5c.3).
            topn_list, default_topn = [], default_topn
        else:
            # One variant per option the control offers -- or, when it offers
            # none, a single variant showing every level. An empty list here
            # would render no chart at all.
            chart_levels = topn_list or [max(1, len(items))]
            chart_html = self._build_categorical_variants(
                col_id,
                items,
                total,
                chart_levels,
                default_topn,
                n_levels=self._distinct_levels(stats, items),
            ) + self._build_coverage_note(stats, items)
        # A high-cardinality column has no ranking to show, so `Common values`
        # is replaced rather than kept: ten bars of one row each imply a
        # frequency that does not exist. What it can show is the *shape* of the
        # values, which is what a reader of an identifier column wants.
        common_table = (
            self._build_shape_pane(stats, high_card)
            if high_card is not None
            else self._build_common_values_table(stats)
        )
        common_label = "Shape" if high_card is not None else "Common values"
        norm_tab_btn, norm_tab_pane = self._build_normalization_section(items, stats)
        missing_table = self._build_missing_values_table(stats, miss_pct)
        length_pane = self._build_length_pane(stats)

        details_html = self._build_details_section(
            col_id,
            common_table,
            norm_tab_btn,
            norm_tab_pane,
            missing_table,
            length_pane,
            common_label,
            # The same rule as every other card kind, now that this one can
            # answer it (#193): the pane only knows something the card face
            # does not when there is more than one chunk -- *where in the read*
            # the gaps fall. With one chunk it restates the header's
            # percentage.
            #
            # This was ungated until the accumulator tracked chunks, and the
            # comment that stood here warned why gating it early would be
            # worse than leaving it: `getattr(stats, "chunk_metadata", None)`
            # returns `None` rather than raising, so the gate would have looked
            # applied while hiding the pane permanently.
            has_missing=(
                int(getattr(stats, "missing", 0) or 0) > 0
                and len(getattr(stats, "chunk_metadata", None) or []) > 1
            ),
        )
        controls_html = self._build_controls_section(col_id, topn_list, default_topn)

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            approx_badge,
            quality_flags_html,
            stat_row,
            chart_html,
            details_html,
            controls_html,
        )

    def _compute_categorical_stats(self, stats: CategoricalStats) -> dict:
        """Compute derived stats for categorical data."""
        total = int(getattr(stats, "count", 0))
        items = list(getattr(stats, "top_items", []) or [])
        mode_label, mode_n = items[0] if items else ("—", 0)
        safe_mode_label = self.safe_html_escape(str(mode_label))
        mode_pct = (mode_n / max(1, total)) * 100.0 if total else 0.0

        # Every figure below is computed from the top-k sketch, and the sketch
        # can legitimately come back **empty**. Misra-Gries only guarantees a
        # value survives if it appears more than n/(k+1) times; `Cabin` has 204
        # values over 147 distinct levels and its most frequent appears 4
        # times, against a threshold of exactly 4. Nothing qualifies, so the
        # sketch is empty -- and it is *right* to be empty.
        #
        # What was wrong is what the card did with that. `entropy` became
        # `float("nan")` and rendered as the literal `NaN`, while `Rare levels`
        # and `Top 5 coverage` fell through to their zero initialisers and
        # printed `0 (0.0%)` and `0.0%` -- three statistics stating a fact
        # about the column, when the truth is that this column has no heavy
        # hitters to compute them from. Unknown is not zero.
        tracked = bool(items) and total > 0

        if tracked:
            probs = [c / total for _, c in items]
            entropy = float(-sum(p * math.log2(max(p, 1e-12)) for p in probs))
        else:
            entropy = None

        # Rare levels analysis
        rare_count = 0
        rare_cov = 0.0
        if total > 0 and items:
            for _, c in items:
                pct = c / total * 100.0
                if pct < 1.0:
                    rare_count += 1
                    rare_cov += pct

        rare_cls = "crit" if rare_cov > 60 else ("warn" if rare_cov >= 30 else "")

        # Top-5 coverage
        top5_cov = 0.0
        if total > 0 and items:
            top5_cov = sum(c for _, c in items[:5]) / total * 100.0

        top5_cls = "good" if top5_cov >= 80 else ("warn" if top5_cov <= 40 else "")

        # Empty strings
        empty_zero = int(getattr(stats, "empty_zero", 0))
        empty_cls = "warn" if empty_zero > 0 else ""

        return {
            "tracked": tracked,
            # What `suppressed_statistics` reads. Computed here rather than
            # there so the whole stat row is derived from one pass over the
            # sketch, and so a test can assert the facts without rendering.
            "levels_complete": _levels_are_complete(items, total),
            "n_levels": len(items),
            "all_singletons": bool(items) and all(c == 1 for _, c in items),
            "mode_label": mode_label,
            "safe_mode_label": safe_mode_label,
            "mode_n": int(mode_n),
            "mode_pct": float(mode_pct),
            "entropy": entropy,
            "rare_count": int(rare_count),
            "rare_cov": float(rare_cov),
            "rare_cls": rare_cls,
            "top5_cov": float(top5_cov),
            "top5_cls": top5_cls,
            "empty_zero": empty_zero,
            "empty_cls": empty_cls,
            "unique_est": int(getattr(stats, "unique_est", 0)),
        }

    def _build_quality_flags_html(
        self, flags: QualityFlags, miss_pct: float, stats: CategoricalStats
    ) -> str:
        """The chips, with the number each one already knows on its face.

        `_quality_flags_markup` builds them; this puts the value on the
        chip and the threshold in a title. Splitting it this way means the
        forty-two places that emit a chip carry on emitting the same
        markup, and the annotation lives in one place rather than being
        repeated at every one of them.
        """
        return annotate_flags(self._quality_flags_markup(flags, miss_pct, stats))

    def _quality_flags_markup(
        self,
        flags: QualityFlags,
        miss_pct: float,
        stats: CategoricalStats | None = None,
    ) -> str:
        """Build quality flags HTML for categorical data.

        A chip carries `data-value` only where the number reads as a prefix to
        the label -- `annotate_flags` renders the face as `{value} {label}`, so
        the test is whether the result is a sentence.

        `19.9% missing` and `12 empty strings` pass. `72% high cardinality` and
        `31.2% many rare levels` do not: those need a phrase rather than a
        prefix, and inventing one would be worse than the word on its own. They
        stay bare deliberately, not by omission.
        """
        flag_items = []

        if flags.high_cardinality:
            flag_items.append('<li class="flag warn">High cardinality</li>')

        if flags.dominant_category:
            share = self._dominant_share(stats)
            if share is None:
                flag_items.append('<li class="flag warn">Dominant category</li>')
            else:
                flag_items.append(
                    f'<li class="flag warn" data-threshold="one level dominates" '
                    f'data-value="{share:.1f}%">Dominant category</li>'
                )

        if flags.many_rare_levels:
            flag_items.append('<li class="flag warn">Many rare levels</li>')

        if flags.case_variants:
            flag_items.append('<li class="flag">Case variants</li>')

        if flags.trim_variants:
            flag_items.append('<li class="flag">Trim variants</li>')

        if flags.empty_strings:
            # "Empty or zero", not "Empty strings". The accumulator counts
            # `value == "" or value == "0"`, and putting the number on the chip
            # is what made that visible: titanic's `SibSp` and `Parch` profile
            # as categorical and rendered `608 empty strings` and `678 empty
            # strings`, when what they have is 608 and 678 *zeros* and not one
            # empty string between them. The vague label had been hiding a
            # false one.
            empty = int(getattr(stats, "empty_zero", 0) or 0) if stats else 0
            if empty > 0:
                flag_items.append(
                    f'<li class="flag" data-threshold=\'empty string or "0"\' '
                    f'data-value="{empty:,}">Empty or zero</li>'
                )
            else:
                flag_items.append('<li class="flag">Empty or zero</li>')

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            threshold = ">20%" if miss_pct > 20 else "≤20%"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{miss_pct:.1f}%">Missing</li>'
            )

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    @staticmethod
    def _dominant_share(stats: CategoricalStats | None) -> float | None:
        """The mode's share of non-missing rows, or None if it cannot be known.

        None rather than 0.0 when the top-k sketch is empty -- the same
        distinction `_compute_categorical_stats` makes. A chip reading
        `0.0% dominant category` would be a contradiction.
        """
        if stats is None:
            return None
        items = list(getattr(stats, "top_items", None) or [])
        count = int(getattr(stats, "count", 0) or 0)
        if not items or count <= 0:
            return None
        return items[0][1] / count * 100.0

    def _left_stats(
        self, stats: CategoricalStats, miss_cls: str, miss_pct: float, cat_stats: dict
    ) -> list[tuple[str, str, str | None]]:
        """The counting half of the stat row: how many, and how many of each."""
        data = [
            ("Count", f"{int(getattr(stats, 'count', 0)):,}", "num"),
            (
                f"Unique{'' if getattr(stats, 'unique_est_exact', False) else ' (≈)'}",
                f"{int(getattr(stats, 'unique_est', 0)):,}",
                "num",
            ),
            (
                "Missing",
                f"{int(getattr(stats, 'missing', 0)):,} ({miss_pct:.1f}%)",
                f"num {miss_cls}",
            ),
            # Same reasoning as Entropy below: with an empty top-k sketch there
            # is no mode to report, and `0.0%` claims there is one and that it
            # covers nothing.
            ("Mode", f"<code>{cat_stats['safe_mode_label']}</code>", None),
            (
                "Mode %",
                self._unknown_cell(
                    "no value repeats often enough to be tracked in the top-k sketch"
                )
                if not cat_stats.get("tracked", True)
                else f"{cat_stats['mode_pct']:.1f}%",
                "num",
            ),
            (
                "Empty or zero",
                f"{int(cat_stats['empty_zero']):,}",
                f"num {cat_stats['empty_cls']}",
            ),
        ]

        return data

    def _unknown_cell(self, reason: str) -> str:
        """An em dash that says why, rather than a number that is not true.

        The `title` matters more than the dash. A reader who sees `—` where
        they expected a percentage will want to know whether the report failed
        or the column has nothing to report, and those are opposite
        conclusions about their data.
        """
        return f'<span title="{self.safe_html_escape(reason)}">—</span>'

    def _length_display(self, value) -> str:
        """A length, or an em dash that says why there is not one.

        The em dash means *absent*. It used to appear for values that were
        merely being read from the wrong object, which is a different thing and
        is what hid #155 -- `Embarked` reported a mean label length of `NaN`
        for a column whose three labels are all one character long.

        That is fixed, and the dash now survives in exactly one case: a column
        with no non-missing values has no labels, so it has no label length.
        It reads through `_unknown_cell` like every other unknown on this card,
        because a bare dash leaves a reader unable to tell a column with
        nothing to measure from a report that failed to measure it -- opposite
        conclusions about their data (#155, 5c.2).
        """
        absent = self._unknown_cell(
            "this column has no non-missing values, so it has no labels to measure"
        )
        if value is None:
            return absent
        try:
            number = float(value)
        except (TypeError, ValueError):
            return absent
        if number != number:  # NaN
            return absent
        return self.format_number(number)

    def _right_stats(
        self, stats: CategoricalStats, cat_stats: dict
    ) -> list[tuple[str, str, str | None]]:
        """The shape half: how the levels are distributed.

        `avg_len` and `len_p90` come from `stats`, not from `cat_stats`.
        `_compute_categorical_stats` has never built those keys, so the `.get()`
        defaults were what rendered -- `NaN` and an em dash -- for **every**
        categorical column in every report. The handoff reported it as an
        `Embarked` quirk about one-character labels; `Name`, whose labels
        average 26.97 characters, printed `NaN` just the same.
        """
        # An em dash where the top-k sketch found nothing to summarise. See
        # `_compute_categorical_stats`: an empty sketch is a real answer about
        # a column with no repeated values, and `0.0%` is a different, false
        # answer that looks equally confident.
        unknown = not cat_stats.get("tracked", True)
        no_heavy_hitters = (
            "no value repeats often enough to be tracked in the top-k sketch"
        )

        # Three of the six describe a spread, and not every column has one.
        # Where a statistic cannot carry information the row is **not emitted**
        # -- the grid closes up rather than printing a dash, because a dash
        # says *this could not be measured* and the truth here is *this does
        # not apply* (#295, 5f.1). `Sex` renders nine slots, not twelve.
        silenced = suppressed_statistics(cat_stats)

        data = []

        if ENTROPY not in silenced:
            data.append(
                (
                    "Entropy",
                    self._unknown_cell(no_heavy_hitters)
                    if unknown
                    else self.format_number(cat_stats["entropy"]),
                    "num",
                )
            )

        if RARE_LEVELS not in silenced:
            data.append(
                (
                    "Rare levels",
                    self._unknown_cell(no_heavy_hitters)
                    if unknown
                    else (
                        f"{int(cat_stats['rare_count']):,} "
                        f"({cat_stats['rare_cov']:.1f}%)"
                    ),
                    f"num {cat_stats['rare_cls']}",
                )
            )

        if TOP5_COVERAGE not in silenced:
            data.append(
                (
                    "Top 5 coverage",
                    self._unknown_cell(no_heavy_hitters)
                    if unknown
                    else f"{cat_stats['top5_cov']:.1f}%",
                    f"num {cat_stats['top5_cls']}",
                )
            )

        data += [
            (
                "Label length (avg)",
                self._length_display(getattr(stats, "avg_len", None)),
                "num",
            ),
            (
                "Length p90",
                self._length_display(getattr(stats, "len_p90", None)),
                "num",
            ),
            (
                # `stats`, not `cat_stats`. The same defect as `avg_len` and
                # `len_p90` above and found the same way -- looking at the
                # card. `_compute_categorical_stats` has never built a
                # `mem_bytes` key, so the `.get()` default was what rendered,
                # and **every** categorical column in every report has claimed
                # to have processed `0.0 B`. `Sex` really is 11.4 KB.
                #
                # `_left_stats` computed the right number from the right
                # object and threw the result away, which is the fossil of the
                # move that broke this: the cell used to sit in the left half.
                "Processed bytes (≈)",
                self.format_bytes(int(getattr(stats, "mem_bytes", 0))),
                "num",
            ),
        ]

        return data

    def _get_topn_candidates(
        self, items: Sequence[tuple[str, int]]
    ) -> tuple[list[int], int]:
        """How many levels the chart may show, and how many it shows by default.

        Returns an empty list when there is no choice to offer (#155, 5c.3).

        `max_n` used to be folded in beside 5/10/15, so a column with two
        levels got `{5, 10, 15, 2}` filtered to `{2}` -- and then rendered
        three buttons, every one of them reading `2`. `Sex` shipped that: a
        chooser offering the same choice three times. `Cabin` rendered two
        buttons both reading `1`.

        A control with one option is not a control. Below the smallest step
        the chart already shows every level, so there is nothing to choose.

        **The chart is not the control.** Returning an empty list here must
        remove the buttons and leave the chart alone -- the first version of
        this fed the same list to both, so `Sex` and `Embarked` lost their bar
        chart entirely. A pre-existing test caught it.
        """
        levels = len(items)
        steps = [n for n in (5, 10, 15) if n < levels]
        if not steps:
            return [], levels
        # The last step shows everything, so it is labelled with the true
        # level count rather than a round number that overstates it.
        topn_list = sorted({*steps, levels})
        default_topn = 10 if 10 in topn_list else topn_list[0]
        return topn_list, default_topn

    def _build_high_cardinality_note(self, stats: CategoricalStats, facts: dict) -> str:
        """The sentence that replaces the chart, plus what is worth knowing.

        `Name`, `Ticket` and `Cabin` used to render ten bars of one row each --
        a chart that says nothing, drawn at the same size as one that does.
        """
        items = list(getattr(stats, "top_items", None) or [])
        extras = []
        if items:
            lengths = [(len(str(v)), str(v)) for v, _ in items]
            shortest = min(lengths)[1]
            longest = max(lengths)[1]
            extras.append(
                f'<li><span class="k">Shortest seen</span>'
                f'<span class="v">{self.safe_html_escape(shortest)}</span></li>'
            )
            extras.append(
                f'<li><span class="k">Longest seen</span>'
                f'<span class="v">{self.safe_html_escape(longest)}</span></li>'
            )
        flag = (
            '<span class="flag warn">identifier-like</span>'
            if facts["identifier_like"]
            else ""
        )
        extra_html = (
            f'<ul class="nochart__facts">{"".join(extras)}</ul>' if extras else ""
        )
        return (
            f'<div class="nochart">{flag}'
            f'<p class="nochart__why">{high_cardinality_sentence(facts)}</p>'
            f"{extra_html}</div>"
        )

    def _build_shape_pane(self, stats: CategoricalStats, facts: dict) -> str:
        """What a high-cardinality column can say instead of a ranking.

        Phase 5c.3 (#155). Phase 5.4 replaced the meaningless top-values
        *chart* on the card; the details pane still opened on `Common values`
        -- the same ten bars of one row each, 0.1% apiece. On the card the fix
        was to say there is nothing to plot. Here there *is* something to plot,
        just not that.

        Everything below is already computed. No sample of raw values is shown:
        putting arbitrary cell contents into a shared HTML file is a privacy
        question, and the design says to decide it deliberately rather than
        inherit it from the sample table. The answer here is no -- the two
        extremes the card already shows are enough to recognise a format, and
        ten more rows would be ten more values to leak.
        """
        name = self.safe_html_escape(stats.name)
        count = int(getattr(stats, "count", 0) or 0)
        unique = int(getattr(stats, "unique_est", 0) or 0)
        empty = int(getattr(stats, "empty_zero", 0) or 0)
        bins = [
            (int(length), int(number))
            for length, number in (getattr(stats, "len_hist", None) or [])
            if number > 0
        ]

        rows: list[tuple[str, str]] = [
            ("Distinct", f"{unique:,} of {count:,} rows"),
            (
                "Repeats",
                "none — every value is different"
                if facts["identifier_like"]
                else (
                    "not measured — the counters are not kept for a column this varied"
                    if facts["coverage"] is None
                    else f"the five most common cover {facts['coverage'] * 100:.1f}%"
                ),
            ),
        ]
        if bins:
            rows.append(
                (
                    "Length",
                    f"{bins[0][0]} to {bins[-1][0]} characters"
                    if bins[0][0] != bins[-1][0]
                    else f"{bins[0][0]} characters, all of them",
                )
            )
        if empty:
            rows.append(("Empty or zero", f"{empty:,}"))

        body = "".join(
            f'<div class="shape__row">'
            f'<span class="shape__key">{key}</span>'
            f'<span class="shape__val">{value}</span>'
            f"</div>"
            for key, value in rows
        )
        return (
            '<div class="fence-pane">'
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · shape</span>'
            '<span class="fence-head__rule"></span>'
            "</div>"
            f'<p class="fence-lede">{high_cardinality_sentence(facts)}</p>'
            f'<div class="shape__rows">{body}</div>'
            "</div>"
        )

    @staticmethod
    def _trim(pct: float) -> str:
        """`100`, not `100.0`; `5.9`, not `6`.

        A whole percentage carries no information in its decimal, and a
        fractional one loses the difference between 5.9% and 6% -- which on a
        147-level column is the difference between the bars covering a
        twentieth of the data and appearing to cover more.
        """
        return f"{pct:.1f}".removesuffix(".0")

    @staticmethod
    def _distinct_levels(stats: CategoricalStats, items: Sequence) -> int:
        """Levels in the column, not levels in the chart.

        The sketch estimate can come in below the number of levels actually
        held, so it is floored at what the chart already shows -- a column
        cannot have fewer levels than the bars drawn for it.
        """
        shown = len(items)
        return max(int(getattr(stats, "unique_est", shown) or shown), shown)

    def _build_coverage_note(self, stats: CategoricalStats, items: list) -> str:
        """How much of the column the bars actually account for.

        A top-N chart is a sample of the levels, and without this line there is
        nothing to say whether the bars are the whole column or a tenth of it.
        """
        count = int(getattr(stats, "count", 0) or 0)
        if not items or count <= 0:
            return ""
        shown = len(items)
        total_levels = self._distinct_levels(stats, items)
        covered = sum(c for _, c in items) / count * 100.0
        levels = "level" if total_levels == 1 else "levels"
        # Of the **non-missing** rows, and it says so. `Cabin` is 77.1% empty,
        # so the same bars are 5.9% of its 204 non-missing rows and 1.3% of the
        # frame -- a share of the whole would say something quite different
        # from what it appears to say (#296).
        # The rule's value, said once per column rather than in a tooltip on
        # every chart variant. Only when there is a rule to explain.
        mark = ""
        if total_levels >= 2:
            mark = (
                f" · rule at {self._trim(even_split_pct(total_levels))}%, an even split"
            )
        return (
            f'<p class="coverage-note">{shown:,} of {total_levels:,} {levels} shown '
            f"· covers {self._trim(covered)}% of the {count:,} non-missing rows"
            f"{mark}</p>"
        )

    def _build_categorical_variants(
        self,
        col_id: str,
        items: Sequence[tuple[str, int]],
        total: int,
        topn_list: list[int],
        default_topn: int,
        n_levels: int | None = None,
    ) -> str:
        """Build categorical chart variants."""
        parts = []
        for n in topn_list:
            if len(items) > n:
                keep = max(1, n - 1)
                head = list(items[:keep])
                other = int(sum(c for _, c in items[keep:]))
                data = head + [("Other", other)]
            else:
                data = list(items[:n])

            # `n_levels` and not `len(data)`: every variant of the same column
            # is read against the same mark, so switching Top-5 to Top-10 moves
            # the bars and not the rule. A mark that moved with the control
            # would be measuring the chart rather than the column.
            svg = self._build_categorical_bar_svg(
                data, total=max(1, int(total)), n_levels=n_levels
            )
            active_class = " active" if n == default_topn else ""
            parts.append(
                f'<div class="cat variant{active_class}" id="{col_id}-cat-top-{n}" data-topn="{n}">{svg}</div>'
            )

        return f"""
        <div class="topn-chart">
            <div class="hist-variants">{"".join(parts)}</div>
        </div>
        """

    def _build_categorical_bar_svg(
        self,
        items: list[tuple[str, int]],
        total: int,
        *,
        scale: str = "count",
        n_levels: int | None = None,
    ) -> str:
        """Build categorical bar chart SVG."""
        if total <= 0 or not items:
            return self.create_empty_svg(
                "cat-svg", self.chart_dims.width, self.chart_dims.height
            )

        bar_data = self._prepare_bar_data(items, total, scale, n_levels)
        return self._render_bar_svg(bar_data)

    def _prepare_bar_data(
        self,
        items: list[tuple[str, int]],
        total: int,
        scale: str,
        n_levels: int | None = None,
    ) -> BarData:
        """Prepare bar chart data, and the even-split mark to read it against.

        `Embarked`'s S at 72.4% against a 33.3% rule says *dominated by one
        port* without asking the reader to divide anything (phase 5f.2, #296).
        It is the same device as the flat-calendar rule on the datetime card
        and the fence on the numeric one, which is the point: one reading
        convention across the report.

        Nothing new is computed -- `even_split_pct` is arithmetic on the level
        count, and the level count is already on the card.

        Args:
            n_levels: Distinct levels in the *column*, which is not
                ``len(items)`` when the chart is a top-N with an ``Other``
                bucket. The rule answers "what would each level hold if this
                column were even", so it is the column's count that matters,
                not the chart's.
        """
        labels = [self.safe_html_escape(str(k)) for k, _ in items]
        counts = [int(c) for _, c in items]
        pcts = [(c / total * 100.0) for c in counts]

        if scale == "pct":
            values = pcts
        else:
            values = counts

        # One level splits evenly into itself, so the rule would sit exactly on
        # the only bar and say nothing. Below two, draw none.
        even_value: float | None = None
        even_share: float | None = None
        levels = int(n_levels or len(items))
        if levels >= 2:
            even_share = even_split_pct(levels)
            even_value = even_share if scale == "pct" else total * even_share / 100.0

        return BarData(
            labels=labels,
            counts=counts,
            percentages=pcts,
            values=values,
            even_split_value=even_value,
            even_split_share=even_share,
        )

    def _render_bar_svg(self, bar_data: BarData) -> str:
        """Render bar chart SVG.

        The height follows the number of bars rather than the other way round.
        Dividing a fixed height among the bars made thickness a function of
        cardinality -- two levels drew two 218px slabs where five drew 87px --
        so the same chart read differently for every column.
        """
        n = len(bar_data.labels)
        width = self.cat_config.chart_width
        margin_top = margin_bottom = self.cat_config.chart_margin_y
        margin_right = 12
        bar_h = self.cat_config.bar_height
        bar_gap = self.cat_config.bar_gap

        # Calculate label width
        max_label_len = max((len(label) for label in bar_data.labels), default=0)
        char_w = self.cat_config.char_width
        gutter = max(
            self.cat_config.min_gutter,
            min(self.cat_config.max_gutter, char_w * min(max_label_len, 28) + 16),
        )
        margin_left = max(120, gutter)

        height = margin_top + margin_bottom + n * bar_h + max(0, n - 1) * bar_gap
        iw = width - margin_left - margin_right

        if n <= 0 or iw <= 0:
            return self.create_empty_svg("cat-svg", width, self.chart_dims.height)

        vmax = max(bar_data.values) or 1.0

        def sx(v: float) -> float:
            return margin_left + (v / vmax) * iw

        parts = [
            f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">'
        ]

        # The even-split rule, drawn **before** the bars. Rule 2 in the token
        # file: a mark crossing a data fill must protrude onto the paper or
        # paint underneath it. Source order is the "underneath" half, so a bar
        # reaching past the mark occludes it rather than crossing it; the 3px
        # of protrusion above and below each bar is the other half, and is what
        # keeps it findable at 390px.
        #
        # Off the right edge when the column is even enough that the mark would
        # land past the longest bar -- there is nothing to compare against
        # there, and a rule outside the plot is a rendering artefact.
        even_value = bar_data.even_split_value
        even_share = bar_data.even_split_share
        if even_value is not None and even_share is not None and 0 < even_value <= vmax:
            ex = sx(float(even_value))
            # Nothing but geometry on this element. It carried a `<title>` and
            # a `data-even-pct`, and both went the way 4b.2 sent the chip
            # tooltips: a measure repeated on every mark cost 5,548 bytes to
            # say fourteen distinct things, and a tooltip is invisible on a
            # phone and absent from paper. The rule's value is stated once per
            # column in the coverage note, where it can be read; nothing reads
            # the attribute (`report_fingerprint.py` keys on `data-pct`, and
            # neither the stylesheet nor `functionality.js` mentions it).
            #
            # One decimal on x, none on y. The viewBox is in pixels here rather
            # than 0..100, so the second decimal was a hundredth of a pixel.
            parts.append(
                f'<line class="cat-even" x1="{ex:.1f}" y1="{margin_top - 3:.0f}" '
                f'x2="{ex:.1f}" y2="{height - margin_bottom + 3:.0f}"/>'
            )

        for i, (label, c, p, val) in enumerate(
            zip(
                bar_data.labels,
                bar_data.counts,
                bar_data.percentages,
                bar_data.values,
                strict=False,
            )
        ):
            y = margin_top + i * (bar_h + bar_gap)
            x0 = margin_left
            x1 = sx(float(val))
            w = max(1.0, x1 - x0)
            short = (
                (label[: self.cat_config.max_label_length] + "…")
                if len(label) > self.cat_config.max_label_length
                else label
            )

            # The value sits inside the bar when the bar is wide enough to hold
            # it, and just past the end when it is not. Those are two different
            # backgrounds -- a saturated fill and the paper -- so they cannot
            # share one colour. They did: `--muted` grey on the bar measured
            # **1.20:1**, against the 4.5:1 that AA asks of text this size, so
            # the count was effectively invisible on every bar wide enough to
            # contain it. The class says which background the text is on and
            # the stylesheet colours it accordingly.
            inside = w >= 56
            placement = "is-inside" if inside else "is-outside"

            parts.append(
                f'<g class="bar-row">'
                f'<rect class="bar" x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="2" ry="2">'
                f"<title>{label}\n{c:,} rows ({p:.1f}%)</title>"
                f"</rect>"
                f'<text class="bar-label" x="{margin_left - 6}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="end">{short}</text>'
                f'<text class="bar-value {placement}" x="{(x1 - 6 if inside else x1 + 4):.2f}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="{("end" if inside else "start")}">{c:,} ({p:.1f}%)</text>'
                f"</g>"
            )

        parts.append("</svg>")
        return "".join(parts)

    def _build_top_values_table(
        self, items: Sequence[tuple[str, int]], count: int, max_rows: int = 15
    ) -> str:
        """Build top values table."""
        rows = []
        total_nonnull = max(1, int(count))
        acc = 0

        for val, c in list(items)[: max_rows - 1]:
            acc += int(c)
            rows.append(
                f"<tr><td><code>{self.safe_html_escape(str(val))}</code></td>"
                f"<td class='num'>{int(c):,}</td>"
                f"<td class='num'>{(int(c) / total_nonnull * 100.0):.1f}%</td></tr>"
            )

        other_n = max(0, total_nonnull - acc)
        if len(items) > (max_rows - 1) or other_n > 0:
            rows.append(
                f"<tr><td><code>Other</code></td>"
                f"<td class='num'>{other_n:,}</td>"
                f"<td class='num'>{(other_n / total_nonnull * 100.0):.1f}%</td></tr>"
            )

        body = "".join(rows) if rows else "<tr><td colspan=3>—</td></tr>"
        return (
            '<table class="kv"><thead><tr><th>Value</th><th>Count</th><th>%</th></tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    def _build_normalization_section(
        self, items: Sequence[tuple[str, int]], stats: CategoricalStats
    ) -> tuple[str, str]:
        """Whether normalising would *merge* levels, not what it would print.

        Phase 5c.1 (#155). The pane printed original / `lower()` / `strip()`
        per level, so for `Embarked` it said `S -> s -> S`. That is a
        transformation nobody asked about. The question the pane exists to
        answer is whether normalising changes the number of categories -- the
        difference between three levels and two -- and it costs one
        `len(set(...))` over levels already held.

        **The verdict is hedged, deliberately.** Only the top-k levels are
        tracked, so "nothing merges" is a claim about those and not about the
        column. Saying it unqualified would be a stronger statement than the
        sketch can support.

        When nothing merges the pane returns empty and the tab does not
        render, which is 5b.4's rule: a tab has to earn itself, and "I checked
        and found nothing" is a sentence, not a pane.
        """
        try:
            levels = [str(value) for value, _ in list(items or [])]
            if not levels:
                return "", ""

            groups = self._collisions(levels)
            if not groups:
                return "", ""

            rows = "".join(
                f'<div class="norm__row">'
                f'<span class="norm__rule">{self.safe_html_escape(rule)}</span>'
                f'<span class="norm__merged">'
                + " · ".join(
                    f"<code>{self.safe_html_escape(member)}</code>"
                    for member in members
                )
                + "</span>"
                f'<span class="norm__into">→ <code>'
                f"{self.safe_html_escape(into)}</code></span>"
                f"</div>"
                for rule, members, into in groups
            )

            merged = sum(len(members) - 1 for _, members, _ in groups)
            plural = (
                f"{len(groups)} groups merge"
                if len(groups) != 1
                else "one group merges"
            )
            verdict = (
                f"{len(levels)} tracked levels become {len(levels) - merged} "
                f"under normalisation: {plural}."
            )

            pane = (
                '<div class="fence-pane">'
                '<div class="fence-head">'
                '<span class="fence-head__title">normalisation</span>'
                '<span class="fence-head__rule"></span>'
                f'<span class="fence-head__count">{len(groups)} '
                f"collision{'s' if len(groups) != 1 else ''}</span>"
                "</div>"
                f'<p class="fence-lede">{verdict}</p>'
                f'<div class="norm__rows">{rows}</div>'
                '<p class="common__caption">only the tracked levels are '
                "checked, so a collision outside the top-k would not appear "
                "here</p>"
                "</div>"
            )
            return (
                '<button role="tab" data-tab="normalize">Normalization</button>',
                f'<section class="tab-pane" data-tab="normalize">{pane}</section>',
            )
        except Exception:
            return "", ""

    #: The normalisations worth asking about, and the order they are reported
    #: in. `casefold` rather than `lower` because it is the one that folds the
    #: cases `lower` misses, and a report that says "nothing merges" should not
    #: be wrong about German or Turkish text.
    _NORMALISERS = (
        ("lower()", lambda text: text.casefold()),
        ("strip()", lambda text: text.strip()),
        ("both", lambda text: text.strip().casefold()),
    )

    def _collisions(self, levels: list[str]) -> list[tuple[str, list[str], str]]:
        """Groups of levels that a normalisation would merge into one.

        A level is reported under the *first* rule that merges it, so a pair
        differing in both case and whitespace is one finding rather than three.
        """
        seen: set[frozenset[str]] = set()
        found: list[tuple[str, list[str], str]] = []

        for rule, normalise in self._NORMALISERS:
            buckets: dict[str, list[str]] = {}
            for level in levels:
                buckets.setdefault(normalise(level), []).append(level)
            for key, members in buckets.items():
                if len(members) < 2:
                    continue
                identity = frozenset(members)
                if identity in seen:
                    continue
                seen.add(identity)
                found.append((rule, sorted(members), key))
        return found

    def _build_length_pane(self, stats: CategoricalStats) -> str:
        """The label-length distribution, which was being spent on two numbers.

        Phase 5c.2 (#155). `categorical.py` has kept a 5,000-value reservoir of
        label lengths all along and the report spent it on `avg_len` and
        `len_p90`. The whole distribution was sitting in it, and on an
        identifier column the shape *is* the finding: `Ticket` clusters at 4-7
        characters and at 8-10 with a tail to 18, which is two ticket formats
        in one column and is available no other way.

        **Suppressed below three distinct lengths.** `Embarked` has one, and a
        chart of one bar at full height says only "all of them", which is what
        the sentence says in fewer pixels and without implying a distribution.
        """
        bins = [
            (int(length), int(count))
            for length, count in (getattr(stats, "len_hist", None) or [])
            if count > 0
        ]
        if not bins:
            return ""

        name = self.safe_html_escape(stats.name)
        header = (
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · label length</span>'
            '<span class="fence-head__rule"></span>'
            f'<span class="fence-head__count">{bins[0][0]} to {bins[-1][0]} '
            "characters</span>"
            "</div>"
        )

        if len(bins) < 3:
            if len(bins) == 1:
                sentence = (
                    f"Every label is {bins[0][0]} character"
                    f"{'s' if bins[0][0] != 1 else ''} long."
                )
            else:
                sentence = (
                    f"Labels are either {bins[0][0]} or {bins[1][0]} characters "
                    f"long — {bins[0][1]:,} and {bins[1][1]:,} of them."
                )
            return (
                f'<div class="fence-pane">{header}'
                f'<p class="fence-lede">{sentence}</p></div>'
            )

        top = max(count for _, count in bins)
        rows = "".join(
            f'<div class="lenbar" style="height:{count / top * 100:.1f}%" '
            f'title="{count:,} label{"s" if count != 1 else ""} of '
            f'{length} character{"s" if length != 1 else ""}" '
            f'data-col="{self.safe_col_id(stats.name)}" data-count="{count}"></div>'
            for length, count in bins
        )
        ticks = (
            f'<span class="lenaxis__tick">{bins[0][0]}</span>'
            f'<span class="lenaxis__tick">{bins[-1][0]}</span>'
        )
        return (
            f'<div class="fence-pane">{header}'
            f'<p class="fence-lede">{self._length_finding(bins)}</p>'
            f'<div class="lenchart">{rows}</div>'
            f'<div class="lenchart__axis"></div>'
            f'<div class="lenaxis">{ticks}</div>'
            '<p class="common__caption">one bar per label length, over a '
            "sample of the values</p>"
            "</div>"
        )

    #: A gap has to span this much of the length range before it is called a
    #: gap. Without it every sparse tail reads as a finding -- `Name` runs 12
    #: to 82 characters and almost every length above 60 is isolated, which the
    #: first version reported as "27 separate clusters".
    _GAP_MIN_SPAN = 0.12

    #: And each side has to hold this much of the labels. A gap with two
    #: stragglers beyond it is a tail, not a second format.
    _GAP_MIN_MASS = 0.10

    #: A gap has to span this much of the length range before it is called a
    #: gap. Without it every sparse tail reads as a finding -- `Name` runs 12
    #: to 82 characters and almost every length above 60 is isolated, which the
    #: first version reported as "27 separate clusters".
    _GAP_MIN_SPAN = 0.12

    #: And each side has to hold this much of the labels, and at least
    #: `_GAP_MIN_LABELS` of them. A gap with two stragglers beyond it is a
    #: tail, not a second format. 2% rather than 10% because `Ticket`'s second
    #: format is 40 of 891 labels -- rare, and still a real cleaning finding.
    _GAP_MIN_MASS = 0.02
    _GAP_MIN_LABELS = 5

    def _length_finding(self, bins: list[tuple[int, int]]) -> str:
        """What the shape says, when it says something.

        A gap in the middle of a length distribution is two formats sharing a
        column, which is a cleaning finding and the reason this chart exists.
        But only a *substantial* gap is, and two earlier versions of this got
        it wrong in opposite directions:

        - Calling any gap of more than one character a cluster boundary
          reported `Name` -- an ordinary right-skewed spread of 12 to 82
          characters -- as "27 separate clusters". That is sparsity at the
          tail described as a finding.
        - Requiring 10% of the labels on each side then rejected `Ticket`,
          whose 40 long tickets in 891 are exactly the second format this
          chart exists to surface.

        **The bin step matters.** Above `_MAX_LENGTH_BINS` distinct lengths the
        histogram groups them, so adjacent bins sit a full bin apart and *every*
        neighbour looks like a gap. `Name` bins at width 2, and its twenty-seven
        "gaps" were all this artifact. The step is inferred as the smallest
        distance between neighbours, and anything that close is contiguous.
        """
        lengths = [length for length, _ in bins]
        total = sum(count for _, count in bins)
        span = max(1, lengths[-1] - lengths[0])
        spread = f"{lengths[0]} to {lengths[-1]} characters"
        step = min(
            (high - low for low, high in zip(lengths, lengths[1:], strict=False)),
            default=1,
        )

        gaps = []
        for index, (low, high) in enumerate(zip(lengths, lengths[1:], strict=False)):
            if high - low <= step:
                continue
            if (high - low) / span < self._GAP_MIN_SPAN:
                continue
            below = sum(count for _, count in bins[: index + 1])
            above = total - below
            smaller = min(below, above)
            if smaller / total < self._GAP_MIN_MASS or smaller < self._GAP_MIN_LABELS:
                continue
            gaps.append((low, high, below, above))

        if len(gaps) == 1:
            low, high, below, above = gaps[0]
            return (
                f"Lengths run {spread}, with nothing between {low} and {high}: "
                f"{below:,} labels fall below the gap and {above:,} above it. "
                "Two formats in one column look like this."
            )
        if len(gaps) > 1:
            return (
                f"Lengths run {spread} in {len(gaps) + 1} groups separated by "
                "gaps, which is more than one format sharing a column."
            )

        peak_length, peak_count = max(bins, key=lambda item: item[1])
        return (
            f"Lengths run {spread}, with no gap wide enough to suggest a "
            f"second format. The most common is {peak_length} characters "
            f"({peak_count:,} of {total:,})."
        )

    def _build_details_section(
        self,
        col_id: str,
        common_table: str,
        norm_tab_btn: str,
        norm_tab_pane: str,
        missing_table: str,
        length_pane: str = "",
        common_label: str = "Common values",
        *,
        has_missing: bool = True,
    ) -> str:
        """Details tabs, minus the ones with nothing to say (#154, 5b.4).

        `norm_tab_btn` and `norm_tab_pane` were already conditional -- the
        normalization tab is emitted as a pair of strings that are empty when
        there is nothing to normalise. Missing Values was not, so it rendered a
        100%-present bar on every complete column.
        """
        normalization = norm_tab_pane.strip()
        return self._build_tabbed_details(
            col_id,
            [
                ("common", common_label, common_table, bool(common_table.strip())),
                (
                    "normalize",
                    "Normalization",
                    normalization,
                    bool(normalization) and bool(norm_tab_btn.strip()),
                ),
                (
                    "length",
                    "Label length",
                    length_pane,
                    bool(length_pane.strip()),
                ),
                (
                    "missing",
                    "Missing Values",
                    f'<div class="sub"><div class="hdr">Missing Values</div>{missing_table}</div>',
                    has_missing,
                ),
            ],
        )

    def _build_controls_section(
        self, col_id: str, topn_list: list[int], default_topn: int
    ) -> str:
        """Build controls section.

        `topn_list` is empty when the chart shows every level, or when there is
        no chart at all -- a high-cardinality column gets a sentence instead.
        Either way there is nothing to control, so the group is not rendered.
        """
        if topn_list:
            topn_buttons = " ".join(
                f'<button type="button" class="btn-soft'
                f'{" active" if n == default_topn else ""}" '
                f'data-topn="{n}">{n}</button>'
                for n in topn_list
            )
            controls = (
                f'<div class="hist-controls" data-topn="{default_topn}">'
                '<div class="center-controls"><span>Top‑N:</span>'
                f'<div class="bin-group">{topn_buttons}</div></div></div>'
            )
        else:
            controls = ""

        return f"""
        <div class="card-controls" role="group" aria-label="Column controls">
            <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
            </div>
            <div class="controls-slot">{controls}</div>
        </div>
        """

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: CategoricalStats,
        approx_badge: str,
        quality_flags_html: str,
        stat_row: str,
        chart_html: str,
        details_html: str,
        controls_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
        docs_url = "https://alvarodiez20.github.io/pysuricata/stats/categorical/"
        info_button = f'''<a href="{docs_url}" target="_blank" rel="noopener noreferrer" class="info-link" title="View documentation for Categorical analysis" aria-label="View Categorical analysis documentation">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path fill="currentColor" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM6.5 5a.75.75 0 0 0 0 1.5h.5v2.5h-.5a.75.75 0 0 0 0 1.5h3a.75.75 0 0 0 0-1.5h-.5V6h-.5A.75.75 0 0 0 8 5.25H6.5zM8 3.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5z"/>
            </svg>
        </a>'''

        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname">{safe_name}</span>
                    <span class="badge">Categorical</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {approx_badge}
                    {quality_flags_html}
                </div>
                {info_button}
            </header>
            <div class="var-card__body">
                <div class="var-chart">{chart_html}</div>
                {controls_html}
                {stat_row}
                {details_html}
            </div>
        </article>
        """

    def _build_common_values_table(self, stats: CategoricalStats) -> str:
        """Build common values table with enhanced formatting and functionality.

        This method creates a professional, feature-rich table that provides
        comprehensive insights into the most frequent categorical values in the dataset.

        Args:
            stats: CategoricalStats object containing the data

        Returns:
            HTML string for the enhanced common values table
        """
        try:
            top_items = list(getattr(stats, "top_items", []) or [])
        except Exception:
            top_items = []

        if not top_items:
            return '<div class="muted">No common values to display</div>'

        rows = []
        total_nonnull = max(1, int(getattr(stats, "count", 0)))

        # Take only top 10 values for better display and performance
        top_items = top_items[:10]

        for i, (value, count) in enumerate(top_items):
            pct = (int(count) / total_nonnull) * 100.0 if total_nonnull else 0.0

            # Add ranking indicator for top values
            rank_icon = ordinal_number(i + 1)

            # Format categorical value with proper escaping
            formatted_value = self.safe_html_escape(str(value))

            rows.append(
                f"<tr class='common-row rank-{i + 1}'>"
                f"<td class='rank'>{rank_icon}</td>"
                f"<td class='cat common-value'>{formatted_value}</td>"
                f"<td class='num common-count'>{int(count):,}</td>"
                f"<td class='num common-pct'>{pct:.1f}%</td>"
                f"<td class='progress-bar'><div class='bar-fill' style='width:{pct:.1f}%'></div></td>"
                f"</tr>"
            )

        body = "".join(rows)
        return (
            '<table class="common-values-table enhanced">'
            "<thead><tr><th>Rank</th><th>Value</th><th>Count</th><th>Frequency</th><th>Distribution</th></tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
        )

    def _build_missing_values_table(
        self, stats: CategoricalStats, miss_pct: float
    ) -> str:
        """Build simple missing values analysis."""
        total_values = stats.count + stats.missing
        present_pct = (
            (stats.count / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )
        return super()._build_missing_values_table(
            stats.count, present_pct, stats.missing, miss_pct, stats, total_values
        )

    def _get_missing_data_severity(self, missing_pct: float) -> tuple[str, str, str]:
        """Get missing data severity classification with clear thresholds.

        Args:
            missing_pct: Percentage of missing data

        Returns:
            Tuple of (severity_class, label, icon)
        """
        if missing_pct >= 50:
            return "critical", "Critical", ""
        elif missing_pct >= 20:
            return "high", "High", ""
        elif missing_pct >= 5:
            return "medium", "Medium", ""
        else:
            return "low", "Low", ""
