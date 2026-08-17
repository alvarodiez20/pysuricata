"""Categorical card rendering functionality."""

import math
from collections.abc import Sequence

from .card_base import CardRenderer
from .card_config import DEFAULT_CAT_CONFIG, DEFAULT_CHART_DIMS
from .card_types import BarData, CategoricalStats, QualityFlags
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
        coverage = 0.0
    else:
        coverage = sum(c for _, c in items[:5]) / count
        if coverage > _LOW_COVERAGE and distinct_ratio < _HIGH_CARDINALITY:
            return None

    return {
        "unique": unique,
        "count": count,
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
    return (
        f"{unique:,} distinct values in {count:,} rows, and the five most "
        f"common cover {facts['coverage'] * 100:.1f}% of them. A top-values "
        "chart would be a row of slivers, so there is nothing worth plotting."
    )


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
        else:
            chart_html = self._build_categorical_variants(
                col_id, items, total, topn_list, default_topn
            ) + self._build_coverage_note(stats, items)
        common_table = self._build_common_values_table(stats)
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
            # NOT gated on chunk count, unlike the numeric and datetime
            # cards. `html.py` calls `finalize()` without chunk metadata for
            # this kind, so the accumulator has none to give -- gating on it
            # would hide the pane permanently rather than tighten the rule.
            # See #193.
            has_missing=int(getattr(stats, "missing", 0) or 0) > 0,
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
        self.format_bytes(int(getattr(stats, "mem_bytes", 0)))

        data = [
            ("Count", f"{int(getattr(stats, 'count', 0)):,}", "num"),
            (
                f"Unique{' (≈)' if getattr(stats, 'approx', False) else ''}",
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

        data = [
            (
                "Entropy",
                self._unknown_cell(no_heavy_hitters)
                if unknown
                else self.format_number(cat_stats["entropy"]),
                "num",
            ),
            (
                "Rare levels",
                self._unknown_cell(no_heavy_hitters)
                if unknown
                else f"{int(cat_stats['rare_count']):,} ({cat_stats['rare_cov']:.1f}%)",
                f"num {cat_stats['rare_cls']}",
            ),
            (
                "Top 5 coverage",
                self._unknown_cell(no_heavy_hitters)
                if unknown
                else f"{cat_stats['top5_cov']:.1f}%",
                f"num {cat_stats['top5_cls']}",
            ),
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
                "Processed bytes (≈)",
                self.format_bytes(int(cat_stats.get("mem_bytes", 0))),
                "num",
            ),
        ]

        return data

    def _get_topn_candidates(
        self, items: Sequence[tuple[str, int]]
    ) -> tuple[list[int], int]:
        """Get Top-N candidates for categorical display."""
        max_n = max(1, min(15, len(items)))
        candidates = [5, 10, 15, max_n]
        topn_list = sorted({n for n in candidates if 1 <= n <= max_n})
        default_topn = (
            10 if 10 in topn_list else (max(topn_list) if topn_list else max_n)
        )
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

    def _build_coverage_note(self, stats: CategoricalStats, items: list) -> str:
        """How much of the column the bars actually account for.

        A top-N chart is a sample of the levels, and without this line there is
        nothing to say whether the bars are the whole column or a tenth of it.
        """
        count = int(getattr(stats, "count", 0) or 0)
        if not items or count <= 0:
            return ""
        shown = len(items)
        total_levels = max(int(getattr(stats, "unique_est", shown) or shown), shown)
        covered = sum(c for _, c in items) / count * 100.0
        levels = "level" if total_levels == 1 else "levels"
        return (
            f'<p class="coverage-note">{shown:,} of {total_levels:,} {levels} shown '
            f"· covers {covered:.0f}% of non-missing rows</p>"
        )

    def _build_categorical_variants(
        self,
        col_id: str,
        items: Sequence[tuple[str, int]],
        total: int,
        topn_list: list[int],
        default_topn: int,
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

            svg = self._build_categorical_bar_svg(data, total=max(1, int(total)))
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
        self, items: list[tuple[str, int]], total: int, *, scale: str = "count"
    ) -> str:
        """Build categorical bar chart SVG."""
        if total <= 0 or not items:
            return self.create_empty_svg(
                "cat-svg", self.chart_dims.width, self.chart_dims.height
            )

        bar_data = self._prepare_bar_data(items, total, scale)
        return self._render_bar_svg(bar_data)

    def _prepare_bar_data(
        self, items: list[tuple[str, int]], total: int, scale: str
    ) -> BarData:
        """Prepare bar chart data."""
        labels = [self.safe_html_escape(str(k)) for k, _ in items]
        counts = [int(c) for _, c in items]
        pcts = [(c / total * 100.0) for c in counts]

        if scale == "pct":
            values = pcts
        else:
            values = counts

        return BarData(labels=labels, counts=counts, percentages=pcts, values=values)

    def _render_bar_svg(self, bar_data: BarData) -> str:
        """Render bar chart SVG."""
        width, height = self.chart_dims.width, self.chart_dims.height
        margin_top, margin_bottom = 8, 8
        margin_right = 12

        # Calculate label width
        max_label_len = max((len(label) for label in bar_data.labels), default=0)
        char_w = self.cat_config.char_width
        gutter = max(
            self.cat_config.min_gutter,
            min(self.cat_config.max_gutter, char_w * min(max_label_len, 28) + 16),
        )
        margin_left = max(120, gutter)

        n = len(bar_data.labels)
        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom

        if n <= 0 or iw <= 0 or ih <= 0:
            return self.create_empty_svg("cat-svg", width, height)

        bar_gap = 6
        bar_h = max(4, (ih - bar_gap * (n - 1)) / max(n, 1))

        vmax = max(bar_data.values) or 1.0

        def sx(v: float) -> float:
            return margin_left + (v / vmax) * iw

        parts = [
            f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">'
        ]

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

            parts.append(
                f'<g class="bar-row">'
                f'<rect class="bar" x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="2" ry="2">'
                f"<title>{label}\n{c:,} rows ({p:.1f}%)</title>"
                f"</rect>"
                f'<text class="bar-label" x="{margin_left - 6}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="end">{short}</text>'
                f'<text class="bar-value" x="{(x1 - 6 if w >= 56 else x1 + 4):.2f}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="{("end" if w >= 56 else "start")}">{c:,} ({p:.1f}%)</text>'
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
                ("common", "Common values", common_table, bool(common_table.strip())),
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
        """Build controls section."""
        topn_buttons = " ".join(
            f'<button type="button" class="btn-soft{" active" if n == default_topn else ""}" data-topn="{n}">{n}</button>'
            for n in topn_list
        )

        return f"""
        <div class="card-controls" role="group" aria-label="Column controls">
            <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
            </div>
            <div class="controls-slot">
                <div class="hist-controls" data-topn="{default_topn}">
                    <div class="center-controls">
                        <span>Top‑N:</span>
                        <div class="bin-group">{topn_buttons}</div>
                    </div>
                </div>
            </div>
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

    def _build_dataprep_spectrum_visualization(self, stats: CategoricalStats) -> str:
        """Legacy method - no longer used."""
        return ""

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

    def _build_dataprep_spectrum_visualization(self, stats: CategoricalStats) -> str:
        """Build DataPrep-style spectrum visualization for missing values per chunk.

        This creates a single horizontal bar with segments representing actual processing
        chunks, colored by missing value density (green-yellow-red gradient).

        Args:
            stats: CategoricalStats object containing chunk metadata and missing data information

        Returns:
            HTML string for the DataPrep-style spectrum visualization
        """
        # Check if we have chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""

        total_values = stats.count + stats.missing
        if total_values == 0:
            return ""

        # Build the spectrum bar segments
        segments_html = ""
        total_width = 0

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            missing_pct = (
                (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )

            # Calculate segment width as percentage of total
            segment_width_pct = (chunk_size / total_values) * 100.0
            total_width += segment_width_pct

            # Determine color based on missing percentage (DataPrep-style)
            if missing_pct <= 5:
                color_class = "spectrum-low"
            elif missing_pct <= 20:
                color_class = "spectrum-medium"
            else:
                color_class = "spectrum-high"

            segments_html += f"""
            <div class="spectrum-segment {color_class}"
                 style="width: {segment_width_pct:.2f}%"
                 data-start="{start_row}"
                 data-end="{end_row}"
                 data-missing="{missing_count}"
                 data-pct="{missing_pct:.1f}">
            </div>
            """

        # Build summary statistics
        total_chunks = len(chunk_metadata)
        max_missing_pct = max(
            (missing_count / (end_row - start_row + 1)) * 100.0
            for start_row, end_row, missing_count in chunk_metadata
        )
        avg_missing_pct = (
            sum(
                (missing_count / (end_row - start_row + 1)) * 100.0
                for start_row, end_row, missing_count in chunk_metadata
            )
            / total_chunks
        )

        # Determine overall severity
        if max_missing_pct >= 50:
            severity = "critical"
        elif max_missing_pct >= 20:
            severity = "high"
        elif max_missing_pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        return f"""
        <div class="dataprep-spectrum">
            <div class="spectrum-header">
                <span class="spectrum-title">Missing Values Distribution</span>
                <span class="spectrum-stats">
                    {total_chunks} chunks • {max_missing_pct:.1f}% max • {avg_missing_pct:.1f}% avg
                </span>
                <span class="spectrum-severity {severity}">
                    {severity.title()} Missing Data
                </span>
            </div>
            <div class="spectrum-bar">
                {segments_html}
            </div>
            <div class="spectrum-legend">
                <span class="legend-item spectrum-low">Low (≤5%)</span>
                <span class="legend-item spectrum-medium">Medium (5-20%)</span>
                <span class="legend-item spectrum-high">High (>20%)</span>
            </div>
        </div>
        """
