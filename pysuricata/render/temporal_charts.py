"""Temporal distribution charts: hour-of-day, day-of-week, month and year.

Each is a `figure.hist` — the structure the numeric histogram uses and the
timeline adopted in #219 — rather than a self-contained SVG. The SVG holds
only marks; every label is HTML beside it.

That is not a style preference. These charts sit in a responsive grid, so the
box they are handed is whatever the grid gives them, and an SVG drawn to fill
its box scales everything inside it including the text. Measured before the
change: the same 11px label rendered between **5.6px and 14.9px** depending on
viewport width, and not even monotonically -- the grid drops from two columns
to one, so 600px gave a *larger* label than 820px. There is no authored size
that fixes this, because the problem is that the text is inside a stretched box
at all.

Reusing the histogram's classes rather than styling a third chart is the point
rather than a shortcut: these inherit its gutter, its edge nudges and its
label rules, all already written and already tested, and they cannot drift
apart from it.
"""

from __future__ import annotations


class TemporalChartRenderer:
    """Renders temporal distribution charts.

    Carries no width, height or margins. It used to hold `width=400`,
    `height=160` and four hardcoded margin constants, which described a box it
    did not control -- the grid decided the real size, and the constants only
    determined how badly the contents were stretched to reach it. Geometry is
    now in percentages of the plot, so there is nothing left to hardcode.
    """

    def render_hour_chart(self, counts: list[int]) -> str:
        """Render hour-of-day distribution (0-23 hours).

        Args:
            counts: List of 24 integers representing counts for each hour

        Returns:
            SVG string
        """
        if not counts or len(counts) != 24:
            counts = [0] * 24

        labels = [f"{h:02d}:00" for h in range(24)]
        # Show every 3rd hour to avoid crowding
        visible_indices = list(range(0, 24, 3))

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Hour of Day Distribution",
        )

    def render_dow_chart(self, counts: list[int]) -> str:
        """Render day-of-week distribution.

        Args:
            counts: List of 7 integers (Monday=0 to Sunday=6)

        Returns:
            SVG string
        """
        if not counts or len(counts) != 7:
            counts = [0] * 7

        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        visible_indices = list(range(7))  # Show all

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Day of Week Distribution",
        )

    def render_month_chart(self, counts: list[int]) -> str:
        """Render month distribution.

        Args:
            counts: List of 12 integers (January=0 to December=11)

        Returns:
            SVG string
        """
        if not counts or len(counts) != 12:
            counts = [0] * 12

        labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        visible_indices = list(range(12))  # Show all

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Month Distribution",
        )

    def render_year_chart(self, year_counts: dict[int, int]) -> str:
        """Render year distribution.

        Args:
            year_counts: Dictionary mapping year -> count

        Returns:
            SVG string
        """
        if not year_counts:
            return self._render_empty_chart("Year Distribution")

        # Sort years and extract counts
        sorted_years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in sorted_years]
        labels = [str(year) for year in sorted_years]

        # Show subset of labels if many years
        if len(labels) > 10:
            step = max(1, len(labels) // 6)
            visible_indices = (
                [0] + list(range(step, len(labels), step)) + [len(labels) - 1]
            )
        else:
            visible_indices = list(range(len(labels)))

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Year Distribution",
        )

    def _render_bar_chart(
        self,
        counts: list[int],
        labels: list[str],
        visible_indices: list[int],
        title: str,
    ) -> str:
        """Render a bar chart with given data.

        Args:
            counts: Data values
            labels: Labels for each bar
            visible_indices: Indices of labels to show
            title: Chart title

        Returns:
            SVG string
        """
        if not counts or max(counts) == 0:
            return self._render_empty_chart(title)

        span = self._SPAN
        max_count = max(counts)
        total_count = sum(counts)
        n_bars = len(counts)
        bar_width = span / n_bars
        bar_padding = bar_width * 0.15

        # Marks only. Every label is HTML, below -- see the class docstring.
        svg_parts = [
            f'<svg class="hist-svg temporal-chart" viewBox="0 0 {span:g} {span:g}" '
            f'preserveAspectRatio="none" role="img" aria-label="{title}">',
            f"<desc>Bar chart showing {title.lower()} with {n_bars} bars</desc>",
        ]

        y_ticks = self._calculate_ticks(0, max_count, 5)

        # `vector-effect="non-scaling-stroke"` throughout: the box is stretched
        # by a different factor on each axis, so without it a 1-unit rule is
        # thick one way and invisible the other.
        for tick in y_ticks:
            if tick == 0:
                continue
            y = span * (1 - tick / max_count)
            svg_parts.append(
                f'<line class="grid" x1="0" y1="{y:.3f}" x2="{span:g}" y2="{y:.3f}" '
                f'vector-effect="non-scaling-stroke"/>'
            )

        for i, count in enumerate(counts):
            if count == 0:
                continue
            x = i * bar_width + bar_padding
            bw = bar_width - 2 * bar_padding
            bh = span * (count / max_count)
            pct = (count / max(1, total_count)) * 100.0
            label = labels[i] if i < len(labels) else ""
            # No `rx`: a corner radius is in user units, so a stretched box
            # rounds the horizontal and vertical corners by different amounts
            # and the bars come out lopsided.
            svg_parts.append(
                f'<rect class="bar temporal-bar" x="{x:.3f}" y="{span - bh:.3f}" '
                f'width="{bw:.3f}" height="{bh:.3f}" '
                f'data-count="{count}" data-pct="{pct:.1f}" data-label="{label}"/>'
            )

        svg_parts.append(
            f'<line class="axis" x1="0" y1="{span:g}" x2="{span:g}" y2="{span:g}" '
            f'vector-effect="non-scaling-stroke"/>'
        )
        svg_parts.append(
            f'<line class="axis" x1="0" y1="0" x2="0" y2="{span:g}" '
            f'vector-effect="non-scaling-stroke"/>'
        )
        svg_parts.append("</svg>")

        return (
            '<figure class="hist temporal-figure">'
            '<div class="hist__plot">'
            f'<div class="hist__gutter">{self._render_count_labels(y_ticks, max_count)}'
            # `ROWS`, not `RECORDS`. The unit sits inside the count gutter, and
            # the gutter is 44px: `RECORDS` measures ~48px at 10px monospace
            # with the tracking this label carries, so it overhung into the
            # first bucket label and rendered as `RECORDS00:00`. `ROWS` is the
            # word the histogram and the timeline already use for the same
            # quantity, so the three charts in this card now agree.
            '<span class="hist__unit">ROWS</span></div>'
            f'<div class="hist__area">{"".join(svg_parts)}'
            f"{self._render_bucket_labels(labels, visible_indices, n_bars)}</div>"
            "</div>"
            "</figure>"
        )

    #: A square viewBox stretched by CSS on both axes, as the histogram and the
    #: timeline use.
    _SPAN = 100.0

    def _render_count_labels(self, y_ticks: list[float], max_count: float) -> str:
        """Count labels in the gutter, positioned as a percentage of the plot.

        `data-edge` marks the two extremes so the stylesheet can nudge them
        inward; without it the top label overhangs the plot and the `0` drops
        into the tick row below. Same contract as the histogram's gutter, whose
        rules these reuse.
        """
        if not max_count:
            return ""
        out = []
        for tick in y_ticks:
            top = (1 - tick / max_count) * 100.0
            edge = ""
            if top <= 0.0:
                edge = ' data-edge="top"'
            elif top >= 100.0:
                edge = ' data-edge="bottom"'
            out.append(
                f'<span class="hist__y"{edge} style="top:{top:.3f}%">'
                f"{self._format_count(tick)}</span>"
            )
        return "".join(out)

    def _render_bucket_labels(
        self, labels: list[str], visible_indices: list[int], n_bars: int
    ) -> str:
        """Bucket names under the axis, centred on their bar.

        The end labels anchor to the plot edge rather than centring on their
        bar, so `00:00` and `Dec` sit inside the chart instead of overhanging
        it -- the same correction the timeline's dates carry.
        """
        if not n_bars:
            return ""
        shown = [i for i in visible_indices if i < len(labels)]
        if not shown:
            return ""
        # Three tiers, halving twice: tier 3 goes first, then tier 2, leaving an
        # evenly spaced quarter of the labels. Stride 4 / stride 2 rather than a
        # single alternation, because one halving is not enough -- twelve month
        # names still collide in a 166px plot after dropping six of them.
        #
        # The thinning is driven by a *container* query on the chart, not a
        # media query on the viewport. These are small multiples in a two-column
        # grid, so chart width and viewport width come apart: at a 1,024px
        # viewport each chart is only 374px, and a viewport-keyed rule reads
        # that as roomy and leaves every label on.
        last = len(shown) - 1
        tiers = [1 if p % 4 == 0 else (2 if p % 2 == 0 else 3) for p in range(last + 1)]
        if last >= 0:
            # The final label always survives, so the axis keeps its right
            # endpoint however far it thins. Its neighbour is pushed to tier 3
            # in the same move: promoting the last one without demoting the one
            # beside it is what put `18:00` on top of `21:00` and `Nov` on top
            # of `Dec`, since the two then survived every thinning together.
            tiers[last] = 1
            if last >= 1:
                tiers[last - 1] = 3
        out = []
        for position, i in enumerate(shown):
            tier = tiers[position]
            fraction = (i + 0.5) / n_bars
            if position == 0:
                anchor = ' data-anchor="start"'
            elif position == last:
                anchor = ' data-anchor="end"'
            else:
                anchor = ""
            # `data-ttier`, not `data-tier`. The histogram thins its own ticks
            # with viewport media queries on `data-tier`, and those are wrong
            # here: at a 700px viewport this chart is 544px wide and perfectly
            # roomy, but the viewport rule read 700 and dropped half the labels
            # anyway. A separate attribute leaves the container query below as
            # the only thing thinning these, with no specificity fight and no
            # dependence on what `display` the histogram's spans compute to.
            out.append(
                f'<span class="hist__tick" data-ttier="{tier}"{anchor} '
                f'style="left:{fraction * 100:.3f}%">{labels[i]}</span>'
            )
        return f'<div class="hist__x">{"".join(out)}</div>'

    def _render_empty_chart(self, title: str) -> str:
        """Render an empty chart placeholder.

        Args:
            title: Chart title

        Returns:
            SVG string
        """
        # HTML, not an SVG with a `<text>` in it. The placeholder used to be a
        # 14px label inside a box the grid stretches, so "No data available"
        # was set at a different size in every column of the grid.
        return (
            '<figure class="hist temporal-figure temporal-figure--empty">'
            f'<p class="hist__empty">No data available<span class="sr-only"> '
            f"for {title.lower()}</span></p>"
            "</figure>"
        )

    def _calculate_ticks(
        self, min_val: float, max_val: float, target_count: int
    ) -> list[float]:
        """Calculate nice tick values for an axis.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            target_count: Target number of ticks

        Returns:
            List of tick values
        """
        if max_val <= min_val:
            return [0]

        range_val = max_val - min_val
        rough_step = range_val / (target_count - 1)

        # Find nice step size
        magnitude = (
            10 ** int(f"{rough_step:.0e}".split("e")[1]) if rough_step > 0 else 1
        )
        residual = rough_step / magnitude

        if residual > 5:
            nice_step = 10 * magnitude
        elif residual > 2:
            nice_step = 5 * magnitude
        elif residual > 1:
            nice_step = 2 * magnitude
        else:
            nice_step = magnitude

        # Generate ticks
        ticks = []
        tick = 0
        while tick <= max_val:
            ticks.append(tick)
            tick += nice_step

        return ticks

    def _format_count(self, count: float) -> str:
        """Format count for display.

        Args:
            count: Count value

        Returns:
            Formatted string
        """
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        else:
            return f"{int(count)}"
