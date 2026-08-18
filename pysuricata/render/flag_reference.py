"""Flag reference: what each quality chip measures, and what it means.

Drop into ``pysuricata/render/`` and render from it in two places:

1. **On the chip** — the threshold goes on the chip's face, not in a ``title``.
   ``annotate_flags`` currently writes ``33.20 heavy-tailed`` with the limit in a
   tooltip, which is invisible on a phone and absent from a PDF. With this table
   the chip reads ``33.20 kurtosis · limit 10``.
2. **In a reference block** under the needs-attention list — one row per flag the
   report actually raised. Four rows on Titanic, nothing on a clean frame.

Phase 4b.2 of ``integration.md``. Design: ``Variables Section.dc.html`` 15b.

The ``means`` sentences are the part worth having: they are what a tooltip never
said. Each one states a consequence for the data without recommending an action
— see open question 7 in the plan for why that line is drawn here.

Keys are the output of ``triage.flag_slug()``, so a chip label maps to its entry
with no extra bookkeeping.

That claim was written before the slugs were measured, and five of the ten the
Titanic report raises had no entry when this arrived: two near-misses
(``heaped`` for ``heaping``, ``skewed`` for ``skewed-right``) and three absent
outright. The keys below are now taken from the 28 labels the renderers
actually emit, checked with ``flag_slug`` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlagMeaning:
    """One row of the reference.

    Attributes:
        measure: What is counted or computed. Names the statistic, so a reader
            who sees ``33.20`` knows it is excess kurtosis and not a percentage.
        limit: The threshold that fires the flag, as it should appear on the
            chip face. ``"any"`` where the flag fires on presence rather than
            on crossing a number.
        means: What it means for the data. Never what to do about it.
        unit: Optional short unit for the chip face, when the value alone is
            ambiguous.
    """

    measure: str
    limit: str
    means: str
    unit: str = ""


#: Every flag the renderers can raise, keyed by ``triage.flag_slug(label)``.
#:
#: Missing entries must degrade gracefully: a chip with no entry keeps its
#: current rendering rather than raising, because a new flag added in the
#: renderers should not break a report.
FLAG_MEANINGS: dict[str, FlagMeaning] = {
    "missing": FlagMeaning(
        measure="Share of rows with no value",
        limit="20%",
        means=(
            "Every use of the column has to say what happens to those rows. "
            "Above about half, imputation invents more data than it fills."
        ),
    ),
    "high-cardinality": FlagMeaning(
        measure="Distinct values ÷ rows",
        limit="90%",
        means=(
            "The column identifies rows rather than grouping them. Useful as a "
            "key or a join, not as a category."
        ),
    ),
    "many-outliers": FlagMeaning(
        measure="Share beyond 1.5× IQR from the quartiles",
        limit="5%",
        means=(
            "Values outside the usual range. They may be errors or may be real "
            "— the Outliers pane lists them so you can tell which."
        ),
    ),
    "heavy-tailed": FlagMeaning(
        measure="Excess kurtosis of the values",
        limit="10",
        means=(
            "The tail is far heavier than a normal distribution, where this is "
            "0. Means and standard deviations are pulled by a few extreme rows."
        ),
    ),
    "skewed-right": FlagMeaning(
        measure="Fisher–Pearson skewness",
        limit="±2",
        means=(
            "The distribution leans one way, so the mean sits away from the "
            "median. A log or rank transform will make it symmetric."
        ),
    ),
    "case-variants": FlagMeaning(
        measure="Levels that merge under lower()",
        limit="any",
        means=(
            "The same category is spelled two ways, so counts are split "
            "between them. Normalising would reduce the level count."
        ),
    ),
    "trim-variants": FlagMeaning(
        measure="Levels that merge under strip()",
        limit="any",
        means=(
            "Leading or trailing whitespace makes one category look like two. "
            "Invisible in the report and in most editors."
        ),
    ),
    "high-uniqueness": FlagMeaning(
        measure="Distinct timestamps ÷ rows",
        limit="90%",
        means=(
            "Almost every timestamp is different, so the column marks events "
            "rather than grouping them into periods."
        ),
    ),
    "irregular-intervals": FlagMeaning(
        measure="Standard deviation ÷ mean of the gaps",
        limit="1.0",
        means=(
            "The gaps between records vary as much as their average, so this "
            "is an event stream rather than a schedule."
        ),
    ),
    "constant": FlagMeaning(
        measure="Distinct non-missing values",
        limit="1",
        means=(
            "Every row holds the same value, so the column cannot distinguish "
            "rows and carries no information."
        ),
    ),
    "zero-variance": FlagMeaning(
        measure="Standard deviation",
        limit="0",
        means=(
            "No spread at all. Any model treating this as a feature is fitting "
            "a constant."
        ),
    ),
    "positive-only": FlagMeaning(
        measure="Count of values below zero",
        limit="0",
        means=(
            "No negative values were seen. A log transform is safe, and a "
            "negative in future data is a validation failure."
        ),
    ),
    "heaping": FlagMeaning(
        measure="Share of values on round numbers",
        limit="20%",
        means=(
            "Values cluster on round numbers, which usually means they were "
            "reported rather than measured."
        ),
    ),
    # --- raised by the renderers, absent from the handoff's table ----------
    #
    # Written in the same voice as the entries above: what was measured, the
    # limit that fired it, and a consequence for the data. No advice -- that
    # line is open question 7 and is not mine to close.
    "dominant-category": FlagMeaning(
        # 70%, not the 50% this said until #314. `dominant_category_threshold`
        # is 0.7 and has been throughout; a 60%-dominant column does not fire,
        # so the table was explaining the flag with a limit that was not the
        # one being applied. This block exists to tell a reader why a chip is
        # on their column, and a wrong number there is worse than no table.
        measure="Share held by the most common level",
        limit="70%",
        means=(
            "One level accounts for most of the column, so the rest are a "
            "small minority however many of them there are."
        ),
    ),
    "skewed-left": FlagMeaning(
        measure="Fisher-Pearson skewness",
        limit="|1|",
        means=(
            "The long tail runs toward the low end, so the mean sits below "
            "the median and neither describes a typical row on its own."
        ),
    ),
    "empty-or-zero": FlagMeaning(
        measure="Share of values that are empty text or zero",
        limit="any",
        means=(
            "Empty and zero are both present, and they are usually two "
            "different things: one is a value, the other is a value missing "
            "in a way the missing count does not see."
        ),
    ),
    "monotonic": FlagMeaning(
        measure="Whether every step moves the same way",
        limit="any",
        means=(
            "The column only ever increases or only ever decreases, which is "
            "the shape of a key, a counter or a timestamp rather than a "
            "measurement."
        ),
    ),
    "quasi-constant": FlagMeaning(
        measure="Share held by the single most common value",
        limit="95%",
        means=(
            "Nearly every row carries the same value, so the column separates "
            "almost nothing."
        ),
    ),
    "many-rare-levels": FlagMeaning(
        measure="Share of levels seen only once or twice",
        limit="50%",
        means=(
            "Most levels are near-singletons, so counts per level rest on one "
            "or two rows each."
        ),
    ),
    "imbalanced": FlagMeaning(
        measure="Ratio between the largest and smallest class",
        limit="10:1",
        means=(
            "The classes are far apart in size, so a share computed over all "
            "of them is dominated by the largest."
        ),
    ),
    "zero-inflated": FlagMeaning(
        measure="Share of values that are exactly zero",
        limit="50%",
        means=(
            "Zero is a large fraction of the column, so summaries computed "
            "over every row describe the zeros more than the rest."
        ),
    ),
}


#: Flat-calendar baselines for the datetime card's two ratios (phase 5e.2).
#:
#: These are arithmetic, not estimates, and they are what makes ``27.0%``
#: readable. Without them the card prints two percentages that look like
#: findings and are noise.
WEEKEND_FLAT_PCT: float = 2 / 7 * 100  # 28.57 — 2 of 7 days
BUSINESS_HOURS_FLAT_PCT: float = (
    8 / 24 * (5 / 7) * 100
)  # 23.81 — 8 of 24 hours, 5 of 7 days

#: How far from the flat baseline counts as a real skew rather than noise, in
#: percentage points. Below this the card should say "flat" rather than print a
#: direction, or every dataset reads as slightly weekend-heavy.
FLAT_TOLERANCE_PP: float = 3.0

#: When the weekend share is loud enough to raise the `Weekend-heavy` chip.
#: 35% is roughly the flat share plus 6.4pp -- a little over twice the noise
#: tolerance above, which is what makes it a finding rather than a wobble.
WEEKEND_HEAVY_FLAG_PCT: float = 35.0

#: When the business-hours share raises the `Business hours` chip. 50% is not
#: baseline-plus-a-margin but *more than twice* the flat share: a column where
#: half the rows fall in a fifth of the week is office-generated data, and the
#: chip says so.
BUSINESS_HOURS_FLAG_PCT: float = 50.0


def flat_verdict(actual_pct: float, flat_pct: float) -> tuple[str, str]:
    """Read a calendar share against the flat baseline it should be judged on.

    ``Weekend % 27.0`` is not a finding — a flat calendar gives 28.6%, so 27.0
    is the *absence* of a weekend effect. The card cannot say that without the
    baseline, which is why this returns the verdict rather than the number
    (phase 5e.2, #291).

    Args:
        actual_pct: The observed share, already in percent.
        flat_pct: What a flat calendar would give, in percent. One of
            :data:`WEEKEND_FLAT_PCT` or :data:`BUSINESS_HOURS_FLAT_PCT`.

    Returns:
        ``(verdict, tone)``. The verdict always states the gap in percentage
        points against the baseline, so the reader can check the reading
        against the mark. The tone is a quality slug — ``"good"`` when the
        column is within :data:`FLAT_TOLERANCE_PP` of flat, ``"warn"``
        otherwise — never a colour, so the token layer stays the one place a
        colour is chosen.
    """
    delta = actual_pct - flat_pct
    sign = "+" if delta >= 0 else "\u2212"
    gap = f"{sign}{abs(delta):.1f}pp vs {flat_pct:.1f}%"
    if abs(delta) < FLAT_TOLERANCE_PP:
        return f"flat \u00b7 {gap}", "good"
    direction = "over" if delta > 0 else "under"
    return f"{direction}-represented \u00b7 {gap}", "warn"


def even_split_pct(n_levels: int) -> float:
    """The share each level would hold if a categorical column were even.

    Drawn as a rule beside every level bar (phase 5f.2). ``Embarked``'s S at
    72.4% against a 33.3% mark says *dominated by one port* without asking the
    reader to divide anything.

    Args:
        n_levels: Number of distinct levels. Values below 1 return 0.0 so a
            column with nothing to draw gets no rule rather than a division
            error.
    """
    if n_levels < 1:
        return 0.0
    return 100.0 / n_levels


def raised_flags(slugs: list[str]) -> list[tuple[str, FlagMeaning]]:
    """The reference rows for a report, in the order given, deduplicated.

    Rendering only the flags a report raised is what keeps the block at four
    rows on Titanic and empty on a clean frame. A slug with no entry is dropped
    rather than rendered blank — a new flag in the renderers must not put an
    empty row in the reference.

    Args:
        slugs: Flag slugs collected from the rendered cards, via
            ``triage.flag_slug`` on each chip label.

    Returns:
        ``(slug, meaning)`` pairs for the slugs that have an entry.
    """
    seen: set[str] = set()
    out: list[tuple[str, FlagMeaning]] = []
    for slug in slugs:
        if slug in seen or slug not in FLAG_MEANINGS:
            continue
        seen.add(slug)
        out.append((slug, FLAG_MEANINGS[slug]))
    return out
