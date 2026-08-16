"""Turn the quality chips from decoration into navigation.

A sixty-column frame renders sixty identically sized cards in source order. The
question a reader arrives with -- *which columns are broken?* -- is never
answered, even though the answer is already computed: every card carries quality
chips, and the ones marked ``warn`` or ``bad`` are exactly the columns worth
looking at first.

Nothing new is measured here. This reads the chips each card already emitted and
puts them where the reader lands.
"""

from __future__ import annotations

import html as _html
import re

# The chips are emitted by the card renderers as
# <li class="flag warn" data-threshold="..." data-value="...">Skewed Right</li>
# so the shape is ours, not arbitrary markup. Severity is the second class.
#
# The attribute run has to be quote-aware: thresholds like data-threshold=">1"
# and data-threshold="|kurtosis| > 3" contain a literal '>', so matching [^>]*
# ends the tag early and swallows half the attributes into the label.
_CHIP = re.compile(
    r'<li class="flag(?P<severity>[^"]*)"'
    r'(?P<attrs>(?:[^>"]|"[^"]*")*)>'
    r"(?P<label>[^<]+)</li>"
)

# A `bad` chip is always worth surfacing -- the card renderer already decided it
# was serious. A `warn` chip is not, and treating every one as an issue makes
# the block useless: a plain standard normal earns "Has negatives" (half its
# values are) and "Some outliers" (0.3-1% beyond 1.5 IQR is what a normal
# distribution does), so nine well-behaved columns would report as nine
# problems. The warnings below are the ones that describe a defect in the data
# rather than the shape of a distribution.
_ACTIONABLE_WARNINGS = frozenset(
    {
        "missing",
        "zero-inflated",
        "constant",
        "quasi-constant",
        "imbalanced",
        "high-cardinality",
        "dominant-category",
        "many-rare-levels",
        "case-variants",
        "trim-variants",
        "empty-strings",
    }
)

_SEVERITY_RANK = {"bad": 0, "warn": 1}


_ATTR = re.compile(r'(?P<name>data-(?:threshold|value))="(?P<value>[^"]*)"')


def annotate_flags(flags_html: str) -> str:
    """Put the number a chip already knows on the face of the chip.

    Every chip carries ``data-threshold`` and ``data-value`` in the DOM and
    displayed neither -- so a card said ``Missing`` where it could have said
    ``19.9% missing``, and the reader had to open the details pane, or the
    inspector, to learn whether that meant two rows or two hundred.

    Done here rather than at each of the forty-two places a chip is emitted:
    those attributes are a contract every one of them already satisfies, so one
    transform over the contract is both less code and less to keep in step.

    The threshold moves into a ``title``. It answers a different question --
    *why is this flagged* rather than *what is it* -- and putting both on the
    face turns the chip into a sentence.

    Args:
        flags_html: A rendered quality-flag list.

    Returns:
        The same markup with each chip's value on its face and its threshold in
        a title. A chip carrying no value is returned untouched.
    """

    def rewrite(match: re.Match[str]) -> str:
        severity = match.group("severity")
        attrs = match.group("attrs")
        label = match.group("label").strip()
        found = {m.group("name"): m.group("value") for m in _ATTR.finditer(attrs)}
        value = (found.get("data-value") or "").strip()
        threshold = (found.get("data-threshold") or "").strip()
        if not value or not label:
            return match.group(0)

        title = f' title="threshold: {_html.escape(threshold)}"' if threshold else ""
        # The value leads: it is the fact, and the label says what the fact is
        # about. `48.7% has negatives` reads; `Has negatives 48.7%` does not.
        face = f"{value} {label[0].lower() + label[1:]}"
        return f'<li class="flag{severity}"{attrs}{title}>{_html.escape(face)}</li>'

    return _CHIP.sub(rewrite, flags_html)


def extract_chips(card_html: str) -> list[tuple[str, str]]:
    """Read the quality chips out of a rendered card.

    Args:
        card_html: A rendered ``<article class="var-card">`` fragment.

    Returns:
        (severity, label) pairs in the order the card emitted them. Severity is
        one of ``bad``, ``warn``, ``good`` or ``""``.
    """
    chips: list[tuple[str, str]] = []
    for match in _CHIP.finditer(card_html):
        severity = match.group("severity").strip()
        label = _html.unescape(match.group("label")).strip()
        if label:
            chips.append((severity, label))
    return chips


def actionable_chips(chips: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the chips that mean a column needs attention.

    Args:
        chips: (severity, label) pairs from :func:`extract_chips`.

    Returns:
        The subset a reader should act on: everything the card marked ``bad``,
        plus the ``warn`` chips that describe a defect rather than a shape.
    """
    return [
        (sev, label)
        for sev, label in chips
        if sev == "bad" or (sev == "warn" and flag_slug(label) in _ACTIONABLE_WARNINGS)
    ]


def flag_slug(label: str) -> str:
    """A stable attribute-safe token for a chip label.

    Labels carry non-breaking hyphens and other typography, so they cannot be
    used directly in a ``data-`` attribute or a CSS selector.
    """
    normalised = label.replace("‑", "-").replace("‐", "-")
    return re.sub(r"[^a-z0-9]+", "-", normalised.lower()).strip("-")


def build_attention_block(columns: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    """Build the "needs attention" summary.

    Args:
        columns: ``(column_name, card_id, chips)`` for every column, in report
            order. Chips are the full list; the actionable ones are selected
            here so callers do not have to know the rule.

    Returns:
        HTML for the block, or an empty string when nothing needs attention --
        an empty "0 of 60 columns have issues" banner is noise.
    """
    flagged = [
        (name, card_id, actionable_chips(chips))
        for name, card_id, chips in columns
        if actionable_chips(chips)
    ]
    if not flagged:
        return ""

    # Worst first: a column with a `bad` chip outranks one with only warnings,
    # and among equals the one with more findings.
    def rank(entry: tuple[str, str, list[tuple[str, str]]]) -> tuple[int, int]:
        _, _, chips = entry
        worst = min(_SEVERITY_RANK.get(sev, 9) for sev, _ in chips)
        return (worst, -len(chips))

    flagged.sort(key=rank)

    items = []
    for name, card_id, chips in flagged:
        chip_html = "".join(
            f'<span class="flag {sev}" data-flag="{flag_slug(label)}">'
            f"{_html.escape(label)}</span>"
            for sev, label in chips
        )
        items.append(
            f'<li class="attention-item">'
            f'<a class="attention-col" href="#{card_id}">{_html.escape(name)}</a>'
            f'<span class="attention-flags">{chip_html}</span>'
            f"</li>"
        )

    return f"""
          <section class="needs-attention" id="needs-attention">
            <h3 class="attention-title">
              <strong>{len(flagged)}</strong> of {len(columns)} columns need a look
            </h3>
            <ul class="attention-list">{"".join(items)}</ul>
            <p class="muted small attention-hint">
              Click a column to jump to its card, or a chip to filter the list below.
            </p>
          </section>
    """
