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

from .flag_reference import FLAG_MEANINGS, raised_flags

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

#: The identity :func:`annotate_flags` stamps on, read back by
#: :func:`extract_chips`.
_FLAG_ATTR = re.compile(r'data-flag="([^"]*)"')


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

        # The identity of the flag, stamped before the face is rewritten and
        # taken from the label as it was *emitted*. Everything downstream --
        # triage, the card's data-flags, the chip filter -- has to ask "which
        # flag is this", and the face stops being able to answer the moment the
        # value is prepended to it: `Missing` slugs to `missing`, and
        # `19.9% missing` slugs to `19-9-missing`, which matches nothing and is
        # unique per column into the bargain. See #238.
        slug = flag_slug(label)
        identity = f' data-flag="{slug}"'
        # The value leads: it is the fact, and the label says what the fact is
        # about. `48.7% has negatives` reads; `Has negatives 48.7%` does not.
        face = f"{value} {label[0].lower() + label[1:]}"

        # The limit goes on the face, not in a `title` (phase 4b.2). A tooltip
        # is invisible on a phone and absent from a printed report, so `33.20`
        # had nothing to be judged against in either -- and the reader who
        # cannot hover is the one with the least context, not the most.
        #
        # Preferred from the reference, which states it once per flag, so the
        # face reads the same wherever a flag is raised. The renderer's own
        # `data-threshold` is the fallback: it is per-site and worded forty-two
        # different ways, but a flag with no entry should still say its limit.
        # Not on a `good` chip. A limit reads as "this is how close you are to
        # a problem", and a good chip is not near one -- `48.7% positive-only ·
        # limit 0` invites a judgement where the card is reporting a property.
        meaning = FLAG_MEANINGS.get(slug)
        limit = meaning.limit if meaning else threshold
        if limit and limit != "any" and "good" not in severity:
            face = f"{face} · limit {limit}"

        # And no `title`. A tooltip is the thing 4b.2 exists to get rid of --
        # invisible on a phone, absent from a printed report, and read by the
        # reader who needs it least. What the number *is* now lives in the flag
        # reference, once per flag, where it can be read on any device and on
        # paper. Repeating it on all 154 chips of a Titanic report cost 5,548
        # bytes to say fourteen distinct things.
        return f'<li class="flag{severity}"{attrs}{identity}>{_html.escape(face)}</li>'

    return _CHIP.sub(rewrite, flags_html)


def extract_chips(card_html: str) -> list[tuple[str, str, str]]:
    """Read the quality chips out of a rendered card.

    Args:
        card_html: A rendered ``<article class="var-card">`` fragment.

    Returns:
        ``(severity, label, slug)`` in the order the card emitted them.
        Severity is one of ``bad``, ``warn``, ``good`` or ``""``. The label is
        what the chip displays; the slug is what it *is*, and the two stopped
        agreeing once :func:`annotate_flags` began putting the value on the
        face. The slug is read from ``data-flag`` when the chip carries one and
        derived from the label otherwise, so markup that never passed through
        :func:`annotate_flags` still behaves.
    """
    chips: list[tuple[str, str, str]] = []
    for match in _CHIP.finditer(card_html):
        severity = match.group("severity").strip()
        label = _html.unescape(match.group("label")).strip()
        if not label:
            continue
        stamped = _FLAG_ATTR.search(match.group("attrs"))
        slug = stamped.group(1) if stamped else flag_slug(label)
        chips.append((severity, label, slug))
    return chips


def actionable_chips(chips: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Keep only the chips that mean a column needs attention.

    Args:
        chips: ``(severity, label, slug)`` triples from :func:`extract_chips`.

    Returns:
        The subset a reader should act on: everything the card marked ``bad``,
        plus the ``warn`` chips that describe a defect rather than a shape.

    The membership test is on the slug, never the label. It used to be on the
    label, and since every chip's label carries its value the set matched
    nothing at all -- eleven entries of dead configuration, and an attention
    block that was ``bad``-only without saying so.
    """
    return [
        (sev, label, slug)
        for sev, label, slug in chips
        if sev == "bad" or (sev == "warn" and slug in _ACTIONABLE_WARNINGS)
    ]


def flag_slug(label: str) -> str:
    """A stable attribute-safe token for a chip label.

    Labels carry non-breaking hyphens and other typography, so they cannot be
    used directly in a ``data-`` attribute or a CSS selector.
    """
    normalised = label.replace("‑", "-").replace("‐", "-")
    return re.sub(r"[^a-z0-9]+", "-", normalised.lower()).strip("-")


def build_attention_block(
    columns: list[tuple[str, str, list[tuple[str, str, str]]]],
) -> str:
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
    def rank(entry: tuple[str, str, list[tuple[str, str, str]]]) -> tuple[int, int]:
        _, _, chips = entry
        worst = min(_SEVERITY_RANK.get(sev, 9) for sev, _, _ in chips)
        return (worst, -len(chips))

    flagged.sort(key=rank)

    items = []
    for name, card_id, chips in flagged:
        # The stamped slug, not one derived from the face. The card's
        # `data-flags` is built from the same slugs, so clicking a chip here
        # selects every column sharing that defect -- which it could not do
        # while both sides carried the value, since `77-1-missing` is unique to
        # the one column that happens to be 77.1% missing.
        chip_html = "".join(
            f'<span class="flag {sev}" data-flag="{slug}">{_html.escape(label)}</span>'
            for sev, label, slug in chips
        )
        items.append(
            f'<li class="attention-item">'
            f'<a class="attention-col" href="#{card_id}">{_html.escape(name)}</a>'
            f'<span class="attention-flags">{chip_html}</span>'
            f"</li>"
        )

    raised = [slug for _, _, chips in flagged for _, _, slug in chips]
    return f"""
          <section class="needs-attention" id="needs-attention">
            <h3 class="attention-title">
              <strong>{len(flagged)}</strong> of {len(columns)} columns need a look
            </h3>
            <ul class="attention-list">{"".join(items)}</ul>
            {_flag_reference(raised)}
          </section>
    """


def _flag_reference(raised: list[str]) -> str:
    """What the flags mean, once per report, for the flags it actually raised.

    Design 15b. The chips name a conclusion -- `heavy-tailed`, `dominant
    category` -- and the vocabulary is only decodable if it is written down
    somewhere. Four columns: the flag, what was measured, the limit that fired
    it, and what it means for the data.

    Rendered from the flags raised rather than from the whole table, so it is
    six rows on Titanic and **nothing at all on a clean frame**. A flag with no
    entry is dropped rather than rendered blank: a new flag in the renderers
    must not put an empty row here.

    Deliberately no advice. Every sentence states a consequence for the data
    and stops -- "drop before modelling" is wrong for a reader who is not
    modelling, and whether pysuricata should recommend actions at all is open
    question 7 of the design package rather than something to settle here.

    It also replaces the hint that used to close this block. "Click a column to
    jump to its card" told the reader what to do instead of showing it, and was
    untrue on paper.
    """
    rows = raised_flags(raised)
    if not rows:
        return ""
    body = "".join(
        f'<tr id="flagref-{slug}">'
        f'<th scope="row"><span class="flag {slug}">{_html.escape(slug.replace("-", " "))}</span></th>'
        f'<td data-label="Measures">{_html.escape(meaning.measure)}</td>'
        f'<td class="flagref__limit" data-label="Fires above">'
        f"{_html.escape(meaning.limit)}</td>"
        f'<td data-label="Means">{_html.escape(meaning.means)}</td>'
        "</tr>"
        for slug, meaning in rows
    )
    return f"""
            <details class="flagref">
              <summary>What these flags mean</summary>
              <table class="flagref__table">
                <caption class="micro-label">
                  Only the {len(rows)} flag{"" if len(rows) == 1 else "s"} this report raised
                </caption>
                <thead>
                  <tr><th scope="col">Flag</th><th scope="col">What is measured</th>
                  <th scope="col">Fires above</th><th scope="col">What it means</th></tr>
                </thead>
                <tbody>{body}</tbody>
              </table>
            </details>"""
