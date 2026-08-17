"""Per-column chunk tracking, shared by the accumulators that need it.

#154's 5b.7 gates the report's Missing Values pane on **missing > 0 and chunks
> 1** — the only condition under which the pane knows something the card face
does not, namely *where in the read* the gaps fall. With one chunk it restates
a percentage the header already carries.

That rule could only land for numeric and datetime, because categorical and
boolean were finalized without chunk metadata at all (#193). The trap, which
the first attempt fell into: `getattr(stats, "chunk_metadata", None)` returns
`None` rather than raising, so applying the gate to a kind that has no such
field *looks* like it works. It does not tighten the rule — it hides the pane
permanently on those kinds.

This is the numeric accumulator's machinery with the column-specific parts
lifted out, so the two kinds that lacked it get the same behaviour rather than
a second interpretation of it. `NumericAccumulator` predates this and keeps its
own copy; the semantics here are written to match it, including the `max_chunks`
escape hatch that stops an unbounded read from growing an unbounded list.
"""

from __future__ import annotations


class ChunkTracker:
    """Counts rows and missing values per chunk, within a bound.

    The owner calls `note(rows, missing)` as it consumes values and
    `mark_boundary()` when a chunk ends; `metadata()` renders the result as the
    `(start_row, end_row, missing_in_chunk)` triples the report reads.

    Bounded by construction: past `max_chunks` it stops recording and reports
    nothing further, which is the same trade the numeric accumulator makes —
    a report that draws a thousand segments is not readable anyway, and the
    alternative is a list that grows with the input.
    """

    __slots__ = (
        "_boundaries",
        "_current_missing",
        "_current_rows",
        "_enabled",
        "_max_chunks",
        "_missing",
        "_seen",
    )

    def __init__(self, *, enabled: bool = True, max_chunks: int = 1000) -> None:
        self._enabled = bool(enabled)
        self._max_chunks = int(max_chunks)
        self._boundaries: list[int] = []
        self._missing: list[int] = []
        self._current_rows = 0
        self._current_missing = 0
        #: Cumulative rows the owner has handed over, missing included. Kept
        #: here rather than read back off the owner so the tracker does not
        #: need to know what a "count" means on that accumulator.
        self._seen = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def note(self, rows: int, missing: int) -> None:
        """Record a batch of values inside the current chunk."""
        if rows <= 0 and missing <= 0:
            return
        self._seen += int(rows)
        self._current_rows += int(rows)
        self._current_missing += int(missing)

    def mark_boundary(self) -> None:
        """Close the current chunk.

        A boundary with no rows behind it is not a chunk. The engine marks
        after every chunk *and* `finalize()` flushes a pending one, so without
        this an uninterrupted run would record a trailing empty segment.
        """
        if not self._enabled or self._current_rows == 0:
            self._current_rows = 0
            self._current_missing = 0
            return

        if len(self._boundaries) >= self._max_chunks:
            # Past the bound, stop tracking rather than grow without limit.
            self._enabled = False
            self._current_rows = 0
            self._current_missing = 0
            return

        self._boundaries.append(self._seen)
        self._missing.append(self._current_missing)
        self._current_rows = 0
        self._current_missing = 0

    def metadata(self) -> list[tuple[int, int, int]]:
        """`(start_row, end_row, missing_in_chunk)` per chunk, flushing any
        chunk still open."""
        if self._current_rows > 0:
            self.mark_boundary()

        out: list[tuple[int, int, int]] = []
        start = 0
        for index, end_cumulative in enumerate(self._boundaries):
            out.append((start, end_cumulative - 1, self._missing[index]))
            start = end_cumulative
        return out

    def merge(self, other: ChunkTracker) -> None:
        """Append another tracker's chunks after this one's.

        Accumulators must be mergeable, and a merged column's chunks are the
        two runs' chunks in order. Offsetting `other`'s cumulative boundaries by
        this one's row count is what keeps the triples contiguous rather than
        restarting at zero halfway through.
        """
        if not other._boundaries and other._current_rows == 0:
            return
        if not self._enabled or not other._enabled:
            self._enabled = False
            return

        offset = self._seen
        self._boundaries.extend(b + offset for b in other._boundaries)
        self._missing.extend(other._missing)
        self._seen += other._seen
        self._current_rows += other._current_rows
        self._current_missing += other._current_missing

    def reset(self) -> None:
        self._boundaries.clear()
        self._missing.clear()
        self._seen = 0
        self._current_rows = 0
        self._current_missing = 0
