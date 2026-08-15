"""Progress reporting for long runs.

Profiling data that does not fit in memory is the use case this library is
positioned on, and it takes minutes. A 1.8-million-cell profile produced 46
bytes of output, none of it progress: a hung process and a working one looked
identical.

`log_every_n_chunks` already existed, but it routes to a logger that is off by
default, so it is invisible unless the caller configures logging first -- which
is not something you think to do while waiting to find out whether anything is
happening.

Design constraints, in order:

* **Never stdout.** A profile written to a pipe must stay parseable. Everything
  here goes to stderr.
* **`"auto"` means "when a human is watching"** -- enabled only when stderr is a
  terminal, so a cron job or a redirect stays quiet without being configured.
* **No ETA unless the total is knowable.** A generator source has no length, and
  an invented estimate is worse than none.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

# Redraw at most this often. A profile can process hundreds of chunks a second,
# and a progress line rewritten that fast is unreadable and costs real time in
# terminal I/O.
_MIN_REDRAW_SECONDS = 0.1


class ProgressCallback(Protocol):
    """A programmatic consumer of progress events."""

    def __call__(self, *, chunks: int, rows: int, elapsed: float) -> None: ...


def resolve(progress: bool | str | Callable[..., Any] | None) -> ProgressReporter:
    """Turn the public ``progress=`` argument into a reporter.

    Args:
        progress: ``True``, ``False``, ``"auto"``, ``None``, or a callable
            taking ``chunks``, ``rows`` and ``elapsed`` as keywords.

    Returns:
        A reporter. The null reporter is a real object rather than None so the
        engine has no branch on the hot path.

    Raises:
        ValueError: If a string other than ``"auto"`` is given.
    """
    if progress is None or progress is False:
        return _NullProgress()
    if callable(progress):
        return _CallbackProgress(progress)
    if progress is True:
        return _StderrProgress()
    if progress == "auto":
        return _StderrProgress() if _stderr_is_a_terminal() else _NullProgress()
    raise ValueError(
        f"progress must be True, False, 'auto' or a callable, not {progress!r}"
    )


def _stderr_is_a_terminal() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        # A stream that cannot answer is not a terminal for our purposes.
        return False


class ProgressReporter:
    """Base class; the null implementation."""

    def start(self, total_rows: int | None = None) -> None:
        """Begin a run. ``total_rows`` is None when the source has no length."""

    def advance(self, chunks: int, rows: int) -> None:
        """Report cumulative progress."""

    def finish(self, chunks: int, rows: int, cells: int) -> None:
        """Report completion."""


class _NullProgress(ProgressReporter):
    """Does nothing, quietly."""


class _CallbackProgress(ProgressReporter):
    """Forwards to a caller-supplied function."""

    def __init__(self, callback: Callable[..., Any]) -> None:
        self._callback = callback
        self._started = 0.0

    def start(self, total_rows: int | None = None) -> None:
        self._started = time.perf_counter()

    def advance(self, chunks: int, rows: int) -> None:
        self._callback(
            chunks=chunks, rows=rows, elapsed=time.perf_counter() - self._started
        )

    def finish(self, chunks: int, rows: int, cells: int) -> None:
        self.advance(chunks, rows)


class _StderrProgress(ProgressReporter):
    """A single rewritten line on stderr, and one summary at the end."""

    def __init__(self, stream: Any | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._started = 0.0
        self._last_draw = 0.0
        self._total_rows: int | None = None
        self._drew_anything = False

    def start(self, total_rows: int | None = None) -> None:
        self._started = time.perf_counter()
        self._last_draw = 0.0
        self._total_rows = total_rows

    def advance(self, chunks: int, rows: int) -> None:
        now = time.perf_counter()
        if now - self._last_draw < _MIN_REDRAW_SECONDS:
            return
        self._last_draw = now
        elapsed = now - self._started
        rate = rows / elapsed if elapsed > 0 else 0.0

        line = f"  {chunks} chunks · {_compact(rows)} rows · {elapsed:.1f}s"
        if rate > 0:
            line += f" · {_compact(int(rate))}/s"
        if self._total_rows:
            share = min(1.0, rows / self._total_rows)
            line += f" · {share:.0%}"
            if rate > 0 and share < 1.0:
                line += f" · ~{(self._total_rows - rows) / rate:.0f}s left"
        self._write(f"\r\033[K{line}")
        self._drew_anything = True

    def finish(self, chunks: int, rows: int, cells: int) -> None:
        elapsed = time.perf_counter() - self._started
        if self._drew_anything:
            self._write("\r\033[K")
        self._write(f"  profiled {_compact(cells)} cells in {elapsed:.1f}s\n")

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            # Progress reporting must never be the reason a profile fails.
            pass


def _compact(value: int) -> str:
    """Format a count as 1.8M, 300K or 942."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)
