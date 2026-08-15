"""Progress reporting on long runs.

UX-7. A 1.8-million-cell profile produced 46 bytes of output, none of it
progress: for the use case this library is positioned on, a hung process and a
working one looked identical.

The constraint that matters most is the first one below. A profile written to a
pipe has to stay parseable, so progress goes to stderr and nowhere else.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from pysuricata import ConfigurationError, summarize
from pysuricata.progress import (
    _CallbackProgress,
    _compact,
    _NullProgress,
    _StderrProgress,
    resolve,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": np.arange(50_000.0), "b": rng.standard_normal(50_000)})


class TestResolve:
    def test_false_is_silent(self):
        assert isinstance(resolve(False), _NullProgress)

    def test_none_is_silent(self):
        assert isinstance(resolve(None), _NullProgress)

    def test_true_reports(self):
        assert isinstance(resolve(True), _StderrProgress)

    def test_a_callable_is_forwarded_to(self):
        assert isinstance(resolve(lambda **kw: None), _CallbackProgress)

    def test_auto_follows_the_terminal(self, monkeypatch):
        monkeypatch.setattr("pysuricata.progress._stderr_is_a_terminal", lambda: True)
        assert isinstance(resolve("auto"), _StderrProgress)
        monkeypatch.setattr("pysuricata.progress._stderr_is_a_terminal", lambda: False)
        assert isinstance(resolve("auto"), _NullProgress)

    def test_anything_else_is_refused(self):
        with pytest.raises(ValueError, match="'auto'"):
            resolve("loud")


class TestNothingReachesStdout:
    """The one constraint that cannot be relaxed."""

    @pytest.mark.parametrize("mode", [True, "auto", False, None])
    def test_stdout_stays_empty(self, frame, mode, monkeypatch):
        monkeypatch.setattr("pysuricata.progress._stderr_is_a_terminal", lambda: True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            summarize(frame, chunk_size=5_000, progress=mode)
        assert out.getvalue() == ""


class TestStderrReporting:
    def test_progress_true_writes_to_stderr(self, frame):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(frame, chunk_size=5_000, progress=True)
        assert "profiled" in err.getvalue()

    def test_the_summary_line_names_the_cell_count(self, frame):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(frame, chunk_size=5_000, progress=True)
        assert "100K cells" in err.getvalue()

    def test_progress_false_writes_nothing(self, frame):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(frame, chunk_size=5_000, progress=False)
        assert err.getvalue() == ""

    def test_auto_is_silent_when_piped(self, frame, monkeypatch):
        monkeypatch.setattr("pysuricata.progress._stderr_is_a_terminal", lambda: False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(frame, chunk_size=5_000, progress="auto")
        assert err.getvalue() == ""

    def test_auto_reports_on_a_terminal(self, frame, monkeypatch):
        monkeypatch.setattr("pysuricata.progress._stderr_is_a_terminal", lambda: True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(frame, chunk_size=5_000, progress="auto")
        assert "profiled" in err.getvalue()

    def test_a_stream_that_cannot_be_written_to_does_not_break_the_profile(self):
        """Progress must never be the reason a profile fails."""

        class Broken:
            def write(self, _text):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        reporter = _StderrProgress(stream=Broken())
        reporter.start(100)
        reporter.advance(1, 10)
        reporter.finish(1, 10, 20)


class TestCallback:
    def test_the_callback_receives_events(self, frame):
        seen: list[dict] = []
        summarize(frame, chunk_size=5_000, progress=lambda **kw: seen.append(kw))
        assert len(seen) > 1

    def test_the_event_carries_chunks_rows_and_elapsed(self, frame):
        seen: list[dict] = []
        summarize(frame, chunk_size=5_000, progress=lambda **kw: seen.append(kw))
        assert set(seen[-1]) == {"chunks", "rows", "elapsed"}

    def test_rows_reach_the_full_count(self, frame):
        seen: list[dict] = []
        summarize(frame, chunk_size=5_000, progress=lambda **kw: seen.append(kw))
        assert seen[-1]["rows"] == 50_000

    def test_the_callback_is_not_rate_limited(self, frame):
        """The stderr line is throttled to stay readable; a program is not."""
        seen: list[dict] = []
        summarize(frame, chunk_size=5_000, progress=lambda **kw: seen.append(kw))
        assert seen[-1]["chunks"] == 10


class TestGeneratorSources:
    """UX-7 acceptance: works where the total is unknown."""

    def test_a_generator_source_still_reports(self):
        rng = np.random.default_rng(0)

        def chunks():
            for _ in range(4):
                yield pd.DataFrame({"a": rng.standard_normal(5_000)})

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            summarize(chunks(), progress=True)
        assert "profiled" in err.getvalue()

    def test_no_eta_is_invented_without_a_total(self):
        reporter = _StderrProgress(stream=io.StringIO())
        reporter.start(None)
        reporter._last_draw = -1e9  # bypass the redraw throttle
        reporter.advance(2, 1_000)
        assert "left" not in reporter._stream.getvalue()

    def test_an_eta_appears_when_the_total_is_known(self):
        reporter = _StderrProgress(stream=io.StringIO())
        reporter.start(10_000)
        reporter._started -= 1.0  # pretend a second has passed
        reporter._last_draw = -1e9
        reporter.advance(2, 1_000)
        assert "left" in reporter._stream.getvalue()


class TestValidation:
    def test_a_bad_value_fails_at_the_public_boundary(self, frame):
        """_to_engine_config swallows errors in a bare except, so a value that
        only fails deeper in becomes a silently different config."""
        with pytest.raises(ConfigurationError, match="progress must be"):
            summarize(frame, progress="loud")

    def test_the_error_is_still_a_valueerror(self, frame):
        with pytest.raises(ValueError):
            summarize(frame, progress=3.5)


class TestCompactFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [(942, "942"), (1_500, "2K"), (1_800_000, "1.8M"), (2_500_000_000, "2.5B")],
    )
    def test_counts_read_at_a_glance(self, value, expected):
        assert _compact(value) == expected
