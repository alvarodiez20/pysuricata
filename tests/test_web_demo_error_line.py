"""`errLine` in `web/worker.js` reports the exception, not the word "Traceback".

Every failure the demo logs from `runPythonAsync` arrives as a **Python**
traceback, whose first line is always the literal `Traceback (most recent call
last):`. `errLine` took that first line, so the log read:

    pysuricata==0.2.0 would not install here (Traceback (most recent call last):)

which is what a visitor saw on the day 0.2.0 was published. The cause was a
PyPI CDN window -- the page's own version check bypasses the browser cache and
micropip's does not, so for about a quarter of an hour the demo could see a
release micropip could not -- and the underlying `ValueError: Can't find a
pure Python 3 wheel for 'pysuricata==0.2.0'` said so plainly. The message that
would have explained it was thrown away by the line that formats it.

The function is small and pure, so it is exercised directly rather than
through a browser: the definition is lifted out of `worker.js` and run under
node. Skipped where node is absent; GitHub's runners have it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "web" / "worker.js"

#: The real micropip failure, as Pyodide surfaces it -- including the bare
#: `See:` hint after the exception, which is why "the last line" is not the
#: rule and the type name is matched instead.
MICROPIP_TRACEBACK = """Traceback (most recent call last):
  File "/lib/python3.13/asyncio/futures.py", line 202, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/lib/python3.13/site-packages/micropip/_commands/install.py", line 143, in install
    raise ValueError(msg)
ValueError: Can't find a pure Python 3 wheel for 'pysuricata==0.2.0'.
See: https://pyodide.org/en/stable/usage/faq.html"""

CASES = [
    pytest.param(
        MICROPIP_TRACEBACK,
        "ValueError: Can't find a pure Python 3 wheel for 'pysuricata==0.2.0'.",
        id="the micropip failure that motivated this",
    ),
    pytest.param(
        'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\n'
        "micropip._utils.PackageNotFound: no such package",
        "micropip._utils.PackageNotFound: no such package",
        id="a dotted exception type",
    ),
    pytest.param(
        "PyPI answered 503",
        "PyPI answered 503",
        id="a plain JS error is unchanged",
    ),
    pytest.param(
        "NetworkError: failed\n  at fetch (worker.js:1)",
        "NetworkError: failed",
        id="a multi-line JS error still takes its first line",
    ),
    pytest.param(
        'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\n'
        "something went wrong",
        "something went wrong",
        id="no typed exception falls back to the last line",
    ),
]


def _extract() -> str:
    """`EXCEPTION_LINE` and `errLine`, lifted verbatim from `worker.js`.

    Reading the shipped source rather than a copy is the whole point: a copy
    would keep passing after the original changed, which is the failure mode
    `tests/test_js_selectors_match_markup.py` exists to prevent for selectors.
    """
    source = WORKER.read_text(encoding="utf-8")
    start = source.index("const EXCEPTION_LINE")
    end = source.index("};", source.index("const errLine")) + len("};")
    return source[start:end]


@pytest.fixture(scope="module")
def run_err_line():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; `errLine` is JavaScript")

    definition = _extract()

    def call(message: str) -> str:
        script = (
            f"{definition}\n"
            f"const input = JSON.parse(process.argv[1]);\n"
            f"process.stdout.write(errLine(new Error(input)));\n"
        )
        result = subprocess.run(
            [node, "-e", script, json.dumps(message)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    return call


@pytest.mark.parametrize(("message", "expected"), CASES)
def test_it_reports_the_line_that_says_something(run_err_line, message, expected):
    assert run_err_line(message) == expected


def test_it_never_returns_the_traceback_header(run_err_line):
    """The specific regression: whatever else it does, the one line that is
    guaranteed to carry no information must not be what gets logged."""
    assert "Traceback (most recent call last)" not in run_err_line(MICROPIP_TRACEBACK)


def test_the_definition_is_still_where_the_test_looks_for_it():
    """If `errLine` is renamed or restructured, `_extract` would raise inside a
    fixture and every case above would error with a confusing message. Fail
    here instead, with one that says what happened."""
    try:
        definition = _extract()
    except ValueError as exc:  # `str.index` on a missing marker
        pytest.fail(f"could not find errLine in {WORKER.name}: {exc}")

    assert "errLine" in definition
    assert "EXCEPTION_LINE" in definition
