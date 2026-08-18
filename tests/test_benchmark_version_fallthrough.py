"""`versions.py` refuses when a subprocess measures the wrong code (#249).

A round-robin once produced a beautifully flat table -- four versions, no
difference -- because one directory in the comparison was a virtualenv
rather than a source checkout, `sys.path` fell through to the next entry,
and every "version" silently measured the same code. `pysuricata.__version__`
could not have caught it: it resolves through `importlib.metadata`, which
reports the *installed* distribution regardless of what actually imported.
Only `pysuricata.__file__` -- checked against the venv the run meant to
measure -- tells the truth, which is what `RUNNER` now does before timing
anything.

This is worse than an ordinary flake. Every other measurement hazard here is
noisy (contention, drift) and the interleaving discipline is built to cancel
it. A fall-through is silent and systematic: it corrupts every round
identically, so the result carries no spread to flag it as suspect.
"""

from __future__ import annotations

import json
import shutil
import sys
from types import SimpleNamespace

import pytest

from benchmarks import versions


def _fake_stdout(**payload) -> SimpleNamespace:
    """A `subprocess.run` result whose stdout carries one `__RESULT__` line,
    matching the shape `time_once` parses out of a real subprocess."""
    return SimpleNamespace(stdout=f"__RESULT__{json.dumps(payload)}", stderr="")


class TestTimeOnceRaisesOnAFallthroughResult:
    """The parsing/raising contract, isolated from spawning a real subprocess."""

    def test_a_path_fallthrough_status_raises(self, monkeypatch):
        monkeypatch.setattr(
            versions.subprocess,
            "run",
            lambda *a, **k: _fake_stdout(
                status="path_fallthrough",
                wanted="/envs/v0.1.0",
                got="/repo/pysuricata/__init__.py",
            ),
        )

        with pytest.raises(versions.VersionPathFallthrough, match="/envs/v0.1.0"):
            versions.time_once(sys.executable, "/envs/v0.1.0", "mixed", 0.01, 30)

    def test_the_message_names_both_paths(self, monkeypatch):
        monkeypatch.setattr(
            versions.subprocess,
            "run",
            lambda *a, **k: _fake_stdout(
                status="path_fallthrough",
                wanted="/envs/v0.1.0",
                got="/repo/pysuricata/__init__.py",
            ),
        )

        with pytest.raises(versions.VersionPathFallthrough) as excinfo:
            versions.time_once(sys.executable, "/envs/v0.1.0", "mixed", 0.01, 30)

        assert "/envs/v0.1.0" in str(excinfo.value)
        assert "/repo/pysuricata/__init__.py" in str(excinfo.value)

    def test_an_ok_result_still_passes_through_untouched(self, monkeypatch):
        monkeypatch.setattr(
            versions.subprocess,
            "run",
            lambda *a, **k: _fake_stdout(seconds=1.5, version="0.1.0"),
        )

        result = versions.time_once(sys.executable, "/envs/v0.1.0", "mixed", 0.01, 30)

        assert result == {"seconds": 1.5, "version": "0.1.0", "status": "ok"}


class TestRoundRobinPropagatesTheRefusal:
    """The whole run stops rather than gathering more results that are just
    as untrustworthy as the one that tripped the check."""

    def test_a_fallthrough_on_the_second_version_aborts_the_whole_round(
        self, monkeypatch
    ):
        calls = []

        def fake_time_once(python, env_dir, suite, scale, timeout):
            calls.append(python)
            if python == "bad":
                raise versions.VersionPathFallthrough("wanted x, got y")
            return {"status": "ok", "seconds": 1.0}

        monkeypatch.setattr(versions, "time_once", fake_time_once)

        with pytest.raises(versions.VersionPathFallthrough):
            versions.round_robin(
                {"a": ("good", "/envs/a"), "b": ("bad", "/envs/b")},
                "mixed",
                0.01,
                rounds=3,
                timeout=30,
            )

        # Aborted on the first round, at the second (bad) entry -- not after
        # gathering every version's result first.
        assert calls == ["good", "bad"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a venv")
class TestTheRealRunnerScriptCatchesItself:
    """No mocking: a real subprocess, checked against a directory it is not
    installed in. If the assertion in `RUNNER` were removed, this would
    instead time a `NameError` (`summarize` unbound after the early exit) or,
    worse, silently succeed by measuring whatever `pysuricata` the ambient
    `sys.path` happens to resolve -- exactly the failure this guards against.
    """

    def test_a_deliberately_wrong_env_dir_is_refused(self, tmp_path):
        # A fresh, private, empty directory -- not the shared `/tmp` itself,
        # which can carry incidental subdirectories from something else
        # entirely and turn "wrong on purpose" into "wrong for a reason this
        # test did not intend to exercise."
        wrong_dir = tmp_path / "definitely-not-where-python-is-installed"
        wrong_dir.mkdir()

        with pytest.raises(versions.VersionPathFallthrough):
            versions.time_once(sys.executable, str(wrong_dir), "mixed", 0.01, 60)

    def test_a_freshly_installed_venv_measures_its_own_code(self, tmp_path):
        """The positive case, built the same way `make_env` does: a real
        venv, a real (non-editable) install of this checkout, and a
        `pysuricata.__file__` that actually lands inside it."""
        env_dir = tmp_path / "v."
        created = versions.subprocess.run(
            ["uv", "venv", str(env_dir)], capture_output=True, text=True
        )
        assert created.returncode == 0, created.stderr

        python = str(env_dir / "bin" / "python")
        installed = versions.subprocess.run(
            ["uv", "pip", "install", "--python", python, "--quiet", versions.REPO],
            capture_output=True,
            text=True,
        )
        assert installed.returncode == 0, installed.stderr

        result = versions.time_once(python, str(env_dir), "mixed", 0.02, 120)

        assert result["status"] == "ok"
        assert "version" in result
