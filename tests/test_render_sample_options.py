"""`include_sample` and `sample_rows`, which for four releases did nothing.

Both were documented against `config.render` -- one of them as
``config.render.include_sample = False  # No PII in reports``, inside a recipe
headed *Production Data Quality Checks* -- while `RenderOptions` carried exactly
two fields, `title` and `description`. It is a plain dataclass with no slots, so
the assignment succeeded, was discarded, and the sample rows rendered anyway
(#266).

Two things had to be true for that to be a silent no-op rather than an
`AttributeError`, and this file pins both:

1. the fields exist on `RenderOptions` and reach the engine, and
2. **something reads `include_sample`.** It was already a field on
   `EngineConfig` that nothing consumed, so carrying the value across the
   boundary would have changed no output at all. The guard lives in
   `sample_section_html` on each adapter, which is why the pandas and polars
   paths are tested separately rather than through one of them.

What this file deliberately does **not** assert is that a value is absent from
the whole document. The first draft did, on the strength of documentation
claiming the sample was "the only place raw values appear", and it failed --
the categorical card prints *Shortest seen* and *Longest seen* verbatim, and the
top-value labels and the numeric and datetime extremes are raw too. That claim
was wrong and is now corrected (#285). Asserting it here would have put the same
overclaim in a second place and called it a guarantee.

So the scope is what the switch is named after: the sample table.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from pysuricata import ProfileConfig, RenderOptions, profile

polars = pytest.importorskip("polars", reason="polars adapter not installed")


SECRET = "alice@example.invalid"


def _frame() -> pd.DataFrame:
    """Wider than the default 10-row sample, so `sample_rows` has room to vary."""
    return pd.DataFrame(
        {
            "email": [f"user{i}@example.invalid" for i in range(29)] + [SECRET],
            "amount": [float(i) for i in range(30)],
        }
    )


def _small_frame() -> pd.DataFrame:
    """Fewer rows than the sample shows, so every value is guaranteed on the
    page. Which rows a 10-row sample picks out of 30 is the sampler's business
    and differs between the two adapters -- asserting on a particular value
    would be testing that, not `include_sample`."""
    return pd.DataFrame({"email": [SECRET, "b@example.invalid"], "amount": [1.0, 2.0]})


def _config(**render) -> ProfileConfig:
    """`config=` is the full escape hatch and cannot be combined with `seed=`,
    so the seed goes inside it."""
    cfg = ProfileConfig()
    cfg.compute.random_seed = 0
    for name, value in render.items():
        setattr(cfg.render, name, value)
    return cfg


class TestTheFieldsExist:
    """`dir()` on a populated instance cannot tell a declared field from one an
    assignment invented, which is how this went unnoticed. `fields()` can."""

    @pytest.mark.parametrize(
        ("name", "default"), [("include_sample", True), ("sample_rows", 10)]
    )
    def test_declared_on_render_options(self, name, default):
        declared = {f.name: f.default for f in dataclasses.fields(RenderOptions)}
        assert name in declared
        assert declared[name] == default

    def test_the_default_still_renders_a_sample(self):
        assert "<table" in profile(_frame()).html


def _sample_table(html: str) -> str:
    """The sample section only. Every other card has tables of its own, so a
    bare `"<table" in html` says nothing about this one."""
    start = html.find('id="dataset-sample"')
    if start < 0:
        return ""
    end = html.find("</section>", start)
    return html[start : end if end > 0 else len(html)]


class TestIncludeSample:
    def test_off_removes_the_sample_table(self):
        html = profile(_frame(), config=_config(include_sample=False)).html
        assert _sample_table(html) == ""

    def test_on_keeps_it(self):
        html = profile(_small_frame(), config=_config(include_sample=True)).html
        assert SECRET in _sample_table(html)

    def test_it_is_not_a_redaction_switch(self):
        """Pinning the limitation, so nobody re-derives the claim #285 removed.

        If a future change genuinely does redact the cards, this failing is the
        signal to update the warning in `configuration.md` -- which is the
        outcome worth having either way."""
        html = profile(_small_frame(), config=_config(include_sample=False)).html
        assert SECRET in html, (
            "raw values still reach the cards; if this no longer holds, the "
            "warning in docs/configuration.md needs revisiting"
        )

    def test_off_is_smaller_than_on(self):
        off = profile(_frame(), config=_config(include_sample=False)).html
        on = profile(_frame(), config=_config(include_sample=True)).html
        assert len(off) < len(on)

    def test_the_statistics_are_untouched(self):
        """Withholding the sample withholds *rows*, not facts. A privacy switch
        that also changed the numbers would be a different feature."""
        frame = _frame()
        with_sample = profile(frame, config=_config(include_sample=True))
        without = profile(frame, config=_config(include_sample=False))
        assert with_sample.stats["columns"] == without.stats["columns"]
        assert with_sample.stats["dataset"] == without.stats["dataset"]

    def test_polars_honours_it_too(self):
        """The guard is per adapter, so one passing says nothing about the other."""
        frame = polars.DataFrame(_small_frame())
        off = profile(frame, config=_config(include_sample=False)).html
        on = profile(frame, config=_config(include_sample=True)).html
        assert _sample_table(off) == ""
        assert SECRET in _sample_table(on)


class TestSampleRows:
    @pytest.mark.parametrize("adapter", ["pandas", "polars"])
    def test_fewer_rows_is_a_smaller_report(self, adapter):
        frame = _frame() if adapter == "pandas" else polars.DataFrame(_frame())
        few = profile(frame, config=_config(sample_rows=2)).html
        many = profile(frame, config=_config(sample_rows=20)).html
        assert len(few) < len(many)

    def test_ignored_when_the_sample_is_off(self):
        cfg = _config(include_sample=False, sample_rows=20)
        assert _sample_table(profile(_frame(), config=cfg).html) == ""
