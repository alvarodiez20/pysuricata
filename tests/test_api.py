import io
import json
import os
from typing import Iterable

import pandas as pd
import pytest

from pysuricata.api import (
    ComputeOptions,
    Report,
    ReportConfig,
    _coerce_input,  # type: ignore
    _to_engine_config,  # type: ignore
    profile,
    summarize,
)


def test_report_save_and_repr(tmp_path):
    rep = Report(html="<html>ok</html>", stats={"dataset": {"rows": 3}})

    # save_html
    html_path = tmp_path / "out.html"
    rep.save_html(str(html_path))
    assert html_path.exists()
    assert html_path.read_text(encoding="utf-8").startswith("<html>")

    # save_json
    json_path = tmp_path / "out.json"
    rep.save_json(str(json_path))
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "dataset" in data

    # save() by extension
    html2 = tmp_path / "out2.html"
    rep.save(str(html2))
    assert html2.exists()

    json2 = tmp_path / "out2.json"
    rep.save(str(json2))
    assert json2.exists()

    # invalid extension
    with pytest.raises(ValueError):
        rep.save(str(tmp_path / "out.txt"))

    # notebook repr
    assert rep._repr_html_().startswith("<html>")


def test_compute_options_properties_map_to_engine_names():
    c = ComputeOptions(
        chunk_size=123,
        numeric_sample_size=999,
        max_uniques=777,
        top_k=55,
        random_seed=1,
    )
    assert c.numeric_sample_k == 999
    assert c.uniques_k == 777
    assert c.topk_k == 55


def test__coerce_input_accepts_pandas_and_iterable():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = _coerce_input(df)  # type: ignore
    assert out is df

    def gen() -> Iterable[pd.DataFrame]:
        yield df
        yield df

    out2 = _coerce_input(gen())  # type: ignore
    assert hasattr(out2, "__iter__")


def test__coerce_input_rejects_invalid_types():
    with pytest.raises(TypeError):
        _ = _coerce_input(123)  # type: ignore
    with pytest.raises(TypeError):
        _ = _coerce_input({"a": 1})  # type: ignore


def test__to_engine_config_prefers_from_options(monkeypatch):
    # Ensure from_options path is exercised
    cfg = ReportConfig(
        compute=ComputeOptions(
            chunk_size=42, numeric_sample_size=5, max_uniques=6, top_k=7, random_seed=9
        )
    )

    # direct call should work with current engine
    eng = _to_engine_config(cfg)  # type: ignore
    for name in ("chunk_size", "numeric_sample_k", "uniques_k", "topk_k"):
        assert hasattr(eng, name)


def test__to_engine_config_fallback_without_from_options(monkeypatch):
    # Simulate older engine without from_options
    class DummyEngineCfg:
        def __init__(
            self,
            *,
            chunk_size,
            numeric_sample_k,
            uniques_k,
            topk_k,
            engine,
            random_seed,
        ):
            self.chunk_size = chunk_size
            self.numeric_sample_k = numeric_sample_k
            self.uniques_k = uniques_k
            self.topk_k = topk_k
            self.engine = engine
            self.random_seed = random_seed

    import pysuricata.api as api

    monkeypatch.setattr(api, "_EngineReportConfig", DummyEngineCfg, raising=True)

    cfg = ReportConfig(
        compute=ComputeOptions(
            chunk_size=99,
            numeric_sample_size=11,
            max_uniques=22,
            top_k=33,
            random_seed=123,
        )
    )
    eng = _to_engine_config(cfg)  # type: ignore
    assert isinstance(eng, DummyEngineCfg)
    assert eng.chunk_size == 99
    assert eng.numeric_sample_k == 11
    assert eng.uniques_k == 22
    assert eng.topk_k == 33
    assert eng.random_seed == 123


def test_profile_and_summarize_with_pandas():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, None, 6.0]})
    rep = profile(df, config=ReportConfig())
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)

    stats = summarize(df, config=ReportConfig())
    assert "dataset" in stats and "columns" in stats


def test_profile_with_iterable_pandas():
    chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})]
    rep = profile(iter(chunks), config=ReportConfig())
    assert rep.html and isinstance(rep.html, str)


def test_profile_invalid_type_raises():
    with pytest.raises(TypeError):
        _ = profile(123)  # type: ignore
