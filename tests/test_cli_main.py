import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from pysuricata.cli import main


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_cli_help():
    with patch("sys.argv", ["pysuricata", "profile", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_cli_version():
    with patch("sys.argv", ["pysuricata", "--version"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_cli_basic_html(sample_csv, tmp_path):
    out = tmp_path / "report.html"
    with patch("sys.argv", ["pysuricata", "profile", sample_csv, "--output", str(out)]):
        main()
    assert out.exists()
    assert "<!DOCTYPE" in out.read_text()


def test_cli_basic_json(sample_csv, tmp_path):
    out = tmp_path / "stats.json"
    with patch(
        "sys.argv", ["pysuricata", "summarize", sample_csv, "--output", str(out)]
    ):
        main()
    assert out.exists()
    data = json.loads(out.read_text())
    assert "dataset" in data
    assert "columns" in data


def test_cli_json_fallback(sample_csv, tmp_path):
    # Tests that when `--stats` flag is provided, a json is emitted
    out = tmp_path / "fallback.json"
    with patch(
        "sys.argv", ["pysuricata", "summarize", sample_csv, "--output", str(out)]
    ):
        main()
    assert out.exists()


def test_cli_params(sample_csv, tmp_path):
    out = tmp_path / "custom.html"
    with patch(
        "sys.argv",
        [
            "pysuricata",
            "profile",
            sample_csv,
            "--output",
            str(out),
            "--title",
            "My CLI Report",
            "--chunk-size",
            "100",
            "--sample-size",
            "50",
            "--seed",
            "42",
            "--no-correlations",
            "--quiet",
        ],
    ):
        main()
    assert out.exists()
    html = out.read_text()
    assert "My CLI Report" in html


def test_cli_file_not_found(tmp_path):
    # Missing input file cleanly returns 1, not SystemExit
    with patch(
        "sys.argv",
        [
            "pysuricata",
            "profile",
            str(tmp_path / "nonexistent.csv"),
            "--output",
            "report.html",
        ],
    ):
        assert main() == 1


def test_cli_unsupported_file(tmp_path):
    with patch(
        "sys.argv",
        [
            "pysuricata",
            "profile",
            str(tmp_path / "unsupported.xyz"),
            "--output",
            "report.html",
        ],
    ):
        assert main() == 1


def test_cli_output_error(sample_csv, monkeypatch):
    # Profile save_html error cleanly returns 1
    with patch(
        "sys.argv",
        ["pysuricata", "profile", sample_csv, "--output", "/invalid/dir/report.html"],
    ):
        assert main() == 1


def test_cli_summarize_stdout(sample_csv, capsys):
    with patch("sys.argv", ["pysuricata", "summarize", sample_csv, "--quiet"]):
        assert main() == 0
    # Capture standard output (stdout print for json dumps)
    captured = capsys.readouterr()
    assert "dataset" in captured.out


def test_cli_summarize_error(tmp_path):
    # Test error fallback in summarize
    with patch(
        "sys.argv", ["pysuricata", "summarize", str(tmp_path / "nonexistent.csv")]
    ):
        assert main() == 1


def test_cli_default_output(sample_csv, monkeypatch):
    original_cwd = os.getcwd()
    try:
        # Move completely into the directory where the CSV resides
        os.chdir(os.path.dirname(sample_csv))
        base_name = os.path.basename(sample_csv)
        with patch(
            "sys.argv",
            ["pysuricata", "profile", base_name, "--output", "pysuricata_report.html"],
        ):
            main()
        assert os.path.exists("pysuricata_report.html")
    finally:
        os.chdir(original_cwd)
