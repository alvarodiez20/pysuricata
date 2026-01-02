"""Tests for the CLI module."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pysuricata.cli import cmd_profile, cmd_summarize, create_parser, load_data, main


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_parser_profile_command(self):
        """Test profile command parsing."""
        parser = create_parser()
        args = parser.parse_args(["profile", "data.csv", "--output", "report.html"])
        
        assert args.command == "profile"
        assert args.file == "data.csv"
        assert args.output == "report.html"

    def test_parser_profile_with_options(self):
        """Test profile command with all options."""
        parser = create_parser()
        args = parser.parse_args([
            "profile", "data.csv",
            "--output", "report.html",
            "--seed", "42",
            "--chunk-size", "50000",
            "--sample-size", "10000",
            "--no-correlations",
            "--quiet",
            "--title", "My Report"
        ])
        
        assert args.seed == 42
        assert args.chunk_size == 50000
        assert args.sample_size == 10000
        assert args.no_correlations is True
        assert args.quiet is True
        assert args.title == "My Report"

    def test_parser_summarize_command(self):
        """Test summarize command parsing."""
        parser = create_parser()
        args = parser.parse_args(["summarize", "data.csv"])
        
        assert args.command == "summarize"
        assert args.file == "data.csv"
        assert args.output is None

    def test_parser_summarize_with_output(self):
        """Test summarize command with output file."""
        parser = create_parser()
        args = parser.parse_args(["summarize", "data.csv", "--output", "stats.json"])
        
        assert args.output == "stats.json"


class TestLoadData:
    """Tests for data loading functionality."""

    def test_load_csv(self):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,x\n2,y\n")
            temp_path = f.name

        try:
            df = load_data(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ["a", "b"]
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_data(temp_path)
        finally:
            os.unlink(temp_path)


class TestCLICommands:
    """Tests for CLI command execution."""

    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "numeric": np.random.randn(100),
                "categorical": ["a", "b", "c", "d"] * 25,
                "boolean": [True, False] * 50
            })
            df.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_cmd_profile_creates_html(self, sample_csv):
        """Test that profile command creates HTML file."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        parser = create_parser()
        args = parser.parse_args(["profile", sample_csv, "--output", output_path, "--quiet"])
        
        result = cmd_profile(args)
        
        assert result == 0
        assert os.path.exists(output_path)
        
        with open(output_path) as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content or "<html" in content

        os.unlink(output_path)

    def test_cmd_profile_with_seed(self, sample_csv):
        """Test profile command with seed for reproducibility."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        parser = create_parser()
        args = parser.parse_args([
            "profile", sample_csv,
            "--output", output_path,
            "--seed", "42",
            "--quiet"
        ])
        
        result = cmd_profile(args)
        assert result == 0

        os.unlink(output_path)

    def test_cmd_profile_invalid_file(self):
        """Test profile command with invalid file."""
        parser = create_parser()
        args = parser.parse_args([
            "profile", "/nonexistent/file.csv",
            "--output", "/tmp/report.html",
            "--quiet"
        ])
        
        result = cmd_profile(args)
        assert result == 1  # Should fail

    def test_cmd_summarize_outputs_json(self, sample_csv, capsys):
        """Test that summarize command outputs JSON."""
        parser = create_parser()
        args = parser.parse_args(["summarize", sample_csv, "--quiet"])
        
        result = cmd_summarize(args)
        
        assert result == 0
        
        captured = capsys.readouterr()
        stats = json.loads(captured.out)
        
        assert "dataset" in stats
        assert "columns" in stats

    def test_cmd_summarize_saves_to_file(self, sample_csv):
        """Test that summarize command saves to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        parser = create_parser()
        args = parser.parse_args([
            "summarize", sample_csv,
            "--output", output_path,
            "--quiet"
        ])
        
        result = cmd_summarize(args)
        
        assert result == 0
        assert os.path.exists(output_path)
        
        with open(output_path) as f:
            stats = json.load(f)
            assert "dataset" in stats

        os.unlink(output_path)


class TestMain:
    """Tests for main entry point."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        import sys
        original_argv = sys.argv
        sys.argv = ["pysuricata"]
        
        try:
            result = main()
            assert result == 0
            
            captured = capsys.readouterr()
            assert "pysuricata" in captured.out.lower() or "usage" in captured.out.lower()
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
