"""Command-line interface for PySuricata.

Usage:
    pysuricata profile <file> --output <report.html>
    pysuricata summarize <file>
    pysuricata check <file> --baseline <baseline.json>
    pysuricata --version

`check` is the only command with a meaningful exit code: 0 when the gate
passes, 1 when a threshold was crossed, 2 when the run could not happen at all.
Keeping "the data drifted" and "the file was missing" distinguishable is what
lets a pipeline treat one as a data problem and the other as an outage.
"""

import argparse
import json
import sys
import time
from dataclasses import replace

from pysuricata import ComputeOptions, ProfileConfig, __version__, profile, summarize
from pysuricata.check import (
    Thresholds,
    compare,
    make_baseline,
    parse_duration,
    read_baseline,
    read_thresholds,
    render_findings,
    write_baseline,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pysuricata",
        description="PySuricata - Lightweight, streaming data profiling for Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pysuricata profile data.csv --output report.html
  pysuricata profile data.parquet -o report.html --seed 42
  pysuricata summarize data.csv
  pysuricata summarize data.csv --output stats.json
  pysuricata check data.parquet --write-baseline baseline.json
  pysuricata check data.parquet --baseline baseline.json

For more information, visit: https://github.com/alvarodiez20/pysuricata
        """,
    )

    parser.add_argument(
        "--version", "-v", action="version", version=f"pysuricata {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Generate an HTML profiling report",
        description="Analyze a dataset and generate a comprehensive HTML report.",
    )
    profile_parser.add_argument(
        "file",
        type=str,
        help="Path to the data file (CSV, Parquet, JSON, Arrow or Excel)",
    )
    profile_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the HTML report",
    )
    profile_parser.add_argument(
        "--title", "-t", type=str, default=None, help="Custom title for the report"
    )
    profile_parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    profile_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for streaming (default: 100000)",
    )
    profile_parser.add_argument(
        "--sample-size",
        type=int,
        default=20_000,
        help="Sample size for quantile estimation (default: 20000)",
    )
    profile_parser.add_argument(
        "--no-correlations",
        action="store_true",
        help="Disable correlation computation (faster for wide datasets)",
    )
    profile_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Output statistics as JSON (no HTML)",
        description="Analyze a dataset and output statistics as JSON.",
    )
    summarize_parser.add_argument(
        "file",
        type=str,
        help="Path to the data file (CSV, Parquet, JSON, Arrow or Excel)",
    )
    summarize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for JSON (default: stdout)",
    )
    summarize_parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    summarize_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for streaming (default: 100000)",
    )
    summarize_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Gate a dataset on shape drift, with an exit code",
        description=(
            "Compare a dataset against a stored baseline and exit non-zero when "
            "a threshold is crossed. Exit codes: 0 pass, 1 threshold crossed, "
            "2 the check could not run."
        ),
    )
    check_parser.add_argument(
        "file",
        type=str,
        help="Path to the data file (CSV, Parquet, JSON, Arrow or Excel)",
    )
    check_parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default=None,
        help="Baseline JSON to compare against",
    )
    check_parser.add_argument(
        "--write-baseline",
        type=str,
        default=None,
        help="Write a baseline from this dataset and exit",
    )
    check_parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Thresholds file (.json or .toml) overriding the defaults",
    )
    check_parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=None,
        help="Fail if any column is missing more than this percentage",
    )
    check_parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Fail if the dataset has fewer rows than this",
    )
    check_parser.add_argument(
        "--max-rows-drift-pct",
        type=float,
        default=None,
        help="Fail if the row count moved more than this percentage from the baseline",
    )
    check_parser.add_argument(
        "--fail-on-new-column",
        action="store_true",
        help="Treat an added column as a breach (off by default)",
    )
    check_parser.add_argument(
        "--max-age",
        type=str,
        default=None,
        help=(
            "Fail if the newest timestamp in a datetime column is older than "
            "this, e.g. 26h, 3d, 90m. Needs no baseline"
        ),
    )
    check_parser.add_argument(
        "--require-fresh",
        action="store_true",
        help=(
            "Fail if a datetime column's newest timestamp did not advance past "
            "the baseline's — catches a re-run of yesterday's extract"
        ),
    )
    check_parser.add_argument(
        "--fail-on-range-expansion",
        action="store_true",
        help="Treat a new minimum or maximum outside the baseline range as a breach",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as JSON on stdout instead of text",
    )
    check_parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report findings but always exit 0",
    )
    check_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Random seed (default: 0, so a re-run of the same data is a no-op)",
    )
    check_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for streaming (default: 100000)",
    )
    check_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    return parser


def load_data(file_path: str):
    """Load data from a file path.

    Delegates to `pysuricata.api._read_path` -- the same dispatch `profile()`
    and `summarize()` use for a path argument, covering CSV, Parquet, JSON,
    Arrow IPC and Excel, streamed where the format allows it. This function
    used to duplicate that dispatch with a narrower format list that had
    drifted out of sync with it: `pysuricata profile data.arrow` worked from
    a Python call and raised "Unsupported file format" from the CLI.

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame or generator of DataFrames

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file's format is not supported.
    """
    from pysuricata.api import PySuricataError, UnsupportedDataError, _read_path

    try:
        return _read_path(file_path)
    except UnsupportedDataError as e:
        # A TypeError subclass at the API boundary (it can also be raised for
        # a Python object of the wrong type); the CLI's contract here has
        # always been ValueError for a bad file format.
        raise ValueError(str(e)) from e
    except PySuricataError as e:
        raise FileNotFoundError(str(e)) from e


def cmd_profile(args: argparse.Namespace) -> int:
    """Execute the profile command."""
    if not args.quiet:
        print(f"Loading data from: {args.file}")

    start_time = time.perf_counter()

    try:
        data = load_data(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build configuration
    compute_options = ComputeOptions(
        chunk_size=args.chunk_size,
        numeric_sample_size=args.sample_size,
        compute_correlations=not args.no_correlations,
    )

    if args.seed is not None:
        compute_options.random_seed = args.seed

    config = ProfileConfig(compute=compute_options)

    if args.title:
        config.render.title = args.title

    if not args.quiet:
        print("Profiling data...")

    try:
        report = profile(data, config=config)
    except Exception as e:
        print(f"Error during profiling: {e}", file=sys.stderr)
        return 1

    # Save report
    try:
        report.save_html(args.output)
    except Exception as e:
        print(f"Error saving report: {e}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start_time

    if not args.quiet:
        print(f"Report saved to: {args.output}")
        print(f"Completed in {elapsed:.1f} seconds")

    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    """Execute the summarize command."""
    if not args.quiet:
        print(f"Loading data from: {args.file}", file=sys.stderr)

    start_time = time.perf_counter()

    try:
        data = load_data(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build configuration
    compute_options = ComputeOptions(
        chunk_size=args.chunk_size,
    )

    if args.seed is not None:
        compute_options.random_seed = args.seed

    config = ProfileConfig(compute=compute_options)

    if not args.quiet:
        print("Summarizing data...", file=sys.stderr)

    try:
        stats = summarize(data, config=config)
    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        return 1

    # Convert to JSON-serializable format
    def convert_numpy(obj):
        """Convert numpy types to Python types."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        return obj

    stats_json = convert_numpy(stats)

    # Output results
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(stats_json, f, indent=2)
            if not args.quiet:
                print(f"Stats saved to: {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving stats: {e}", file=sys.stderr)
            return 1
    else:
        print(json.dumps(stats_json, indent=2))

    elapsed = time.perf_counter() - start_time

    if not args.quiet:
        print(f"Completed in {elapsed:.1f} seconds", file=sys.stderr)

    return 0


def _resolve_thresholds(args: argparse.Namespace) -> Thresholds:
    """Combine the thresholds file with the command-line overrides.

    Precedence is defaults, then the file, then flags — the same order the
    Python API uses for presets and keywords.

    Args:
        args: Parsed arguments.

    Returns:
        The thresholds to gate on.

    Raises:
        ValueError: If the thresholds file is unreadable or names an unknown
            threshold.
    """
    base = read_thresholds(args.thresholds) if args.thresholds else Thresholds()

    overrides = {
        "max_missing_pct": args.max_missing_pct,
        "min_rows": args.min_rows,
        "max_rows_drift_pct": args.max_rows_drift_pct,
    }
    given = {k: v for k, v in overrides.items() if v is not None}
    if args.max_age is not None:
        given["max_age"] = parse_duration(args.max_age)
    if args.fail_on_new_column:
        given["fail_on_new_column"] = True
    if args.fail_on_range_expansion:
        given["fail_on_range_expansion"] = True
    if args.require_fresh:
        given["require_max_ts_advances"] = True
    if not given:
        return base
    return replace(base, **given)


def cmd_check(args: argparse.Namespace) -> int:
    """Execute the check command.

    Returns:
        0 when the gate passes, 1 when a threshold was crossed, 2 when the
        check could not run.
    """
    if args.baseline is None and args.write_baseline is None:
        print(
            "Error: check needs --baseline to compare against, or "
            "--write-baseline to create one.",
            file=sys.stderr,
        )
        return 2

    try:
        thresholds = _resolve_thresholds(args)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        data = load_data(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        stats = summarize(
            data,
            chunk_size=args.chunk_size,
            seed=args.seed,
            progress=None if args.quiet else "auto",
        )
    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        return 2

    if args.write_baseline is not None:
        try:
            write_baseline(make_baseline(stats, source=args.file), args.write_baseline)
        except OSError as e:
            print(f"Error writing baseline: {e}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"Baseline written to: {args.write_baseline}", file=sys.stderr)
        return 0

    try:
        baseline = read_baseline(args.baseline)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Error reading baseline: {e}", file=sys.stderr)
        return 2

    result = compare(stats, baseline, thresholds)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(render_findings(result))

    if result.passed or args.warn_only:
        return 0
    return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "profile":
        return cmd_profile(args)
    elif args.command == "summarize":
        return cmd_summarize(args)
    elif args.command == "check":
        return cmd_check(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
