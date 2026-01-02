"""Command-line interface for PySuricata.

Usage:
    pysuricata profile <file> --output <report.html>
    pysuricata summarize <file>
    pysuricata --version
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pysuricata import ComputeOptions, ProfileConfig, __version__, profile, summarize


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
    profile_parser.add_argument("file", type=str, help="Path to the data file (CSV or Parquet)")
    profile_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output path for the HTML report"
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
    summarize_parser.add_argument("file", type=str, help="Path to the data file (CSV or Parquet)")
    summarize_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for JSON (default: stdout)"
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

    return parser


def load_data(file_path: str):
    """Load data from a file path.

    Supports CSV and Parquet files. For large files, returns a generator
    that yields chunks.

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame or generator of DataFrames
    """
    import pandas as pd

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif suffix == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use CSV, Parquet, or JSON.")


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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
