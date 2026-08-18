#!/usr/bin/env python3
"""Regenerate the example report used in the docs and the README.

Run this whenever CSS, JS, templates or rendering logic changes, so the
embedded example keeps up with what the library actually renders.

**The dataset is Bike Sharing rather than Titanic (#150).** Titanic could not
exercise what the report does: no datetime column, so the datetime card and its
four temporal panels never appeared in the one example anybody looks at, and no
numeric pair above 0.5, so the correlations section always took the weak-result
route. Three of four card kinds and one of three correlation views, in the
example that exists to demonstrate the library.

Titanic stays as the *test* fixture -- the byte and layout ratchets in
`tests/test_report_layout.py` are pinned to it, and repinning them would throw
away their history for no gain.

Usage:
    python scripts/regenerate_example_report.py
    # or via uv:
    uv run python scripts/regenerate_example_report.py
"""

from __future__ import annotations

import os
import sys
import time

# Ensure the local package is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import pandas as pd

    import pysuricata
    from pysuricata.api import ProfileConfig, RenderOptions

    # Vendored, so this never reaches the network and the generated report is
    # byte-stable across runs. `scripts/build_demo_dataset.py` says where the
    # file came from and how to rebuild it.
    local_path = os.path.join(REPO_ROOT, "docs", "assets", "bike_sharing.csv")
    output_path = os.path.join(REPO_ROOT, "docs", "assets", "example_report.html")

    print(f"📦 PySuricata v{pysuricata.__version__}")

    if not os.path.exists(local_path):
        print(f"   ❌ Missing {local_path}.")
        print("      Rebuild it with scripts/build_demo_dataset.py.")
        return 1

    print(f"📥 Loading the demo dataset from {local_path}...")
    # Parsed here rather than left to inference: the column is a real timestamp
    # and the example is about what the datetime card does with one.
    df = pd.read_csv(local_path, parse_dates=["rented_at"])
    print(f"   ✓ {len(df):,} rows × {len(df.columns)} columns")

    config = ProfileConfig(
        render=RenderOptions(title="PySuricata EDA Report — Bike Sharing"),
    )

    print("⚡ Generating report...")
    start = time.perf_counter()
    report = pysuricata.profile(df, config=config)
    elapsed = time.perf_counter() - start
    print(f"   ✓ Generated in {elapsed:.3f}s ({len(report.html):,} bytes)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"💾 Saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
