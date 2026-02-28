#!/usr/bin/env python3
"""Regenerate the example Titanic report used in docs and README.

Run this script whenever CSS, JS, templates, or rendering logic changes
to keep the embedded example report up-to-date.

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

    # Use Titanic dataset — small, public, and well-known
    titanic_url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    output_path = os.path.join(REPO_ROOT, "docs", "assets", "titanic_report.html")

    print(f"📦 PySuricata v{pysuricata.__version__}")
    print("📥 Loading Titanic dataset from GitHub...")

    try:
        df = pd.read_csv(titanic_url)
    except Exception:
        # Fallback: try local file
        local_path = os.path.join(REPO_ROOT, "docs", "assets", "titanic.csv")
        if os.path.exists(local_path):
            print(f"   ⚠ GitHub unavailable, using local {local_path}")
            df = pd.read_csv(local_path)
        else:
            print("   ❌ Cannot download Titanic CSV and no local copy found.")
            return 1

    print(f"   ✓ {len(df)} rows × {len(df.columns)} columns")

    config = ProfileConfig(
        render=RenderOptions(title="PySuricata EDA Report — Titanic Dataset"),
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
