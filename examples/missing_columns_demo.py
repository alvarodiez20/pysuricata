#!/usr/bin/env python3
"""Demonstration of the new intelligent missing columns functionality.

This script shows how the new system adapts to different dataset sizes
and provides optimal missing columns display.
"""

import numpy as np
import pandas as pd

from pysuricata.config import EngineConfig
from pysuricata.report import build_report


def create_test_datasets():
    """Create test datasets with different sizes and missing patterns."""

    # Small dataset (5 columns, 1000 rows)
    small_data = {
        "id": range(1000),
        "name": [f"user_{i}" if i % 100 != 0 else None for i in range(1000)],
        "age": [
            np.random.randint(18, 80) if i % 50 != 0 else None for i in range(1000)
        ],
        "score": [
            np.random.normal(75, 15) if i % 80 != 0 else None for i in range(1000)
        ],
        "active": [
            np.random.choice([True, False]) if i % 30 != 0 else None
            for i in range(1000)
        ],
    }
    small_df = pd.DataFrame(small_data)

    # Medium dataset (25 columns, 5000 rows)
    medium_data = {}
    for i in range(25):
        col_name = f"feature_{i:02d}"
        # Vary missing percentages: 0%, 1%, 5%, 10%, 15%, 20%, etc.
        missing_pct = (i % 5) * 5
        medium_data[col_name] = [
            np.random.normal(0, 1) if np.random.random() > (missing_pct / 100) else None
            for _ in range(5000)
        ]
    medium_df = pd.DataFrame(medium_data)

    # Large dataset (100 columns, 20000 rows)
    large_data = {}
    for i in range(100):
        col_name = f"var_{i:03d}"
        # Vary missing percentages: 0% to 30%
        missing_pct = (i % 10) * 3
        large_data[col_name] = [
            np.random.normal(0, 1) if np.random.random() > (missing_pct / 100) else None
            for _ in range(20000)
        ]
    large_df = pd.DataFrame(large_data)

    return {"small": small_df, "medium": medium_df, "large": large_df}


def demonstrate_configurations():
    """Show different configuration options for missing columns."""

    configs = {
        "default": EngineConfig(),
        "strict": EngineConfig(
            missing_columns_threshold_pct=5.0,  # Only show >5% missing
            missing_columns_max_initial=5,  # Show only 5 initially
            missing_columns_max_expanded=15,  # Show 15 when expanded
        ),
        "lenient": EngineConfig(
            missing_columns_threshold_pct=0.1,  # Show >0.1% missing
            missing_columns_max_initial=15,  # Show 15 initially
            missing_columns_max_expanded=50,  # Show 50 when expanded
        ),
    }

    return configs


def analyze_missing_patterns(df, name):
    """Analyze and display missing patterns for a dataset."""
    print(f"\n=== {name.upper()} DATASET ANALYSIS ===")
    print(f"Shape: {df.shape}")

    # Calculate missing percentages
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100

    # Show top missing columns
    missing_df = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_pct": missing_pct.values,
        }
    ).sort_values("missing_pct", ascending=False)

    print(f"\nTop 10 missing columns:")
    print(missing_df.head(10).to_string(index=False, float_format="%.1f"))

    # Show summary statistics
    significant_missing = missing_df[missing_df["missing_pct"] >= 0.5]
    print(f"\nSummary:")
    print(f"- Total columns: {len(df.columns)}")
    print(f"- Columns with ≥0.5% missing: {len(significant_missing)}")
    print(
        f"- Columns with ≥5% missing: {len(missing_df[missing_df['missing_pct'] >= 5])}"
    )
    print(
        f"- Columns with ≥20% missing: {len(missing_df[missing_df['missing_pct'] >= 20])}"
    )


def generate_sample_reports():
    """Generate sample reports to demonstrate the functionality."""

    datasets = create_test_datasets()
    configs = demonstrate_configurations()

    print("🔍 INTELLIGENT MISSING COLUMNS DEMONSTRATION")
    print("=" * 50)

    for name, df in datasets.items():
        analyze_missing_patterns(df, name)

        # Generate report with default configuration
        print(f"\n📊 Generating report for {name} dataset...")
        try:
            html = build_report(
                df,
                config=configs["default"],
                report_title=f"{name.title()} Dataset Report",
            )

            # Save report
            filename = f"missing_columns_demo_{name}.html"
            with open(filename, "w") as f:
                f.write(html)
            print(f"✅ Report saved as: {filename}")

        except Exception as e:
            print(f"❌ Error generating report: {e}")

    print(f"\n🎯 CONFIGURATION COMPARISON")
    print("=" * 30)

    # Show how different configurations affect the display
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} Configuration:")
        print(f"- Threshold: {config.missing_columns_threshold_pct}%")
        print(f"- Max initial: {config.missing_columns_max_initial}")
        print(f"- Max expanded: {config.missing_columns_max_expanded}")

    print(f"\n📚 FEATURES DEMONSTRATED")
    print("=" * 25)
    print("✅ Dynamic display limits based on dataset size")
    print("✅ Smart filtering of insignificant missing data")
    print("✅ Expandable UI for large datasets")
    print("✅ Configurable thresholds and limits")
    print("✅ Visual severity indicators (low/medium/high)")
    print("✅ Responsive design and accessibility")


if __name__ == "__main__":
    generate_sample_reports()
