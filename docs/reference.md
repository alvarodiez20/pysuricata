---
title: Generated Reference
description: Signatures and docstrings, generated from the source
---

# Generated Reference

Everything below is generated from the source at build time, so it cannot drift
from the code the way a hand-written page can. For the narrative version — what
to reach for and why — see the [High-Level API](api.md) and the
[Configuration Guide](configuration.md).

## Entry points

::: pysuricata.profile
    options:
      heading_level: 3

::: pysuricata.summarize
    options:
      heading_level: 3

::: pysuricata.compare
    options:
      heading_level: 3

## Configuration

::: pysuricata.ProfileConfig
    options:
      heading_level: 3
      members: []

::: pysuricata.ComputeOptions
    options:
      heading_level: 3
      members:
        - validate
        - checkpoint

::: pysuricata.RenderOptions
    options:
      heading_level: 3
      members: []

## Results

::: pysuricata.Report
    options:
      heading_level: 3
      members:
        - save
        - save_html
        - save_json
        - show
        - display_in_notebook

::: pysuricata.Comparison
    options:
      heading_level: 3

## Errors

::: pysuricata.PySuricataError
    options:
      heading_level: 3
      show_source: false

::: pysuricata.UnsupportedDataError
    options:
      heading_level: 3
      show_source: false

::: pysuricata.ConfigurationError
    options:
      heading_level: 3
      show_source: false

## Readers

The batch readers behind the path, Arrow and DuckDB inputs, for when you want
the batches rather than a profile.

::: pysuricata.sources
    options:
      heading_level: 3
      members:
        - stream_parquet
        - stream_ipc
        - stream_arrow
        - stream_duckdb
        - is_arrow_source
        - is_duckdb_relation
