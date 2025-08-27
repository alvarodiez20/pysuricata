# `pysuricata`
[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo_suricata_transparent.png" alt="pysuricata Logo" width="300">
</div>



A lightweight Python library to generate self-contained HTML reports for exploratory data analysis (EDA).

📖 [Read the documentation](https://alvarodiez20.github.io/pysuricata/)


## Installation

Install `pysuricata` directly from PyPI:

```bash
pip install pysuricata
```

## Why use pysuricata?
- **Instant reports**: Generate clean, self-contained HTML reports directly from pandas DataFrames.
- **No dependencies on heavy frameworks**: Only requires pandas and numpy.
- **Rich insights**: Summaries for numeric, categorical, datetime columns, missing values, duplicates, correlations, and sample rows.
- **Portable**: Reports are standalone HTML (with inline CSS/JS/images) that can be easily shared.
- **Customizable**: Title, sample display, and output path can be tailored to your needs.

## Quick Example

The following example demonstrates how to generate an EDA report using the Iris dataset with Pandas:


```python
import pandas as pd
import pysuricata
from IPython.display import HTML

# Load the Iris dataset directly using Pandas
iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(iris_url)

# Generate the HTML EDA report and save it to a file
html_report = pysuricata.generate_report(iris_df, output_file="iris_report.html")

# Display the report in a Jupyter Notebook
HTML(html_report)
```