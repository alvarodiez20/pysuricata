# `pysuricata`
[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo.png" alt="pysuricata Logo" width="300">
</div>

`pysuricata` is a lightweight Python library for exploratory data analysis (EDA) that supports multiple data formats—Pandas, Dask, and Polars DataFrames. It generates self-contained HTML reports featuring summary statistics, missing values, and correlation matrices with a clean, modern design. 


## Installation

Install `pysuricata` directly from PyPI:

```bash
pip install pysuricata
```

## Quick Example

The following example demonstrates how to generate an EDA report using the Iris dataset with Pandas:


```python
import pandas as pd
from pysuricata import generate_report
from IPython.display import HTML

# Load the Iris dataset directly using Pandas
iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(iris_url)

# Generate the HTML EDA report and save it to a file
html_report = generate_report(iris_df, output_file="iris_report.html")

# Display the report in a Jupyter Notebook
HTML(html_report)
```