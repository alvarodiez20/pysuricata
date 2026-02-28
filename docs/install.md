# Installation

Install from PyPI:

```bash
# using uv (recommended)
uv add pysuricata

# or using pip
pip install pysuricata
```

Optional: install polars support for polars DataFrames:

```bash
uv add pysuricata[polars]
# or: pip install pysuricata[polars]
```

Verify your installation:

```python
>>> import pandas as pd
>>> from pysuricata import profile
>>> df = pd.DataFrame({"x": [1, 2, 3]})
>>> profile(df).html[:15]
'<!DOCTYPE html>'
```
