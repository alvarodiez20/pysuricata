# Installation

Install from PyPI:

```bash
pip install pysuricata
```

Optional: install `polars` to use polars DataFrames directly:

```bash
pip install polars
```

Verify your installation:

```python
>>> import pandas as pd
>>> from pysuricata import profile
>>> df = pd.DataFrame({"x": [1, 2, 3]})
>>> profile(df).html[:15]
'<!DOCTYPE html>'
```
