# Usage

```python
import pandas as pd
from pysuricata.report import generate_report

df = pd.read_csv("data.csv")
html = generate_report(df, report_title="My EDA")
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html)
```