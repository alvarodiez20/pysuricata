---
title: Reservoir Sampling
description: Uniform random sampling from data streams
---

# Reservoir Sampling

Maintain a uniform random sample of fixed size \(k\) from a stream of unknown
length. PySuricata uses the sample for quantiles, the median, the IQR, the MAD
and the histogram — everything that needs order statistics but cannot afford to
hold the column in memory.

!!! info "Examples on this page assume a DataFrame named `df`"

    Every snippet below that does not build its own frame expects one already in
    scope. Paste this first to follow along:

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(5_000),
            "amount": rng.lognormal(3, 1, 5_000),
            "country": rng.choice(["ES", "FR", "DE"], 5_000),
            "signed_up": pd.date_range("2024-01-01", periods=5_000, freq="17min"),
            "active": rng.random(5_000) > 0.3,
        }
    )
    ```

The guarantee is the same for both algorithms below: **every element that has
arrived is in the reservoir with probability \(k/n\)**, independently of where it
sat in the stream and of how the stream happened to be split into chunks. That
last part is what makes chunked results equal unchunked ones.

<figure class="ps-figure" markdown="0">
  <iframe src="../../assets/diagrams/figures.html?only=reservoir" title="Reservoir sampling: Algorithm R versus Algorithm L" loading="lazy"></iframe>
</figure>

## Algorithm R, and why it is not what ships

Algorithm R (Vitter, 1985) is the version usually taught: hold the first \(k\)
elements, then for each subsequent arrival draw a random index and overwrite a
slot if it falls inside the reservoir.

```python
# Reference only -- this is NOT what PySuricata runs. See Algorithm L below.
def algorithm_r(stream, k):
    reservoir = []
    for n, item in enumerate(stream, start=1):
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = random.randint(0, n - 1)
            if j < k:
                reservoir[j] = item
    return reservoir
```

It is correct, and it costs one random draw per element — 10 million draws for
10 million rows, every one of them in Python. That is the problem.

## Algorithm L, which is what ships

Algorithm L (Li, 1994) keeps the same guarantee and pays for it far less often.
Instead of testing every arrival, it draws a *geometric skip* and jumps straight
to the next element it will accept, never touching the ones in between. The
number of draws falls from \(n\) to about

\[
k \ln\!\left(\frac{n}{k}\right)
\]

— roughly 145,000 draws for 10 million rows into \(k = 20{,}000\), instead of
10 million.

`ReservoirSampler` in `pysuricata/accumulators/sketches.py` goes one step
further. The acceptance schedule depends only on the generator and on \(k\) —
never on the data — so there is no reason to derive it one acceptance at a time.
Writing the recurrence out makes every term an array operation:

\[
\log W_i = \frac{1}{k}\sum_{j \le i} \log u_j
\qquad
\text{skip}_i = \left\lfloor \frac{\log v_i}{\log(1 - W_i)} \right\rfloor
\qquad
\text{index}_i = \text{base} + \sum_{j \le i}\text{skip}_j + i
\]

so the schedule is generated in blocks with `np.cumsum` rather than in a Python
loop. `log1p(-W)` rather than `log(1 - W)`: \(W\) is close to 1 early on, where
the subtraction would lose most of its significant digits.

Because the schedule comes from the draw sequence alone, it is identical however
the stream is later split — which is the property the accuracy oracle asserts.

## Reproducibility

Each sampler owns its generator:

```python
import numpy as np

from pysuricata.accumulators.sketches import ReservoirSampler

sampler = ReservoirSampler(k=20_000, rng=np.random.default_rng(42))
```

Nothing reads or writes the process-global RNG. Within a profiling run each
column's seed is derived from the run seed and the **column name**, so a
column's sample does not depend on which other columns are present or on the
order they were built in — profiling a subset reproduces the numbers from
profiling the whole frame.

```python
from pysuricata import profile
from pysuricata.api import ComputeOptions, ProfileConfig

config = ProfileConfig(compute=ComputeOptions(random_seed=42))
report = profile(df, config=config)
```

Sample size is `ComputeOptions.numeric_sample_size` (default 20,000).

## Complexity

| | Algorithm R | Algorithm L |
|---|---|---|
| Space | O(k) | O(k) |
| Random draws | \(n\) | \(\approx k \ln(n/k)\) |
| Time per element | O(1) | O(1) amortised, most elements untouched |
| Uniform guarantee | exact | exact |

## Use cases

- Quantile estimation (sort the sample)
- Histogram construction
- Representative row sampling

## See also

- [Sketch Algorithms](sketches.md) — the bounded-memory counterparts
- [Numeric Analysis](../stats/numeric.md) — using the reservoir for quantiles
