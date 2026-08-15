# PySuricata benchmarks

Measure first. Every performance claim in the README, the docs or a blog post
should be reproducible by someone else running these scripts on their own
machine.

```bash
uv sync --dev
uv pip install pytest pytest-benchmark          # accuracy suite
uv pip install ydata-profiling sweetviz skimpy  # optional, for end_to_end
```

## The four entry points

| Script | Question it answers |
|---|---|
| `hotspots.py` | Where does `profile()` spend its time *today*? |
| `kernels.py` | How fast is each kernel, and how close is it to this machine's memory bandwidth? |
| `accuracy.py` | Are the numbers right — and do they stay right when the data is chunked? |
| `end_to_end.py` | How does PySuricata compare to ydata-profiling / sweetviz / skimpy on identical data? |
| `versions.py` | How much faster is this version than the last five, measured properly? |

### Order of operations

1. **`accuracy.py` first.** Nothing else matters if the numbers are wrong, and
   an optimisation you cannot verify is a liability. The `xfail`-marked tests
   are live bugs with code pointers; they turn to XPASS as fixes land.
2. **`hotspots.py`** to decide *what* to optimise. Its "BY SUBSYSTEM" rollup is
   the one to look at before committing a weekend: if 40% of the wall clock is
   in the render layer, rewriting an accumulator in Rust is the wrong move.
3. **`kernels.py`** to decide *how*. The `% roofline` column is the decision
   rule: under 20% means the instruction stream is the bottleneck (a native
   kernel helps); over 70% means you are already saturating memory bandwidth
   and only *fewer passes over the data* will help.
4. **`end_to_end.py`** and **`versions.py`** to produce the numbers you publish.

## The one rule for anything you publish

**A ratio is only quotable when both sides were measured in the same
round-robin, on the same machine, within the same run.**

This is not a style preference. Two published claims in this project came from
pairing measurements taken at different times, on a container whose available
CPU varies between sessions:

- *"0.0.21 is 1.24x faster than 0.0.16."* Measured properly, in one round-robin,
  it is **0.88x** — a regression, published as an improvement.
- *"3.56x since 0.0.16."* Really **2.48x**. The bigger number paired a slow
  0.0.16 run with a fast 0.0.31 run.

Both harnesses now enforce the schedule rather than leaving it to memory. Every
tool or version is measured in **every round**, interleaved, and each one's best
is reported; a slow patch of machine time penalises everything in that round, so
it cancels in the ratio. Below three rounds, the output labels itself
**Not quotable** in the terminal and in the generated markdown.

```bash
python -m benchmarks.end_to_end --rounds 5 --markdown results.md
python -m benchmarks.versions --versions 0.0.16,0.0.21,0.0.26,0.0.27,. --rounds 5
```

`versions.py` installs each released version into its own throwaway virtualenv
and times it in its own subprocess, so import cost, allocator state and garbage
from one version cannot leak into the next.

Absolute times are never comparable across runs. Ratios inside one table are.

## Reading the roofline column

`kernels.py` first measures how fast this machine streams a float64 array
(`np.sum`), then reports each kernel against that ceiling.

```
kernel                          ns/row   M rows/s   % roofline
moments: numpy 5-pass             12.4       80.6        61.2%
moments: native fused 1-pass       1.9      526.3        11.4%
```

The NumPy version is close to the ceiling *for the traffic it generates* — it
is not badly written, it just reads the column eleven times. The native version
touches memory once, so it sits low against the ceiling with plenty of headroom
left. That distinction is the whole argument for fusing passes rather than
micro-optimising each one, and it is a direct application of CS:APP ch. 6:
the win came from the memory hierarchy, not from better arithmetic.

## Publishing numbers

`end_to_end.py --markdown results.md` emits a table with the environment block
attached. Always publish both. Also publish the failures: "ydata-profiling
raised MemoryError at 5M rows x 40 cols on a 16 GB machine" is a more useful
and more credible datapoint than a bar chart.

Rules for anything that goes in a blog post:

- Same DataFrame object for every tool, generated from a seeded RNG.
- Separate subprocess per tool.
- Report peak RSS and output size next to wall time.
- Never compare a `minimal=True` incumbent run against a full PySuricata run,
  or the reverse, without labelling it.
- State the version of every package involved.

## Adding a kernel

Put the current Python implementation in `kernels.py` as a `py_*` function —
copied from the library, not imported — so the baseline stays fixed even as the
library changes. Add the native counterpart next to it and a pair in
`render_table`'s `pairs` list. Then add the correctness check to `accuracy.py`
before you optimise anything.
