# ADR: a user-specified memory budget

**Status:** proposed
**Verdict:** do it — as a *planner* that derives settings, not as a *cap* that enforces them.

---

## The question

Should a user be able to say `profile(source, memory_budget="512MB")`?

## What I measured first

All figures from the working tree at 0.0.27, streaming sources (a generator of
chunks), peak RSS via `getrusage`, each run in a fresh subprocess.

### The floor is 75 MB before you execute a line

| | peak RSS |
|---|---:|
| after `import numpy, pandas` | 68 MB |
| after `import pysuricata` | 73 MB (+4) |

### Memory is genuinely flat in rows

| rows (8 cols, chunk 50k) | RSS above floor |
|---:|---:|
| 200,000 | 32 MB |
| 1,000,000 | 35 MB |
| 5,000,000 | 35 MB |

A 25× increase in rows costs 3 MB. **The core claim holds under measurement**,
which is worth knowing independently of this decision.

### And linear in everything you control

| varying | measurements | slope |
|---|---|---|
| columns (1M rows) | 4 → 21 MB, 16 → 58, 64 → 214 | ~3.2 MB/column |
| chunk_size (8 cols) | 10k → 18 MB, 50k → 35, 250k → 110 | ~48 B per row per column |
| sample k (16 cols) | 2k → 48 MB, 20k → 58, 100k → 106 | ~37 B per slot per column |

Which fits:

```
peak_MB ≈ 75 + n_cols × (0.5 + k × 37B + chunk_size × 48B)
```

Every term on the right is a knob the library owns. **So the budget is
invertible** — you can solve for `chunk_size` and `k` given a target.

### It works

I implemented the inversion and measured the result:

| budget | cols | chosen | predicted | **actual** |
|---:|---:|---|---:|---:|
| 150 MB | 8 | chunk 110,937 · k 20,000 | 128 | **125** |
| 250 MB | 8 | chunk 200,000 · k 20,000 | 162 | **153** |
| 250 MB | 40 | chunk 48,437 · k 20,000 | 218 | **207** |
| 500 MB | 40 | chunk 126,562 · k 20,000 | 368 | **302** |
| 1000 MB | 100 | chunk 109,375 · k 20,000 | 724 | **560** |

Every case landed under budget, and the model over-predicts — which is the right
direction for a budget to be wrong in.

---

## The decision

**Ship it as `memory_budget`, framed as a target that selects settings, and
report what it chose.** Do not frame it as a guarantee and do not hard-cap.

### Why it is a good idea

1. **It is the missing preset, in the unit users think in.** The UX review found
   21 configuration options and no presets. Nobody knows what
   `numeric_sample_size=20000` costs. Everybody knows their CI runner has 512 MB.
   One argument sets four knobs correctly.
2. **It makes the positioning testable.** "Bounded memory" is currently a claim
   about architecture. `memory_budget="512MB"` turns it into "I asked for 512 and
   it used 480" — a number you can publish, and a test you can run in CI.
3. **The model is real, not hand-waving.** It was fitted from measurements and
   verified against six shapes. This is not a knob that pretends.
4. **It pairs with `pysuricata check`.** In CI you know your runner size exactly,
   which is the case where a budget is most obviously correct.

### Why it must not be a cap

1. **You cannot enforce it.** 75 MB is gone before your first line runs. You do
   not control pandas' allocations, the allocator's fragmentation, or when the GC
   returns pages. You can *plan*; you cannot *police*.
2. **Promising and missing is worse than not promising.** A hard cap that raises
   `MemoryError` reproduces exactly the incumbent failure mode you are positioned
   against.
3. **For an in-memory DataFrame it is nearly meaningless.** A 1M × 8 frame costs
   123 MB resident *before* `profile()` is called, and profiling added nothing
   above that high-water mark. The user already paid. The budget only bites on
   streaming sources — which is, conveniently, exactly where the streaming claim
   matters.

### The real danger, and the mitigation

**Silent accuracy loss.** A tight budget forces `k` down, and every quantile, the
median, IQR, MAD and the histogram come from that sample. Relative error is
`1/√k`:

| k | quantile error |
|---:|---:|
| 20,000 | ±0.7% |
| 5,000 | ±1.4% |
| 1,000 | ±3.2% |

A user who asks for 128 MB and silently receives ±3.2% quantiles has been
mistreated. **The plan must be reported, always** — one line naming the chosen
settings and the resulting error — and there must be a floor below which the call
errors rather than degrades.

---

## What to build

```python
# the common case
profile(source, memory_budget="512MB")

# inspectable, for anyone who wants to see the plan before running
ProfileConfig.for_memory("512MB", n_columns=40)
# -> ComputeOptions(chunk_size=48_437, numeric_sample_size=20_000, ...)
```

Behaviour:

1. **Error immediately below the floor**, with real numbers rather than a generic
   message:
   > `budget 60 MB is below the floor: the interpreter, numpy and pandas cost ~75 MB before profiling starts, plus ~4 MB of fixed state for 8 columns. Minimum workable budget is ~99 MB.`
2. **Report the plan**, at INFO — settings chosen and the accuracy consequence.
3. **Never exceed the caps.** Extra budget beyond `k=20,000` and
   `chunk_size=200,000` buys nothing; leave it unspent rather than inflating.
4. **Split the budget 60/40** between chunk (throughput) and sample (accuracy).
   That ratio is a guess and should be revisited once there is a benchmark for it.
5. **Test it.** Assert measured peak RSS ≤ budget across a matrix of column counts
   and budgets. That test is also the proof of the headline claim, so it earns its
   keep twice.
6. **Document the model in the docs**, including the floor and the fact that a
   budget is a target. A user who understands the formula will trust the number.

## Open questions

- The 60/40 split is unmeasured. Is throughput or accuracy the better marginal
  buy? Worth one benchmark.
- Should `memory_budget` also cap the *report* size? A 100-column profile produces
  a large HTML file; that is a different resource and probably a different
  argument.
- Non-numeric columns are not in the model yet. Categorical accumulators hold a
  Misra-Gries table and string samples; datetime holds its own reservoir. The
  model should be refitted per column kind before this ships.

## Rejected alternatives

- **Hard cap with `MemoryError`.** Reproduces the failure mode you are positioned
  against, and cannot be honoured anyway.
- **Polling RSS mid-run and shrinking sketches adaptively.** Makes results depend
  on machine load, which breaks the chunked-equals-unchunked invariant the
  accuracy oracle enforces. Not worth it.
- **Doing nothing.** The knobs exist; users just have no way to reason about them
  in a unit that means anything.
