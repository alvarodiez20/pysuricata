# pysuricata-core

Native streaming-statistics kernels for [PySuricata](https://github.com/alvarodiez20/pysuricata).

This package is an **optional accelerator**. PySuricata works without it. Installing
`pysuricata[fast]` pulls it in, and the accumulator factory switches to the native
implementations when the import succeeds.

## What's in here

| Export | Replaces | Why |
|---|---|---|
| `NumericKernel` | the moments + reservoir + KMV + monotonicity path in `NumericAccumulator.update` | one fused pass over the chunk instead of ~12 NumPy passes, GIL released |
| `KmvSketch` | `accumulators.sketches.KMV` | fixed-capacity binary heap over a flat `Vec<u64>` instead of a sorted Python list re-sorted every chunk; non-cryptographic hash instead of SHA-1 |
| `Reservoir` | `accumulators.sketches.ReservoirSampler` | Algorithm L (correct + O(k log(n/k)) draws), per-instance seeded RNG instead of the process-global one |
| `mix_u64_batch` | `KMV.add_many` fed by `RowKMV` | row hashes are already uniform u64; stop stringifying and SHA-1-ing them |
| `hash_arrow_utf8` | `Series.to_list()` + per-value hashing | hashes the Arrow value buffer in place, no Python string objects |
| `scan_numeric` | — | stateless single-pass scan, used by the accuracy oracle to diff against NumPy |

## Build

```bash
# dev build into the current environment
maturin develop --release -m native/Cargo.toml

# wheel
maturin build --release -m native/Cargo.toml --out dist/
```

Rust 1.74+ required. `abi3-py310` means one wheel per platform covers CPython 3.10+.

## Tests

```bash
cargo test --lib                          # Rust unit tests (moments, kmv, reservoir, hashing)
python -m pytest benchmarks/accuracy.py   # Python-vs-native-vs-NumPy agreement
```

The Rust tests are the ones that pin the invariants that matter:

- `moments::merge_equals_single_pass` — profiling in 1 chunk and in 7 chunks gives
  the same moments. This is the property the Python fast path currently violates.
- `reservoir::sample_is_uniform_over_position` — the sample's mean position matches
  the population's. This is the property the current `add_many` violates.
- `kmv::merge_matches_union` — merging two sketches gives exactly the same estimate
  as feeding one sketch both streams.

## Licence

MIT, same as PySuricata.
