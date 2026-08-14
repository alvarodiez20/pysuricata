//! PySuricata native kernels.
//!
//! Scope of this crate: the four hot streaming primitives, plus a fused
//! numeric column scan. Everything else — type inference, orchestration,
//! rendering — stays in Python. The Python package works without this crate
//! installed; `pysuricata[fast]` pulls it in and the accumulator factory picks
//! the native implementation when it imports successfully.
//!
//! Design rules for everything in here:
//!   * Take NumPy arrays as read-only views. No copies at the boundary.
//!   * Release the GIL for the duration of every scan longer than a few
//!     microseconds, so column-level work can actually run concurrently.
//!   * Be reproducible from an explicit seed. Never touch a global RNG.
//!   * Produce results identical to the Python reference within documented
//!     tolerance, verified by `benchmarks/accuracy.py`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

mod hashing;
mod kmv;
mod moments;
mod reservoir;

use hashing::{hash_bytes, mix64};
use kmv::Kmv;
use moments::{scan, scan_opts, Moments};
use reservoir::ReservoirL;

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// Mix a batch of already-64-bit values (row hashes, integer keys) into
/// well-distributed sketch hashes.
///
/// This replaces the single worst line in the current Python implementation:
/// `RowKMV.update_from_pandas` computes a vectorised uint64 row hash with
/// `pandas.util.hash_pandas_object`, then hands it to `KMV.add_many`, which
/// does `str(v).encode('utf-8')` and `hashlib.sha1(...)` on every one of those
/// uint64s. For a 10M-row frame that is 10M string formats and 10M SHA-1
/// digests to re-hash values that were already uniformly distributed.
#[pyfunction]
fn mix_u64_batch<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = values.as_slice()?;
    let out = py.detach(|| slice.iter().map(|&v| mix64(v)).collect::<Vec<u64>>());
    Ok(out.into_pyarray(py))
}

/// Hash an Arrow-layout UTF-8 array without copying the strings.
///
/// `data` is the contiguous value buffer and `offsets` the (n+1) offsets, i.e.
/// exactly what `polars.Series` / `pyarrow.StringArray` already hold. The
/// current Python path calls `.to_list()` first, allocating one Python string
/// object per row before it hashes anything.
#[pyfunction]
#[pyo3(signature = (data, offsets, seed = 0))]
fn hash_arrow_utf8<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    offsets: PyReadonlyArray1<'py, i64>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let buf = data.as_slice()?;
    let offs = offsets.as_slice()?;
    if offs.is_empty() {
        return Ok(Vec::<u64>::new().into_pyarray(py));
    }
    let out = py.detach(|| {
        let mut out = Vec::with_capacity(offs.len() - 1);
        for w in offs.windows(2) {
            let (s, e) = (w[0].max(0) as usize, w[1].max(0) as usize);
            let e = e.min(buf.len());
            let s = s.min(e);
            out.push(hash_bytes(&buf[s..e], seed));
        }
        out
    });
    Ok(out.into_pyarray(py))
}

/// Convenience path for a Python list of strings. Slower than
/// `hash_arrow_utf8` because extracting the list already materialises Rust
/// `String`s; use it only where an Arrow buffer is not available.
#[pyfunction]
#[pyo3(signature = (values, seed = 0))]
fn hash_str_batch<'py>(
    py: Python<'py>,
    values: Vec<String>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let out = py.detach(|| {
        values
            .iter()
            .map(|s| hash_bytes(s.as_bytes(), seed))
            .collect::<Vec<u64>>()
    });
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// KMV
// ---------------------------------------------------------------------------

#[pyclass(name = "KmvSketch", module = "pysuricata_core", from_py_object)]
#[derive(Clone)]
struct PyKmv {
    inner: Kmv,
}

#[pymethods]
impl PyKmv {
    #[new]
    #[pyo3(signature = (k = 2048))]
    fn new(k: usize) -> Self {
        Self { inner: Kmv::new(k) }
    }

    /// Ingest a batch of raw u64 values. They are mixed before insertion, so
    /// callers may pass unmixed keys (ids, timestamps) directly.
    fn offer_u64(&mut self, py: Python<'_>, values: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        let inner = &mut self.inner;
        py.detach(|| {
            for &v in slice {
                inner.offer(mix64(v));
            }
        });
        Ok(())
    }

    /// Ingest a batch of hashes that are already well distributed.
    fn offer_hashes(&mut self, py: Python<'_>, hashes: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let slice = hashes.as_slice()?;
        let inner = &mut self.inner;
        py.detach(|| inner.offer_many(slice));
        Ok(())
    }

    /// Ingest float values (finite only), for numeric-column cardinality.
    fn offer_f64(&mut self, py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        let inner = &mut self.inner;
        py.detach(|| {
            for &v in slice {
                if v.is_finite() {
                    // Normalise -0.0 to 0.0 so they are not counted as two
                    // distinct values.
                    let v = if v == 0.0 { 0.0 } else { v };
                    inner.offer(mix64(v.to_bits()));
                }
            }
        });
        Ok(())
    }

    fn merge(&mut self, other: &PyKmv) {
        self.inner.merge(&other.inner);
    }

    fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Distinct-count estimate with its relative standard error, so the report
    /// can say "~12,400 (+/-3%)" instead of asserting a false precision.
    fn estimate_with_error(&self) -> (f64, f64) {
        (self.inner.estimate(), self.inner.relative_error())
    }

    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    #[getter]
    fn seen(&self) -> u64 {
        self.inner.seen()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "KmvSketch(k={}, stored={}, seen={}, estimate={:.0}, exact={})",
            self.inner.k(),
            self.inner.len(),
            self.inner.seen(),
            self.inner.estimate(),
            self.inner.is_exact()
        )
    }

    /// Pickle support — the engine's checkpoint feature pickles accumulators.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ctor = py
            .import("pysuricata_core")?
            .getattr("_kmv_from_state")?
            .unbind();
        let args: Py<PyAny> = (
            self.inner.k(),
            self.inner.hashes().to_vec(),
            self.inner.seen(),
        )
            .into_pyobject(py)?
            .into_any()
            .unbind();
        Ok(PyTuple::new(py, [ctor, args])?.into())
    }
}

#[pyfunction]
fn _kmv_from_state(k: usize, heap: Vec<u64>, seen: u64) -> PyKmv {
    PyKmv {
        inner: Kmv::from_parts(k, heap, seen),
    }
}

// ---------------------------------------------------------------------------
// Reservoir
// ---------------------------------------------------------------------------

#[pyclass(name = "Reservoir", module = "pysuricata_core", from_py_object)]
#[derive(Clone)]
struct PyReservoir {
    inner: ReservoirL,
}

#[pymethods]
impl PyReservoir {
    #[new]
    #[pyo3(signature = (k = 20_000, seed = 0))]
    fn new(k: usize, seed: u64) -> Self {
        Self {
            inner: ReservoirL::new(k, seed),
        }
    }

    fn add_many(&mut self, py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        let inner = &mut self.inner;
        py.detach(|| inner.add_many(slice));
        Ok(())
    }

    fn merge(&mut self, other: &PyReservoir) {
        self.inner.merge(&other.inner);
    }

    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values().to_vec().into_pyarray(py)
    }

    /// Quantiles computed in one sort, matching `np.percentile(..., method="linear")`.
    fn quantiles<'py>(
        &self,
        py: Python<'py>,
        qs: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let inner = &self.inner;
        let out = py.detach(|| {
            let sorted = inner.sorted();
            qs.iter()
                .map(|&q| ReservoirL::quantile_from_sorted(&sorted, q))
                .collect::<Vec<f64>>()
        });
        out.into_pyarray(py)
    }

    #[getter]
    fn seen(&self) -> u64 {
        self.inner.seen()
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn __len__(&self) -> usize {
        self.inner.values().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Reservoir(k={}, stored={}, seen={}, scale={:.2})",
            self.inner.k(),
            self.inner.values().len(),
            self.inner.seen(),
            self.inner.scale()
        )
    }
}

// ---------------------------------------------------------------------------
// Fused numeric scan
// ---------------------------------------------------------------------------

/// Stateless single-pass scan of a chunk. Returned as a dict so the Python
/// side can diff it against the NumPy reference in tests without any binding
/// ceremony.
#[pyfunction]
#[pyo3(signature = (values, prev_last = None, gmean = true))]
fn scan_numeric<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    prev_last: Option<f64>,
    gmean: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = values.as_slice()?;
    let s = py.detach(|| scan_opts(slice, prev_last, gmean));

    let d = PyDict::new(py);
    d.set_item("n_total", s.n_total)?;
    d.set_item("n_valid", s.moments.n)?;
    d.set_item("n_nan", s.n_nan)?;
    d.set_item("n_inf", s.n_inf)?;
    d.set_item("n_zeros", s.n_zeros)?;
    d.set_item("n_negatives", s.n_negatives)?;
    d.set_item("min", s.min)?;
    d.set_item("max", s.max)?;
    d.set_item("mean", s.moments.mean)?;
    d.set_item("m2", s.moments.m2)?;
    d.set_item("m3", s.moments.m3)?;
    d.set_item("m4", s.moments.m4)?;
    d.set_item("variance", s.moments.variance())?;
    d.set_item("std", s.moments.variance().sqrt())?;
    d.set_item("skewness", s.moments.skewness())?;
    d.set_item("kurtosis", s.moments.kurtosis())?;
    d.set_item("log_sum_pos", s.log_sum_pos)?;
    d.set_item("n_pos", s.n_pos)?;
    d.set_item("all_int_like", s.all_int_like)?;
    d.set_item("mono_inc", s.mono_inc)?;
    d.set_item("mono_dec", s.mono_dec)?;
    d.set_item("last_finite", if s.has_finite { Some(s.last_finite) } else { None })?;
    Ok(d)
}

/// Stateful numeric accumulator: the drop-in replacement for
/// `pysuricata.accumulators.numeric.NumericAccumulator`'s hot path.
///
/// One `update()` call does what the Python version spreads across
/// `StreamingMoments.update`, `ReservoirSampler.add_many`, `KMV.add_many`,
/// `MonotonicityDetector.update` and several standalone NumPy reductions —
/// in a single pass, with the GIL released.
#[pyclass(name = "NumericKernel", module = "pysuricata_core")]
struct PyNumericKernel {
    moments: Moments,
    reservoir: ReservoirL,
    uniques: Kmv,
    n_total: u64,
    n_nan: u64,
    n_inf: u64,
    n_zeros: u64,
    n_negatives: u64,
    min: f64,
    max: f64,
    log_sum_pos: f64,
    n_pos: u64,
    all_int_like: bool,
    mono_inc: bool,
    mono_dec: bool,
    last_finite: Option<f64>,
    sample_size: usize,
    sketch_size: usize,
    seed: u64,
}

#[pymethods]
impl PyNumericKernel {
    #[new]
    #[pyo3(signature = (sample_size = 20_000, sketch_size = 2048, seed = 0))]
    fn new(sample_size: usize, sketch_size: usize, seed: u64) -> Self {
        Self {
            moments: Moments::new(),
            reservoir: ReservoirL::new(sample_size, seed),
            uniques: Kmv::new(sketch_size),
            n_total: 0,
            n_nan: 0,
            n_inf: 0,
            n_zeros: 0,
            n_negatives: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            log_sum_pos: 0.0,
            n_pos: 0,
            all_int_like: true,
            mono_inc: true,
            mono_dec: true,
            last_finite: None,
            sample_size,
            sketch_size,
            seed,
        }
    }

    fn update(&mut self, py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        let prev = self.last_finite;

        // Everything below runs without the GIL. Two columns updated from two
        // Python threads genuinely overlap here — which is the point.
        let (batch, ()) = py.detach(|| {
            let b = scan(slice, prev);
            self.reservoir.add_many(slice);
            for &v in slice {
                if v.is_finite() {
                    let v = if v == 0.0 { 0.0 } else { v };
                    self.uniques.offer(mix64(v.to_bits()));
                }
            }
            (b, ())
        });

        self.moments.merge(&batch.moments);
        self.n_total += batch.n_total;
        self.n_nan += batch.n_nan;
        self.n_inf += batch.n_inf;
        self.n_zeros += batch.n_zeros;
        self.n_negatives += batch.n_negatives;
        self.log_sum_pos += batch.log_sum_pos;
        self.n_pos += batch.n_pos;
        self.all_int_like &= batch.all_int_like;
        self.mono_inc &= batch.mono_inc;
        self.mono_dec &= batch.mono_dec;
        if batch.has_finite {
            if batch.min < self.min {
                self.min = batch.min;
            }
            if batch.max > self.max {
                self.max = batch.max;
            }
            self.last_finite = Some(batch.last_finite);
        }
        Ok(())
    }

    fn finalize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let n = self.moments.n;
        d.set_item("count", n)?;
        d.set_item("missing", self.n_nan)?;
        d.set_item("inf", self.n_inf)?;
        d.set_item("zeros", self.n_zeros)?;
        d.set_item("negatives", self.n_negatives)?;
        d.set_item("mean", if n > 0 { self.moments.mean } else { f64::NAN })?;
        d.set_item("variance", self.moments.variance())?;
        d.set_item("std", self.moments.variance().sqrt())?;
        d.set_item("skew", self.moments.skewness())?;
        d.set_item("kurtosis", self.moments.kurtosis())?;
        d.set_item("min", if n > 0 { self.min } else { f64::NAN })?;
        d.set_item("max", if n > 0 { self.max } else { f64::NAN })?;
        d.set_item(
            "gmean",
            if self.n_pos > 0 {
                (self.log_sum_pos / self.n_pos as f64).exp()
            } else {
                f64::NAN
            },
        )?;
        d.set_item("int_like", self.all_int_like)?;
        d.set_item("mono_inc", self.mono_inc)?;
        d.set_item("mono_dec", self.mono_dec)?;

        let (est, err) = (self.uniques.estimate(), self.uniques.relative_error());
        d.set_item("unique_est", est)?;
        d.set_item("unique_rel_err", err)?;
        d.set_item("unique_exact", self.uniques.is_exact())?;

        let sorted = self.reservoir.sorted();
        let q = |p: f64| ReservoirL::quantile_from_sorted(&sorted, p);
        d.set_item("q1", q(0.25))?;
        d.set_item("median", q(0.5))?;
        d.set_item("q3", q(0.75))?;
        d.set_item("iqr", q(0.75) - q(0.25))?;
        d.set_item("p01", q(0.01))?;
        d.set_item("p99", q(0.99))?;
        d.set_item("sample_size", sorted.len())?;
        d.set_item("sample_scale", self.reservoir.scale())?;
        d.set_item("approx", !self.uniques.is_exact() || self.reservoir.scale() > 1.0)?;
        Ok(d)
    }

    fn sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.reservoir.values().to_vec().into_pyarray(py)
    }

    /// Merge another kernel — the primitive that makes per-thread or
    /// per-partition accumulation recombinable.
    fn merge(&mut self, other: &PyNumericKernel) {
        self.moments.merge(&other.moments);
        self.reservoir.merge(&other.reservoir);
        self.uniques.merge(&other.uniques);
        self.n_total += other.n_total;
        self.n_nan += other.n_nan;
        self.n_inf += other.n_inf;
        self.n_zeros += other.n_zeros;
        self.n_negatives += other.n_negatives;
        self.log_sum_pos += other.log_sum_pos;
        self.n_pos += other.n_pos;
        self.all_int_like &= other.all_int_like;
        // Monotonicity is order-dependent; a merge of unordered partitions
        // cannot preserve it, so it degrades to "unknown" (false) rather than
        // silently claiming a property that was never checked at the seam.
        self.mono_inc = false;
        self.mono_dec = false;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NumericKernel(count={}, missing={}, mean={:.6}, sample={}/{})",
            self.moments.n,
            self.n_nan,
            self.moments.mean,
            self.reservoir.values().len(),
            self.sample_size
        )
    }

    #[getter]
    fn config(&self) -> (usize, usize, u64) {
        (self.sample_size, self.sketch_size, self.seed)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Build metadata, so a benchmark run can record what it actually measured.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("profile", if cfg!(debug_assertions) { "debug" } else { "release" })?;
    d.set_item("target_arch", std::env::consts::ARCH)?;
    d.set_item("target_os", std::env::consts::OS)?;
    d.set_item("pointer_width", usize::BITS)?;
    Ok(d)
}

#[pymodule]
fn pysuricata_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(mix_u64_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hash_arrow_utf8, m)?)?;
    m.add_function(wrap_pyfunction!(hash_str_batch, m)?)?;
    m.add_function(wrap_pyfunction!(scan_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(_kmv_from_state, m)?)?;
    m.add_class::<PyKmv>()?;
    m.add_class::<PyReservoir>()?;
    m.add_class::<PyNumericKernel>()?;
    Ok(())
}
