//! Streaming central moments with a numerically-correct pairwise merge.
//!
//! Two things this fixes relative to the pure-Python implementation:
//!
//! 1. **Correctness.** `StreamingMoments._update_vectorized` in
//!    `accumulators/algorithms.py` merges a batch's M3/M4 with formulas marked
//!    "simplified for performance". They are not equivalent to Pebay's:
//!    the M3 cross-term is `3*delta*M2_B/n_B` where it should be
//!    `3*delta*(n_A*M2_B - n_B*M2_A)/n`. The result is that skewness and
//!    kurtosis are wrong whenever more than one chunk is processed — and right
//!    when the data fits in a single chunk, which is exactly why it survived.
//!    The `merge()` method in the same class has the correct formula, so the
//!    two code paths disagree with each other.
//!
//! 2. **Memory traffic.** The Python version evaluates
//!    `deviations * deviations * deviations * deviations` — that is three
//!    full-length temporaries per batch, on top of the `np.sum` reductions and
//!    the `isfinite`/`isnan`/`isinf` masks computed upstream. For a chunk that
//!    does not fit in L2 this is the whole cost: you pay to stream the column
//!    through the cache once per pass. Fusing everything into one pass turns
//!    ~12 passes into 1.
//!
//! Reference: Pebay, P. (2008), "Formulas for Robust, One-Pass Parallel
//! Computation of Covariances and Arbitrary-Order Statistical Moments",
//! Sandia Report SAND2008-6212.

#[derive(Clone, Copy, Debug, Default)]
pub struct Moments {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
}

impl Moments {
    pub fn new() -> Self {
        Self::default()
    }

    /// Welford/Pebay single-value update. Stable, but one value at a time.
    #[inline(always)]
    pub fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// Pebay's pairwise merge. This is the formula the Python fast path is
    /// missing; it is what makes chunked results identical to unchunked ones.
    pub fn merge(&mut self, other: &Moments) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }

        let na = self.n as f64;
        let nb = other.n as f64;
        let n = na + nb;

        let delta = other.mean - self.mean;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d2 * d2;

        let m4 = self.m4
            + other.m4
            + d4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * d2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n;

        let m3 = self.m3 + other.m3 + d3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n;

        let m2 = self.m2 + other.m2 + d2 * na * nb / n;

        self.mean += delta * nb / n;
        self.m2 = m2;
        self.m3 = m3;
        self.m4 = m4;
        self.n = n as u64;
    }

    /// Sample variance (n-1 denominator).
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    /// Population skewness g1 = m3/n / (m2/n)^1.5.
    ///
    /// Note the denominator uses the *population* second moment, not the
    /// sample variance. The Python version divides by `variance**1.5` where
    /// `variance` is the n-1 form, which biases skewness low by a factor of
    /// ((n-1)/n)^1.5 — about 0.15% at n=1000, but it never converges away
    /// because it is a systematic scale error, not noise.
    pub fn skewness(&self) -> f64 {
        if self.n < 3 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        let pop_var = self.m2 / n;
        (self.m3 / n) / pop_var.powf(1.5)
    }

    /// Excess kurtosis g2 = m4/n / (m2/n)^2 - 3.
    pub fn kurtosis(&self) -> f64 {
        if self.n < 4 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        let pop_var = self.m2 / n;
        (self.m4 / n) / (pop_var * pop_var) - 3.0
    }
}

/// Everything a numeric column needs from one scan of a chunk.
///
/// The pure-Python path computes these with separate NumPy calls:
/// `isfinite`, `isnan`, `isinf`, `sum`, `mean`, `>0` mask, `log`, `sum`,
/// `min`, `max`, three `deviations*` temporaries, `argpartition` x2, and a
/// `digitize`. Each one streams the array from memory again. This struct is
/// filled by a single sequential pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchScan {
    pub moments: Moments,
    pub n_total: u64,
    pub n_nan: u64,
    pub n_inf: u64,
    pub n_zeros: u64,
    pub n_negatives: u64,
    pub min: f64,
    pub max: f64,
    pub log_sum_pos: f64,
    pub n_pos: u64,
    pub all_int_like: bool,
    /// Monotonicity across the batch, given the previous batch's last value.
    pub mono_inc: bool,
    pub mono_dec: bool,
    pub last_finite: f64,
    pub has_finite: bool,
}

/// Tile size for the blocked scan. 4096 f64 = 32 KiB, so a tile plus its
/// accumulators sits inside a typical 48 KiB L1d. The second pass over a tile
/// therefore reads from L1, not from DRAM — this is the whole reason a
/// two-pass-per-tile algorithm beats a one-pass-over-everything one.
const TILE: usize = 4096;

/// Extract mantissa in [0.5, 1) and a base-2 exponent, without libm.
///
/// Used to renormalise a running product so the geometric mean can be
/// accumulated with multiplications instead of one `ln()` per element.
/// `ln()` costs roughly 10 ns/value; a multiply costs a fraction of a
/// nanosecond and pipelines.
#[inline(always)]
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i32;
    if raw == 0 {
        // Subnormal (or zero): scale up by 2^54 and retry.
        let scaled = x * (1u64 << 54) as f64;
        let b = scaled.to_bits();
        let r = ((b >> 52) & 0x7ff) as i32;
        let m = f64::from_bits((b & !(0x7ffu64 << 52)) | (1022u64 << 52));
        return (m, r - 1022 - 54);
    }
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, raw - 1022)
}

/// State for the running geometric-mean product.
#[derive(Clone, Copy, Default)]
struct LogProduct {
    /// Four independent chains so the multiplies pipeline instead of
    /// serialising on a single 4-cycle dependency.
    p: [f64; 4],
    exp2: i64,
    n: u64,
}

impl LogProduct {
    fn new() -> Self {
        Self {
            p: [1.0; 4],
            exp2: 0,
            n: 0,
        }
    }

    #[inline(always)]
    fn renormalise(&mut self) {
        for i in 0..4 {
            if self.p[i] != 0.0 && self.p[i].is_finite() {
                let (m, e) = frexp(self.p[i]);
                self.p[i] = m;
                self.exp2 += e as i64;
            }
        }
    }

    /// Natural log of the product of everything pushed.
    fn ln_sum(&self) -> f64 {
        let mut acc = 0.0;
        for i in 0..4 {
            if self.p[i] > 0.0 {
                acc += self.p[i].ln();
            }
        }
        acc + (self.exp2 as f64) * std::f64::consts::LN_2
    }
}

/// Blocked, fused scan over a chunk of f64.
///
/// Structure, and why:
///
/// * The array is walked in 32 KiB tiles. Within a tile there are two passes,
///   but both hit L1, so the cost that matters — DRAM traffic — is paid once.
/// * Pass A is branch-free enough to auto-vectorise: min/max, the four
///   counters, monotonicity via adjacent compares, int-likeness, and the
///   running product for the geometric mean.
/// * Pass B computes the central moments about the *tile* mean, which is both
///   more accurate than a streaming Welford update and fully vectorisable —
///   four independent multiply-accumulate chains with no loop-carried
///   dependency.
/// * Tiles are combined with Pebay's pairwise merge, so the result is
///   independent of `TILE` and of how the caller chunked the data.
///
/// `prev_last` carries the previous chunk's final finite value so monotonicity
/// survives chunk boundaries.
pub fn scan(values: &[f64], prev_last: Option<f64>) -> BatchScan {
    scan_opts(values, prev_last, true)
}

/// `want_gmean = false` skips the running product. Worth it: the geometric
/// mean is rarely looked at, and it is the single most expensive quantity in
/// the scan.
pub fn scan_opts(values: &[f64], prev_last: Option<f64>, want_gmean: bool) -> BatchScan {
    let mut out = BatchScan {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        all_int_like: true,
        mono_inc: true,
        mono_dec: true,
        n_total: values.len() as u64,
        ..Default::default()
    };

    let mut prev = prev_last;
    let mut logprod = LogProduct::new();

    for tile in values.chunks(TILE) {
        // Cheap, vectorisable test: is the whole tile finite? Summing the
        // absolute values and testing the result once is enough — any NaN or
        // infinity poisons the sum. In real columns this is true for almost
        // every tile, so the fast path is the one that runs.
        let mut probe = 0.0f64;
        for &x in tile {
            probe += x.abs();
        }

        let tile_moments = if probe.is_finite() {
            scan_tile_finite(tile, &mut out, &mut prev, &mut logprod, want_gmean)
        } else {
            scan_tile_general(tile, &mut out, &mut prev, &mut logprod, want_gmean)
        };
        out.moments.merge(&tile_moments);
    }

    if want_gmean && logprod.n > 0 {
        out.log_sum_pos = logprod.ln_sum();
        out.n_pos = logprod.n;
    }
    if !out.has_finite {
        out.min = f64::NAN;
        out.max = f64::NAN;
    }
    out
}

/// Fast path: every value in the tile is finite.
#[inline]
fn scan_tile_finite(
    tile: &[f64],
    out: &mut BatchScan,
    prev: &mut Option<f64>,
    logprod: &mut LogProduct,
    want_gmean: bool,
) -> Moments {
    let n = tile.len();
    if n == 0 {
        return Moments::new();
    }

    // ---- pass A: reductions, all independent, all vectorisable ----
    let mut sum = [0.0f64; 4];
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut zeros = 0u64;
    let mut negs = 0u64;
    let mut int_like = true;

    for (i, &x) in tile.iter().enumerate() {
        sum[i & 3] += x;
        mn = if x < mn { x } else { mn };
        mx = if x > mx { x } else { mx };
        zeros += (x == 0.0) as u64;
        negs += (x < 0.0) as u64;
        int_like &= x == x.trunc();
    }

    if want_gmean {
        for (i, &x) in tile.iter().enumerate() {
            if x > 0.0 {
                logprod.p[i & 3] *= x;
                logprod.n += 1;
            }
        }
        logprod.renormalise();
    }

    // Monotonicity: adjacent compares, plus the seam with the previous tile.
    let mut inc = true;
    let mut dec = true;
    if let Some(p) = *prev {
        inc &= tile[0] >= p;
        dec &= tile[0] <= p;
    }
    for w in tile.windows(2) {
        inc &= w[1] >= w[0];
        dec &= w[1] <= w[0];
    }
    out.mono_inc &= inc;
    out.mono_dec &= dec;
    *prev = Some(tile[n - 1]);

    let total: f64 = sum[0] + sum[1] + sum[2] + sum[3];
    let mean = total / n as f64;

    // ---- pass B: central moments about the tile mean, from L1 ----
    let mut m2 = [0.0f64; 4];
    let mut m3 = [0.0f64; 4];
    let mut m4 = [0.0f64; 4];
    for (i, &x) in tile.iter().enumerate() {
        let d = x - mean;
        let d2 = d * d;
        let j = i & 3;
        m2[j] += d2;
        m3[j] += d2 * d;
        m4[j] += d2 * d2;
    }

    out.min = out.min.min(mn);
    out.max = out.max.max(mx);
    out.n_zeros += zeros;
    out.n_negatives += negs;
    out.all_int_like &= int_like;
    out.has_finite = true;
    out.last_finite = tile[n - 1];

    Moments {
        n: n as u64,
        mean,
        m2: m2[0] + m2[1] + m2[2] + m2[3],
        m3: m3[0] + m3[1] + m3[2] + m3[3],
        m4: m4[0] + m4[1] + m4[2] + m4[3],
    }
}

/// Slow path: the tile contains NaN or infinity. Same algorithm, scalar, with
/// the non-finite values counted and excluded.
#[inline]
fn scan_tile_general(
    tile: &[f64],
    out: &mut BatchScan,
    prev: &mut Option<f64>,
    logprod: &mut LogProduct,
    want_gmean: bool,
) -> Moments {
    let mut buf: Vec<f64> = Vec::with_capacity(tile.len());
    for &x in tile {
        if x.is_nan() {
            out.n_nan += 1;
        } else if x.is_infinite() {
            out.n_inf += 1;
        } else {
            buf.push(x);
        }
    }
    if buf.is_empty() {
        return Moments::new();
    }
    scan_tile_finite(&buf, out, prev, logprod, want_gmean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_moments(xs: &[f64]) -> (f64, f64, f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let m2: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum();
        let m3: f64 = xs.iter().map(|x| (x - mean).powi(3)).sum();
        let m4: f64 = xs.iter().map(|x| (x - mean).powi(4)).sum();
        (mean, m2, m3, m4)
    }

    #[test]
    fn push_matches_two_pass() {
        // Deliberately skewed: a symmetric sample drives m3 to ~0, where any
        // relative tolerance is meaningless and the two-pass "reference" is
        // itself dominated by cancellation. Tolerances below are relative to
        // each moment's natural scale (m2^(p/2)), not to its own value.
        let xs: Vec<f64> = (0..10_000)
            .map(|i| {
                let u = ((i * 7919) % 10_007) as f64 / 10_007.0;
                -(1.0 - u).ln() * 3.0 + 1.0 // exponential(1/3), shifted
            })
            .collect();

        let mut m = Moments::new();
        for &x in &xs {
            m.push(x);
        }
        let (mean, m2, m3, m4) = naive_moments(&xs);

        let s = (m2 / xs.len() as f64).sqrt(); // population sd
        let n = xs.len() as f64;
        assert!((m.mean - mean).abs() / s < 1e-12, "mean {} vs {mean}", m.mean);
        assert!((m.m2 - m2).abs() / (n * s * s) < 1e-12);
        assert!((m.m3 - m3).abs() / (n * s.powi(3)) < 1e-10);
        assert!((m.m4 - m4).abs() / (n * s.powi(4)) < 1e-10);

        // Skewness of an exponential distribution is 2. Sanity-check that the
        // sign and magnitude are right, not just self-consistent.
        assert!(
            (m.skewness() - 2.0).abs() < 0.15,
            "skewness {} should be near 2",
            m.skewness()
        );
    }

    #[test]
    fn merge_equals_single_pass() {
        // This is the invariant the Python fast path violates: profiling the
        // same data in 1 chunk vs 7 chunks must give the same moments.
        let xs: Vec<f64> = (0..9_999).map(|i| ((i * 31 % 977) as f64).sin() * 100.0 + 5.0).collect();

        let mut whole = Moments::new();
        for &x in &xs {
            whole.push(x);
        }

        let mut chunked = Moments::new();
        for chunk in xs.chunks(1_429) {
            let mut part = Moments::new();
            for &x in chunk {
                part.push(x);
            }
            chunked.merge(&part);
        }

        assert_eq!(whole.n, chunked.n);
        assert!((whole.mean - chunked.mean).abs() < 1e-9);
        assert!((whole.m2 - chunked.m2).abs() / whole.m2 < 1e-10);
        assert!((whole.m3 - chunked.m3).abs() / whole.m3.abs().max(1.0) < 1e-8);
        assert!((whole.m4 - chunked.m4).abs() / whole.m4 < 1e-9);
        assert!((whole.skewness() - chunked.skewness()).abs() < 1e-9);
        assert!((whole.kurtosis() - chunked.kurtosis()).abs() < 1e-9);
    }

    #[test]
    fn scan_counts_and_monotonicity() {
        let vals = vec![1.0, 2.0, f64::NAN, 3.0, f64::INFINITY, 4.0, 0.0, -1.0];
        let s = scan(&vals, None);
        assert_eq!(s.n_total, 8);
        assert_eq!(s.n_nan, 1);
        assert_eq!(s.n_inf, 1);
        assert_eq!(s.n_zeros, 1);
        assert_eq!(s.n_negatives, 1);
        assert_eq!(s.moments.n, 6); // 1,2,3,4,0,-1
        assert_eq!(s.min, -1.0);
        assert_eq!(s.max, 4.0);
        assert!(!s.mono_inc);
        assert!(!s.mono_dec);
        assert!(s.all_int_like);
    }

    #[test]
    fn scan_monotonic_across_chunks() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let sa = scan(&a, None);
        assert!(sa.mono_inc);
        let sb = scan(&b, Some(sa.last_finite));
        assert!(sb.mono_inc);

        // ...and a chunk boundary that breaks it is detected.
        let c = vec![2.0, 6.0];
        let sc = scan(&c, Some(sa.last_finite));
        assert!(!sc.mono_inc);
    }

    #[test]
    fn all_nan_chunk_is_safe() {
        let vals = vec![f64::NAN; 100];
        let s = scan(&vals, None);
        assert_eq!(s.n_nan, 100);
        assert_eq!(s.moments.n, 0);
        assert!(s.min.is_nan() && s.max.is_nan());
        assert_eq!(s.moments.variance(), 0.0);
    }
}
